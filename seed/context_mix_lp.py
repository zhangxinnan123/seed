"""Ported verbatim from RLCSD (`src/context_mix_lp.py`); see `seed/mix_pair.py`
for the SEED-side pairing helpers and `agent_system/multi_turn_rollout/rollout_loop.py`
for the caller.

vLLM V1 batch-level logits processor: per-token arithmetic mixture of two
contexts that share one set of weights.

The caller submits each rollout as TWO requests -- one per context -- tied
together by a pair id (see `mix_pair.py` for the builder):

    context1 -> extra_args={"mix_pair_id": p, "mix_role": 1, "mix_alpha": a}
    context2 -> extra_args={"mix_pair_id": p, "mix_role": 2, "mix_alpha": a}

`mix_alpha` is the weight on context2. At each decode step, for every pair whose
BOTH members are present in the current batch, both rows' logits are overwritten
with the log of the ARITHMETIC mixture

    mu = (1 - alpha) * softmax(logits_1) + alpha * softmax(logits_2)

which is evaluated in log space as

    logaddexp( log(1-alpha) + log_softmax(logits_1),
               log(alpha)   + log_softmax(logits_2) )

The two forms are algebraically identical, but the log-space one needs no
epsilon floor (the naive form has to add ~1e-30 inside the log, which clamps
tail log-probs at about -69 instead of their true value) and it collapses six
elementwise passes over a (n_pairs, vocab) tensor into a handful. So it is both
cheaper and strictly more faithful.

Arithmetic (not geometric) mixing is deliberate: the mixture of two normalized
distributions is itself normalized, so log mu of the sampled token is
recoverable downstream from the two per-context log-probs alone (logaddexp
again, see `mix_pair.mixture_logprob`), with no full-vocab partition function.
That keeps the PPO importance ratio exact. It is also why temperature must stay
at 1.0 -- see "Sampling-param contract" below.

Both rows receive identical logits and are given identical seeds by the caller,
so they sample the same token and the pair stays in lockstep.


Degenerate alpha: exact reduction to single-context sampling
-----------------------------------------------------------
alpha=0 must reproduce plain sampling on context1 *bit for bit*, and alpha=1
plain sampling on context2. That is stronger than "same distribution": going
through the fusion path would replace the surviving row's logits with
log_softmax(logits), which is the same distribution but not the same floats --
softmax(z) and softmax(log_softmax(z)) differ in the last bits, which flips the
argmax on near-ties. So the degenerate cases take a separate path that leaves
the surviving row **completely untouched** and only copies its logits onto the
other row (to keep the pair in lockstep so the caller's bookkeeping still
works). The LP also consumes no randomness of its own, so the surviving row's
sampling is byte-identical to a request that never met this processor.

The test for this lives in `test_alpha_extremes.py`.

The `_ALPHA_EPS` band matters: an alpha annealed to 1e-8 would otherwise take
the fusion path and lose that bit-exactness. Snapping |alpha| < 1e-6 to the
degenerate path costs O(1e-6) of distributional error, far below bf16 noise.


Sampling-param contract
-----------------------
Everything vLLM does around this LP is checked at request-admission time,
because a silently-invalid mixture is worse than a loud crash:

  * temperature must be 1.0 (or greedy). vLLM divides by temperature *after*
    this LP, so with T != 1 the behavior distribution becomes
    softmax(log mu / T), whose log is no longer recoverable from the two
    per-context log-probs without a partition function. Set
    MIXLP_ALLOW_TEMPERATURE=1 to opt in anyway: the LP then pre-multiplies the
    fused log-probs by T so the engine's later division restores a mixture of
    *tempered* per-context distributions -- correct as a behavior policy, but
    you lose the cheap downstream log mu recovery.
  * penalties must be off. `apply_penalties` runs after this LP and keys off
    `prompt_token_ids`, which differ between the two contexts -- so a penalty
    both corrupts mu and breaks lockstep.
  * top_p/top_k/min_p must be disabled. They truncate the mixture after the
    fact, so the sampled distribution would no longer be mu.
  * logit_bias / allowed_token_ids / bad_words must be unset. They run *before*
    this LP, so they would redefine each per-context distribution.
  * seed is required (lockstep depends on the two rows drawing identical noise)
    and n must be 1 (vLLM expands n>1 into child requests that share
    extra_args, which would collide on the pair id).


Scheduling contract
-------------------
Lockstep additionally requires that the two rows draw the *same* noise. vLLM
seeds a per-request generator when the request enters the persistent batch and
advances it once per step the row is in the batch -- including steps whose
sampled token is discarded because prefill is still in progress. So run with

    enable_chunked_prefill=False, max_num_batched_tokens>=max_model_len

which makes the V1 scheduler defer a request that does not fit the token budget
instead of splitting it (v1/core/sched/scheduler.py). Every prefill is then
atomic, "output token t" always uses "generator draw t", and the RNG stream
becomes a function of (seed, t) alone -- independent of batch composition.

What that does NOT fix: if the two members are admitted on *different* steps,
the leading one produces a token while its partner is still waiting, and the
resulting length offset is permanent (every row in the batch appends exactly one
token per step, so the gap can never close). Such pairs are detected and marked
dead; the caller is expected to drop and re-run them.

Also required: no speculative decoding (it breaks the row<->request mapping this
LP relies on), and with multiple engine replicas both members of a pair must be
routed to the same replica, since this state is per-engine.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

# Roles, i.e. which context a row carries.
ROLE_C1 = 1
ROLE_C2 = 2

# Snap-to-degenerate band; see the module docstring.
_ALPHA_EPS = 1e-6
# Mirrors vllm.v1.sample.sampler._SAMPLING_EPS (below this, a row is greedy).
_SAMPLING_EPS = 1e-5

# Set MIXLP_STRICT=1 to raise on pair desync instead of dropping the pair.
_STRICT = os.environ.get("MIXLP_STRICT", "0") == "1"
# See "Sampling-param contract" in the module docstring.
_ALLOW_TEMPERATURE = os.environ.get("MIXLP_ALLOW_TEMPERATURE", "0") == "1"

# How many desync diagnostics to keep.
_MAX_BREAK_RECORDS = 64


class _Row:
    __slots__ = ("pid", "role", "alpha", "scale", "out")

    def __init__(self, pid: str, role: int, alpha: float, scale: float,
                 out: list[int]):
        self.pid = pid
        self.role = role
        self.alpha = alpha
        self.scale = scale
        # Live reference to the request's running output-token list.
        self.out = out


class _Pair:
    __slots__ = ("row1", "row2", "dead")

    def __init__(self):
        self.row1: Optional[int] = None
        self.row2: Optional[int] = None
        # Sticky: once the two members' token streams diverge, the offset can
        # never be closed, so there is nothing to re-check.
        self.dead = False

    def slot(self, role: int) -> Optional[int]:
        return self.row1 if role == ROLE_C1 else self.row2

    def set_slot(self, role: int, row: Optional[int]) -> None:
        if role == ROLE_C1:
            self.row1 = row
        else:
            self.row2 = row

    def empty(self) -> bool:
        return self.row1 is None and self.row2 is None


def validate_mix_params(params, *, where: str = "request") -> float:
    """Check a mix request's sampling params; return the fused-logit scale.

    Shared with the caller-side builder in `mix_pair.py` so a bad config fails
    before the engine ever sees it. Returns the factor the fused log-probs must
    be multiplied by to survive vLLM's later division by temperature (1.0 in the
    only fully-supported configuration).
    """
    bad: list[str] = []

    if getattr(params, "n", 1) != 1:
        bad.append(f"n={params.n} (must be 1: vLLM expands n>1 into child "
                   "requests that share extra_args, colliding on the pair id)")
    if getattr(params, "seed", None) is None:
        bad.append("seed=None (lockstep needs both rows to draw identical "
                   "noise, which requires a per-request seed)")

    # Penalties run after this LP and key off prompt_token_ids, which differ
    # between the two contexts -- they both corrupt mu and break lockstep.
    _PENALTY_WHY = " (penalties are keyed on the prompt, which differs per context)"
    if getattr(params, "presence_penalty", 0.0) != 0.0:
        bad.append(f"presence_penalty={params.presence_penalty}" + _PENALTY_WHY)
    if getattr(params, "frequency_penalty", 0.0) != 0.0:
        bad.append(f"frequency_penalty={params.frequency_penalty}" + _PENALTY_WHY)
    if getattr(params, "repetition_penalty", 1.0) != 1.0:
        bad.append(f"repetition_penalty={params.repetition_penalty}" + _PENALTY_WHY)

    if getattr(params, "top_p", 1.0) != 1.0:
        bad.append(f"top_p={params.top_p} (truncates the mixture)")
    if getattr(params, "top_k", 0) not in (0, -1):
        bad.append(f"top_k={params.top_k} (truncates the mixture)")
    if getattr(params, "min_p", 0.0) != 0.0:
        bad.append(f"min_p={params.min_p} (truncates the mixture)")

    if getattr(params, "logit_bias", None):
        bad.append("logit_bias (applied before this LP, redefines each "
                   "per-context distribution)")
    if getattr(params, "allowed_token_ids", None):
        bad.append("allowed_token_ids (applied before this LP)")
    if getattr(params, "bad_words", None):
        bad.append("bad_words (applied before this LP)")

    temp = float(getattr(params, "temperature", 1.0))
    scale = 1.0
    if temp < _SAMPLING_EPS or abs(temp - 1.0) <= 1e-6:
        # Greedy rows: vLLM either skips temperature entirely (all_greedy) or
        # clamps the divisor to 1.0, so no rescale is needed either way.
        scale = 1.0
    elif _ALLOW_TEMPERATURE:
        scale = temp
    else:
        bad.append(f"temperature={temp} (must be 1.0; set "
                   "MIXLP_ALLOW_TEMPERATURE=1 to mix tempered distributions "
                   "instead, at the cost of cheap log-mu recovery)")

    if bad:
        raise ValueError(
            f"{where}: incompatible sampling params for two-context mixing: "
            + "; ".join(bad))
    return scale


class ContextMixLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config=None, device=None, is_pin_memory=False):
        self.device = device
        self._rows: dict[int, _Row] = {}
        self._pairs: dict[str, _Pair] = {}
        # diagnostics
        self.n_steps = 0
        self.n_fused_pair_steps = 0
        self.n_incomplete_pair_steps = 0
        self.n_out_of_range_pair_steps = 0
        self.n_dead_pairs = 0
        self.breaks: list[dict] = []

    # ---------------- required interface ----------------

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if batch_update is None:
            return
        # interface.py requires this order: removed -> added -> moved
        for row in batch_update.removed:
            self._forget(row)

        for row, params, _prompt_ids, output_tok_ids in batch_update.added:
            # A row may replace an existing one at the same index.
            self._forget(row)
            role, pid, alpha = _read_extra_args(params)
            if role is None:
                # Not a mix request -> leave it entirely alone.
                continue
            scale = validate_mix_params(params, where=f"mix pair {pid} role {role}")
            pair = self._pairs.setdefault(pid, _Pair())
            live = pair.slot(role)
            if live is not None:
                raise RuntimeError(
                    f"mix pair {pid} already has a live row for role {role} "
                    f"(row {live}, new row {row}): pair ids must be unique "
                    "among concurrent requests")
            self._rows[row] = _Row(pid, role, alpha, scale, output_tok_ids)
            pair.set_slot(role, row)
            self._check_pair_agreement(pid, pair)

        for i1, i2, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                a, b = self._rows.get(i1), self._rows.get(i2)
                self._forget(i1)
                self._forget(i2)
                self._rebind(i1, b)
                self._rebind(i2, a)
            else:  # UNIDIRECTIONAL: i1 -> i2
                a = self._rows.get(i1)
                self._forget(i1)
                self._forget(i2)
                self._rebind(i2, a)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        self.n_steps += 1
        if not self._pairs:
            return logits

        n_rows = logits.shape[0]
        # alpha=0 / alpha=1: copy the surviving row onto the other one, never
        # the reverse, so the surviving row stays bit-identical.
        copy_1_to_2: list[tuple[int, int]] = []
        copy_2_to_1: list[tuple[int, int]] = []
        # (alpha, scale) -> (rows_c1, rows_c2). Grouping keeps the log-weights
        # scalar instead of an (n, 1) broadcast tensor.
        groups: dict[tuple[float, float], tuple[list[int], list[int]]] = {}
        n_pairs_here = 0

        for pid, pair in self._pairs.items():
            if pair.dead:
                continue
            r1, r2 = pair.row1, pair.row2
            if r1 is None or r2 is None:
                self.n_incomplete_pair_steps += 1
                continue
            if r1 >= n_rows or r2 >= n_rows:
                # Our row bookkeeping disagrees with the batch we were handed.
                self.n_out_of_range_pair_steps += 1
                if _STRICT:
                    raise RuntimeError(
                        f"mix pair {pid}: rows ({r1}, {r2}) outside batch of "
                        f"{n_rows}")
                continue
            row1, row2 = self._rows[r1], self._rows[r2]
            o1, o2 = row1.out, row2.out
            # O(1) lockstep check. A full list compare would be O(t) per pair
            # per step -- i.e. O(t^2) over a rollout, inside the hot path -- and
            # detects nothing earlier: a divergence at step t shows up as a
            # last-token mismatch at step t+1 either way.
            if len(o1) != len(o2) or (o1 and o1[-1] != o2[-1]):
                pair.dead = True
                self.n_dead_pairs += 1
                if len(self.breaks) < _MAX_BREAK_RECORDS:
                    self.breaks.append({"pair_id": pid, "len_c1": len(o1),
                                        "len_c2": len(o2),
                                        "step": self.n_steps})
                if _STRICT:
                    raise RuntimeError(
                        f"mix pair {pid} desynced at lengths "
                        f"({len(o1)}, {len(o2)})")
                continue

            n_pairs_here += 1
            if row1.alpha <= _ALPHA_EPS:
                copy_1_to_2.append((r1, r2))
            elif row1.alpha >= 1.0 - _ALPHA_EPS:
                copy_2_to_1.append((r1, r2))
            else:
                g = groups.setdefault((row1.alpha, row1.scale), ([], []))
                g[0].append(r1)
                g[1].append(r2)

        if not n_pairs_here:
            return logits

        dev = logits.device
        # keep_at is the position in the (r1, r2) tuple of the row that must stay
        # bit-identical; the other row is overwritten with a copy of it.
        for src, keep_at in ((copy_1_to_2, 0), (copy_2_to_1, 1)):
            if not src:
                continue
            keep = _idx([p[keep_at] for p in src], dev)
            over = _idx([p[1 - keep_at] for p in src], dev)
            logits.index_copy_(0, over, logits.index_select(0, keep))

        for (alpha, scale), (g1, g2) in groups.items():
            i1 = _idx(g1, dev)
            i2 = _idx(g2, dev)
            # log[(1-a) * softmax(z1) + a * softmax(z2)]
            #   == logaddexp(log(1-a) + log_softmax(z1), log(a) + log_softmax(z2))
            l1 = torch.log_softmax(logits.index_select(0, i1), dim=-1)
            l2 = torch.log_softmax(logits.index_select(0, i2), dim=-1)
            if scale != 1.0:
                # Mix the *tempered* distributions, then pre-multiply so the
                # engine's later div_(temperature) lands back on log mu_T.
                l1 /= scale
                l2 /= scale
                l1 -= torch.logsumexp(l1, dim=-1, keepdim=True)
                l2 -= torch.logsumexp(l2, dim=-1, keepdim=True)
            l1 += math.log1p(-alpha)
            l2 += math.log(alpha)
            fused = torch.logaddexp(l1, l2)
            if scale != 1.0:
                fused *= scale
            logits.index_copy_(0, i1, fused)
            logits.index_copy_(0, i2, fused)

        self.n_fused_pair_steps += n_pairs_here
        return logits

    # ---------------- helpers ----------------

    def _forget(self, row: int) -> None:
        info = self._rows.pop(row, None)
        if info is None:
            return
        pair = self._pairs.get(info.pid)
        if pair is None:
            return
        if pair.slot(info.role) == row:
            pair.set_slot(info.role, None)
        if pair.empty():
            # Both members gone: drop the entry so a reused pair id starts
            # clean (and a dead pair does not poison the reuse).
            self._pairs.pop(info.pid, None)

    def _rebind(self, row: int, info: Optional[_Row]) -> None:
        if info is None:
            return
        self._rows[row] = info
        self._pairs.setdefault(info.pid, _Pair()).set_slot(info.role, row)

    def _check_pair_agreement(self, pid: str, pair: _Pair) -> None:
        if pair.row1 is None or pair.row2 is None:
            return
        a, b = self._rows[pair.row1], self._rows[pair.row2]
        if a.alpha != b.alpha or a.scale != b.scale:
            raise RuntimeError(
                f"mix pair {pid}: members disagree "
                f"(alpha {a.alpha} vs {b.alpha}, scale {a.scale} vs {b.scale})")

    def stats(self) -> dict:
        return {
            "steps": self.n_steps,
            "fused_pair_steps": self.n_fused_pair_steps,
            "incomplete_pair_steps": self.n_incomplete_pair_steps,
            "out_of_range_pair_steps": self.n_out_of_range_pair_steps,
            "dead_pairs": self.n_dead_pairs,
            "breaks": self.breaks,
        }


def _idx(rows: list[int], device) -> torch.Tensor:
    return torch.as_tensor(rows, device=device, dtype=torch.long)


def _read_extra_args(params) -> tuple[Optional[int], str, float]:
    """Return (role, pair_id, alpha); role is None for non-mix requests."""
    ea = getattr(params, "extra_args", None) or {}
    pid = ea.get("mix_pair_id")
    raw_role = ea.get("mix_role")
    if pid is None and raw_role is None:
        return None, "", 0.0
    if pid is None or raw_role is None:
        raise ValueError(f"mix request needs both mix_pair_id and mix_role, "
                         f"got {ea!r}")
    try:
        role = int(raw_role)
    except (TypeError, ValueError):
        raise ValueError(f"mix_role must be {ROLE_C1} or {ROLE_C2}, "
                         f"got {raw_role!r}") from None
    if role not in (ROLE_C1, ROLE_C2):
        raise ValueError(f"mix_role must be {ROLE_C1} or {ROLE_C2}, got {role}")
    alpha = float(ea.get("mix_alpha", 0.5))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"mix_alpha must be in [0, 1], got {alpha}")
    return role, str(pid), alpha
