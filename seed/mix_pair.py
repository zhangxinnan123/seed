"""Caller-side helpers for two-context mixture rollout.

The engine-side half is `seed.context_mix_lp.ContextMixLogitsProcessor`, which
fuses the logits of a request *pair* into the arithmetic mixture

    mu(. | y_<t) = (1 - alpha) * pi(. | c1, y_<t) + alpha * pi(. | c2, y_<t)

Here c1 is the deployable (skill-free) context and c2 the privileged
(skill-augmented) one. Both members share one set of weights and one seed, so
they decode the same token stream.

Because the mixture is arithmetic it is itself normalized, so log mu of the
sampled token is recoverable from the two per-context log-probs alone -- no
full-vocab partition function. vLLM returns log-probs computed *before* logits
processors run, which is exactly what makes those two inputs available. That
keeps the importance ratio pi_theta(.|c1) / mu exact, which is the whole reason
to mix per token instead of picking one context per rollout.

alpha = 0 reduces to plain single-context rollout bit for bit (the processor
leaves the surviving row untouched inside `ALPHA_EPS` of 0 or 1), so a run with
`algorithm.seed.mix_alpha=0.0` is byte-identical to SEED without this feature.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Sequence

from seed.context_mix_lp import ROLE_C1, ROLE_C2  # noqa: F401  (re-exported)

# Must match _ALPHA_EPS in seed/context_mix_lp.py: inside this band of 0 or 1 the
# mixture is degenerate and the second context is not worth a forward pass.
ALPHA_EPS = 1e-6

# non-tensor column carrying per-row {pair_id, role, alpha, seed} into the
# rollout worker, which turns it into per-request vLLM SamplingParams.
MIX_META_KEY = "mix_meta"


def mixture_logprob(
    lp1: Sequence[float],
    lp2: Sequence[float],
    alpha: float,
    n_fused: int,
) -> List[float]:
    """log mu for the first `n_fused` tokens, then log pi(.|c1) for the rest.

    Past a desync the pair was no longer fused, so the surviving row sampled
    from its own context alone. Switching the reported value there (instead of
    dropping the row) keeps the importance weights honest per token: a desynced
    pair degrades into an alpha=0 sample rather than a mislabeled one.
    """
    if alpha <= ALPHA_EPS:
        return list(lp1)
    log_w1 = math.log1p(-alpha)
    log_w2 = math.log(alpha)
    out: List[float] = []
    for t, a in enumerate(lp1):
        if t >= n_fused or t >= len(lp2):
            out.append(a)
            continue
        hi, lo = log_w1 + a, log_w2 + lp2[t]
        if lo > hi:
            hi, lo = lo, hi
        out.append(hi + math.log1p(math.exp(lo - hi)))
    return out


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def pair_seed(base_seed: int, pair_id: str) -> int:
    """Stable 31-bit seed shared by the two members of a pair.

    Derived from the pair id alone: fresh per pair (so the N rollouts of one
    prompt do not collapse onto the same sample) and identical across the two
    members (so they draw the same noise and stay in lockstep).
    """
    digest = hashlib.blake2b(f"{base_seed}:{pair_id}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big") % (2**31 - 1)


def task_key(text: object) -> str:
    """Stable identifier for the task an observation belongs to.

    The skill bank has to survive across training steps, and neither candidate id
    in the rollout batch does: `sample_id` is a position in the batch (so it moves
    when the dataloader shuffles) and `uid` is a fresh uuid4 per group per step.
    The task description is carried inside the observation itself and is stable,
    so key on that. Returns "" when no description can be found, which the caller
    treats as "no skill" (alpha 0 for that row).
    """
    from seed.analysis import _clean_task_description, _extract_task_description_from_text

    text = str(text or "")
    task = _extract_task_description_from_text(text)
    if not task:
        task = _clean_task_description(text)[:200]
    return task.lower()


def interleave_indices(batch_size: int) -> List[int]:
    """Indices that turn concat([c1_rows, c2_rows]) into c1_0, c2_0, c1_1, ...

    Pair members must land on the same engine, because the logits processor's
    pair state is per-engine. verl dispatches a DataProto by chunking it into
    `world_size` contiguous pieces, so adjacent rows stay together as long as
    the per-rank chunk size is even -- see the assertion in the caller.
    """
    out: List[int] = []
    for i in range(batch_size):
        out.append(i)
        out.append(batch_size + i)
    return out
