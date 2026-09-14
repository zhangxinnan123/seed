# Two-Context Mixture Rollout for SEED

Record of everything added on top of the upstream SEED repo (baseline commit `2cf2fad`).
Everything here is **off by default**: with `algorithm.seed.mix_alpha=0.0` (the default)
none of the new code paths execute and runs are identical to upstream SEED.

## 1. What this adds, in one paragraph

Upstream SEED uses hindsight skills only to *re-score* already-sampled actions: after an
episode ends the analyzer writes an `episode_skill`, the skill is spliced into that
episode's own prompts, and the same action tokens are re-scored to produce
`teacher_log_prob` for the gated OPD loss. The skill never influences behaviour. This
change additionally lets the skill influence **sampling**: every token is drawn from the
per-token arithmetic mixture of two contexts that share one set of weights,

```
mu(. | y_<t) = (1 - alpha) * pi(. | c_clean, y_<t) + alpha * pi(. | c_skill, y_<t)
```

with `c_clean` the deployable prompt and `c_skill` the same prompt plus an
`**Episode-Level Skill**` block. Because the mixture is arithmetic it is itself
normalised, so `log mu` of the sampled token is recoverable from the two per-context
log-probs by `logaddexp` alone — which keeps the PPO importance ratio exact.

Provenance: the logits processor is ported verbatim from the RLCSD project
(`src/context_mix_lp.py`); the SEED-side pairing, skill bank, importance correction and
tests are new.

## 2. Files

### New

| File | Contents |
|---|---|
| `seed/context_mix_lp.py` | vLLM V1 batch-level logits processor. Fuses paired rows into `log mu` in log space; bit-exact degenerate path at `alpha≈0/1`; O(1) desync detection. Ported verbatim from RLCSD. |
| `seed/mix_pair.py` | Caller-side helpers: `mixture_logprob`, `common_prefix_len`, `pair_seed`, `interleave_indices`, `task_key`, `ALPHA_EPS`, `MIX_META_KEY`. |
| `tests/trainer/ppo/test_mix_pair.py` | 24 tests (see §7). |

### Modified

| File | Change |
|---|---|
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | Registers the logits processor when `rollout.mix_enable=True`; `_build_mix_sampling_params` turns a per-row `mix_meta` column into per-request `SamplingParams` (`extra_args`, shared `seed`, truncation disabled), validated by `validate_mix_params`. |
| `agent_system/multi_turn_rollout/rollout_loop.py` | `_mix_alpha` (schedule + the two hard guards), `_build_mix_pair_batch` (builds `c_skill`, interleaves, asserts DP chunk parity), `_fold_mix_pair_output` (vectorised `log mu` with cumprod desync fallback), `set_mix_skills` / `update_mix_skills` / `_mix_skill_for` / `pop_mix_stats`, and the `mix_row_alpha` column. |
| `verl/trainer/ppo/ray_trainer.py` | Publishes each step's skills as `meta_info["seed_mix_skills"]` (with success flags, success-priority collapse); merges them into the collector inside `_merge_async_seed_teacher_signals`; folds `mix/*` into step metrics. |
| `verl/trainer/ppo/core_algos.py` | `compute_mix_is_weight` (importance correction); `compute_opd_loss` gains `gate_enable` for the ungated ablation. |
| `verl/workers/actor/dp_actor.py` | Applies the importance weight to advantages before the policy loss; passes `gate_enable`; logs `actor/mix_is_*`. |
| `verl/trainer/config/ppo_trainer.yaml` | The new keys in §3. |

## 3. Config keys

All default to the upstream behaviour.

| Key | Default | Meaning |
|---|---|---|
| `algorithm.seed.mix_alpha` | `0.0` | Weight on the skill-augmented context. `0.0` disables everything below. |
| `algorithm.seed.mix_alpha_end` | `null` | Linear anneal target; `null` = constant `mix_alpha`. |
| `algorithm.seed.mix_alpha_anneal_steps` | `null` | Anneal horizon; falls back to `trainer.total_training_steps`, then `trainer.total_epochs`. |
| `algorithm.seed.mix_skill_overwrite` | `success_priority` | Bank overwrite policy: `success_priority` (a failed episode's skill never displaces a successful one) or `latest` (unconditional). |
| `actor_rollout_ref.rollout.mix_enable` | `False` | Register the logits processor with the engine. Must be `True` whenever `mix_alpha > 0`. |
| `actor_rollout_ref.actor.mix_is_log_clip` | `2.0` | Symmetric clamp (log space) on the importance weight. |
| `actor_rollout_ref.actor.opd_gate_enable` | `True` | `False` = ungated distillation (`gate == 1`), the ablation for the OPD confidence gate. |

Debug only: `SEED_MIX_DEBUG_SKILL` forces a synthetic skill on every row so the pair
mechanism can be smoke-tested before the bank fills. Not a training path.

## 4. Two invariants, enforced in code and tested

1. **Evaluation never mixes.** `_mix_alpha` returns `0.0` for any `phase != "train"`, and
   validation is dispatched with `phase="val"` (`rollout_loop.py`, `multi_turn_loop`). So
   `val/*` metrics are always single-context, no skills, no analyzer — matching deployment.
   A second, independent net: the processor requires `temperature=1.0`, while validation
   uses `val_kwargs.temperature=0.4`, so a mixed validation batch would raise rather than
   silently mix.
2. **Mixture is SEED-only.** `mix_alpha > 0` with any `algorithm.adv_estimator != seed`
   raises. The privileged context is built from SEED's hindsight skills and the
   importance correction lives on the SEED path, so mixing outside SEED is meaningless.

Corollary for reading metrics: in a mixture run `critic/score/mean` is measured on
`mu`-sampled (privilege-assisted) trajectories and is **not** a capability number. Only
`val/*` is comparable across arms.

## 5. The skill bank — and why it is a real departure from SEED

Mixture sampling needs `c_skill` *before* the episode runs, but hindsight skills only
exist *after* it ends. Upstream SEED resolves this by never using skills for acting. To
make mixture possible at all, a carry-over bank was added:

```
step N     rollout(task A) -> analyzer -> skill_A^N -> bank[task_key(A)] = skill_A^N
                                                                 |
step N+k   task A resampled -> c_skill uses skill_A^N  <----------+
```

- Key: `task_key()` = the task description extracted from the observation, lowercased.
  Neither `sample_id` (batch position) nor `uid` (per-step uuid4) is stable across steps.
- Value: `{"skill": str, "success": bool}` — **one** entry per task.
- Merged on the main thread inside `_merge_async_seed_teacher_signals`, i.e. after the
  step's rollout, so the earliest a skill can be used is the next step, and in practice
  much later (the task must be resampled).

**This is an algorithmic change, not an implementation detail, and must be stated as such
in any writeup.** The paper claims SEED needs "no analyzer, no skill bank, no retrieval
module" — that remains true at inference (evaluation is single-context) but is no longer
true of training, which now carries persistent cross-step state and acts on a skill
distilled from a *different* attempt at the same task under an *older* policy.

Known bank limitations:

- 8 rollouts produce 8 skills but only 1 is kept. `success_priority` prevents the worst
  case (an avoidance rule displacing a workflow) but among equals it is still "last in
  batch order wins", which is unrelated to quality. Success and failure skills are
  complementary and only one survives.
- Not persisted. It lives in the driver process, so it is lost on job end and is **not**
  restored by `resume_mode=auto`; a restarted run rebuilds it from empty.
- Exact string keys: paraphrases of the same task (`put two pot in shelf.` vs
  `find two pot and put them in shelf.`) are distinct keys. Measured hit rate 75–81%.
- No staleness decay: a skill persists until its task is resampled, however far the
  policy has moved on.

## 6. Loss-side changes

**Importance correction** (`compute_mix_is_weight`, self-gated on the `mix_row_alpha`
column so it only ever touches mixture runs):

```
w = exp(clamp(old_log_probs - log mu, ±mix_is_log_clip))
pg_loss uses advantages * w
```

Applied by scaling advantages, which is *exact* rather than an approximation: `w > 0`
commutes with the `min()`/`clip()` of the PPO objective, so it equals a per-token weight
on the policy-gradient term (pinned by a test). Rows with `alpha == 0` get `w == 1`
exactly — without that carve-out, unmixed rows would import the vLLM-vs-FSDP log-prob
discrepancy (tracked upstream as `rollout_probs_diff_max`) as a spurious weight. KL,
entropy and OPD are deliberately left uncorrected.

**OPD gate ablation**: `opd_gate_enable=False` replaces `gate = sigmoid(beta * gap)` with
`1`. Note the gated mean gate runs ≈0.48 in practice, so at a fixed `opd_loss_coef` the
ungated loss is ~2x stronger — a strength-matched arm needs `opd_loss_coef≈0.005`.

## 7. Tests

`pytest tests/trainer/ppo` — 81 passed (24 new + 57 pre-existing, no regressions).
The new tests cover: `log mu` against a probability-space reference across alphas;
`alpha=0` identity (both scalar and vectorised paths); desync fallback; vectorised fold
vs scalar reference; eval-never-mixes; SEED-estimator-only enforcement; alpha schedule
(midpoint, `ALPHA_EPS` snap, hold-past-horizon, constant-by-default); `task_key`
stability across differing history; bank merge semantics and all four success/failure
overwrite cases; importance-weight identity on unmixed rows, clamping, mask handling,
and advantage-scaling equivalence; ungated OPD.

## 8. Metrics added

`mix/alpha_mean`, `mix/active_row_ratio`, `mix/fused_token_ratio`, `mix/in_sync_ratio`,
`mix/logmu_minus_lp1_mean`, `mix/bank_size`, `mix/bank_success_ratio`, `mix/env_steps`,
`actor/mix_is_weight_mean`, `actor/mix_is_weight_max`, `actor/mix_is_log_ratio_mean`,
`actor/mix_is_clipped_ratio`.

Health reading: `in_sync_ratio` should sit at ~1.0 (pair lockstep holding);
`active_row_ratio` near 0 means `task_key` is not matching and the bank is inert;
`mix_is_clipped_ratio` above a few percent means alpha is too aggressive for the clamp.

## 9. Launchers

`run_seed_alfworld_mix.sbatch` (alpha=0.3) plus `_01/_02/_04/_05`, `_anneal`
(0.3 -> 0), `_nogate`, `_smoke` (2 steps, synthetic skill); `run_alfworld_grpo_baseline.sbatch`
(wraps `examples/grpo_trainer/run_alfworld.sh`); `run_alfworld_rollouts.sbatch`
(Stage-1 rollout stage only, via `--stop-after-baseline-rollouts`).

Cluster-specific workarounds baked into these (SFM-P5):

- `logs/inner_launch*.sh` unsets `ROCR_VISIBLE_DEVICES` **inside** the srun step, because
  Slurm's GRES plugin re-exports it per step and `verl.single_controller.base.worker`
  refuses to start when it is set alongside `CUDA_VISIBLE_DEVICES`.
- `TMPDIR`/triton/inductor caches are forced onto `/fsx`; the login node's root
  filesystem is 100% full.
- Ray's temp dir is pinned to `/dev/shm/ray_$SLURM_JOB_ID` via `+ray_init._temp_dir`
  (works because `main_ppo.py` passes `ray_init` straight into `ray.init`).
- `--exclude=ip-10-1-105-237`: that node cannot resolve `api.wandb.ai`, which makes
  `WANDB_MODE=online` kill the job at `wandb.init`. Other nodes resolve it fine.

## 10. Open items

1. **Bank quality selection.** Port Stage-1's validation filter (inject the candidate
   skill, re-roll, keep only if `delta_success_rate >= 0.125`) so the survivor is the best
   rather than the latest. Costs extra rollouts.
2. **Bank persistence.** `algorithm.seed.mix_skill_bank_path` to load/dump JSON — fixes
   resume loss, makes the bank auditable, and allows preheating a new arm with skills
   accumulated by a previous run.
3. **Multi-skill entries.** Store a workflow *and* an avoidance rule per task and inject
   both (`skill_teacher_mode=additive` already supports two blocks downstream).
4. **Key granularity.** A `task|task_type` switch: keying by the 6 ALFWorld task types
   would push hit rate to ~100%, and the generated skills are already task-type level
   generalisations ("When tasked to place multiple identical objects, ...").
5. **Contrastive analysis.** The analyzer sees one trajectory at a time and cannot compare
   the successful and failed rollouts of the same task, which is likely the single largest
   information loss in the current hindsight step.
6. **Multimodal.** `_build_mix_pair_batch` raises `NotImplementedError` when
   `multi_modal_data` is present; EZPoints/Sokoban would need image-token repacking for
   the second context.
