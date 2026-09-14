# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo Purpose

SEED — Self-Evolving On-Policy Distillation for agentic RL. The repo is a fork/superset of [veRL](https://github.com/volcengine/verl) that plugs the SEED algorithm (`seed/`, hindsight-skill analyzer + OPD-gated distillation loss) into veRL's Ray+FSDP PPO/GRPO/GiGPO stack and drives it on long-horizon agent environments from `verl-agent` (`agent_system/`). The distributable pip package is still named `verl` (see `pyproject.toml`, `setup.py`), so imports use `verl.*`.

## Environment

- Python 3.12; primary conda envs used by scripts: `seed` (top-level README) / `skillrl` (default `CONDA_ENV` in RL launchers) / `copd` (default `CONDA_ENV` in SFT prepare scripts). WebShop needs a separate `seed-webshop` (Python 3.10). Search-QA also uses a separate `retriever` env.
- Installs are pinned via `pip install -e .` + `vllm==0.11.0` + `flash-attn==2.7.4.post1`. See README "Installation" for the per-environment install (ALFWorld / WebShop / Search-QA + retrieval server).
- All scripts source `<repo>/.env` for `MODELS_ROOT`, `DATA_ROOT`, `OPENAI_*` (used by hindsight-skill annotation), and `CUDA_VISIBLE_DEVICES`. `.env` is gitignored and **not** present in a fresh clone — create it from the key list in README "Training". Scripts hard-fail with a message when a required var is missing.

## Common Commands

Training uses shell entrypoints under `scripts/sft/` (Stage 1 hindsight-skill SFT) and `examples/seed_trainer/` (Stage 2 self-evolving RL). Both call shared implementations under `_common/`; dataset wrappers only set model, LR, context length, and env-specific flags.

```bash
# Stage 1 — build hindsight-skill SFT data + train analyzer-capable policy (per dataset):
bash scripts/sft/<dataset>/prepare_data.sh   # spins a local vLLM policy server on $HOST:$PORT, collects rollouts, annotates skills via OPENAI_*, writes parquet
bash scripts/sft/<dataset>/train_sft.sh      # torchrun FSDP SFT via verl.trainer.fsdp_sft_trainer; exports HF ckpt under $MODELS_ROOT
# datasets: alfworld | search | webshop | ezpoints | sokoban  (per-dataset defaults table: scripts/sft/README.md)

# Stage 2 — SEED self-evolving OPD RL:
bash examples/seed_trainer/run_<env>_sft_<teacher>_self.sh
# → sources .env, sets SEED_* env vars, execs examples/seed_trainer/_common/<env>.sh
# → which runs data preprocess + python -m verl.trainer.main_ppo with algorithm.adv_estimator=seed and dozens of algorithm.seed.* overrides.
# DRY_RUN=true prints the resolved config without launching.

# Merge FSDP shards into a HF checkpoint:
bash scripts/merge.sh                        # thin wrapper around scripts/model_merger.py merge --backend fsdp

# Data prep for Search-QA (Search-R1-style dataset):
python examples/data_preprocess/preprocess_search_r1_dataset.py
python examples/data_preprocess/prepare.py --mode text --train_data_size 16 --val_data_size 128  # verl-agent parquets (called automatically by _common/alfworld.sh)

# Retrieval server for Search-QA (separate conda env `retriever`):
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log
```

Tests and lint (inherited from veRL; there is no repo-wide pytest target — `tests/` mixes GPU/distributed suites with cheap ones):

```bash
pytest tests/sanity                                  # cheap import/license checks
pytest tests/trainer/ppo                             # CPU-only SEED unit tests (test_opd_loss, test_seed_advantage,
                                                     #   test_seed_analyzer, test_seed_skill_gen_reward, test_episode_skill_guidance)
pytest tests/trainer/ppo/test_opd_loss.py -k <name>   # single file / single test
bash tests/e2e/run_ray_trainer.sh                    # end-to-end Ray+PPO smoke test on the arithmetic_sequence toy env
ruff check . && ruff format .                        # config in pyproject.toml (line-length 300)
```

Overrides commonly threaded through the environment: `MODEL_PATH`, `DATA_DIR`, `NPROC_PER_NODE`, `N_GPUS_PER_NODE`, `TOTAL_EPOCHS`, `TRAIN_BATCH_SIZE`, `MICRO_BATCH_SIZE_PER_GPU`, `MAX_LENGTH`, `EXPORT_MODEL_DIR`, `HISTORY_LENGTH`, and the full `SEED_*` set (see `examples/seed_trainer/_common/*.sh` for the authoritative list).

## Architecture

Three code layers stack on top of each other; understanding SEED requires reading across all three.

### 1. veRL core (`verl/`)

Standard veRL: Ray + FSDP/Megatron PPO. Key files to know:

- `verl/trainer/main_ppo.py` — Hydra entrypoint (`config_path=config`, `config_name=ppo_trainer`) that spins up a Ray `TaskRunner` and hands off to `RayPPOTrainer`.
- `verl/trainer/config/ppo_trainer.yaml` — one big config; the SEED-specific keys live under `algorithm.seed.*` (skill mode, teacher weights, OPD gating/windowing, analysis backend, analyzer prompt version, `skill_gen.*`, per-run dump dirs). `algorithm.adv_estimator` is an `AdvantageEstimator` enum defined in `verl/trainer/ppo/ray_trainer.py` (values: `gae`, `grpo`, `reinforce_plus_plus[_baseline]`, `remax`, `rloo`, `grpo_passk`, `gigpo`, `seed`).
- `verl/trainer/ppo/ray_trainer.py` — the largest SEED surface. `RayPPOTrainer` owns analyzer construction (`_init_seed_analyzer`), the policy-vLLM analysis path (`_finalize_policy_vllm_seed_analysis`), teacher-signal re-scoring, dump helpers (`_dump_seed_analysis`, augmented observations, state-group metrics), the enable/disable windows (`_is_seed_analysis_enabled`, `_is_seed_opd_loss_enabled`), and the `AdvantageEstimator.SEED` branch in `compute_advantage`.
- `verl/trainer/ppo/core_algos.py` — `compute_opd_loss`: `gate = sigmoid(beta * (teacher_log_prob - log_prob.detach()))`, `loss = gate * (teacher_log_prob - log_prob)`, with teacher log-probs and gate detached so gradients flow only through the policy.
- `verl/workers/{actor,critic,rollout,reward_manager}` + `fsdp_workers.py` — Ray actor classes. `DataParallelPPOActor` (`verl/workers/actor/dp_actor.py`) consumes `actor.opd_loss_coef` / `actor.opd_gate_beta` and has a separate `backward_skill_gen_loss` path driven by `meta_info["seed_skill_gen"]` and `actor.skill_gen_micro_batch_size_per_gpu`.
- `verl/trainer/fsdp_sft_trainer.py` — standalone FSDP SFT trainer used by Stage 1 (`scripts/sft/_common/trainer.sh` shells out to `python -m verl.trainer.fsdp_sft_trainer` via torchrun).

### 2. Agent runtime (`agent_system/`, from `verl-agent`)

- `environments/env_manager.py` + `environments/base.py` — vectorized environment wrappers. Each supported task lives under `environments/env_package/{alfworld,webshop,search,sokoban,gym_cards,sciworld,appworld}` with its own installer and text/visual observation format.
- `multi_turn_rollout/rollout_loop.py` — `TrajectoryCollector` runs multi-turn rollouts through the vLLM/SGLang engine, packs image tokens for VL models, and hands `DataProto` back to the trainer.
- `memory/` — pluggable observation-history managers (`memory.py` = default, `retrieval_memory.py`, `skills_only_memory.py`, `skill_updater.py`). Selected via env config.
- `reward_manager/` — per-env reward shaping applied on top of the environment-returned reward.

### 3. SEED algorithm (`seed/`, `gigpo/`)

- `seed/prompting.py` — enums (`SKILL_MODES = episode_step | step_only | episode_only`, `SKILL_TEACHER_MODES = step_priority | additive`), `select_skill_teacher_sources`, and `build_augmented_observation_text` that splices `episode_skill` / `step_skill` teacher blocks into the rollout prompt at a well-defined anchor.
- `seed/analysis.py` — the hindsight analyzer. `SEEDEpisodeAnalyzer` takes completed trajectories (`build_episode_records`, `select_critical_steps_by_stats`), calls an OpenAI-compatible or `policy_vllm` backend (i.e. the current policy itself), extracts JSON `{episode_skill, step_skills}`, and returns skill-augmented contexts for OPD scoring. `SEED_ANALYSIS_BACKEND=policy_vllm` is what makes the loop "self-evolving" — the current policy serves as its own analyzer, so hindsight supervision co-evolves with the policy.
- `seed/skill_gen.py` — `SkillGenRewardConfig` and `compute_skill_gen_reward` compute an auxiliary reward on the skill-generation act itself (downstream log-prob gain, valid-JSON bonus, length penalty, failed-episode handling). Used when `algorithm.seed.skill_gen.enable=true`.
- `gigpo/core_gigpo.py` — `compute_seed_outcome_advantage` / `compute_seed_advantage_components` are the SEED advantage estimator, reusing GiGPO's state-grouping and mean-norm logic.

### SEED training step, end to end

1. `TrajectoryCollector` rolls out multi-turn episodes → `DataProto` with `traj_index`, rewards, and per-step prompts.
2. If analysis is enabled for this step, `SEEDEpisodeAnalyzer` turns each (optionally failure-filtered) episode into `episode_skill` + up to `analysis_max_step_skills_per_traj` `step_skills`.
3. `build_augmented_observation_text` rebuilds the selected steps' prompts with skill blocks; the actor re-scores the *same sampled action tokens* under those prompts, producing `teacher_log_prob` / `episode_teacher_log_prob` / `step_teacher_log_prob` (zero-filled when disabled).
4. The skill-induced log-prob shift either gates the dense OPD loss in the actor (`opd_loss_coef > 0`) or is folded into advantages via `compute_seed_outcome_advantage`.

### Config invariants (enforced in `RayPPOTrainer._validate_config`)

With `adv_estimator=seed`: `algorithm.seed.step_advantage_w` must be `0.0`, `algorithm.seed.selector` must be `llm`, and `analysis_backend` ∈ {`openai`, `policy_vllm`}. `policy_vllm` additionally requires `rollout.name=vllm` and `rollout.max_model_len >= analysis_context_length + analysis_max_completion_tokens` — this is why the `_common/*.sh` scripts pin `SEED_ANALYSIS_MAX_MODEL_LEN`. `episode_skill_teacher_advantage_w` / `step_skill_teacher_advantage_w` are **ignored when `actor.opd_loss_coef > 0`** (dense OPD loss replaces the teacher-advantage path), which is the default in the shipped launchers.

## Notes for Editing

- The distributed package name in `pyproject.toml`/`setup.py` is still `verl` and its version file is `verl/version/version` — do not rename.
- `agent_system/`, `seed/`, `gigpo/`, `utils/`, `scripts/` are all top-level source dirs (not under `verl/`); when adding modules, use one of these packages rather than nesting under `verl/`.
- New `algorithm.seed.*` knobs need three edits to be reachable: the default in `verl/trainer/config/ppo_trainer.yaml`, a `SEED_*` env var + `algorithm.seed.<key>=` override line in each `examples/seed_trainer/_common/*.sh`, and the read site in `ray_trainer.py` (use `OmegaConf.select` so older configs stay loadable).
- Teacher naming (`SFT_TEACHER_SHORT`) is inferred from `OPENAI_MODEL` in `scripts/sft/_common/teacher_naming.sh`; override explicitly if the model name is not recognized. Data-dir and export-model suffixes both include this teacher tag (e.g. `glm_self` / `glm-self`) — the Stage 2 launcher `run_<env>_sft_<teacher>_self.sh` expects a matching checkpoint at `$MODELS_ROOT/<...>-<teacher>-self`.
- Cluster runs: this project is developed on the SFM-P5 HyperPod cluster (see the user's global `hyperpod-dev-workflow.md`); Python jobs go over SSH via `sbatch`, not locally. Edit locally, let sync-rsync push, then invoke via `ssh sfm-science-sfm-p5-cluster "cd /fsx/xinnanzh/SEED && ..."`.
