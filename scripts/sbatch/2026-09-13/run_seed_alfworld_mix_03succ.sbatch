#!/bin/bash
#SBATCH --job-name=seed_mix03_succ
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
# ip-10-1-105-237 cannot resolve api.wandb.ai, which makes WANDB_MODE=online kill
# the job at wandb.init. Other nodes resolve it fine.
#SBATCH --exclude=ip-10-1-105-237
#SBATCH --output=/fsx/xinnanzh/SEED/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/SEED/logs/%x_%j.err

# SEED with two-context mixture rollout: the ONLY change from
# run_alfworld_sft_glm_self.sh is the rollout policy, which samples every token from
#     mu = (1 - alpha) * pi(. | clean) + alpha * pi(. | skill-augmented)
# Analysis, teacher re-scoring, the OPD gate/loss, the advantage estimator and KL
# are all untouched. The privileged context uses the carry-over skill bank, filled
# from each step's hindsight analysis and keyed by task description, so epoch 1
# runs at alpha=0 until the bank fills.
#
# Importance correction IS applied, but only to mu-sampled rows: the policy-gradient
# term is weighted by exp(clamp(old_log_probs - log mu, +-mix_is_log_clip)), which
# self-gates on the mix_row_alpha column. KL, entropy and the OPD term stay
# uncorrected, and rows with alpha=0 get weight exactly 1.

set -x
export TMPDIR=/fsx/xinnanzh/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export TRITON_CACHE_DIR=$TMPDIR/triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor_$SLURM_JOB_ID
RAY_TEMP=/dev/shm/ray_$SLURM_JOB_ID
mkdir -p "$RAY_TEMP" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export CONDA_ENV=seed
unset ROCR_VISIBLE_DEVICES
source /fsx/xinnanzh/miniconda3/etc/profile.d/conda.sh
conda activate seed

export WANDB_MODE=online
export WANDB_ENTITY=sean-zhang
export EXPERIMENT_NAME=seed_mix03succ_qwen2.5_3b_alfworld_sft_glm_self_mean-std-norm

cd /fsx/xinnanzh/SEED
nvidia-smi
srun --unbuffered bash /fsx/xinnanzh/SEED/logs/inner_launch.sh \
    +ray_init._temp_dir="$RAY_TEMP" \
    algorithm.seed.mix_alpha=0.3 \
    actor_rollout_ref.rollout.mix_enable=True \
    algorithm.seed.mix_skill_overwrite=success_priority \
    trainer.save_freq=-1
