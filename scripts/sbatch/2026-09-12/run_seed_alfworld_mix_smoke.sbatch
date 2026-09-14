#!/bin/bash
#SBATCH --job-name=seed_mix_smoke
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/fsx/xinnanzh/SEED/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/SEED/logs/%x_%j.err

# 2-step smoke test for two-context mixture rollout (P1 + P2).
# Forces a synthetic episode skill so the pair path is exercised before the
# skill bank exists; this is not a training configuration.

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

export EXPERIMENT_NAME=smoke_mix_alpha03
export SEED_MIX_DEBUG_SKILL="Check likely receptacles in a fixed order, pick up the target object before moving to the destination, and avoid repeating a failed action."

cd /fsx/xinnanzh/SEED
srun --unbuffered bash /fsx/xinnanzh/SEED/logs/inner_launch.sh \
    +ray_init._temp_dir="$RAY_TEMP" \
    algorithm.seed.mix_alpha=0.3 \
    actor_rollout_ref.rollout.mix_enable=True \
    trainer.total_training_steps=2 \
    trainer.total_epochs=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    'trainer.logger=[console]'
