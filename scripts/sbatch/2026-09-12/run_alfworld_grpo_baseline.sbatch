#!/bin/bash
#SBATCH --job-name=grpo_alfworld
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
# ip-10-1-105-237 cannot resolve api.wandb.ai (wandb-core: "lookup api.wandb.ai on
# 10.0.0.2:53"), which makes WANDB_MODE=online time out and kill the job. Other
# nodes resolve it fine, so exclude just that one.
#SBATCH --exclude=ip-10-1-105-237
#SBATCH --output=/fsx/xinnanzh/SEED/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/SEED/logs/%x_%j.err

# GRPO baseline via the repo's own launcher, examples/grpo_trainer/run_alfworld.sh.
# That script uses adv_estimator=gigpo with GIGPO_STEP_ADV_W=0.0, which drops the
# step-advantage term and leaves episode-level group-normalized advantage == GRPO.

set -x
export TMPDIR=/fsx/xinnanzh/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export TRITON_CACHE_DIR=$TMPDIR/triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor_$SLURM_JOB_ID
RAY_TEMP=/dev/shm/ray_$SLURM_JOB_ID
mkdir -p "$RAY_TEMP" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

source /fsx/xinnanzh/miniconda3/etc/profile.d/conda.sh
conda activate seed

# run_alfworld.sh does not source .env itself.
set -a
source /fsx/xinnanzh/SEED/.env
set +a
export WANDB_MODE=online
export WANDB_ENTITY=sean-zhang

# The script defaults to Qwen2.5-7B-Instruct, which is not staged; use the 3B base
# so the arm matches the SEED run's scale.
export MODEL_PATH=$MODELS_ROOT/Qwen2.5-3B-Instruct
export EXPERIMENT_NAME=grpo_qwen2.5_3b_alfworld_mean-std-norm

cd /fsx/xinnanzh/SEED
nvidia-smi
srun --unbuffered bash /fsx/xinnanzh/SEED/logs/inner_launch_grpo.sh \
    +ray_init._temp_dir="$RAY_TEMP"
