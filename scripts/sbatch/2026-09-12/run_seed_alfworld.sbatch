#!/bin/bash
#SBATCH --job-name=seed_alfworld
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/fsx/xinnanzh/SEED/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/SEED/logs/%x_%j.err

set -x
# Keep every scratch path off the root filesystem (it is 100% full on this cluster).
export TMPDIR=/fsx/xinnanzh/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export TRITON_CACHE_DIR=$TMPDIR/triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor_$SLURM_JOB_ID
# Ray needs unix sockets in its temp dir -> use tmpfs, not Lustre.
RAY_TEMP=/dev/shm/ray_$SLURM_JOB_ID
mkdir -p "$RAY_TEMP" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export CONDA_ENV=seed
unset ROCR_VISIBLE_DEVICES
source /fsx/xinnanzh/miniconda3/etc/profile.d/conda.sh
conda activate seed

cd /fsx/xinnanzh/SEED
nvidia-smi
srun --unbuffered bash /fsx/xinnanzh/SEED/logs/inner_launch.sh \
    +ray_init._temp_dir="$RAY_TEMP"
