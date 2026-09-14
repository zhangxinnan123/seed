#!/bin/bash
#SBATCH --job-name=alfworld_rollouts
#SBATCH --partition=ml.p5.48xlarge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/fsx/xinnanzh/SEED/logs/%x_%j.out
#SBATCH --error=/fsx/xinnanzh/SEED/logs/%x_%j.err

# Stage-1, rollout stage ONLY: collect baseline trajectories from the base policy
# (Qwen2.5-3B-Instruct) and stop before any skill annotation.
#
# Calls scripts/sft/alfworld/pipeline.py directly with --stop-after-baseline-rollouts,
# because prepare_data.sh does not forward that flag. The skill-* endpoint args are
# placeholders: pipeline.py builds the endpoint config up front but never calls it
# when stopping after rollouts.

set -x
export TMPDIR=/fsx/xinnanzh/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
mkdir -p "$TMPDIR"

unset ROCR_VISIBLE_DEVICES
source /fsx/xinnanzh/miniconda3/etc/profile.d/conda.sh
conda activate seed

cd /fsx/xinnanzh/SEED
set -a
source /fsx/xinnanzh/SEED/.env
set +a

MODEL_PATH=${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-3B-Instruct}
SERVED_NAME=Qwen2.5-3B-Instruct
HOST=127.0.0.1
PORT=60001
BASE_URL="http://${HOST}:${PORT}/v1"
OUTPUT_DIR=${OUTPUT_DIR:-/fsx/xinnanzh/SEED/outputs/alfworld_rollouts_qwen25_3b}
VLLM_LOG=/fsx/xinnanzh/SEED/logs/vllm_rollouts_${SLURM_JOB_ID}.log

nvidia-smi
test -f "$MODEL_PATH/config.json" || { echo "base policy not found: $MODEL_PATH" >&2; exit 1; }
test -d "$ALFWORLD_DATA/json_2.1.1/train" || { echo "ALFWORLD_DATA incomplete: $ALFWORLD_DATA" >&2; exit 1; }

server_pid=""
cleanup() { [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null; }
trap cleanup EXIT

# 8 GPUs: TP=1 x DP=8, purely for rollout throughput (3B fits on one card).
vllm serve "$MODEL_PATH" \
    --host "$HOST" --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --dtype auto > "$VLLM_LOG" 2>&1 &
server_pid=$!

deadline=$((SECONDS + 900))
until curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; do
    kill -0 "$server_pid" 2>/dev/null || { echo "vLLM died, see $VLLM_LOG" >&2; tail -30 "$VLLM_LOG" >&2; exit 1; }
    (( SECONDS >= deadline )) && { echo "vLLM startup timed out, see $VLLM_LOG" >&2; exit 1; }
    sleep 5
done
echo "vLLM ready at $BASE_URL"

python scripts/sft/alfworld/pipeline.py \
    --env-file /fsx/xinnanzh/SEED/.env \
    --alf-config /fsx/xinnanzh/SEED/agent_system/environments/env_package/alfworld/configs/config_tw.yaml \
    --output-dir "$OUTPUT_DIR" \
    --tasks-per-type "${TASKS_PER_TYPE:-30}" \
    --rollouts-per-task "${ROLLOUTS_PER_TASK:-8}" \
    --task-batch-size "${TASK_BATCH_SIZE:-128}" \
    --max-steps 30 \
    --history-length 5 \
    --request-workers "${REQUEST_WORKERS:-1024}" \
    --policy-base-url "$BASE_URL" \
    --policy-api-key EMPTY \
    --policy-model "$SERVED_NAME" \
    --policy-temperature 0.4 \
    --policy-max-completion-tokens 512 \
    --fallback-action look \
    --skill-base-url "$BASE_URL" \
    --skill-api-key EMPTY \
    --skill-model "$SERVED_NAME" \
    --seed 2026 \
    --overwrite \
    --stop-after-baseline-rollouts

echo "ROLLOUT_STAGE_DONE -> $OUTPUT_DIR/baseline_rollouts.jsonl"
