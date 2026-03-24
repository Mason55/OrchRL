#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
CONFIG_DIR="$EXAMPLE_DIR/configs"

# Trainer mode:
#   multi_agent (default) — orchrl.trainer.train  (MultiAgentsPPOTrainer)
#   verl_base             — examples.akg_kernel_gen.train_verl_base (RayPPOTrainer, single model)
DEFAULT_TRAINER_MODE="verl_base"
DEFAULT_CONFIG_NAME="kernel_gen_level1_verl_base"
DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_LOG_PATH="$REPO_ROOT/logs/kernel_gen_train_${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
TRAINER_MODE="${TRAINER_MODE:-$DEFAULT_TRAINER_MODE}"
CONFIG_NAME="${CONFIG_NAME:-$DEFAULT_CONFIG_NAME}"
LOG_PATH="${LOG_PATH:-$DEFAULT_LOG_PATH}"

mkdir -p "$REPO_ROOT/logs"

CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

cd "$REPO_ROOT"

echo "[INFO] ========================================"
echo "[INFO] AKG Kernel Gen Training"
echo "[INFO] ========================================"
echo "[INFO] Example dir: $EXAMPLE_DIR"
echo "[INFO] Trainer mode: $TRAINER_MODE"
echo "[INFO] Config: $CONFIG_NAME"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] Log path: $LOG_PATH"

if [[ "$TRAINER_MODE" == "verl_base" ]]; then
  # Single-model base training: RayPPOTrainer + agent_trajectory_engine rollout
  python3 -m examples.akg_kernel_gen.train_verl_base \
    --config-path "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    "hydra.searchpath=[file://$REPO_ROOT/orchrl/config]" 2>&1 | tee "$LOG_PATH"
else
  # Multi-agent training: MultiAgentsPPOTrainer (original)
  python3 -m orchrl.trainer.train \
    --config-path "$CONFIG_DIR" \
    --config-name "$CONFIG_NAME" \
    "hydra.searchpath=[file://$REPO_ROOT/orchrl/config]" 2>&1 | tee "$LOG_PATH"
fi
