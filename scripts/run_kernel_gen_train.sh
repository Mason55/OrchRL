#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/examples/akg_kernel_gen/configs"

DEFAULT_CONFIG_NAME="kernel_gen_level1"
DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_LOG_PATH="$REPO_ROOT/logs/kernel_gen_train_${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
CONFIG_NAME="${CONFIG_NAME:-$DEFAULT_CONFIG_NAME}"
LOG_PATH="${LOG_PATH:-$DEFAULT_LOG_PATH}"

mkdir -p "$REPO_ROOT/logs"
mkdir -p "$(dirname "$LOG_PATH")"

CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

ORCHRL_CONFIG_DIR="$REPO_ROOT/orchrl/config"

if ! eval "$(CONFIG_DIR="$CONFIG_DIR" CONFIG_NAME="$CONFIG_NAME" ORCHRL_CONFIG_DIR="$ORCHRL_CONFIG_DIR" python3 - <<'PY'
import os
import shlex
from hydra import compose, initialize_config_dir

config_dir = os.environ['CONFIG_DIR']
config_name = os.environ['CONFIG_NAME']
orchrl_config_dir = os.environ['ORCHRL_CONFIG_DIR']
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(
        config_name=config_name,
        overrides=[f'hydra.searchpath=[file://{orchrl_config_dir}]'],
    )

values = {
    'MAS_WORK_DIR': cfg.training.mate.mas_work_dir,
    'CONFIG_TEMPLATE_PATH': cfg.training.mate.config_template_path,
    'PROMPT_DATA_PATH': cfg.training.mate.prompt_loader.path,
    'MODEL_PATH_0': cfg.base_models.policy_0.path,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"; then
  echo "[ERROR] Failed to resolve runtime paths from Hydra config: $CONFIG_NAME" >&2
  exit 1
fi

for required_dir in "$MAS_WORK_DIR"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "[ERROR] Required directory not found: $required_dir" >&2
    exit 1
  fi
done

for required_file in "$CONFIG_TEMPLATE_PATH" "$PROMPT_DATA_PATH"; do
  if [[ ! -e "$required_file" ]]; then
    echo "[ERROR] Required path not found: $required_file" >&2
    exit 1
  fi
done

if [[ ! -d "$MODEL_PATH_0" ]]; then
  echo "[ERROR] Model path not found: $MODEL_PATH_0" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export PYTHONUNBUFFERED=1

if [[ -z "$WANDB_API_KEY" ]]; then
  WANDB_API_KEY=$(python3 -c "
import netrc
try:
    n = netrc.netrc()
    auth = n.authenticators('api.wandb.ai')
    print(auth[2] if auth else '')
except Exception:
    print('')
" 2>/dev/null)
  export WANDB_API_KEY
fi

if [[ -n "$WANDB_API_KEY" ]]; then
  echo "[INFO] Wandb API key detected, logging to wandb online"
  export WANDB_MODE="${WANDB_MODE:-online}"
else
  echo "[WARN] No Wandb API key found, using offline mode"
  export WANDB_MODE=offline
fi

cd "$REPO_ROOT"

echo "=============================================="
echo "  OrchRL Kernel Gen Training"
echo "=============================================="
echo "[INFO] Repo root:             $REPO_ROOT"
echo "[INFO] Config:                $CONFIG_NAME"
echo "[INFO] Config dir:            $CONFIG_DIR"
echo "[INFO] CUDA_VISIBLE_DEVICES:  $CUDA_VISIBLE_DEVICES"
echo "[INFO] Log path:              $LOG_PATH"
echo "[INFO] MAS work dir:          $MAS_WORK_DIR"
echo "[INFO] Prompt data:           $PROMPT_DATA_PATH"
echo "[INFO] Model path:            $MODEL_PATH_0"
echo "[INFO] Wandb mode:            $WANDB_MODE"
echo "=============================================="

python3 -m orchrl.trainer.train \
  --config-path "$CONFIG_DIR" \
  --config-name "$CONFIG_NAME" \
  "hydra.searchpath=[file://$ORCHRL_CONFIG_DIR]" \
  2>&1 | tee "$LOG_PATH"
