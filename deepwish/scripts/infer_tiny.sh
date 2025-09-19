#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/deepseekv3_tiny.yaml}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/runs/deepseekv3_tiny/model.pt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-0.6B}"
PROMPT="${PROMPT:-Write a short haiku about gradients.}"
DEVICE="${DEVICE:-cuda:0}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model weights not found at ${MODEL_PATH}. Train the model first or point MODEL_PATH to a checkpoint." >&2
  exit 1
fi

python "${ROOT_DIR}/inference/inference.py" \
  --config "${CONFIG_PATH}" \
  --model_save_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --prompt "${PROMPT}" \
  --device "${DEVICE}"
