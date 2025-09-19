#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/deepseekv3_tiny.yaml}"
GPUS="${GPUS:-1}"
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-0.6B}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found at ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/examples/sample_conversations.csv" ]]; then
  echo "Sample dataset missing. Did the repo checkout complete correctly?" >&2
  exit 1
fi

TORCHRUN=${TORCHRUN:-torchrun}

${TORCHRUN} --standalone --nnodes=1 --nproc_per_node="${GPUS}" "${ROOT_DIR}/train/train.py" \
  --config "${CONFIG_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}"
