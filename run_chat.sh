#!/usr/bin/env bash
set -euo pipefail
# Download models/datasets from ModelScope (hf-mirror is broken through the local proxy)
USE_MODELSCOPE_HUB="${USE_MODELSCOPE_HUB:-1}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
llamafactory-cli chat configs/qwen3_4b_lora_infer.yaml
