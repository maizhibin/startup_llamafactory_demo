#!/usr/bin/env bash
set -euo pipefail
# Download models/datasets from ModelScope (hf-mirror is broken through the local proxy)
USE_MODELSCOPE_HUB="${USE_MODELSCOPE_HUB:-1}" \
llamafactory-cli export configs/qwen3_4b_lora_merge.yaml
