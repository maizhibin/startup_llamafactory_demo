#!/usr/bin/env bash
set -euo pipefail
python scripts/generate_demo_dataset.py
python scripts/validate_dataset.py
