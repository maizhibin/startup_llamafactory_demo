#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import csv
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def validate_sft(name: str, manifest_name: str):
    data = json.loads((PROC / name).read_text(encoding="utf-8"))
    required = {"instruction", "input", "output", "system"}
    errors = []
    for i, x in enumerate(data):
        missing = required - set(x)
        if missing:
            errors.append(f"row {i}: missing {sorted(missing)}")
        for k in required:
            if not isinstance(x.get(k), str) or not x.get(k).strip():
                errors.append(f"row {i}: {k} empty/non-string")
        try:
            json.loads(x["input"])
            json.loads(x["output"])
        except Exception as e:
            errors.append(f"row {i}: input/output not valid JSON text: {e}")
    with (PROC / manifest_name).open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    counts = Counter(x["task_type"] for x in manifest)
    if len(manifest) != len(data):
        errors.append("manifest length mismatch")
    return len(data), counts, errors

for name, manifest in [
    ("startup_sft_train.json", "startup_sft_train_manifest.csv"),
    ("startup_sft_eval.json", "startup_sft_eval_manifest.csv"),
]:
    n, counts, errors = validate_sft(name, manifest)
    print(f"{name}: {n} samples")
    print(" task distribution:", dict(counts))
    if errors:
        print(" ERRORS:")
        for e in errors[:20]: print("  -", e)
        raise SystemExit(1)

# Check all derived JSONL parse
for path in sorted(PROC.glob("*.jsonl")):
    n = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            json.loads(line)
            n += 1
    print(f"{path.name}: {n} valid JSONL rows")

# YAML syntax check if PyYAML is installed
try:
    import yaml
    for p in sorted((ROOT / "configs").glob("*.yaml")):
        yaml.safe_load(p.read_text(encoding="utf-8"))
        print(f"{p.name}: YAML OK")
except ImportError:
    print("PyYAML not installed; skipped YAML parse check")

print("ALL VALIDATIONS PASSED")
