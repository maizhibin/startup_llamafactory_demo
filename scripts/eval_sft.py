#!/usr/bin/env python3
"""Generation-based evaluation for the LoRA SFT model on startup_sft_eval.

Metrics (outputs are structured JSON, so we score structure, not just loss):
  - json_valid:   model output parses as JSON
  - exact_match:  parsed output == parsed reference (order-insensitive)
  - field_acc:    per-leaf-field accuracy over flattened reference JSON

Usage:
  USE_MODELSCOPE_HUB=1 python scripts/eval_sft.py                 # fine-tuned (LoRA)
  USE_MODELSCOPE_HUB=1 python scripts/eval_sft.py --baseline      # base model, no adapter
  USE_MODELSCOPE_HUB=1 python scripts/eval_sft.py --limit 8       # quick smoke run
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "data/processed/startup_sft_eval.json"
ADAPTER_DIR = ROOT / "saves/startup-qwen3-4b/lora/sft"
BASE_MODEL = "Qwen/Qwen3.5-4B"


def build_prompt(tokenizer, sample):
    user = sample["instruction"]
    if sample.get("input"):
        user = f"{user}\n{sample['input']}"
    messages = [
        {"role": "system", "content": sample.get("system") or ""},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_json(text):
    """Parse the first JSON object in the model output; None if unparseable."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def flatten(obj, prefix=""):
    """Flatten nested dicts into {dot.path: leaf}; lists stay as leaves."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def score(pred, ref):
    if pred is None or not isinstance(pred, dict):
        return 0.0, pred == ref
    if pred == ref:
        return 1.0, True
    pf, rf = flatten(pred), flatten(ref)
    if not rf:
        return 0.0, False
    hits = sum(1 for k, v in rf.items() if pf.get(k) == v)
    return hits / len(rf), False


@torch.inference_mode()
def generate_all(model, tokenizer, samples, batch_size, max_new_tokens):
    prompts = [build_prompt(tokenizer, s) for s in samples]
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for row in gen[:, enc["input_ids"].shape[1] :]:
            outputs.append(tokenizer.decode(row, skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="evaluate base model without LoRA adapter")
    ap.add_argument("--limit", type=int, default=0, help="only evaluate first N samples")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    samples = json.load(open(EVAL_FILE))
    if args.limit:
        samples = samples[: args.limit]
    print(f"eval samples: {len(samples)}  baseline: {args.baseline}")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    if not args.baseline:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()

    outputs = generate_all(model, tokenizer, samples, args.batch_size, args.max_new_tokens)

    n_valid = n_exact = 0
    field_scores = []
    records = []
    for s, out in zip(samples, outputs):
        pred = extract_json(out)
        ref = json.loads(s["output"])
        fscore, exact = score(pred, ref)
        n_valid += pred is not None
        n_exact += exact
        field_scores.append(fscore)
        records.append(
            {"instruction": s["instruction"], "prediction": out,
             "reference": s["output"], "json_valid": pred is not None,
             "exact_match": exact, "field_acc": round(fscore, 4)}
        )

    n = len(samples)
    tag = "baseline" if args.baseline else "lora"
    print("\n===== results =====")
    print(f"json_valid : {n_valid}/{n} = {n_valid / n:.2%}")
    print(f"exact_match: {n_exact}/{n} = {n_exact / n:.2%}")
    print(f"field_acc  : {sum(field_scores) / n:.2%}")

    out_file = ROOT / f"data/processed/eval_predictions_{tag}.jsonl"
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"predictions -> {out_file}")


if __name__ == "__main__":
    main()
