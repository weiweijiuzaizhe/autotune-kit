#!/usr/bin/env python3
"""Build LF train yaml for Unsloth QLoRA run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_yaml", required=True)
    ap.add_argument("--out_yaml", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--cutoff_len", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--grad_acc", type=int, default=4)
    args = ap.parse_args()

    tpl = Path(args.template_yaml).expanduser().resolve()
    out_yaml = Path(args.out_yaml).expanduser().resolve()
    train_jsonl = Path(args.train_jsonl).expanduser().resolve()

    raw = yaml.safe_load(tpl.read_text(encoding="utf-8"))

    raw["model_name_or_path"] = args.base_model
    raw["template"] = "qwen"
    raw["dataset_dir"] = str(train_jsonl.parent)
    raw["dataset"] = train_jsonl.stem
    raw["cutoff_len"] = int(args.cutoff_len)
    raw["max_steps"] = int(args.max_steps)
    raw["per_device_train_batch_size"] = int(args.micro_batch)
    raw["gradient_accumulation_steps"] = int(args.grad_acc)
    raw["output_dir"] = str(out_yaml.parent / "output_lora")
    raw["overwrite_output_dir"] = True
    raw["report_to"] = "none"

    raw["finetuning_type"] = "lora"
    raw["quantization_bit"] = 4
    raw["quantization_method"] = "bnb"
    raw["use_unsloth"] = True

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(out_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
