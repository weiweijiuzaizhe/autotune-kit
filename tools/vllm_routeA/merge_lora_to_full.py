#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a base model and save a standalone HF model dir.

This is Route A for vLLM: produce a merged full-weights model directory so vLLM
can load it without depending on PEFT at runtime.

Design goals:
- Never modify the base model cache.
- Write outputs to a new directory.
- Keep it reusable: CLI args + JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _copy_if_exists(src_dir: Path, dst_dir: Path, rel: str) -> None:
    src = src_dir / rel
    if src.exists():
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model name or path")
    ap.add_argument("--adapter", required=True, help="LoRA adapter directory (PEFT)")
    ap.add_argument("--out", required=True, help="Output directory for merged model")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Where to do the merge")
    ap.add_argument("--max_shard_size", default="2GB")
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=args.trust_remote_code)

    # Keep base cache untouched; just load weights.
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
        device_map={"": args.device},
        trust_remote_code=args.trust_remote_code,
    )

    peft_model = PeftModel.from_pretrained(model, args.adapter)

    # Merge LoRA into base weights.
    merged = peft_model.merge_and_unload()
    merged.eval()

    # Save standalone model directory.
    merged.save_pretrained(
        out_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tok.save_pretrained(out_dir)

    # Preserve common extra files when present in adapter output.
    adapter_dir = Path(args.adapter).expanduser().resolve()
    _copy_if_exists(adapter_dir, out_dir, "chat_template.jinja")

    info = {
        "base": args.base,
        "adapter": str(adapter_dir),
        "out": str(out_dir),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (out_dir / "merge_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
