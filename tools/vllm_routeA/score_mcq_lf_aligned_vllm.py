#!/usr/bin/env python3
"""Score LF-aligned items with vLLM using A/B/C/D constrained decoding.

This aligns with LLaMA-Factory evaluator behavior:
- Input uses LF tokenized prompt (`prompt_token_ids`).
- Prediction chooses among token ids for A/B/C/D only.
- No regex parsing of free-form generated text.

Quantization alignment:
- Default to bitsandbytes 4bit-style loading in vLLM (closest to LF QLoRA/eval workflows).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _load_items(path: Path, limit: int | None = None) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def _build_llm(args: argparse.Namespace, tok_name: str) -> tuple[LLM, dict]:
    llm_kwargs = {
        "model": args.model,
        "tokenizer": tok_name,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem_util,
        "tensor_parallel_size": 1,
    }

    used_quant = "none"
    used_load_format = "auto"
    fallback_reason = ""

    if args.quantization != "none":
        llm_kwargs["quantization"] = args.quantization
        llm_kwargs["load_format"] = args.load_format
        used_quant = args.quantization
        used_load_format = args.load_format

    try:
        llm = LLM(**llm_kwargs)
        return llm, {
            "quantization": used_quant,
            "load_format": used_load_format,
            "dtype": args.dtype,
            "fallback": False,
            "fallback_reason": fallback_reason,
        }
    except Exception as e:  # pragma: no cover
        if not args.fallback_fp16_on_quant_error or args.quantization == "none":
            raise
        fallback_reason = str(e)
        llm_kwargs.pop("quantization", None)
        llm_kwargs.pop("load_format", None)
        llm = LLM(**llm_kwargs)
        return llm, {
            "quantization": "none",
            "load_format": "auto",
            "dtype": args.dtype,
            "fallback": True,
            "fallback_reason": fallback_reason,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None, help="Tokenizer path/name. Default uses --model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--quantization", choices=["none", "bitsandbytes"], default="bitsandbytes")
    ap.add_argument("--load_format", default="bitsandbytes")
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--fallback_fp16_on_quant_error", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    items_path = Path(args.items).expanduser().resolve()
    limit = None if args.limit == 0 else args.limit
    items = _load_items(items_path, limit=limit)
    if not items:
        (out_dir / "result.json").write_text(json.dumps({"error": "no items"}, indent=2) + "\n", encoding="utf-8")
        return 2

    tok_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    choices = ["A", "B", "C", "D"]
    choice_token_ids = [tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in choices]
    id_to_choice = {tid: ch for tid, ch in zip(choice_token_ids, choices)}

    llm, runtime_cfg = _build_llm(args, tok_name)

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        allowed_token_ids=choice_token_ids,
    )

    total = len(items)
    t0 = time.time()
    report_step = max(1, total // 20)
    next_report = report_step

    correct = 0
    per_task = defaultdict(lambda: {"n": 0, "correct": 0})

    for off in range(0, total, args.batch):
        batch = items[off : off + args.batch]
        prompts = [{"prompt_token_ids": x["prompt_token_ids"]} for x in batch]
        outs = llm.generate(prompts, sp)

        for x, o in zip(batch, outs):
            pred = ""
            if o.outputs and o.outputs[0].token_ids:
                pred = id_to_choice.get(o.outputs[0].token_ids[0], "")

            gold = str(x["answer"]).strip().upper()[:1]
            ok = pred == gold
            per_task[x["task"]]["n"] += 1
            per_task[x["task"]]["correct"] += 1 if ok else 0
            correct += 1 if ok else 0

        done = min(off + args.batch, total)
        if done >= next_report or done == total:
            elapsed = time.time() - t0
            ips = done / elapsed if elapsed > 0 else 0.0
            pct = 100.0 * done / total
            print(f"progress: {done}/{total} ({pct:.1f}%), {ips:.2f} items/s", flush=True)
            while next_report <= done:
                next_report += report_step

    elapsed = time.time() - t0
    result = {
        "model": args.model,
        "tokenizer": tok_name,
        "items": str(items_path),
        "count": total,
        "elapsed_sec": round(elapsed, 3),
        "overall": {"acc": correct / total, "correct": correct, "n": total},
        "by_task": {
            k: {"acc": (v["correct"] / v["n"] if v["n"] else 0.0), "correct": v["correct"], "n": v["n"]}
            for k, v in sorted(per_task.items())
        },
        "lf_aligned": {
            "decision": "argmax among token(A/B/C/D) via allowed_token_ids",
            "choice_token_ids": {
                "A": choice_token_ids[0],
                "B": choice_token_ids[1],
                "C": choice_token_ids[2],
                "D": choice_token_ids[3],
            },
            "runtime": runtime_cfg,
        },
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
