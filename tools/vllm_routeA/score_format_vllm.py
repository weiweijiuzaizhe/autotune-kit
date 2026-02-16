#!/usr/bin/env python3
"""Score format compliance with vLLM.

Metric:
- format_score = ratio of outputs that are valid JSON and contain keys: name, age, city.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams

PROMPTS = [
    "输出严格JSON: name=Alice age=23 city=Beijing",
    "请只返回JSON，信息是 Bob, 31, Shanghai",
    "Return strict JSON only for: Charlie, 45, Shenzhen",
    "仅返回 JSON：Diana 29 Hangzhou",
    "Output JSON only: Ethan, 38, Guangzhou",
    "只输出JSON: Fiona, 22, Chengdu",
    "JSON only: Grace 41 Nanjing",
    "严格JSON返回：Henry, 19, Wuhan",
    "Return only JSON: Iris, 27, Suzhou",
    "仅JSON: Jack, 33, Xi'an",
]


def _strip_code_fence(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip()


def _ok_format(text: str) -> bool:
    t = _strip_code_fence(text)
    try:
        obj = json.loads(t)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    return {"name", "age", "city"}.issubset(set(obj.keys()))


def _build_llm(args: argparse.Namespace) -> tuple[LLM, dict]:
    kwargs = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem_util,
        "tensor_parallel_size": 1,
    }
    used = {
        "quantization": "none",
        "load_format": "auto",
        "dtype": args.dtype,
        "fallback": False,
        "fallback_reason": "",
    }

    if args.quantization != "none":
        kwargs["quantization"] = args.quantization
        kwargs["load_format"] = args.load_format
        used["quantization"] = args.quantization
        used["load_format"] = args.load_format

    try:
        llm = LLM(**kwargs)
        return llm, used
    except Exception as e:
        if not args.fallback_fp16_on_quant_error or args.quantization == "none":
            raise
        kwargs.pop("quantization", None)
        kwargs.pop("load_format", None)
        llm = LLM(**kwargs)
        used.update({
            "quantization": "none",
            "load_format": "auto",
            "fallback": True,
            "fallback_reason": str(e),
        })
        return llm, used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--quantization", choices=["none", "bitsandbytes"], default="none")
    ap.add_argument("--load_format", default="auto")
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--fallback_fp16_on_quant_error", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    llm, runtime = _build_llm(args)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    ok = 0
    details = []
    total = len(PROMPTS)
    for off in range(0, total, args.batch):
        batch_prompts = PROMPTS[off : off + args.batch]
        outs = llm.generate(batch_prompts, sp)
        for p, o in zip(batch_prompts, outs):
            text = (o.outputs[0].text if o.outputs else "") or ""
            passed = _ok_format(text)
            ok += int(passed)
            details.append({"prompt": p, "output": text, "ok": passed})

    elapsed = time.time() - t0
    res = {
        "model": args.model,
        "count": total,
        "format_score": ok / total if total else 0.0,
        "ok": ok,
        "elapsed_sec": round(elapsed, 3),
        "runtime": runtime,
        "details": details,
    }

    (out_dir / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"format_score": res["format_score"], "elapsed_sec": res["elapsed_sec"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
