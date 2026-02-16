#!/usr/bin/env python3
"""Backward-compatible MCQ eval entrypoint (safe, no single-process dual-engine).

This file used to load base + tuned vLLM engines in one Python process, which
can OOM on a 24GB GPU. It now runs base and tuned scoring in separate
subprocesses (Route A safe path).

Preferred entrypoint: run_base_vs_tuned.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _run(cmd: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(cmd) + "\n")
        lf.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            lf.write(line)
            lf.flush()
            print(line, end="")
        rc = p.wait()
        if rc != 0:
            raise SystemExit(f"command failed (rc={rc}): {' '.join(cmd)}; see {log_path}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model name or path")
    ap.add_argument("--tuned", required=True, help="Merged tuned model path")
    ap.add_argument("--out", required=True, help="Output run dir")
    ap.add_argument("--conda_env", default="vllm014")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_util", type=float, default=0.90)
    ap.add_argument("--chunk_size", type=int, default=64, help="Batch size")
    ap.add_argument("--limit_mmlu", type=int, default=0, help="0 = full")
    ap.add_argument("--limit_per_subject", type=int, default=0, help="0 = full for ceval/cmmlu")
    ap.add_argument("--ceval_subject_limit", type=int, default=0)
    ap.add_argument("--cmmlu_subject_limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval.log"

    items_dir = out_dir / "items"
    base_dir = out_dir / "base"
    tuned_dir = out_dir / "tuned"
    items_dir.mkdir(exist_ok=True)
    base_dir.mkdir(exist_ok=True)
    tuned_dir.mkdir(exist_ok=True)

    meta = {
        "base": args.base,
        "tuned": str(Path(args.tuned).expanduser().resolve()),
        "conda_env": args.conda_env,
        "max_model_len": args.max_model_len,
        "gpu_util": args.gpu_util,
        "chunk_size": args.chunk_size,
        "limits": {
            "mmlu": args.limit_mmlu,
            "per_subject": args.limit_per_subject,
            "ceval_subject_limit": args.ceval_subject_limit,
            "cmmlu_subject_limit": args.cmmlu_subject_limit,
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    build_cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "/root/atk_project/tools/vllm_routeA/build_mcq_items.py",
        "--out",
        str(items_dir),
        "--limit_mmlu",
        str(args.limit_mmlu),
        "--limit_per_subject",
        str(args.limit_per_subject),
        "--ceval_subject_limit",
        str(args.ceval_subject_limit),
        "--cmmlu_subject_limit",
        str(args.cmmlu_subject_limit),
    ]
    _run(build_cmd, log_path)

    items_path = items_dir / "items.jsonl"
    if not items_path.exists() or items_path.stat().st_size == 0:
        raise SystemExit(f"items not built: {items_path}")

    base_cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "/root/atk_project/tools/vllm_routeA/score_mcq_items_vllm.py",
        "--items",
        str(items_path),
        "--model",
        args.base,
        "--out",
        str(base_dir),
        "--batch",
        str(args.chunk_size),
        "--max_model_len",
        str(args.max_model_len),
        "--gpu_mem_util",
        str(args.gpu_util),
    ]
    _run(base_cmd, log_path)

    tuned_cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "/root/atk_project/tools/vllm_routeA/score_mcq_items_vllm.py",
        "--items",
        str(items_path),
        "--model",
        str(Path(args.tuned).expanduser().resolve()),
        "--tokenizer",
        args.base,
        "--out",
        str(tuned_dir),
        "--batch",
        str(args.chunk_size),
        "--max_model_len",
        str(args.max_model_len),
        "--gpu_mem_util",
        str(args.gpu_util),
    ]
    _run(tuned_cmd, log_path)

    base_res = _read_json(base_dir / "result.json")
    tuned_res = _read_json(tuned_dir / "result.json")

    def acc(x: dict, task: str | None = None) -> float:
        if task is None:
            return float(x["overall"]["acc"])
        return float(x["by_task"].get(task, {}).get("acc", 0.0))

    results = {
        "base": base_res,
        "tuned": tuned_res,
        "delta": {
            "overall_acc": acc(tuned_res) - acc(base_res),
            "mmlu": acc(tuned_res, "mmlu") - acc(base_res, "mmlu"),
            "ceval": acc(tuned_res, "ceval") - acc(base_res, "ceval"),
            "cmmlu": acc(tuned_res, "cmmlu") - acc(base_res, "cmmlu"),
        },
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(results["delta"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
