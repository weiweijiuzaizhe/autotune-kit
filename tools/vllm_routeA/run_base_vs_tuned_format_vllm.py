#!/usr/bin/env python3
"""Run base vs tuned format evaluation with vLLM.

By default runs fp16 and bnb4, outputs summary.json and comparison.md.
Delta is relative: (tuned - base) / base.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent.parent
_DEFAULT_OUT_ROOT = _REPO_ROOT / "artifacts" / "vllm_routeA"
_DEFAULT_CONDA_BIN = os.environ.get("CONDA_EXE", "conda")


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


def _safe_rel(base: float, tuned: float) -> float:
    if base == 0:
        return 0.0
    return (tuned - base) / base


def _score_one_mode(*, mode_tag: str, quantization: str, load_format: str, args: argparse.Namespace, run_dir: Path, log_path: Path) -> dict:
    base_dir = run_dir / f"base_{mode_tag}"
    tuned_dir = run_dir / f"tuned_{mode_tag}"
    base_dir.mkdir(exist_ok=True)
    tuned_dir.mkdir(exist_ok=True)

    common = [
        "--batch", str(args.batch),
        "--max_tokens", str(args.max_tokens),
        "--max_model_len", str(args.max_model_len),
        "--gpu_mem_util", str(args.gpu_mem_util),
        "--quantization", quantization,
        "--load_format", load_format,
        "--dtype", args.dtype,
    ]
    if args.fallback_fp16_on_quant_error:
        common.append("--fallback_fp16_on_quant_error")

    base_cmd = [
        args.conda_bin, "run", "-n", args.score_env, "python",
        str(_TOOLS_DIR / "score_format_vllm.py"),
        "--model", args.base,
        "--tokenizer", args.base,
        "--out", str(base_dir),
    ] + common
    _run(base_cmd, log_path)

    tuned_cmd = [
        args.conda_bin, "run", "-n", args.score_env, "python",
        str(_TOOLS_DIR / "score_format_vllm.py"),
        "--model", str(Path(args.tuned).expanduser().resolve()),
        "--tokenizer", args.base,
        "--out", str(tuned_dir),
    ] + common
    _run(tuned_cmd, log_path)

    b = _read_json(base_dir / "result.json")
    t = _read_json(tuned_dir / "result.json")

    base_score = float(b["format_score"])
    tuned_score = float(t["format_score"])

    return {
        "quantization": {
            "mode": quantization,
            "load_format": load_format,
            "dtype": args.dtype,
            "fallback_fp16_on_quant_error": args.fallback_fp16_on_quant_error,
        },
        "base": {
            "model": args.base,
            "result": b,
            "result_path": str(base_dir / "result.json"),
        },
        "tuned": {
            "model": str(Path(args.tuned).expanduser()),
            "result": t,
            "result_path": str(tuned_dir / "result.json"),
        },
        "delta_abs": tuned_score - base_score,
        "delta": _safe_rel(base_score, tuned_score),
    }


def _to_md(summary: dict) -> str:
    lines = []
    lines.append("# Format Eval (vLLM)")
    lines.append("")
    lines.append(f"- run_dir: `{summary['run_dir']}`")
    lines.append(f"- compare_quant_nonquant: `{summary['compare_quant_nonquant']}`")
    lines.append("")
    lines.append("| mode | quant | format(base) | format(tuned) | Δ ((tuned-base)/base) |")
    lines.append("|---|---|---:|---:|---:|")

    for mode, m in summary["modes"].items():
        b = float(m["base"]["result"]["format_score"])
        t = float(m["tuned"]["result"]["format_score"])
        d = float(m["delta"]) * 100.0
        lines.append(f"| {mode} | {m['quantization']['mode']} | {b:.6f} | {t:.6f} | {d:+.2f}% |")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--out_root", default=str(_DEFAULT_OUT_ROOT))
    ap.add_argument("--score_env", default="vllm014")
    ap.add_argument("--conda_bin", default=_DEFAULT_CONDA_BIN)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)

    ap.add_argument("--compare_quant_nonquant", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quantization", choices=["none", "bitsandbytes"], default="bitsandbytes")
    ap.add_argument("--load_format", default="bitsandbytes")
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--fallback_fp16_on_quant_error", action="store_true")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root).expanduser().resolve() / f"format_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "eval.log"

    mode_plan: list[tuple[str, str, str]] = []
    if args.compare_quant_nonquant:
        mode_plan.append(("fp16", "none", "auto"))
        mode_plan.append(("bnb4", "bitsandbytes", "bitsandbytes"))
    else:
        load = args.load_format if args.quantization != "none" else "auto"
        tag = "bnb4" if args.quantization == "bitsandbytes" else "fp16"
        mode_plan.append((tag, args.quantization, load))

    modes = {}
    for tag, quant, load in mode_plan:
        modes[tag] = _score_one_mode(
            mode_tag=tag,
            quantization=quant,
            load_format=load,
            args=args,
            run_dir=run_dir,
            log_path=log_path,
        )

    primary = "bnb4" if "bnb4" in modes else next(iter(modes.keys()))
    summary = {
        "run_dir": str(run_dir),
        "compare_quant_nonquant": args.compare_quant_nonquant,
        "modes": modes,
        "primary_mode": primary,
        "primary_delta": modes[primary]["delta"],
        "primary_delta_abs": modes[primary]["delta_abs"],
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "comparison.md").write_text(_to_md(summary), encoding="utf-8")

    print(json.dumps({
        "run_dir": str(run_dir),
        "primary_mode": primary,
        "primary_delta": summary["primary_delta"],
        "modes": list(modes.keys()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
