#!/usr/bin/env python3
"""Run LF-aligned base vs tuned evaluation with vLLM.

Protocol alignment to LF:
- Build items with LF template + encode_oneturn.
- Score by constrained A/B/C/D token selection (no free-form parse).
- Use mmlu/en and ceval+cmmlu/zh defaults as LF run_all.sh.

Quantization comparison:
- By default, run BOTH non-quantized(fp16) and bitsandbytes modes.
- Output mode-wise and delta comparisons in summary.json.
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


def _acc(x: dict, task: str | None = None) -> float:
    if task is None:
        return float(x["overall"]["acc"])
    return float(x.get("by_task", {}).get(task, {}).get("acc", 0.0))


def _safe_rel(tuned: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (tuned - base) / base


def _delta_abs(base_res: dict, tuned_res: dict) -> dict:
    return {
        "overall_acc": _acc(tuned_res) - _acc(base_res),
        "mmlu": _acc(tuned_res, "mmlu") - _acc(base_res, "mmlu"),
        "ceval": _acc(tuned_res, "ceval") - _acc(base_res, "ceval"),
        "cmmlu": _acc(tuned_res, "cmmlu") - _acc(base_res, "cmmlu"),
    }


def _delta_rel(base_res: dict, tuned_res: dict) -> dict:
    return {
        "overall_acc": _safe_rel(_acc(tuned_res), _acc(base_res)),
        "mmlu": _safe_rel(_acc(tuned_res, "mmlu"), _acc(base_res, "mmlu")),
        "ceval": _safe_rel(_acc(tuned_res, "ceval"), _acc(base_res, "ceval")),
        "cmmlu": _safe_rel(_acc(tuned_res, "cmmlu"), _acc(base_res, "cmmlu")),
    }


def _score_one_mode(
    *,
    mode_tag: str,
    quantization: str,
    load_format: str,
    args: argparse.Namespace,
    run_dir: Path,
    log_path: Path,
    items_path: Path,
) -> dict:
    base_dir = run_dir / f"base_{mode_tag}"
    tuned_dir = run_dir / f"tuned_{mode_tag}"
    base_dir.mkdir(exist_ok=True)
    tuned_dir.mkdir(exist_ok=True)

    score_common = [
        "--batch", str(args.batch),
        "--max_model_len", str(args.max_model_len),
        "--gpu_mem_util", str(args.gpu_mem_util),
        "--quantization", quantization,
        "--load_format", load_format,
        "--dtype", args.dtype,
    ]
    if args.fallback_fp16_on_quant_error:
        score_common.append("--fallback_fp16_on_quant_error")

    base_cmd = [
        args.conda_bin, "run", "-n", args.score_env, "python",
        str(_TOOLS_DIR / "score_mcq_lf_aligned_vllm.py"),
        "--items", str(items_path),
        "--model", args.base,
        "--tokenizer", args.base,
        "--out", str(base_dir),
    ] + score_common
    _run(base_cmd, log_path)

    tuned_cmd = [
        args.conda_bin, "run", "-n", args.score_env, "python",
        str(_TOOLS_DIR / "score_mcq_lf_aligned_vllm.py"),
        "--items", str(items_path),
        "--model", str(Path(args.tuned).expanduser().resolve()),
        "--tokenizer", args.base,
        "--out", str(tuned_dir),
    ] + score_common
    _run(tuned_cmd, log_path)

    base_res = _read_json(base_dir / "result.json")
    tuned_res = _read_json(tuned_dir / "result.json")

    return {
        "quantization": {
            "mode": quantization,
            "load_format": load_format,
            "dtype": args.dtype,
            "fallback_fp16_on_quant_error": args.fallback_fp16_on_quant_error,
        },
        "base": {
            "model": args.base,
            "overall": base_res["overall"],
            "by_task": base_res.get("by_task", {}),
            "elapsed_sec": base_res.get("elapsed_sec", 0.0),
            "runtime": base_res.get("lf_aligned", {}).get("runtime", {}),
            "result_path": str(base_dir / "result.json"),
        },
        "tuned": {
            "model": str(Path(args.tuned).expanduser()),
            "overall": tuned_res["overall"],
            "by_task": tuned_res.get("by_task", {}),
            "elapsed_sec": tuned_res.get("elapsed_sec", 0.0),
            "runtime": tuned_res.get("lf_aligned", {}).get("runtime", {}),
            "result_path": str(tuned_dir / "result.json"),
        },
        "delta_abs": _delta_abs(base_res, tuned_res),
        "delta": _delta_rel(base_res, tuned_res),
    }


def _make_markdown(summary: dict) -> str:
    lines = []
    lines.append("# LF-Aligned vLLM Eval Summary")
    lines.append("")
    lines.append(f"- run_dir: `{summary['run_dir']}`")
    lines.append(f"- compare_quant_nonquant: `{summary['lf_align']['compare_quant_nonquant']}`")
    lines.append("")
    lines.append("## Mode Table")
    lines.append("")
    lines.append("| mode | quant | overall(base) | overall(tuned) | Δ ((tuned-base)/base) |")
    lines.append("|---|---|---:|---:|---:|")
    for mode, x in summary["modes"].items():
        qb = x["quantization"]["mode"]
        b = x["base"]["overall"]["acc"]
        t = x["tuned"]["overall"]["acc"]
        d = x["delta"]["overall_acc"]
        lines.append(f"| {mode} | {qb} | {b:.6f} | {t:.6f} | {d*100:+.2f}% |")

    if "quant_vs_nonquant" in summary:
        q = summary["quant_vs_nonquant"]
        lines.append("")
        lines.append("## Quant vs Non-Quant")
        lines.append("")
        lines.append(f"- base(bnb4 - fp16): `{q['overall']['base_bnb4_minus_fp16']:+.6f}`")
        lines.append(f"- tuned(bnb4 - fp16): `{q['overall']['tuned_bnb4_minus_fp16']:+.6f}`")
        lines.append(f"- (tuned-base) drift (bnb4 - fp16): `{q['overall']['tuned_minus_base_drift_bnb4_minus_fp16']:+.6f}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tuned", required=True, help="Merged model dir")
    ap.add_argument("--out_root", default=str(_DEFAULT_OUT_ROOT))

    ap.add_argument("--build_env", default="py310", help="Env with llamafactory")
    ap.add_argument("--score_env", default="vllm014", help="Env with vllm")
    ap.add_argument("--conda_bin", default=_DEFAULT_CONDA_BIN, help="Conda executable path")

    ap.add_argument("--n_shot_en", type=int, default=0)
    ap.add_argument("--n_shot_zh", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16, help="LF default batch size is 16")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)

    ap.add_argument(
        "--compare_quant_nonquant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run both fp16(non-quant) and bnb4(bitsandbytes). Default: true.",
    )
    ap.add_argument("--quantization", choices=["none", "bitsandbytes"], default="bitsandbytes", help="Used only when --no-compare_quant_nonquant")
    ap.add_argument("--load_format", default="bitsandbytes", help="Used only when --no-compare_quant_nonquant")
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--fallback_fp16_on_quant_error", action="store_true")

    ap.add_argument("--limit_mmlu", type=int, default=0)
    ap.add_argument("--limit_per_subject", type=int, default=0)
    ap.add_argument("--ceval_subject_limit", type=int, default=0)
    ap.add_argument("--cmmlu_subject_limit", type=int, default=0)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root).expanduser().resolve() / f"eval_run_{ts}_lf_aligned"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "eval.log"

    items_dir = run_dir / "items"
    items_dir.mkdir(exist_ok=True)

    build_cmd = [
        args.conda_bin, "run", "-n", args.build_env, "python",
        str(_TOOLS_DIR / "build_lf_aligned_items.py"),
        "--out", str(items_dir),
        "--base_model", args.base,
        "--n_shot_en", str(args.n_shot_en),
        "--n_shot_zh", str(args.n_shot_zh),
        "--limit_mmlu", str(args.limit_mmlu),
        "--limit_per_subject", str(args.limit_per_subject),
        "--ceval_subject_limit", str(args.ceval_subject_limit),
        "--cmmlu_subject_limit", str(args.cmmlu_subject_limit),
    ]
    _run(build_cmd, log_path)

    items_path = items_dir / "items.jsonl"
    if not items_path.exists() or items_path.stat().st_size == 0:
        raise SystemExit(f"items not built: {items_path}")

    mode_plan: list[tuple[str, str, str]] = []
    if args.compare_quant_nonquant:
        mode_plan.append(("fp16", "none", "auto"))
        mode_plan.append(("bnb4", "bitsandbytes", "bitsandbytes"))
    else:
        load = args.load_format if args.quantization != "none" else "auto"
        tag = "bnb4" if args.quantization == "bitsandbytes" else "fp16"
        mode_plan.append((tag, args.quantization, load))

    modes: dict[str, dict] = {}
    for tag, quant, load_format in mode_plan:
        modes[tag] = _score_one_mode(
            mode_tag=tag,
            quantization=quant,
            load_format=load_format,
            args=args,
            run_dir=run_dir,
            log_path=log_path,
            items_path=items_path,
        )

    data_info = _read_json(items_dir / "data_info.json")

    primary_mode = "bnb4" if "bnb4" in modes else next(iter(modes.keys()))
    p = modes[primary_mode]

    summary = {
        "run_dir": str(run_dir),
        "env": {"build_env": args.build_env, "score_env": args.score_env},
        "items": data_info,
        "lf_align": {
            "template": "fewshot",
            "n_shot_en": args.n_shot_en,
            "n_shot_zh": args.n_shot_zh,
            "decision": "argmax among A/B/C/D token logits",
            "compare_quant_nonquant": args.compare_quant_nonquant,
        },
        "modes": modes,
        "primary_mode": primary_mode,
        "base": p["base"],
        "tuned": p["tuned"],
        "delta": p["delta"],
        "delta_abs": p["delta_abs"],
    }

    if "fp16" in modes and "bnb4" in modes:
        f = modes["fp16"]
        b = modes["bnb4"]
        summary["quant_vs_nonquant"] = {
            "overall": {
                "base_bnb4_minus_fp16": _acc(b["base"]) - _acc(f["base"]),
                "tuned_bnb4_minus_fp16": _acc(b["tuned"]) - _acc(f["tuned"]),
                "tuned_minus_base_drift_bnb4_minus_fp16": b["delta_abs"]["overall_acc"] - f["delta_abs"]["overall_acc"],
            },
            "mmlu": {
                "base_bnb4_minus_fp16": _acc(b["base"], "mmlu") - _acc(f["base"], "mmlu"),
                "tuned_bnb4_minus_fp16": _acc(b["tuned"], "mmlu") - _acc(f["tuned"], "mmlu"),
                "tuned_minus_base_drift_bnb4_minus_fp16": b["delta_abs"]["mmlu"] - f["delta_abs"]["mmlu"],
            },
            "ceval": {
                "base_bnb4_minus_fp16": _acc(b["base"], "ceval") - _acc(f["base"], "ceval"),
                "tuned_bnb4_minus_fp16": _acc(b["tuned"], "ceval") - _acc(f["tuned"], "ceval"),
                "tuned_minus_base_drift_bnb4_minus_fp16": b["delta_abs"]["ceval"] - f["delta_abs"]["ceval"],
            },
            "cmmlu": {
                "base_bnb4_minus_fp16": _acc(b["base"], "cmmlu") - _acc(f["base"], "cmmlu"),
                "tuned_bnb4_minus_fp16": _acc(b["tuned"], "cmmlu") - _acc(f["tuned"], "cmmlu"),
                "tuned_minus_base_drift_bnb4_minus_fp16": b["delta_abs"]["cmmlu"] - f["delta_abs"]["cmmlu"],
            },
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "comparison.md").write_text(_make_markdown(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "primary_mode": primary_mode,
                "primary_delta": summary["delta"],
                "modes": list(modes.keys()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
