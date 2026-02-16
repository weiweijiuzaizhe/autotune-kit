#!/usr/bin/env python3
"""Run base vs tuned on factuality + safety with public datasets.

- Factuality: TruthfulQA multiple-choice (truthfulness proxy).
- Safety: JailbreakBench harmful/benign refusal behavior.

By default runs BOTH fp16 and bnb4, then compares:
- base vs tuned per mode
- bnb4 vs fp16 per model
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


def _safe_rel(tuned: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (tuned - base) / base


def _delta_abs(base: dict, tuned: dict) -> dict:
    return {
        "factual_acc": tuned["factual"]["acc"] - base["factual"]["acc"],
        "safety_balanced": tuned["safety"]["balanced_score"] - base["safety"]["balanced_score"],
        "harmful_refusal_rate": tuned["safety"]["harmful"]["refusal_rate"] - base["safety"]["harmful"]["refusal_rate"],
        "benign_helpful_rate": tuned["safety"]["benign"]["helpful_rate"] - base["safety"]["benign"]["helpful_rate"],
    }


def _delta_rel(base: dict, tuned: dict) -> dict:
    return {
        "factual_acc": _safe_rel(tuned["factual"]["acc"], base["factual"]["acc"]),
        "safety_balanced": _safe_rel(tuned["safety"]["balanced_score"], base["safety"]["balanced_score"]),
        "harmful_refusal_rate": _safe_rel(tuned["safety"]["harmful"]["refusal_rate"], base["safety"]["harmful"]["refusal_rate"]),
        "benign_helpful_rate": _safe_rel(tuned["safety"]["benign"]["helpful_rate"], base["safety"]["benign"]["helpful_rate"]),
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

    common = [
        "--items", str(items_path),
        "--batch_factual", str(args.batch_factual),
        "--batch_safety", str(args.batch_safety),
        "--max_tokens_safety", str(args.max_tokens_safety),
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
        "/root/atk_project/tools/vllm_routeA/score_factual_safety_vllm.py",
        "--model", args.base,
        "--tokenizer", args.base,
        "--out", str(base_dir),
    ] + common
    _run(base_cmd, log_path)

    tuned_cmd = [
        args.conda_bin, "run", "-n", args.score_env, "python",
        "/root/atk_project/tools/vllm_routeA/score_factual_safety_vllm.py",
        "--model", str(Path(args.tuned).expanduser().resolve()),
        "--tokenizer", args.base,
        "--out", str(tuned_dir),
    ] + common
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
            "result": base_res,
            "result_path": str(base_dir / "result.json"),
        },
        "tuned": {
            "model": str(Path(args.tuned).expanduser()),
            "result": tuned_res,
            "result_path": str(tuned_dir / "result.json"),
        },
        "delta_abs": _delta_abs(base_res, tuned_res),
        "delta": _delta_rel(base_res, tuned_res),
    }


def _to_md(summary: dict) -> str:
    lines = []
    lines.append("# Factuality + Safety (vLLM, LF-style workflow)")
    lines.append("")
    lines.append(f"- run_dir: `{summary['run_dir']}`")
    lines.append(f"- compare_quant_nonquant: `{summary['compare_quant_nonquant']}`")
    lines.append("")
    lines.append("| mode | quant | factual(base) | factual(tuned) | Δ factual ((tuned-base)/base) | safety(base) | safety(tuned) | Δ safety ((tuned-base)/base) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for mode, m in summary["modes"].items():
        b = m["base"]["result"]
        t = m["tuned"]["result"]
        d = m["delta"]
        lines.append(
            f"| {mode} | {m['quantization']['mode']} | "
            f"{b['factual']['acc']:.6f} | {t['factual']['acc']:.6f} | {d['factual_acc']*100:+.2f}% | "
            f"{b['safety']['balanced_score']:.6f} | {t['safety']['balanced_score']:.6f} | {d['safety_balanced']*100:+.2f}% |"
        )

    if "quant_vs_nonquant" in summary:
        q = summary["quant_vs_nonquant"]
        lines.append("")
        lines.append("## Quant vs Non-Quant")
        lines.append("")
        lines.append(f"- base factual(bnb4-fp16): `{q['factual']['base_bnb4_minus_fp16']:+.6f}`")
        lines.append(f"- tuned factual(bnb4-fp16): `{q['factual']['tuned_bnb4_minus_fp16']:+.6f}`")
        lines.append(f"- base safety(bnb4-fp16): `{q['safety']['base_bnb4_minus_fp16']:+.6f}`")
        lines.append(f"- tuned safety(bnb4-fp16): `{q['safety']['tuned_bnb4_minus_fp16']:+.6f}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--out_root", default="/root/atk_project/artifacts/vllm_routeA")

    ap.add_argument("--build_env", default="py310")
    ap.add_argument("--score_env", default="vllm014")
    ap.add_argument("--conda_bin", default="/usr/local/miniconda3/bin/conda")

    ap.add_argument("--truthful_limit", type=int, default=0)
    ap.add_argument("--jbb_harmful_limit", type=int, default=0)
    ap.add_argument("--jbb_benign_limit", type=int, default=0)

    ap.add_argument("--batch_factual", type=int, default=32)
    ap.add_argument("--batch_safety", type=int, default=16)
    ap.add_argument("--max_tokens_safety", type=int, default=128)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)

    ap.add_argument(
        "--compare_quant_nonquant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--quantization", choices=["none", "bitsandbytes"], default="bitsandbytes")
    ap.add_argument("--load_format", default="bitsandbytes")
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--fallback_fp16_on_quant_error", action="store_true")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root).expanduser().resolve() / f"factual_safety_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "eval.log"

    items_dir = run_dir / "items"
    items_dir.mkdir(exist_ok=True)

    build_cmd = [
        args.conda_bin, "run", "-n", args.build_env, "python",
        "/root/atk_project/tools/vllm_routeA/build_factual_safety_items.py",
        "--out", str(items_dir),
        "--truthful_limit", str(args.truthful_limit),
        "--jbb_harmful_limit", str(args.jbb_harmful_limit),
        "--jbb_benign_limit", str(args.jbb_benign_limit),
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

    modes = {}
    for tag, quant, load in mode_plan:
        modes[tag] = _score_one_mode(
            mode_tag=tag,
            quantization=quant,
            load_format=load,
            args=args,
            run_dir=run_dir,
            log_path=log_path,
            items_path=items_path,
        )

    info = _read_json(items_dir / "data_info.json")

    primary = "bnb4" if "bnb4" in modes else next(iter(modes.keys()))
    summary = {
        "run_dir": str(run_dir),
        "env": {"build_env": args.build_env, "score_env": args.score_env},
        "datasets": info.get("datasets", {}),
        "items": info,
        "compare_quant_nonquant": args.compare_quant_nonquant,
        "modes": modes,
        "primary_mode": primary,
        "primary_delta": modes[primary]["delta"],
        "primary_delta_abs": modes[primary]["delta_abs"],
    }

    if "fp16" in modes and "bnb4" in modes:
        f = modes["fp16"]
        b = modes["bnb4"]
        summary["quant_vs_nonquant"] = {
            "factual": {
                "base_bnb4_minus_fp16": b["base"]["result"]["factual"]["acc"] - f["base"]["result"]["factual"]["acc"],
                "tuned_bnb4_minus_fp16": b["tuned"]["result"]["factual"]["acc"] - f["tuned"]["result"]["factual"]["acc"],
            },
            "safety": {
                "base_bnb4_minus_fp16": b["base"]["result"]["safety"]["balanced_score"] - f["base"]["result"]["safety"]["balanced_score"],
                "tuned_bnb4_minus_fp16": b["tuned"]["result"]["safety"]["balanced_score"] - f["tuned"]["result"]["safety"]["balanced_score"],
            },
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
