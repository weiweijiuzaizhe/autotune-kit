#!/usr/bin/env python3
"""Generate unified 4D table with relative delta.

Delta formula: (tuned - base) / base
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent.parent
_DEFAULT_OUT_ROOT = _REPO_ROOT / "artifacts" / "vllm_routeA"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_rel(base: float, tuned: float) -> float:
    if base == 0:
        return 0.0
    return (tuned - base) / base


def _fmt_cell(base: float | None, tuned: float | None) -> str:
    if base is None or tuned is None:
        return "N/A"
    return f"base {base:.4f} / tuned {tuned:.4f} / Δ {_safe_rel(base, tuned) * 100:+.2f}%"


def _mode_order(modes: set[str]) -> list[str]:
    out: list[str] = []
    for m in ["fp16", "bnb4"]:
        if m in modes:
            out.append(m)
    for m in sorted(modes):
        if m not in out:
            out.append(m)
    return out


def _list_latest(pattern: str) -> list[Path]:
    return sorted(_REPO_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def _has_modes(path: Path, required: set[str]) -> bool:
    try:
        data = _read_json(path)
    except Exception:
        return False
    modes = data.get("modes")
    if not isinstance(modes, dict):
        return False
    return required.issubset(set(modes.keys()))


def _prefer_with_modes(pattern: str, required: set[str]) -> Path | None:
    paths = _list_latest(pattern)
    for p in paths:
        if _has_modes(p, required):
            return p
    return paths[0] if paths else None


def _default_paths() -> tuple[Path | None, Path | None, Path | None, Path | None]:
    preferred_modes = {"fp16", "bnb4"}
    reasoning = _prefer_with_modes(
        "artifacts/vllm_routeA/eval_run_*_lf_aligned/summary.json",
        preferred_modes,
    )
    factual_safety = _prefer_with_modes(
        "artifacts/vllm_routeA/factual_safety_run_*/summary.json",
        preferred_modes,
    )
    format_summary = _prefer_with_modes(
        "artifacts/vllm_routeA/format_run_*/summary.json",
        preferred_modes,
    )

    sanity = None
    sanity_candidates = sorted((_REPO_ROOT / "atk_runs").glob("run_*/sanity_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if sanity_candidates:
        sanity = sanity_candidates[0]

    return reasoning, factual_safety, format_summary, sanity


def _extract_reasoning(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    modes = summary.get("modes")

    if isinstance(modes, dict) and modes:
        for mode, blk in modes.items():
            out[mode] = {
                "base": float(blk["base"]["overall"]["acc"]),
                "tuned": float(blk["tuned"]["overall"]["acc"]),
            }
        return out

    base_blk = summary.get("base", {})
    tuned_blk = summary.get("tuned", {})
    mode = str(summary.get("primary_mode") or "fp16")
    out[mode] = {
        "base": float(base_blk["overall"]["acc"]),
        "tuned": float(tuned_blk["overall"]["acc"]),
    }
    return out


def _extract_factual_safety(summary: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    hallucination: dict[str, dict[str, float]] = {}
    safety: dict[str, dict[str, float]] = {}

    for mode, blk in summary.get("modes", {}).items():
        b = blk["base"]["result"]
        t = blk["tuned"]["result"]
        hallucination[mode] = {
            "base": float(b["factual"]["acc"]),
            "tuned": float(t["factual"]["acc"]),
        }
        safety[mode] = {
            "base": float(b["safety"]["balanced_score"]),
            "tuned": float(t["safety"]["balanced_score"]),
        }

    return hallucination, safety


def _extract_format_from_sanity(sanity: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        "shared": {
            "base": float(sanity["base"]["format_score"]),
            "tuned": float(sanity["tuned"]["format_score"]),
        }
    }


def _extract_format_from_summary(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    modes = summary.get("modes", {})
    if isinstance(modes, dict):
        for mode, blk in modes.items():
            out[mode] = {
                "base": float(blk["base"]["result"]["format_score"]),
                "tuned": float(blk["tuned"]["result"]["format_score"]),
            }
    return out


def _build_rows(
    reasoning: dict[str, dict[str, float]],
    fmt_by_mode: dict[str, dict[str, float]],
    hallucination: dict[str, dict[str, float]],
    safety: dict[str, dict[str, float]],
) -> list[dict[str, str]]:
    mode_set = set(reasoning.keys()) | set(hallucination.keys()) | set(safety.keys())
    if "shared" not in fmt_by_mode:
        mode_set = mode_set | set(fmt_by_mode.keys())
    rows: list[dict[str, str]] = []

    for mode in _mode_order(mode_set):
        r = reasoning.get(mode)
        h = hallucination.get(mode)
        s = safety.get(mode)

        f = fmt_by_mode.get(mode) or fmt_by_mode.get("shared")

        rows.append(
            {
                "mode": mode,
                "reasoning": _fmt_cell(r["base"], r["tuned"]) if r else "N/A",
                "format": _fmt_cell(f["base"], f["tuned"]) if f else "N/A",
                "hallucination": _fmt_cell(h["base"], h["tuned"]) if h else "N/A",
                "safety": _fmt_cell(s["base"], s["tuned"]) if s else "N/A",
            }
        )

    return rows


def _make_md(rows: list[dict[str, str]], meta: dict[str, str]) -> str:
    lines = [
        "# 4D Evaluation Summary",
        "",
        f"- generated_at: `{meta['generated_at']}`",
        f"- reasoning_source: `{meta['reasoning_source']}`",
        f"- format_source: `{meta['format_source']}`",
        f"- factual_safety_source: `{meta['factual_safety_source']}`",
        "",
        "| 模式 | 推理（MCQ acc） | 格式（format_score） | 幻觉（TruthfulQA acc） | 安全（balanced_score） |",
        "|---|---|---|---|---|",
    ]

    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['reasoning']} | {r['format']} | {r['hallucination']} | {r['safety']} |"
        )

    lines.append("")
    lines.append("注：Δ 统一为 `(tuned - base) / base`。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning_summary", default="")
    ap.add_argument("--factual_safety_summary", default="")
    ap.add_argument("--format_summary", default="")
    ap.add_argument("--sanity_report", default="")
    ap.add_argument("--out_root", default=str(_DEFAULT_OUT_ROOT))
    ap.add_argument("--out_name", default="")
    args = ap.parse_args()

    auto_reasoning, auto_factual_safety, auto_format, auto_sanity = _default_paths()

    reasoning_path = Path(args.reasoning_summary) if args.reasoning_summary else auto_reasoning
    factual_safety_path = Path(args.factual_safety_summary) if args.factual_safety_summary else auto_factual_safety
    format_summary_path = Path(args.format_summary) if args.format_summary else auto_format
    sanity_path = Path(args.sanity_report) if args.sanity_report else auto_sanity

    if not reasoning_path or not reasoning_path.exists():
        raise SystemExit("missing reasoning summary.json")
    if not factual_safety_path or not factual_safety_path.exists():
        raise SystemExit("missing factual_safety summary.json")

    if format_summary_path and format_summary_path.exists():
        fmt_by_mode = _extract_format_from_summary(_read_json(format_summary_path))
        fmt_source = format_summary_path
    elif sanity_path and sanity_path.exists():
        fmt_by_mode = _extract_format_from_sanity(_read_json(sanity_path))
        fmt_source = sanity_path
    else:
        raise SystemExit("missing format summary or sanity report")

    reasoning = _extract_reasoning(_read_json(reasoning_path))
    hallucination, safety = _extract_factual_safety(_read_json(factual_safety_path))

    rows = _build_rows(reasoning, fmt_by_mode, hallucination, safety)

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"four_dim_report_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = out_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reasoning_source": str(reasoning_path),
        "format_source": str(fmt_source),
        "factual_safety_source": str(factual_safety_path),
    }

    md = _make_md(rows, meta)
    payload = {
        "meta": meta,
        "rows": rows,
        "formula": "(tuned - base) / base",
    }

    md_path = out_dir / "four_dim_table.md"
    json_path = out_dir / "four_dim_table.json"

    md_path.write_text(md + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_dir": str(out_dir),
        "table_md": str(md_path),
        "table_json": str(json_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
