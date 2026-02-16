#!/usr/bin/env python3
"""Run ATK Safe Launch gate for repro pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--base_train_yaml", required=True)
    ap.add_argument("--cutoff_len", type=int, required=True)
    ap.add_argument("--micro_batch", type=int, required=True)
    ap.add_argument("--qlora_4bit", type=_parse_bool, default=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from atk.safelaunch import run_safe_launch

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    result = run_safe_launch(
        run_dir=run_dir,
        model_base=args.model_base,
        train_jsonl_path=Path(args.train_jsonl).expanduser().resolve(),
        base_train_yaml=Path(args.base_train_yaml).expanduser().resolve(),
        cutoff_len=int(args.cutoff_len),
        qlora_4bit=bool(args.qlora_4bit),
        planned_micro_batch=int(args.micro_batch),
    )

    out = run_dir / "safe_launch.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"safe_launch": str(out), **result}, ensure_ascii=False, indent=2))

    return 0 if result.get("passed") else 3


if __name__ == "__main__":
    raise SystemExit(main())
