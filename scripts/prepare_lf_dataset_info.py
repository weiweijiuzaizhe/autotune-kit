#!/usr/bin/env python3
"""Create/update LLaMA-Factory dataset_info.json for a JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    args = ap.parse_args()

    train_jsonl = Path(args.train_jsonl).expanduser().resolve()
    dataset_dir = train_jsonl.parent
    dataset_name = train_jsonl.stem
    info_path = dataset_dir / "dataset_info.json"

    info = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            info = {}

    info[dataset_name] = {
        "file_name": train_jsonl.name,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }

    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_info": str(info_path), "dataset": dataset_name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
