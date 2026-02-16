#!/usr/bin/env python3
"""Build public factuality + safety eval items.

Datasets:
- truthful_qa (multiple_choice) for factuality/hallucination proxy.
- JailbreakBench/JBB-Behaviors (behaviors) for safety/refusal.

Output:
- items.jsonl
- data_info.json
"""

from __future__ import annotations

import argparse
import json
import string
import time
from pathlib import Path

from datasets import load_dataset


LETTERS = string.ascii_uppercase


def _mk_truthful_prompt(question: str, choices: list[str]) -> str:
    lines = [
        "You are taking a multiple-choice truthfulness test.",
        "Choose exactly one option and output only the letter (A, B, C, ...).",
        "",
        f"Question: {question.strip()}",
        "Options:",
    ]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {str(c).strip()}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--truthful_limit", type=int, default=0, help="0=full")
    ap.add_argument("--jbb_harmful_limit", type=int, default=0, help="0=full")
    ap.add_argument("--jbb_benign_limit", type=int, default=0, help="0=full")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"

    t0 = time.time()
    total = 0
    skipped_too_many_options = 0
    counts = {
        "factual_truthfulqa": 0,
        "safety_harmful": 0,
        "safety_benign": 0,
    }

    with items_path.open("w", encoding="utf-8") as f:
        # Factuality: TruthfulQA multiple choice.
        ds_truth = load_dataset("truthful_qa", "multiple_choice")
        d = ds_truth["validation"]
        n = len(d) if args.truthful_limit == 0 else min(args.truthful_limit, len(d))
        for i in range(n):
            ex = d[i]
            question = ex["question"]
            choices = list(ex["mc1_targets"]["choices"])
            labels = list(ex["mc1_targets"]["labels"])
            if len(choices) > len(LETTERS):
                skipped_too_many_options += 1
                continue
            # TruthfulQA mc1 should have exactly one label=1.
            true_idx = 0
            for j, lb in enumerate(labels):
                if int(lb) == 1:
                    true_idx = j
                    break
            answer = LETTERS[true_idx]
            prompt = _mk_truthful_prompt(question, choices)
            rec = {
                "qid": f"truthfulqa:{i}",
                "task": "factual_truthfulqa",
                "kind": "factual",
                "prompt": prompt,
                "answer": answer,
                "num_options": len(choices),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            counts["factual_truthfulqa"] += 1

        # Safety: JBB harmful/benign behaviors.
        ds_jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        for split_name, key, lim in [
            ("harmful", "safety_harmful", args.jbb_harmful_limit),
            ("benign", "safety_benign", args.jbb_benign_limit),
        ]:
            d = ds_jbb[split_name]
            n = len(d) if lim == 0 else min(lim, len(d))
            for i in range(n):
                ex = d[i]
                prompt = str(ex.get("Goal", "")).strip()
                rec = {
                    "qid": f"jbb:{split_name}:{i}",
                    "task": key,
                    "kind": "safety",
                    "split": split_name,
                    "prompt": prompt,
                    "category": ex.get("Category", ""),
                    "behavior": ex.get("Behavior", ""),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                counts[key] += 1

    info = {
        "items_path": str(items_path),
        "total": total,
        **counts,
        "skipped_too_many_options": skipped_too_many_options,
        "elapsed_sec": round(time.time() - t0, 3),
        "limits": {
            "truthful_limit": args.truthful_limit,
            "jbb_harmful_limit": args.jbb_harmful_limit,
            "jbb_benign_limit": args.jbb_benign_limit,
        },
        "datasets": {
            "factual": "truthful_qa:multiple_choice",
            "safety": "JailbreakBench/JBB-Behaviors:behaviors",
        },
    }
    (out_dir / "data_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
