#!/usr/bin/env python3
"""Build LF-aligned evaluation items (prompt_token_ids + gold label).

This script aligns with LLaMA-Factory evaluator protocol:
- Uses LF eval template (`en` / `zh`).
- Uses LF data template (`template=fewshot`) and `encode_oneturn`.
- Produces tokenized prompts (prompt_token_ids), not plain text.

Run in LF environment (py310 with llamafactory installed).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.eval.template import get_eval_template
from llamafactory.hparams import DataArguments


_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent.parent
_DEFAULT_TASK_DIR = _REPO_ROOT / "assets" / "lf_eval_tasks"


def _load_mapping(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_answer(ans) -> str:
    if isinstance(ans, int):
        return "ABCD"[ans] if 0 <= ans < 4 else "A"
    s = str(ans).strip().upper()
    return s[:1] if s else "A"


def _to_std(task: str, ex: dict) -> dict:
    if task == "mmlu":
        return {
            "question": str(ex["question"]),
            "A": str(ex["choices"][0]),
            "B": str(ex["choices"][1]),
            "C": str(ex["choices"][2]),
            "D": str(ex["choices"][3]),
            "answer": _norm_answer(ex["answer"]),
        }

    # ceval / cmmlu follow A-D + answer style.
    q = ex.get("question")
    if q is None:
        q = ex.get("Question")
    return {
        "question": str(q),
        "A": str(ex["A"]),
        "B": str(ex["B"]),
        "C": str(ex["C"]),
        "D": str(ex["D"]),
        "answer": _norm_answer(ex.get("answer", ex.get("Answer", "A"))),
    }


def _iter_subjects(mapping: dict, limit: int) -> list[str]:
    subs = sorted(mapping.keys())
    if limit > 0:
        subs = subs[:limit]
    return subs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--task_dir", default=str(_DEFAULT_TASK_DIR))
    ap.add_argument("--template", default="fewshot")
    ap.add_argument("--n_shot_en", type=int, default=0)
    ap.add_argument("--n_shot_zh", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--limit_mmlu", type=int, default=0, help="0=full")
    ap.add_argument("--limit_per_subject", type=int, default=0, help="0=full")
    ap.add_argument("--ceval_subject_limit", type=int, default=0, help="0=all")
    ap.add_argument("--cmmlu_subject_limit", type=int, default=0, help="0=all")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    data_args = DataArguments(template=args.template)
    lf_template = get_template_and_fix_tokenizer(tokenizer, data_args)
    eval_template_en = get_eval_template("en")
    eval_template_zh = get_eval_template("zh")

    task_dir = Path(args.task_dir)
    mmlu_map = _load_mapping(task_dir / "mmlu" / "mapping.json")
    ceval_map = _load_mapping(task_dir / "ceval" / "mapping.json")
    cmmlu_map = _load_mapping(task_dir / "cmmlu" / "mapping.json")

    t0 = time.time()
    total = 0
    counts = {"mmlu": 0, "ceval": 0, "cmmlu": 0}
    lens = []

    with items_path.open("w", encoding="utf-8") as f:
        # MMLU (single subject "all")
        subject = "all"
        subject_name = mmlu_map[subject]["name"]
        ds_eval = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        ds_train = None
        if args.n_shot_en > 0:
            ds_train = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=True)

        n = len(ds_eval) if args.limit_mmlu == 0 else min(args.limit_mmlu, len(ds_eval))
        for i in range(n):
            target = _to_std("mmlu", ds_eval[i])
            support = []
            if ds_train is not None and args.n_shot_en > 0:
                support = ds_train.shuffle(seed=args.seed + i).select(range(min(args.n_shot_en, len(ds_train))))
                support = [_to_std("mmlu", x) for x in support]
            messages = eval_template_en.format_example(target_data=target, support_set=support, subject_name=subject_name)
            input_ids, _ = lf_template.encode_oneturn(tokenizer=tokenizer, messages=messages)
            gold = str(messages[-1]["content"]).strip().upper()[:1]

            rec = {
                "qid": f"mmlu:{i}",
                "task": "mmlu",
                "subject": subject,
                "subject_name": subject_name,
                "lang": "en",
                "answer": gold,
                "prompt_token_ids": input_ids,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            counts["mmlu"] += 1
            lens.append(len(input_ids))
        print(f"mmlu loaded: {n}")

        # C-Eval
        ceval_subjects = _iter_subjects(ceval_map, args.ceval_subject_limit)
        for si, subject in enumerate(ceval_subjects, start=1):
            subject_name = ceval_map[subject]["name"]
            ds_eval = load_dataset("ceval/ceval-exam", subject, split="val")
            ds_train = None
            if args.n_shot_zh > 0:
                ds_train = load_dataset("ceval/ceval-exam", subject, split="dev")

            n = len(ds_eval) if args.limit_per_subject == 0 else min(args.limit_per_subject, len(ds_eval))
            for i in range(n):
                target = _to_std("ceval", ds_eval[i])
                support = []
                if ds_train is not None and args.n_shot_zh > 0:
                    support = ds_train.shuffle(seed=args.seed + i).select(range(min(args.n_shot_zh, len(ds_train))))
                    support = [_to_std("ceval", x) for x in support]
                messages = eval_template_zh.format_example(target_data=target, support_set=support, subject_name=subject_name)
                input_ids, _ = lf_template.encode_oneturn(tokenizer=tokenizer, messages=messages)
                gold = str(messages[-1]["content"]).strip().upper()[:1]

                rec = {
                    "qid": f"ceval:{subject}:{i}",
                    "task": "ceval",
                    "subject": subject,
                    "subject_name": subject_name,
                    "lang": "zh",
                    "answer": gold,
                    "prompt_token_ids": input_ids,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                counts["ceval"] += 1
                lens.append(len(input_ids))

            if si % 5 == 0 or si == len(ceval_subjects):
                print(f"ceval subjects: {si}/{len(ceval_subjects)}")

        # CMMLU
        cmmlu_subjects = _iter_subjects(cmmlu_map, args.cmmlu_subject_limit)
        for si, subject in enumerate(cmmlu_subjects, start=1):
            subject_name = cmmlu_map[subject]["name"]
            ds_eval = load_dataset("haonan-li/cmmlu", subject, split="test", trust_remote_code=True)
            ds_train = None
            if args.n_shot_zh > 0:
                ds_train = load_dataset("haonan-li/cmmlu", subject, split="dev", trust_remote_code=True)

            n = len(ds_eval) if args.limit_per_subject == 0 else min(args.limit_per_subject, len(ds_eval))
            for i in range(n):
                target = _to_std("cmmlu", ds_eval[i])
                support = []
                if ds_train is not None and args.n_shot_zh > 0:
                    support = ds_train.shuffle(seed=args.seed + i).select(range(min(args.n_shot_zh, len(ds_train))))
                    support = [_to_std("cmmlu", x) for x in support]
                messages = eval_template_zh.format_example(target_data=target, support_set=support, subject_name=subject_name)
                input_ids, _ = lf_template.encode_oneturn(tokenizer=tokenizer, messages=messages)
                gold = str(messages[-1]["content"]).strip().upper()[:1]

                rec = {
                    "qid": f"cmmlu:{subject}:{i}",
                    "task": "cmmlu",
                    "subject": subject,
                    "subject_name": subject_name,
                    "lang": "zh",
                    "answer": gold,
                    "prompt_token_ids": input_ids,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                counts["cmmlu"] += 1
                lens.append(len(input_ids))

            if si % 5 == 0 or si == len(cmmlu_subjects):
                print(f"cmmlu subjects: {si}/{len(cmmlu_subjects)}")

    lens_sorted = sorted(lens)
    def _pct(p: float) -> int:
        if not lens_sorted:
            return 0
        idx = int((len(lens_sorted) - 1) * p)
        return lens_sorted[idx]

    info = {
        "items_path": str(items_path),
        "total": total,
        **counts,
        "elapsed_sec": round(time.time() - t0, 3),
        "limits": {
            "mmlu": None if args.limit_mmlu == 0 else args.limit_mmlu,
            "per_subject": None if args.limit_per_subject == 0 else args.limit_per_subject,
            "ceval_subject_limit": args.ceval_subject_limit,
            "cmmlu_subject_limit": args.cmmlu_subject_limit,
        },
        "lf_align": {
            "template": args.template,
            "n_shot_en": args.n_shot_en,
            "n_shot_zh": args.n_shot_zh,
            "base_model": args.base_model,
        },
        "token_len_stats": {
            "min": min(lens_sorted) if lens_sorted else 0,
            "p50": _pct(0.50),
            "p95": _pct(0.95),
            "max": max(lens_sorted) if lens_sorted else 0,
        },
    }
    (out_dir / "data_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
