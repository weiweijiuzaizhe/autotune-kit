#!/usr/bin/env python3
"""Generate demo Alpaca JSONL for ATK training.

Creates 200 regular samples + 3 long samples.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "Grace", "Henry", "Iris", "Jack",
    "Kate", "Leo", "Mia", "Noah", "Olivia", "Peter", "Queenie", "Ryan", "Sophia", "Tom",
]
CITIES = [
    "Beijing", "Shanghai", "Shenzhen", "Guangzhou", "Chengdu", "Hangzhou", "Nanjing", "Wuhan",
    "Xi'an", "Suzhou", "Tianjin", "Chongqing", "Qingdao", "Xiamen", "Harbin",
]


def _record(name: str, age: int, city: str, extra: str = "") -> dict:
    instruction = (
        "You must output STRICT JSON only. "
        "Required keys: name, age, city. "
        "No markdown, no extra keys, no explanation."
    )
    input_text = f"name={name}; age={age}; city={city}. {extra}".strip()
    output = json.dumps({"name": name, "age": age, "city": city}, ensure_ascii=False)
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regular", type=int, default=200)
    ap.add_argument("--long", type=int, default=3)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.regular):
        name = random.choice(NAMES)
        age = random.randint(18, 65)
        city = random.choice(CITIES)
        extra = f"sample_id={i}" if i % 3 == 0 else ""
        rows.append(_record(name, age, city, extra))

    long_phrase = "This is a stress test token. "
    for i in range(args.long):
        name = f"LongCase{i+1}"
        age = 30 + i
        city = "Beijing"
        extra = long_phrase * 1200
        rows.append(_record(name, age, city, extra))

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(out), "count": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
