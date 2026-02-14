import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from atk.utils import run_subprocess, save_json, save_text, shell_join


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _token_stats(model_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    lengths: List[Tuple[int, int]] = []
    for i, r in enumerate(rows):
        text = f"Instruction: {r.get('instruction','')}\nInput: {r.get('input','')}\nOutput: {r.get('output','')}"
        l = len(tok(text, add_special_tokens=True)["input_ids"])
        lengths.append((i, l))

    vals = sorted(l for _, l in lengths)

    def pct(p: float) -> int:
        if not vals:
            return 0
        k = (len(vals) - 1) * p
        f = int(k)
        c = min(f + 1, len(vals) - 1)
        if f == c:
            return vals[f]
        return int(vals[f] + (vals[c] - vals[f]) * (k - f))

    top_idx, top_len = max(lengths, key=lambda x: x[1])
    return {
        "num_samples": len(rows),
        "min": min(vals) if vals else 0,
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": max(vals) if vals else 0,
        "top1": {"index": int(top_idx), "length": int(top_len)},
    }


def _mk_temp_dataset(run_dir: Path, top_row: Dict[str, Any]) -> Path:
    d = run_dir / "safe_launch" / "dataset"
    d.mkdir(parents=True, exist_ok=True)
    train_jsonl = d / "train.jsonl"

    # repeat 8 times to avoid empty dataloader edge cases
    with train_jsonl.open("w", encoding="utf-8") as f:
        for _ in range(8):
            f.write(json.dumps(top_row, ensure_ascii=False) + "\n")

    dataset_info = {
        "atk_safe": {
            "file_name": "train.jsonl",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    }
    (d / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return d


def run_safe_launch(
    *,
    run_dir: Path,
    model_base: str,
    train_jsonl_path: Path,
    base_train_yaml: Path,
    cutoff_len: int,
    qlora_4bit: bool,
    planned_micro_batch: int,
) -> Dict[str, Any]:
    """v0.1 Safe Launch

    1) Token stats + top1 sample
    2) Mini trial run with micro_batch=1, grad_acc=1, max_steps=5
    3) Risk gate for the *planned* training config. If too risky, intercept early.

    This keeps the implementation minimal while still preventing obviously risky runs.
    """

    rows = _load_jsonl(train_jsonl_path)
    stats = _token_stats(model_base, rows)
    save_json(run_dir / "data_stats.json", stats)

    top_row = rows[stats["top1"]["index"]]
    ds_dir = _mk_temp_dataset(run_dir, top_row)

    import yaml

    cfg = yaml.safe_load(base_train_yaml.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("base train.yaml must be mapping")

    out_dir = run_dir / "safe_launch" / "output_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    # LLaMA-Factory compatible minimal overrides (trial)
    cfg["model_name_or_path"] = model_base
    cfg["dataset_dir"] = str(ds_dir)
    cfg["dataset"] = "atk_safe"
    cfg["cutoff_len"] = int(cutoff_len)
    cfg["max_steps"] = 5
    cfg["per_device_train_batch_size"] = 1
    cfg["gradient_accumulation_steps"] = 1
    cfg["output_dir"] = str(out_dir)
    cfg["overwrite_output_dir"] = True
    cfg["report_to"] = "none"

    if qlora_4bit:
        cfg["finetuning_type"] = "lora"
        cfg["quantization_bit"] = 4
    else:
        cfg.pop("quantization_bit", None)

    ypath = run_dir / "safe_launch" / "train.safe.yaml"
    save_text(ypath, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    cmd = ["llamafactory-cli", "train", str(ypath)]
    log_path = run_dir / "safe_launch" / "lf_train.safe.log"
    rc = run_subprocess(cmd, log_path=log_path)

    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    err_kw = ""
    if re.search(r"oom|out of memory", text, re.I):
        err_kw = "OOM"
    elif re.search(r"\bnan\b", text, re.I):
        err_kw = "NaN"

    trial_passed = rc == 0

    # Risk gate: deliberately intercept risky planned configs.
    risk_level = "LOW"
    if planned_micro_batch >= 2 and cutoff_len >= 4096:
        risk_level = "HIGH"

    passed = trial_passed and (risk_level != "HIGH")

    suggestion = ""
    if not trial_passed:
        suggestion = "Trial run failed. Try lowering cutoff_len or enabling QLoRA 4bit; if OOM, lower cutoff_len first."
    elif risk_level == "HIGH":
        err_kw = err_kw or "HIGH_RISK"
        suggestion = "Planned config is HIGH RISK. Recommended: cutoff_len=1024 and micro_batch=1."

    return {
        "passed": passed,
        "trial_passed": trial_passed,
        "exit_code": rc,
        "error_keyword": err_kw,
        "risk_level": risk_level,
        "planned": {"cutoff_len": int(cutoff_len), "micro_batch": int(planned_micro_batch)},
        "suggestion": suggestion,
        "cmd": shell_join(cmd),
        "log": str(log_path),
    }
