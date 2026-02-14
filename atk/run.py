import os
import subprocess
from pathlib import Path
from typing import Dict, Any

from atk.config import load_config, dump_yaml
from atk.safelaunch import run_safe_launch
from atk.sanity import run_sanity
from atk.report import build_report, write_report
from atk.utils import create_run_dir, run_subprocess, save_text, save_json, shell_join


LF_BASE_TRAIN_YAML = Path(__file__).resolve().parent / "templates" / "lf_train_template.yaml"


def _save_env(run_dir: Path) -> str:
    parts = []
    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as e:
        smi = f"nvidia-smi failed: {e}\n"

    parts.append("=== nvidia-smi ===\n" + smi)

    try:
        import torch
        parts.append("=== torch ===\n" + f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n")
    except Exception as e:
        parts.append(f"torch check failed: {e}\n")

    env_text = "\n".join(parts)
    save_text(run_dir / "env.txt", env_text)
    return env_text


def _prepare_dataset(run_dir: Path, train_jsonl: Path) -> Path:
    d = run_dir / "dataset"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.jsonl").write_bytes(train_jsonl.read_bytes())

    dataset_info = {
        "atk_train": {
            "file_name": "train.jsonl",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
    }
    import json
    save_text(d / "dataset_info.json", json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n")
    return d


def _make_lf_yaml(run_dir: Path, cfg: Dict[str, Any], dataset_dir: Path, output_model_dir: Path) -> Path:
    import yaml

    base_yaml = LF_BASE_TRAIN_YAML
    if not base_yaml.exists():
        raise FileNotFoundError(f"missing base train.yaml template: {base_yaml}")

    y = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    if not isinstance(y, dict):
        raise ValueError("base train.yaml must be mapping")

    t = cfg["train"]

    y["model_name_or_path"] = cfg["model"]["base"]
    y["template"] = t["template"]
    y["dataset_dir"] = str(dataset_dir)
    y["dataset"] = "atk_train"
    y["cutoff_len"] = int(t["cutoff_len"])
    y["max_steps"] = int(t["max_steps"])
    y["per_device_train_batch_size"] = int(t["micro_batch"])
    y["gradient_accumulation_steps"] = int(t["grad_acc"])
    y["learning_rate"] = float(t["lr"])
    y["num_train_epochs"] = float(t["epochs"])

    y["output_dir"] = str(output_model_dir)
    y["overwrite_output_dir"] = True
    y["report_to"] = "none"

    if bool(t.get("qlora_4bit")):
        y["finetuning_type"] = "lora"
        y["quantization_bit"] = 4
    else:
        y.pop("quantization_bit", None)

    out = run_dir / "train.yaml"
    out.write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def run_pipeline(cfg_path: str, raw_cfg: Dict[str, Any]) -> int:
    run_dir = create_run_dir()

    # persist effective config
    save_text(run_dir / "config.effective.yaml", dump_yaml(raw_cfg))

    env_text = _save_env(run_dir)

    # dataset staging
    ds_dir = _prepare_dataset(run_dir, Path(raw_cfg["data"]["train"]))

    # safe launch
    safe = run_safe_launch(
        run_dir=run_dir,
        model_base=raw_cfg["model"]["base"],
        train_jsonl_path=Path(raw_cfg["data"]["train"]),
        base_train_yaml=LF_BASE_TRAIN_YAML,
        cutoff_len=int(raw_cfg["train"]["cutoff_len"]),
        qlora_4bit=bool(raw_cfg["train"].get("qlora_4bit")),
        planned_micro_batch=int(raw_cfg["train"].get("micro_batch", 1)),
    )
    save_json(run_dir / "safe_launch.json", safe)

    if not safe.get("passed"):
        import json
        data_stats = json.loads((run_dir / "data_stats.json").read_text(encoding="utf-8"))
        content = build_report(
            run_dir=run_dir,
            env_text=env_text,
            data_stats=data_stats,
            safe_launch=safe,
            train_exit_code=999,
            sanity={"base": {}, "tuned": {}, "delta": {}},
        )
        write_report(run_dir, content)
        print(f"[ATK] safe launch failed, exiting. run_dir={run_dir}")
        return 1

    # training output dir under cfg.output.dir/run_<ts>
    out_root = Path(raw_cfg["output"]["dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    output_model_dir = out_root / run_dir.name
    output_model_dir.mkdir(parents=True, exist_ok=True)

    lf_yaml = _make_lf_yaml(run_dir, raw_cfg, ds_dir, output_model_dir)

    cmd = ["llamafactory-cli", "train", str(lf_yaml)]
    save_text(run_dir / "train.cmd.txt", shell_join(cmd) + "\n")
    rc = run_subprocess(cmd, log_path=run_dir / "lf_train.log")

    sanity = {}
    if rc == 0:
        sanity = run_sanity(
            run_dir=run_dir,
            model_base=raw_cfg["model"]["base"],
            adapter_dir=output_model_dir,
            qlora_4bit=bool(raw_cfg["train"].get("qlora_4bit")),
        )
    else:
        sanity = {"base": {}, "tuned": {}, "delta": {}}

    import json
    data_stats = json.loads((run_dir / "data_stats.json").read_text(encoding="utf-8"))

    content = build_report(
        run_dir=run_dir,
        env_text=env_text,
        data_stats=data_stats,
        safe_launch=safe,
        train_exit_code=rc,
        sanity=sanity,
    )
    report_path = write_report(run_dir, content)

    print(f"[ATK] run_dir={run_dir}")
    print(f"[ATK] report={report_path}")
    try:
        head = report_path.read_text(encoding="utf-8").splitlines()[:20]
        print("[ATK] report head:")
        for ln in head:
            print(ln)
    except Exception:
        pass

    return 0 if rc == 0 else 2


def run_from_config_path(path: str) -> int:
    cfg = load_config(path)
    return run_pipeline(path, cfg.raw)


def rerun_from_run_dir(run_dir: str) -> int:
    p = Path(run_dir) / "config.effective.yaml"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    import yaml
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config.effective.yaml must be mapping")
    return run_pipeline(str(p), raw)
