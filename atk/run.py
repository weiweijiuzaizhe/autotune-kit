import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from atk.config import load_config, dump_yaml
from atk.safelaunch import run_safe_launch
from atk.sanity import run_sanity, run_quick_sanity
from atk.report import build_report, write_report
from atk.utils import create_run_dir, run_subprocess, save_text, save_json, shell_join


LF_BASE_TRAIN_YAML = Path(__file__).resolve().parent / "templates" / "lf_train_template.yaml"
VLLM_QUICK4D_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "vllm_routeA" / "quick4d_vllm_eval.py"


def _save_env(run_dir: Path) -> str:
    parts = []
    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as e:
        smi = f"nvidia-smi failed: {e}\n"

    parts.append("=== nvidia-smi ===\n" + smi)

    try:
        import torch

        parts.append(
            "=== torch ===\n"
            + f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n"
        )
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
    save_text(d / "dataset_info.json", json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n")
    return d


def _fallback_data_stats(run_dir: Path, train_jsonl: Path) -> Dict[str, Any]:
    # Fast fallback stats for quick mode when Safe Launch is disabled.
    num = 0
    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                num += 1

    stats = {
        "num_samples": num,
        "min": 0,
        "p50": 0,
        "p95": 0,
        "max": 0,
        "top1": {"index": 0, "length": 0},
        "note": "safe_launch_disabled_stats",
    }
    save_json(run_dir / "data_stats.json", stats)
    return stats


def _load_data_stats(run_dir: Path, train_jsonl: Path) -> Dict[str, Any]:
    p = run_dir / "data_stats.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return _fallback_data_stats(run_dir, train_jsonl)


def _is_enabled(raw_cfg: Dict[str, Any], section: str, default: bool = True) -> bool:
    v = raw_cfg.get(section)
    if isinstance(v, dict):
        return bool(v.get("enabled", default))
    return default


def _get_int(raw_cfg: Dict[str, Any], section: str, key: str, default: int) -> int:
    v = raw_cfg.get(section)
    if isinstance(v, dict) and key in v:
        try:
            return int(v[key])
        except Exception:
            return default
    return default


def _get_str(raw_cfg: Dict[str, Any], section: str, key: str, default: str) -> str:
    v = raw_cfg.get(section)
    if isinstance(v, dict) and key in v and v[key] is not None:
        return str(v[key])
    return default


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


def _run_quick_eval_vllm(
    *,
    run_dir: Path,
    model_base: str,
    adapter_dir: Path,
    n_per_dim: int,
    score_env: str,
    batch_size: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not VLLM_QUICK4D_SCRIPT.exists():
        return None, f"vllm_script_missing:{VLLM_QUICK4D_SCRIPT}"

    out_path = run_dir / "sanity_report.json"
    log_path = run_dir / "quick_eval.log"

    if score_env:
        conda_bin = shutil.which("conda")
        if not conda_bin:
            return None, "conda_not_found"
        cmd = [
            conda_bin,
            "run",
            "-n",
            score_env,
            "python",
            str(VLLM_QUICK4D_SCRIPT),
            "--base-model",
            model_base,
            "--adapter-dir",
            str(adapter_dir),
            "--n-per-dim",
            str(n_per_dim),
            "--out",
            str(out_path),
            "--max-num-seqs",
            str(max(32, batch_size)),
        ]
    else:
        cmd = [
            sys.executable,
            str(VLLM_QUICK4D_SCRIPT),
            "--base-model",
            model_base,
            "--adapter-dir",
            str(adapter_dir),
            "--n-per-dim",
            str(n_per_dim),
            "--out",
            str(out_path),
            "--max-num-seqs",
            str(max(32, batch_size)),
        ]

    save_text(run_dir / "quick_eval.cmd.txt", shell_join(cmd) + "\n")
    rc = run_subprocess(cmd, log_path=log_path)
    if rc != 0:
        return None, f"vllm_exit_code:{rc}"

    try:
        report = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"vllm_bad_output:{e}"

    return report, ""


def run_pipeline(cfg_path: str, raw_cfg: Dict[str, Any]) -> int:
    run_dir = create_run_dir()

    # persist effective config
    save_text(run_dir / "config.effective.yaml", dump_yaml(raw_cfg))

    env_text = _save_env(run_dir)

    train_jsonl = Path(raw_cfg["data"]["train"])

    # dataset staging
    ds_dir = _prepare_dataset(run_dir, train_jsonl)

    safe_launch_enabled = _is_enabled(raw_cfg, "safe_launch", default=True)
    sanity_enabled = _is_enabled(raw_cfg, "sanity", default=True)
    quick_eval_enabled = _is_enabled(raw_cfg, "quick_eval", default=False)
    quick_eval_n_per_dim = _get_int(raw_cfg, "quick_eval", "n_per_dim", 120)
    quick_eval_batch_size = _get_int(raw_cfg, "quick_eval", "batch_size", 64)
    quick_eval_backend = _get_str(raw_cfg, "quick_eval", "backend", "vllm").strip().lower()
    quick_eval_score_env = _get_str(raw_cfg, "quick_eval", "score_env", "vllm014").strip()

    # safe launch
    if safe_launch_enabled:
        safe = run_safe_launch(
            run_dir=run_dir,
            model_base=raw_cfg["model"]["base"],
            train_jsonl_path=train_jsonl,
            base_train_yaml=LF_BASE_TRAIN_YAML,
            cutoff_len=int(raw_cfg["train"]["cutoff_len"]),
            qlora_4bit=bool(raw_cfg["train"].get("qlora_4bit")),
            planned_micro_batch=int(raw_cfg["train"].get("micro_batch", 1)),
        )
    else:
        safe = {
            "passed": True,
            "skipped": True,
            "trial_passed": None,
            "exit_code": None,
            "error_keyword": "SAFE_LAUNCH_DISABLED",
            "risk_level": "SKIPPED",
            "planned": {
                "cutoff_len": int(raw_cfg["train"]["cutoff_len"]),
                "micro_batch": int(raw_cfg["train"].get("micro_batch", 1)),
            },
            "suggestion": "Set safe_launch.enabled=true for preflight OOM interception.",
            "cmd": "",
            "log": "",
        }

    save_json(run_dir / "safe_launch.json", safe)
    data_stats = _load_data_stats(run_dir, train_jsonl)

    if not safe.get("passed"):
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

    if rc == 0 and sanity_enabled:
        sanity = run_sanity(
            run_dir=run_dir,
            model_base=raw_cfg["model"]["base"],
            adapter_dir=output_model_dir,
            qlora_4bit=bool(raw_cfg["train"].get("qlora_4bit")),
        )
    elif rc == 0 and quick_eval_enabled:
        if quick_eval_backend == "vllm":
            sanity, err = _run_quick_eval_vllm(
                run_dir=run_dir,
                model_base=raw_cfg["model"]["base"],
                adapter_dir=output_model_dir,
                n_per_dim=quick_eval_n_per_dim,
                score_env=quick_eval_score_env,
                batch_size=quick_eval_batch_size,
            )
            if sanity is None:
                sanity = run_quick_sanity(
                    run_dir=run_dir,
                    model_base=raw_cfg["model"]["base"],
                    adapter_dir=output_model_dir,
                    qlora_4bit=bool(raw_cfg["train"].get("qlora_4bit")),
                    n_per_dim=quick_eval_n_per_dim,
                    batch_size=quick_eval_batch_size,
                )
                old_note = str(sanity.get("note", "")).strip()
                sanity["backend"] = "hf_batch_fallback"
                sanity["note"] = f"{old_note}; vllm_fallback_reason={err}".strip("; ")
                save_json(run_dir / "quick_sanity_report.json", sanity)
                save_json(run_dir / "sanity_report.json", sanity)
        else:
            sanity = run_quick_sanity(
                run_dir=run_dir,
                model_base=raw_cfg["model"]["base"],
                adapter_dir=output_model_dir,
                qlora_4bit=bool(raw_cfg["train"].get("qlora_4bit")),
                n_per_dim=quick_eval_n_per_dim,
                batch_size=quick_eval_batch_size,
            )
    elif rc == 0:
        sanity = {
            "skipped": True,
            "base": {},
            "tuned": {},
            "delta": {},
            "note": "sanity.disabled=true",
        }
    else:
        sanity = {"base": {}, "tuned": {}, "delta": {}}

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
