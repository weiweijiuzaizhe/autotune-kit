from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ATKConfig:
    raw: Dict[str, Any]

    @property
    def model_base(self) -> str:
        return self.raw["model"]["base"]

    @property
    def data_train(self) -> str:
        return self.raw["data"]["train"]

    @property
    def output_dir(self) -> str:
        return self.raw["output"]["dir"]

    @property
    def train(self) -> Dict[str, Any]:
        return self.raw["train"]


def load_config(path: str) -> ATKConfig:
    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("config must be a YAML mapping")

    # minimal validation
    for k in ["model", "data", "output", "train"]:
        if k not in obj:
            raise ValueError(f"missing top-level key: {k}")

    for k in ["base"]:
        if k not in obj["model"]:
            raise ValueError(f"missing model.{k}")

    for k in ["train"]:
        if k not in obj["data"]:
            raise ValueError(f"missing data.{k}")

    for k in ["dir"]:
        if k not in obj["output"]:
            raise ValueError(f"missing output.{k}")

    t = obj["train"]
    required = [
        "template",
        "cutoff_len",
        "max_steps",
        "micro_batch",
        "grad_acc",
        "lr",
        "epochs",
        "qlora_4bit",
    ]
    for k in required:
        if k not in t:
            raise ValueError(f"missing train.{k}")

    return ATKConfig(raw=obj)


def dump_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
