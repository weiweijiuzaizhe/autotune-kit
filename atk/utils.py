import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def create_run_dir(base: str = "/root/atk_runs") -> Path:
    base_p = Path(base)
    base_p.mkdir(parents=True, exist_ok=True)
    run_dir = base_p / f"run_{now_ts()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _merged_env(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Ensure the current Python's bin dir is on PATH.

    This makes subprocess calls (e.g. llamafactory-cli) work even when users
    invoke /path/to/atk directly without activating a shell env.
    """

    env = os.environ.copy()
    py_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = py_bin + os.pathsep + env.get("PATH", "")
    if extra_env:
        env.update(extra_env)
    return env


def run_subprocess(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    timeout_s: Optional[int] = None,
) -> int:
    """Run a subprocess and stream output to log_path (and return code)."""

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_path, "w", encoding="utf-8")
    else:
        f = None

    merged_env = _merged_env(env)

    try:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            if f:
                f.write(line)
            else:
                print(line, end="")
        return p.wait(timeout=timeout_s)
    finally:
        if f:
            f.close()


def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)
