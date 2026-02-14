import shutil
import subprocess
import sys
from typing import List, Optional


_PYTORCH_INDEX = {
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


def _run(cmd: List[str]) -> int:
    print("[ATK][bootstrap] $", " ".join(cmd), flush=True)
    p = subprocess.run(cmd)
    return int(p.returncode)


def bootstrap(
    *,
    cuda: str = "cu126",
    install_torch: bool = True,
    torch_spec: Optional[str] = None,
) -> int:
    """Install runtime dependencies so a bare machine can run `atk run`.

    Note: Installing PyTorch CUDA wheels typically requires an extra index URL.
    This helper hides that detail so users don't have to type it.
    """

    if shutil.which("git") is None:
        print("[ATK][bootstrap] ERROR: git not found. Install git first.", file=sys.stderr)
        return 2

    py = sys.executable

    # Upgrade pip first (best-effort).
    rc = _run([py, "-m", "pip", "install", "-U", "pip"])
    if rc != 0:
        return rc

    if install_torch:
        idx = _PYTORCH_INDEX.get(cuda)
        if not idx:
            print(
                f"[ATK][bootstrap] ERROR: unsupported --cuda {cuda}. Use: {', '.join(_PYTORCH_INDEX)}",
                file=sys.stderr,
            )
            return 2

        spec = torch_spec
        if spec is None:
            # Default to the repo's baseline; users can override.
            spec = "torch==2.7.0+cu126" if cuda == "cu126" else "torch"

        rc = _run([py, "-m", "pip", "install", "--extra-index-url", idx, spec])
        if rc != 0:
            return rc

    deps = [
        "transformers>=4.40",
        "accelerate",
        "datasets",
        "peft",
        "sentencepiece",
        "llamafactory",
        "bitsandbytes",
    ]
    return _run([py, "-m", "pip", "install", *deps])
