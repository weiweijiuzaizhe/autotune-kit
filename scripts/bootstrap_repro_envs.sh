#!/usr/bin/env bash
set -euo pipefail

TRAIN_ENV="${TRAIN_ENV:-py310}"
SCORE_ENV="${SCORE_ENV:-vllm014}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif [[ -x "/usr/local/miniconda3/bin/conda" ]]; then
  CONDA_BIN="/usr/local/miniconda3/bin/conda"
else
  CONDA_BIN="conda"
fi

log(){ echo "[bootstrap] $*"; }

have_env(){
  "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "$1"
}

ensure_env(){
  local env_name="$1"
  if have_env "${env_name}"; then
    log "conda env exists: ${env_name}"
  else
    log "creating conda env: ${env_name}"
    "${CONDA_BIN}" create -y -n "${env_name}" python=3.10
  fi
}

ensure_env "${TRAIN_ENV}"
ensure_env "${SCORE_ENV}"

log "installing train stack in ${TRAIN_ENV}"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python -m pip install -U pip
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python -m pip install \
  "transformers==4.52.4" accelerate datasets peft sentencepiece \
  "bitsandbytes==0.49.1" "llamafactory==0.9.3" "unsloth==2025.8.10" hf_transfer

log "installing vLLM stack in ${SCORE_ENV}"
"${CONDA_BIN}" run -n "${SCORE_ENV}" python -m pip install -U pip
"${CONDA_BIN}" run -n "${SCORE_ENV}" python -m pip install \
  "vllm==0.14.1" "transformers==4.57.6" datasets

log "writing HF cache env to ~/.bashrc"
if ! grep -q 'HF_HOME=~/.cache/hf' ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<'EOF'
export HF_HOME=~/.cache/hf
export HF_HUB_CACHE=~/.cache/hf/hub
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF
fi
export HF_HOME=~/.cache/hf
export HF_HUB_CACHE=~/.cache/hf/hub
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HUB_CACHE}"

log "warmup tokenizer cache for ${BASE_MODEL}"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(\"${BASE_MODEL}\", trust_remote_code=True); print(\"tokenizer_ok\")"

log "version check"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python -c "import torch,transformers; print('[train_env] torch', torch.__version__, 'cuda', torch.version.cuda); print('[train_env] transformers', transformers.__version__)"
"${CONDA_BIN}" run -n "${SCORE_ENV}" python -c "import torch,transformers,vllm; print('[score_env] torch', torch.__version__, 'cuda', torch.version.cuda); print('[score_env] transformers', transformers.__version__); print('[score_env] vllm', vllm.__version__)"

log "done"
