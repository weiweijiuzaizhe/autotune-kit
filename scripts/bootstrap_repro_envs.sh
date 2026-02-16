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

has_module(){
  local env_name="$1"
  local module_name="$2"
  "${CONDA_BIN}" run -n "${env_name}" python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('${module_name}') else 1)" >/dev/null 2>&1
}

pip_install(){
  local env_name="$1"
  shift
  "${CONDA_BIN}" run -n "${env_name}" python -m pip install --upgrade-strategy only-if-needed "$@"
}

pip_install_if_missing(){
  local env_name="$1"
  local module_name="$2"
  shift 2
  if has_module "${env_name}" "${module_name}"; then
    log "${env_name}: ${module_name} already installed"
  else
    log "${env_name}: installing ${module_name}"
    pip_install "${env_name}" "$@"
  fi
}

ensure_env "${TRAIN_ENV}"
ensure_env "${SCORE_ENV}"

# Keep existing torch in train env if already present.
if has_module "${TRAIN_ENV}" torch; then
  TORCH_BEFORE="$("${CONDA_BIN}" run -n "${TRAIN_ENV}" python -c "import torch; print(torch.__version__)")"
  log "${TRAIN_ENV}: keep existing torch=${TORCH_BEFORE}"
else
  log "${TRAIN_ENV}: torch missing, installing torch 2.7 cu126"
  "${CONDA_BIN}" run -n "${TRAIN_ENV}" python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.0 torchvision torchaudio
fi

log "installing train stack in ${TRAIN_ENV}"
pip_install_if_missing "${TRAIN_ENV}" transformers "transformers==4.52.4"
pip_install_if_missing "${TRAIN_ENV}" accelerate accelerate
pip_install_if_missing "${TRAIN_ENV}" datasets datasets
pip_install_if_missing "${TRAIN_ENV}" peft peft
pip_install_if_missing "${TRAIN_ENV}" sentencepiece sentencepiece
pip_install_if_missing "${TRAIN_ENV}" bitsandbytes "bitsandbytes==0.49.1"
pip_install_if_missing "${TRAIN_ENV}" llamafactory "llamafactory==0.9.3"
pip_install_if_missing "${TRAIN_ENV}" hf_transfer hf_transfer

if has_module "${TRAIN_ENV}" unsloth; then
  log "${TRAIN_ENV}: unsloth already installed"
else
  log "${TRAIN_ENV}: installing unsloth without deps to avoid torch drift"
  "${CONDA_BIN}" run -n "${TRAIN_ENV}" python -m pip install --no-deps "unsloth==2025.8.10" \
    || pip_install "${TRAIN_ENV}" "unsloth==2025.8.10"
fi

if has_module "${TRAIN_ENV}" torch; then
  TORCH_AFTER="$("${CONDA_BIN}" run -n "${TRAIN_ENV}" python -c "import torch; print(torch.__version__)")"
  log "${TRAIN_ENV}: torch after install=${TORCH_AFTER}"
fi

log "installing vLLM stack in ${SCORE_ENV}"
pip_install_if_missing "${SCORE_ENV}" vllm "vllm==0.14.1"
pip_install_if_missing "${SCORE_ENV}" transformers "transformers==4.57.6"
pip_install_if_missing "${SCORE_ENV}" datasets datasets
pip_install_if_missing "${SCORE_ENV}" bitsandbytes "bitsandbytes>=0.46.1"
pip_install_if_missing "${SCORE_ENV}" hf_transfer hf_transfer

log "writing HF cache env to ~/.bashrc"
if ! grep -q 'HF_HOME=~/.cache/hf' ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<'BASHRC_EOF'
export HF_HOME=~/.cache/hf
export HF_HUB_CACHE=~/.cache/hf/hub
export HF_HUB_ENABLE_HF_TRANSFER=1
BASHRC_EOF
fi
export HF_HOME=~/.cache/hf
export HF_HUB_CACHE=~/.cache/hf/hub
mkdir -p "${HF_HUB_CACHE}"

if has_module "${TRAIN_ENV}" hf_transfer && has_module "${SCORE_ENV}" hf_transfer; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  log "HF transfer enabled"
else
  export HF_HUB_ENABLE_HF_TRANSFER=0
  log "HF transfer disabled because hf_transfer missing in one env"
fi

log "warmup tokenizer cache for ${BASE_MODEL}"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${BASE_MODEL}', trust_remote_code=True); print('tokenizer_ok')"

log "version check"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python -c "import torch,transformers; print('[train_env] torch', torch.__version__, 'cuda', torch.version.cuda); print('[train_env] transformers', transformers.__version__)"
"${CONDA_BIN}" run -n "${SCORE_ENV}" python -c "import torch,transformers,vllm; print('[score_env] torch', torch.__version__, 'cuda', torch.version.cuda); print('[score_env] transformers', transformers.__version__); print('[score_env] vllm', vllm.__version__)"

log "done"
