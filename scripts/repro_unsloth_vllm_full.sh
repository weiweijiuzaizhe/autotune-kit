#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TRAIN_ENV="${TRAIN_ENV:-py310}"
SCORE_ENV="${SCORE_ENV:-vllm014}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-${ROOT_DIR}/data/train.jsonl}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/artifacts/repro_runs}"
MAX_STEPS="${MAX_STEPS:-200}"
CUTOFF_LEN="${CUTOFF_LEN:-1024}"
MICRO_BATCH="${MICRO_BATCH:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
LIMIT_MMLU="${LIMIT_MMLU:-0}"
LIMIT_PER_SUBJECT="${LIMIT_PER_SUBJECT:-0}"
CEVAL_SUBJECT_LIMIT="${CEVAL_SUBJECT_LIMIT:-0}"
CMMLU_SUBJECT_LIMIT="${CMMLU_SUBJECT_LIMIT:-0}"
TRUTHFUL_LIMIT="${TRUTHFUL_LIMIT:-0}"
JBB_HARMFUL_LIMIT="${JBB_HARMFUL_LIMIT:-0}"
JBB_BENIGN_LIMIT="${JBB_BENIGN_LIMIT:-0}"
SKIP_BOOTSTRAP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-bootstrap) SKIP_BOOTSTRAP=1; shift ;;
    --train-jsonl) TRAIN_JSONL="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --train-env) TRAIN_ENV="$2"; shift 2 ;;
    --score-env) SCORE_ENV="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif [[ -x "/usr/local/miniconda3/bin/conda" ]]; then
  CONDA_BIN="/usr/local/miniconda3/bin/conda"
else
  CONDA_BIN="conda"
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_ROOT}/repro_${TS}"
mkdir -p "${RUN_DIR}"
mkdir -p "${RUN_DIR}/evals"

LOG_PATH="${RUN_DIR}/pipeline.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

step() {
  local idx="$1"
  local total="$2"
  local msg="$3"
  echo ""
  echo "[$idx/$total] ${msg}"
}

echo "[ATK] run_dir=${RUN_DIR}"

echo "[ATK] settings: TRAIN_ENV=${TRAIN_ENV}, SCORE_ENV=${SCORE_ENV}, BASE_MODEL=${BASE_MODEL}"

TOTAL=9

if [[ ${SKIP_BOOTSTRAP} -eq 0 ]]; then
  step 1 ${TOTAL} "bootstrap envs + cache warmup"
  TRAIN_ENV="${TRAIN_ENV}" SCORE_ENV="${SCORE_ENV}" BASE_MODEL="${BASE_MODEL}" \
    bash "${ROOT_DIR}/scripts/bootstrap_repro_envs.sh"
else
  step 1 ${TOTAL} "skip bootstrap (user flag)"
fi

step 2 ${TOTAL} "prepare training data"
if [[ ! -s "${TRAIN_JSONL}" ]]; then
  "${CONDA_BIN}" run -n "${TRAIN_ENV}" python "${ROOT_DIR}/scripts/generate_demo_train_jsonl.py" --out "${TRAIN_JSONL}"
else
  echo "[ATK] use existing train data: ${TRAIN_JSONL}"
fi
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python "${ROOT_DIR}/scripts/prepare_lf_dataset_info.py" --train_jsonl "${TRAIN_JSONL}"

step 3 ${TOTAL} "build unsloth train yaml"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python "${ROOT_DIR}/scripts/build_unsloth_train_yaml.py" \
  --template_yaml "${ROOT_DIR}/atk/templates/lf_train_template.yaml" \
  --out_yaml "${RUN_DIR}/train_unsloth.yaml" \
  --base_model "${BASE_MODEL}" \
  --train_jsonl "${TRAIN_JSONL}" \
  --cutoff_len "${CUTOFF_LEN}" \
  --max_steps "${MAX_STEPS}" \
  --micro_batch "${MICRO_BATCH}" \
  --grad_acc "${GRAD_ACC}"

step 4 ${TOTAL} "run unsloth finetuning"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" llamafactory-cli train "${RUN_DIR}/train_unsloth.yaml" | tee "${RUN_DIR}/lf_train.log"

step 5 ${TOTAL} "merge LoRA adapter -> full model"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python "${ROOT_DIR}/tools/vllm_routeA/merge_lora_to_full.py" \
  --base "${BASE_MODEL}" \
  --adapter "${RUN_DIR}/output_lora" \
  --out "${RUN_DIR}/merged_model" | tee "${RUN_DIR}/merge.log"

step 6 ${TOTAL} "run LF-aligned reasoning eval (vLLM, fp16+bnb4)"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python "${ROOT_DIR}/tools/vllm_routeA/run_base_vs_tuned_lf_aligned.py" \
  --base "${BASE_MODEL}" \
  --tuned "${RUN_DIR}/merged_model" \
  --out_root "${RUN_DIR}/evals" \
  --build_env "${TRAIN_ENV}" \
  --score_env "${SCORE_ENV}" \
  --conda_bin "${CONDA_BIN}" \
  --limit_mmlu "${LIMIT_MMLU}" \
  --limit_per_subject "${LIMIT_PER_SUBJECT}" \
  --ceval_subject_limit "${CEVAL_SUBJECT_LIMIT}" \
  --cmmlu_subject_limit "${CMMLU_SUBJECT_LIMIT}" \
  --compare_quant_nonquant | tee "${RUN_DIR}/eval_reasoning.log"

REASONING_SUMMARY="$(ls -td "${RUN_DIR}"/evals/eval_run_*_lf_aligned 2>/dev/null | head -n 1)/summary.json"

step 7 ${TOTAL} "run format eval (vLLM, fp16+bnb4)"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python "${ROOT_DIR}/tools/vllm_routeA/run_base_vs_tuned_format_vllm.py" \
  --base "${BASE_MODEL}" \
  --tuned "${RUN_DIR}/merged_model" \
  --out_root "${RUN_DIR}/evals" \
  --score_env "${SCORE_ENV}" \
  --conda_bin "${CONDA_BIN}" \
  --compare_quant_nonquant | tee "${RUN_DIR}/eval_format.log"

FORMAT_SUMMARY="$(ls -td "${RUN_DIR}"/evals/format_run_* 2>/dev/null | head -n 1)/summary.json"

step 8 ${TOTAL} "run factual + safety eval (vLLM, fp16+bnb4)"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python "${ROOT_DIR}/tools/vllm_routeA/run_base_vs_tuned_factual_safety.py" \
  --base "${BASE_MODEL}" \
  --tuned "${RUN_DIR}/merged_model" \
  --out_root "${RUN_DIR}/evals" \
  --build_env "${TRAIN_ENV}" \
  --score_env "${SCORE_ENV}" \
  --conda_bin "${CONDA_BIN}" \
  --truthful_limit "${TRUTHFUL_LIMIT}" \
  --jbb_harmful_limit "${JBB_HARMFUL_LIMIT}" \
  --jbb_benign_limit "${JBB_BENIGN_LIMIT}" \
  --compare_quant_nonquant | tee "${RUN_DIR}/eval_factual_safety.log"

FACTUAL_SUMMARY="$(ls -td "${RUN_DIR}"/evals/factual_safety_run_* 2>/dev/null | head -n 1)/summary.json"

step 9 ${TOTAL} "generate unified 4D report"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python "${ROOT_DIR}/tools/vllm_routeA/generate_four_dim_report.py" \
  --reasoning_summary "${REASONING_SUMMARY}" \
  --factual_safety_summary "${FACTUAL_SUMMARY}" \
  --format_summary "${FORMAT_SUMMARY}" \
  --out_root "${RUN_DIR}" \
  --out_name "four_dim" | tee "${RUN_DIR}/four_dim.log"

FOUR_DIM_MD="${RUN_DIR}/four_dim/four_dim_table.md"

cat > "${RUN_DIR}/README_RUN.md" <<EOF
# ATK Repro Run

- run_dir: ${RUN_DIR}
- base_model: ${BASE_MODEL}
- train_jsonl: ${TRAIN_JSONL}
- train_env: ${TRAIN_ENV}
- score_env: ${SCORE_ENV}
- lora_output: ${RUN_DIR}/output_lora
- merged_model: ${RUN_DIR}/merged_model
- reasoning_summary: ${REASONING_SUMMARY}
- format_summary: ${FORMAT_SUMMARY}
- factual_safety_summary: ${FACTUAL_SUMMARY}
- four_dim_table: ${FOUR_DIM_MD}
EOF

echo ""
echo "[ATK] DONE"
echo "[ATK] run_dir=${RUN_DIR}"
echo "[ATK] four_dim_table=${FOUR_DIM_MD}"
if [[ -f "${FOUR_DIM_MD}" ]]; then
  echo "[ATK] four_dim_table (head)"
  head -n 30 "${FOUR_DIM_MD}"
fi
