# ATK (AutoTune-Kit)

ATK 是一个单卡微调与评测工具集，目标是把「安全起跑 + 快速训练 + 统一评测」做成可复现的 CLI 流程。

## 一键复现：Unsloth 微调 + vLLM 全链路评测

适用场景：新开一台 GPU 机器后，希望一条命令复现你这次已经跑通的流程。

```bash
git clone https://github.com/weiweijiuzaizhe/autotune-kit.git
cd autotune-kit
bash scripts/repro_unsloth_vllm_full.sh
```

该命令默认会执行：
1. `bootstrap` 两个 conda 环境（训练 env + vLLM 评测 env）
2. 自动准备训练数据（若 `./data/train.jsonl` 不存在）
3. Safe Launch 预跑拦截（trial + 风险判断，不通过则直接退出）
4. 用 `LLaMA-Factory + Unsloth + QLoRA 4bit` 做一次 SFT 微调
5. 合并 LoRA adapter 为完整模型目录（给 vLLM 加载）
6. 跑 vLLM 全链路评测（推理/格式/幻觉/安全，含 fp16 与 bnb4）
7. 生成统一四维报告表

运行结束后会打印 `run_dir`，默认在：
- `./artifacts/repro_runs/repro_<timestamp>/`

核心产物：
- `safe_launch.json`
- `safe_launch/lf_train.safe.log`
- `lf_train.log`
- `merged_model/`
- `evals/eval_run_*_lf_aligned/summary.json`
- `evals/format_run_*/summary.json`
- `evals/factual_safety_run_*/summary.json`
- `four_dim/four_dim_table.md`

## 可调参数

```bash
# 示例：跳过 bootstrap 和 safe launch，指定训练步数与数据路径
bash scripts/repro_unsloth_vllm_full.sh \
  --skip-bootstrap \
  --skip-safe-launch \
  --max-steps 200 \
  --train-jsonl ./data/train.jsonl
```

环境变量可覆盖：
- `TRAIN_ENV`（默认 `py310`）
- `SCORE_ENV`（默认 `vllm014`）
- `BASE_MODEL`（默认 `Qwen/Qwen2.5-7B-Instruct`）
- `RUN_ROOT`（默认 `./artifacts/repro_runs`）
- `MAX_STEPS`、`CUTOFF_LEN`、`MICRO_BATCH`、`GRAD_ACC`
- `LIMIT_MMLU`、`LIMIT_PER_SUBJECT`、`CEVAL_SUBJECT_LIMIT`、`CMMLU_SUBJECT_LIMIT`
- `TRUTHFUL_LIMIT`、`JBB_HARMFUL_LIMIT`、`JBB_BENIGN_LIMIT`

## 目录说明

```text
.
├── atk/                              # ATK CLI 核心逻辑
├── assets/lf_eval_tasks/             # LF 对齐评测映射（mmlu/ceval/cmmlu）
├── examples/                         # 最小配置示例
├── scripts/
│   ├── bootstrap_repro_envs.sh       # 新机器依赖与缓存初始化
│   ├── generate_demo_train_jsonl.py  # 生成 200+3 样本训练集
│   ├── prepare_lf_dataset_info.py    # 自动生成 dataset_info.json
│   ├── run_safe_launch_gate.py       # Safe Launch 预跑与拦截
│   └── repro_unsloth_vllm_full.sh    # 一键全链路复现入口
└── tools/vllm_routeA/                # vLLM 评测与四维报告工具
```

## 仍可单独使用 ATK CLI

```bash
pip install -e .
atk run --config examples/atk.yaml
```

## 备注

- 仓库内脚本已去除 `/root/...` 硬编码，默认使用仓库相对路径。
- HF 缓存默认使用 `~/.cache/hf`（bootstrap 会写入 `~/.bashrc`）。
