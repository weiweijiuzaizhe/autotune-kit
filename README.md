# ATK (AutoTune-Kit)

ATK 是单卡微调与评测工具集，目标是把「Safe Launch + 快速训练 + 统一评测」做成可复现 CLI 流程。

## 快速上手

### 方式 A：GitHub 直接安装（推荐）

```bash
pip install git+https://github.com/weiweijiuzaizhe/autotune-kit.git
atk run --config examples/atk.yaml
```

### 方式 B：开发者模式

```bash
git clone https://github.com/weiweijiuzaizhe/autotune-kit.git
cd autotune-kit
pip install -e .
```

## 运行模式

### 3 分钟冒烟（默认 quick）

```bash
atk run
# 等价于：atk run --config examples/atk.yaml
```

默认 `examples/atk.yaml` 指向 quick 配置（小样本 + 少步数），并在微调后自动执行 `quick_eval`：
- 默认后端 `vllm`（失败自动回退到 HF batch）
- 输出 base vs tuned 四维对比：推理/格式/幻觉/安全

### 全量闭环（full）

```bash
atk run --config examples/atk.full.yaml
```

`full` 配置会启用 `safe_launch` 与 `sanity`，用于更稳健的训练后体检与对比。

## 配置文件说明

- `examples/atk.yaml`: 默认 quick（建议新机器先跑）
- `examples/atk.quick.yaml`: quick 显式配置（训练后自动 quick_eval）
- `examples/atk.full.yaml`: full 配置（safe_launch + full sanity，更完整更耗时）

## 一键复现脚本（Unsloth + vLLM 四维评测）

```bash
bash scripts/repro_unsloth_vllm_full.sh
```

该脚本会执行：
1. bootstrap 两个 conda 环境（训练 env + vLLM 评测 env）
2. 自动准备训练数据（若 `./data/train.jsonl` 不存在）
3. Safe Launch 预跑拦截（trial + 风险判断）
4. `LLaMA-Factory + Unsloth + QLoRA 4bit` 训练
5. 合并 LoRA adapter 为完整模型目录（供 vLLM）
6. vLLM 四维评测（推理/格式/幻觉/安全，含 fp16 与 bnb4）
7. 生成统一报告

默认输出目录：
- `./artifacts/repro_runs/repro_<timestamp>/`

## 评测子集 limit 默认值

以下参数由统一配置文件管理（保证跨模型口径一致）：
- `LIMIT_MMLU`
- `LIMIT_PER_SUBJECT`
- `CEVAL_SUBJECT_LIMIT`
- `CMMLU_SUBJECT_LIMIT`

配置文件：
- `configs/eval_limits.conf`

可在该文件修改默认值，或用环境变量临时覆盖。

## 目录结构

```text
.
├── atk/                              # ATK CLI 核心逻辑
├── assets/lf_eval_tasks/             # LF 对齐评测映射（mmlu/ceval/cmmlu）
├── configs/                          # 评测默认参数配置
├── data/                             # 训练样本（含 quick 小样本）
├── examples/                         # quick/full 示例配置
├── scripts/
│   ├── bootstrap_repro_envs.sh       # 新机器依赖与缓存初始化
│   ├── generate_demo_train_jsonl.py  # 生成 200+3 样本训练集
│   ├── prepare_lf_dataset_info.py    # 自动生成 dataset_info.json
│   ├── run_safe_launch_gate.py       # Safe Launch 预跑与拦截
│   └── repro_unsloth_vllm_full.sh    # 一键全链路复现入口
└── tools/vllm_routeA/                # vLLM 评测与四维报告工具
```

## 备注

- 仓库脚本默认使用相对路径（无 `/root/...` 硬编码）。
- HF 缓存默认使用 `~/.cache/hf`（bootstrap 会写入 `~/.bashrc`）。
