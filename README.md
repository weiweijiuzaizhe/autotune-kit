# ATK

ATK — 单卡微调安全护盾（预跑 OOM 拦截 + 体检报告）

ATK 是一个极简 CLI，用于在训练前做 Safe Launch 预跑拦截显存高风险配置，并在训练后生成 base vs tuned 的 sanity 对比与一页报告。

Safe Launch HIGH RISK 拦截示例（节选）：

```text
## 1) Safe Launch
- passed: False
- risk_level: HIGH
- planned: {'cutoff_len': 4096, 'micro_batch': 2}
- trial_passed: True
- exit_code: 0
- error_keyword: HIGH_RISK
- suggestion: Planned config is HIGH RISK. Recommended: cutoff_len=1024 and micro_batch=1.
```

推荐修改（可直接复制到配置里）：

```yaml
train:
  cutoff_len: 1024
  micro_batch: 1
```

## 📦 Quickstart（更像产品的安装方式）

```bash
pip install git+https://github.com/weiweijiuzaizhe/autotune-kit.git
atk run --config examples/atk.yaml
```

默认输出目录：`./atk_runs/run_<timestamp>/`（以当前工作目录为根）。

## ⭐ 核心功能亮点
- Safe Launch：训练前自动统计数据长度，进行 5 steps 试跑，并对明显高风险配置给出可复制修复建议
- Sanity Delta：训练后自动生成 JSON 格式遵循率 + 简单 QA 的 base/tuned 对比与 delta
- 面向单卡 24GB GPU（如 RTX 4090），覆盖 7B–32B 的量化微调工作流（取决于配置与数据长度）
- 无 Web UI，仅 CLI；ATK 自身依赖极少（仅 `PyYAML`），其余依赖由训练栈提供

## 📌 示例输出
每次运行会创建一个 `run_dir`，例如：

```text
./atk_runs/run_YYYYmmdd_HHMMSS/
  atk_report.md
  lf_train.log
  safe_launch.json
  sanity_report.json
  ...
```

## 🧠 安装说明
推荐环境：CUDA 12 + Python 3.10。

ATK 会调用 `llamafactory-cli train` 完成训练，因此运行前需要准备训练栈依赖（示例）：

```bash
pip install -U "transformers>=4.40" accelerate datasets peft bitsandbytes sentencepiece
pip install llamafactory
```

注意：`torch` 安装请按你的 CUDA/驱动环境选择官方推荐方式。

## 🧑‍💻 开发者模式（本地开发/二次开发）

```bash
git clone https://github.com/weiweijiuzaizhe/autotune-kit.git
cd autotune-kit
pip install -e .
```

## 🧪 演示案例
一个最小 `examples/atk.yaml` 示例（请按实际路径修改）：

```yaml
model:
  base: Qwen/Qwen2.5-7B-Instruct
data:
  train: ./data/train.jsonl
output:
  dir: ./atk_outputs/demo1
train:
  template: qwen
  cutoff_len: 1024
  max_steps: 50
  micro_batch: 1
  grad_acc: 4
  lr: 0.0001
  epochs: 1
  qlora_4bit: true
```

字段说明（最小必需）：
- `model.base`：base 模型名称或本地路径
- `data.train`：Alpaca JSONL（包含 instruction/input/output）
- `output.dir`：训练产物输出根目录
- `train.*`：训练关键参数（ATK 会写入一份 LLaMA-Factory 兼容的 `train.yaml`）

## 📍 注意事项
- 模型首次下载可能较慢，建议固定 Hugging Face 缓存目录，例如：

```bash
export HF_HOME=~/.cache/hf
export HF_HUB_CACHE=~/.cache/hf/hub
```

- Safe Launch 依赖训练数据统计，训练数据需为 JSONL 格式（每行一个 JSON）

## ❓ 常见坑
- bitsandbytes 4bit 安装失败：请确认 CUDA 与 bnb 版本兼容；必要时降级/固定版本
- llamafactory 版本不兼容：优先用与你的 `train.yaml` 模板兼容的版本
- 数据分布过长：先降低 `cutoff_len`，再降低 `micro_batch`；必要时移除极长样本

## 🗂️ 目录结构
```text
.
  LICENSE
  README.md
  pyproject.toml
  examples/
    atk.yaml
  atk/
    __init__.py
    cli.py
    config.py
    init.py
    report.py
    run.py
    safelaunch.py
    sanity.py
    utils.py
    templates/
      lf_train_template.yaml
```

## 🤝 反馈与贡献
欢迎提 issue、PR、star！

- 贡献指南：优先提交最小可复现的 bug report（包含配置、日志关键片段）。PR 建议包含：改动说明 + 本地验证命令 + 影响面评估。
- 反馈链接：Issues https://github.com/weiweijiuzaizhe/autotune-kit/issues
