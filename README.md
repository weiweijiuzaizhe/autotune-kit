# autotune-kit

`autotune-kit` 是一个最小可运行的 ATK CLI 工具集，用于在单机单卡环境中调用 `llamafactory-cli train` 跑通微调闭环。

目标：闭环优先，不追求训练效果。

## 特性 (v0.1)
- `atk run --config <yaml>`: 一键跑通 Safe Launch + 训练 + sanity + 报告
- Safe Launch: 自动统计 token 长度，先 5 steps 试跑；并对明显高风险配置做拦截
- 报告: `atk_report.md`，Safe Launch 失败时提供可复制的“推荐修改”配置片段

## 环境假设
- Python 3.10
- 已安装并可执行 `llamafactory-cli`
- 有可用 GPU

## 安装
建议在你的 py310 环境中：

```bash
cd /root/autotune-kit
pip install -U pip
pip install -e .
```

注意：如果你使用的是 conda 环境，请确保运行的是对应环境的 `pip`/`python`。

## 使用
示例配置：`examples/atk.yaml`

```bash
atk run --config /root/autotune-kit/examples/atk.yaml
```

重跑(读取历史 run 的有效配置 `config.effective.yaml` 再跑一遍)：

```bash
atk rerun /root/atk_runs/run_YYYYmmdd_HHMMSS
```

## 输出
每次运行会创建：`/root/atk_runs/run_<timestamp>/`，包含：
- `config.effective.yaml`
- `env.txt`
- `data_stats.json`
- `safe_launch.json`
- `lf_train.log`
- `sanity_report.json` (训练成功时)
- `atk_report.md`

## 目录结构
```text
autotune-kit/
  pyproject.toml
  README.md
  .gitignore
  examples/
    atk.yaml
  atk/
    __init__.py
    cli.py
    config.py
    run.py
    safelaunch.py
    sanity.py
    report.py
    utils.py
```
