# v0.1.0 (Draft)

ATK v0.1.0: 单卡微调安全护盾（Safe Launch 预跑拦截 + sanity 体检报告）。

## 核心亮点
- Safe Launch: 训练前做最长样本统计 + 5 steps 试跑，拦截显存高风险配置并给出可复制修复建议
- Sanity Delta: 训练后自动输出 base vs tuned 的格式/QA 对比与 delta
- 仅 CLI，无 Web UI；单机单卡闭环优先

## 示例命令
```bash
pip install git+https://github.com/weiweijiuzaizhe/autotune-kit
atk run --config examples/atk.yaml
```

## URLs
- Repo: https://github.com/weiweijiuzaizhe/autotune-kit
