import re
from pathlib import Path
from typing import Any, Dict

from atk.utils import save_text


def _recommended_changes_block() -> str:
    lines = []
    lines.append("## 推荐修改")
    lines.append("Safe Launch 未通过或判定为高风险。建议将参数调整为更安全的组合：")
    lines.append("")
    lines.append("```yaml")
    lines.append("train:")
    lines.append("  cutoff_len: 1024")
    lines.append("  micro_batch: 1")
    lines.append("```")
    lines.append("")
    lines.append("可复制的命令(直接回写示例配置)：")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 -c \"import yaml; p='/root/atk_project/examples/atk.yaml'; o=yaml.safe_load(open(p,'r',encoding='utf-8')); o['train']['cutoff_len']=1024; o['train']['micro_batch']=1; open(p,'w',encoding='utf-8').write(yaml.safe_dump(o, sort_keys=False, allow_unicode=True)); print('updated', p)\"")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def build_report(
    *,
    run_dir: Path,
    env_text: str,
    data_stats: Dict[str, Any],
    safe_launch: Dict[str, Any],
    train_exit_code: int,
    sanity: Dict[str, Any],
) -> str:
    has_oom = False
    has_nan = False
    lf_log = run_dir / 'lf_train.log'
    if lf_log.exists():
        txt = lf_log.read_text(encoding='utf-8', errors='ignore')
        has_oom = bool(re.search(r"oom|out of memory", txt, re.I))
        has_nan = bool(re.search(r"\bnan\b", txt, re.I))

    lines = []
    lines.append('# ATK Report')
    lines.append('')

    # Put Safe Launch (and fixes) first so it appears in head -n 30.
    lines.append('## 1) Safe Launch')
    lines.append(f"- passed: {safe_launch.get('passed')}")
    if 'risk_level' in safe_launch:
        lines.append(f"- risk_level: {safe_launch.get('risk_level')}")
    if 'planned' in safe_launch:
        lines.append(f"- planned: {safe_launch.get('planned')}")
    if 'trial_passed' in safe_launch:
        lines.append(f"- trial_passed: {safe_launch.get('trial_passed')}")
    lines.append(f"- exit_code: {safe_launch.get('exit_code')}")
    lines.append(f"- error_keyword: {safe_launch.get('error_keyword')}")
    if safe_launch.get('suggestion'):
        lines.append(f"- suggestion: {safe_launch.get('suggestion')}")
    lines.append('')

    if not safe_launch.get('passed', True):
        lines.append(_recommended_changes_block())
        lines.append('')

    lines.append('## 2) 训练状态')
    lines.append(f"- train_exit_code: {train_exit_code}")
    lines.append(f"- OOM keyword found: {has_oom}")
    lines.append(f"- NaN keyword found: {has_nan}")
    lines.append('')

    lines.append('## 3) Sanity Delta')
    b = sanity.get('base', {})
    t = sanity.get('tuned', {})
    d = sanity.get('delta', {})
    lines.append(f"- base format/qa: {b.get('format_score')}/{b.get('qa_score')}")
    lines.append(f"- tuned format/qa: {t.get('format_score')}/{t.get('qa_score')}")
    lines.append(f"- delta format/qa: {d.get('format_delta')}/{d.get('qa_delta')}")
    lines.append('')

    lines.append('## 4) 数据长度统计')
    lines.append(f"- num_samples: {data_stats.get('num_samples')}")
    lines.append(
        f"- min/p50/p95/max: {data_stats.get('min')}/{data_stats.get('p50')}/{data_stats.get('p95')}/{data_stats.get('max')}"
    )
    lines.append(f"- top1: {data_stats.get('top1')}")
    lines.append('')

    lines.append('## 5) 环境摘要')
    lines.append('```text')
    for ln in env_text.strip().splitlines()[:20]:
        lines.append(ln)
    lines.append('```')
    lines.append('')

    lines.append('## 6) 下一步建议')
    lines.append('- 提升 max_steps 到 300~1000 观察 delta 是否稳定提升。')
    lines.append('- 如果显存紧张，优先降低 cutoff_len，其次降低 micro_batch。')

    return '\n'.join(lines) + '\n'


def write_report(run_dir: Path, content: str) -> Path:
    p = run_dir / 'atk_report.md'
    save_text(p, content)
    return p
