import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from atk.utils import save_json


def _parse_json_obj(text: str):
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _build_input(tok, prompt: str) -> str:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _gen_one(model, tok, prompt: str) -> str:
    import torch

    text = _build_input(tok, prompt)
    inputs = tok(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


def _eval_model(model, tok, format_prompts: List[str], qa_items: List[Tuple[str, str]]):
    format_hits = 0
    qa_hits = 0

    for p in format_prompts:
        out = _gen_one(model, tok, p)
        obj = _parse_json_obj(out)
        ok = bool(obj and all(k in obj for k in ["name", "age", "city"]))
        format_hits += int(ok)

    for q, exp in qa_items:
        out = _gen_one(model, tok, q)
        ok = exp.lower() in out.lower()
        qa_hits += int(ok)

    return {
        "format_score": format_hits / len(format_prompts),
        "qa_score": qa_hits / len(qa_items),
    }


def run_sanity(*, run_dir: Path, model_base: str, adapter_dir: Path, qlora_4bit: bool) -> Dict:
    # Lazy imports: allow `pip install autotune-kit` and `atk --help` on machines without torch.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    format_prompts = [
        "输出严格JSON: name=Alice age=23 city=Beijing",
        "请只返回JSON，信息是 Bob, 31, Shanghai",
        "Return strict JSON only for: Charlie, 45, Shenzhen",
        "仅返回 JSON：Diana 29 Hangzhou",
        "Output JSON only: Ethan, 38, Guangzhou",
        "只输出JSON: Fiona, 22, Chengdu",
        "JSON only: Grace 41 Nanjing",
        "严格JSON返回：Henry, 19, Wuhan",
        "Return only JSON: Iris, 27, Suzhou",
        "仅JSON: Jack, 33, Xi'an",
    ]

    qa_items = [
        ("1+1=?", "2"),
        ("2*3=?", "6"),
        ("北京是中国的首都吗？只回答是或否。", "是"),
        ("地球是平的吗？只回答是或否。", "否"),
        ("英文中cat是什么意思？可简短", "猫"),
        ("中国的首都是哪里？", "北京"),
        ("水在100摄氏度会沸腾吗？是/否", "是"),
        ("太阳从哪边升起？", "东"),
        ("5减2等于几？", "3"),
        ("Python 是编程语言吗？是/否", "是"),
    ]

    tok = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_cfg = None
    if qlora_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_base,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    base_model.eval()
    base_res = _eval_model(base_model, tok, format_prompts, qa_items)

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tuned_base = AutoModelForCausalLM.from_pretrained(
        model_base,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    tuned_model = PeftModel.from_pretrained(tuned_base, str(adapter_dir))
    tuned_model.eval()
    tuned_res = _eval_model(tuned_model, tok, format_prompts, qa_items)

    report = {
        "base": base_res,
        "tuned": tuned_res,
        "delta": {
            "format_delta": tuned_res["format_score"] - base_res["format_score"],
            "qa_delta": tuned_res["qa_score"] - base_res["qa_score"],
        },
    }

    save_json(run_dir / "sanity_report.json", report)
    return report
