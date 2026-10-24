"""模型准入测：同参数段（1.7B~4.5B）谁是合适的润色器底座？

## 测什么

只测**能不能接这个任务**，不测训练后的效果（那要各训一版，成本高）。
输入是同一批表达指令、同一份 restyle system prompt，看三个机械指标：

1. **标签序列遵从率**——指令给了 `语气: A > B > C`，输出是否原样照做
2. **词表外标签**——冒出自创情绪词（温柔/认真/期待）就等于下游动画映射不了
3. **中英混杂 / 重复语气词**——小模型在中文上的基本功

## 支持两类模型

* 纯文本因果 LM（Qwen3 等）：`AutoModelForCausalLM`
* 多模态包装（Gemma 4 这类 `*ForConditionalGeneration`）：退到 `AutoModelForImageTextToText`，
  只喂文本，用 processor 套 chat template

    python tools/model_survey.py --list
    python tools/model_survey.py --model data/models/gemma-4-E4B-it --n 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from bench_v2 import BENCH  # noqa: E402
from moneaka import contract, dataset  # noqa: E402
from moneaka.destyle_tags import parse_tagged_plan  # noqa: E402
from moneaka.prompts import restyle_system  # noqa: E402

WHITE = set(contract.EMOTION_TAGS) | {"很开心"}
_EN_RUN = re.compile(r"[A-Za-z]+(?:\s+[A-Za-z]+){2,}")
_DUP = re.compile(r"([么呢吧啊哦呀嘛])" + chr(92) + "1")


def tags_of(t: str) -> list[str]:
    return [m.group(1) for m in contract.TAG_RE.finditer(contract.PLACEHOLDER_RE.sub("\x00", t))]


def load(model_dir: str, load_4bit: bool = True):
    """返回 (generate_fn, kind)。两类架构都能吃。"""
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    kw = {"dtype": torch.bfloat16, "trust_remote_code": True}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)

    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    arch = (cfg.get("architectures") or [""])[0]

    # 一律先试纯文本路径（AutoModelForCausalLM + tokenizer）。
    # Gemma 4 的 checkpoint 虽然标着 `*ForConditionalGeneration`，但这份权重能被
    # 当因果 LM 加载，只喂文本就行——省掉 processor（它还依赖 torchvision）。
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", **kw)
    tok = AutoTokenizer.from_pretrained(model_dir)

    def _tmpl(msgs) -> str:
        """Qwen3/Gemma 默认可能吐 think 块，关掉——生产路径也是这么做的。"""
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def gen(system: str, user: str) -> str:
        text = _tmpl([{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=260, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()

    kind = f"causal({type(model).__name__})"
    try:
        print(f"  显存 {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    except Exception:  # noqa: BLE001
        pass
    return gen, kind


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="模型目录")
    ap.add_argument("--n", type=int, default=12, help="题数（12=官方题，>12 用留出集）")
    ap.add_argument("--fp16", action="store_true", help="不量化（仅 1.7B 这类小模型可用）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.model:
        d = ROOT / "data" / "models"
        print("可用模型：")
        for p in sorted(d.iterdir()):
            if (p / "config.json").is_file():
                print("  ", p.relative_to(ROOT))
        return 0

    if args.n > 12:
        samples, _ = dataset.build("data/pairs/split_holdout210.jsonl")
        cases = [(s.user_msg, s.plan) for s in samples[: args.n]]
    else:
        # 官方 12 题没有现成 plan，用我们的编剧现场生成？不——准入测只关心
        # 「给了 plan 能不能照做」，所以取留出集里前 n 条（有真 plan）
        samples, _ = dataset.build("data/pairs/split_holdout210.jsonl")
        cases = [(s.user_msg, s.plan) for s in samples[: args.n]]
        assert BENCH  # 保留导入，说明官方题在 bench_v2 里

    gen, kind = load(args.model, load_4bit=not args.fp16)
    print(f"加载完成：{args.model}  [{kind}]  测 {len(cases)} 条")

    sys_p = restyle_system(few_shot=False)
    rows = []
    for i, (user, plan) in enumerate(cases, 1):
        try:
            out = gen(sys_p, f"表达指令:\n{plan}\n\n莫妮卡会说:")
        except Exception as exc:  # noqa: BLE001
            out = f"__ERROR__ {exc}"
        want = parse_tagged_plan(plan)[0]
        got = tags_of(out)
        rows.append({"user": user, "plan": plan, "out": out,
                     "tag_ok": got == want, "tags": got, "want": want,
                     "oov": [t for t in got if t not in WHITE],
                     "en": bool(_EN_RUN.search(out)), "dup": bool(_DUP.search(out)),
                     "len": contract.cjk_len(contract.PLACEHOLDER_RE.sub(
                         "", contract.TAG_RE.sub("", out)))})
        print(f"  {i}/{len(cases)} 标签{'✓' if rows[-1]['tag_ok'] else '✗'}"
              f" 自创{rows[-1]['oov']} 夹英{rows[-1]['en']}", flush=True)

    n = len(rows)
    print(f"\n=== {Path(args.model).name} ===")
    print(f"标签序列遵从 {sum(r['tag_ok'] for r in rows)}/{n}"
          f"   含自创标签 {sum(1 for r in rows if r['oov'])}/{n}"
          f"   夹英文 {sum(1 for r in rows if r['en'])}/{n}"
          f"   重复语气词 {sum(1 for r in rows if r['dup'])}/{n}")
    print(f"平均字数 {sum(r['len'] for r in rows) / n:.0f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())