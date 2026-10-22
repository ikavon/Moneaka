"""DPO：在 SFT 模型之上，用**受控负例**做风格偏序训练。

SFT 教「什么是莫妮卡」，DPO 教「哪个地方不对」。区别在于负例的构造方式：

- 不用官方 rejected —— 它和 chosen 说的不是同一件事（字序相似度中位 0.18），
  梯度会被内容差异吃干净。
- 用 moneaka.negatives 的四条轴（tags/order/merge/flatten），每条只动一个风格维度，
  正文其余逐字相同 → 学到的偏序可以归因到具体那条轴。

显存：先把 SFT adapter merge 成独立基座，再在它上面挂新 LoRA。
trl 的 DPOTrainer 在传 peft_config 时用同一个权重上的 "ref" adapter 当参照模型，
**不会**复制第二份 3.4GB，8GB 卡能跑。

    python tools/train_dpo.py --init data/adapters/run3/checkpoint-200 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "models"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(ROOT / "src"))

from moneaka import dataset, negatives  # noqa: E402


def merge_sft(base: str, adapter: str, out_dir: str) -> str:
    """SFT adapter 合并成独立基座（DPOTrainer 不允许 PeftModel + peft_config 并存）。"""
    tgt = str(Path(out_dir).resolve())
    if (Path(tgt) / "config.json").exists():
        print(f"合并基座已存在: {tgt}")
        return tgt
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"合并 {adapter} → {tgt}")
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
    m = PeftModel.from_pretrained(m, adapter)
    m = m.merge_and_unload()
    m.save_pretrained(tgt)
    AutoTokenizer.from_pretrained(base).save_pretrained(tgt)
    del m
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return tgt


def build_pref_rows(samples: list[dataset.Sample], axes: list[str],
                    held_only: bool = True, seed: int = 0):
    """每条样本 × 每条轴 = 一个偏序对。prompt 与 SFT 完全一致（短 system）。"""
    rows = []
    for si, s in enumerate(samples):
        msgs = s.as_messages()
        prompt = msgs[:-1]                      # system + user(表达指令)
        negs = negatives.negatives(s.target, seed=si + seed)
        for ax in axes:
            neg = negs.get(ax)
            if not neg:
                continue
            rows.append({"prompt": prompt,
                         "chosen": [{"role": "assistant", "content": s.target}],
                         "rejected": [{"role": "assistant", "content": neg}],
                         "axis": ax, "turn_id": s.turn_id})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/plan.jsonl")
    ap.add_argument("--base", default="data/models/Qwen3-1.7B")
    ap.add_argument("--init", required=True, help="SFT adapter 目录")
    ap.add_argument("--out", default="data/adapters/dpo1")
    ap.add_argument("--merged", default="data/models/Qwen3-1.7B-sft-merged")
    ap.add_argument("--axes", default="tags,order,merge,flatten")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--held-out", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from datasets import Dataset as HFDataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    samples, stats = dataset.build(args.pairs)
    train_s, _eval_s = dataset.held_out(samples, n=args.held_out, seed=args.seed)
    if args.smoke:
        train_s = train_s[:24]
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    rows = build_pref_rows(train_s, axes, seed=args.seed)
    per_axis: dict[str, int] = {}
    for r in rows:
        per_axis[r["axis"]] = per_axis.get(r["axis"], 0) + 1
    print(f"偏好对 {len(rows)} 条（{len(train_s)} 样本 × 轴 {axes}）→ {per_axis}")
    if not rows:
        print("没有可用偏序对"); return 1

    merged = merge_sft(args.base, args.init, args.merged)
    tok = AutoTokenizer.from_pretrained(merged)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = HFDataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")}
                              for r in rows])
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    n_steps = max(1, math.ceil(len(ds) / args.accum) * int(math.ceil(args.epochs)))
    cfg = DPOConfig(
        output_dir=args.out,
        beta=args.beta,
        max_length=args.max_len,
        max_prompt_length=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(n_steps * 0.1)),
        num_train_epochs=args.epochs,
        max_steps=30 if args.smoke else -1,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=5,
        save_strategy="no" if args.smoke else "steps",
        save_steps=max(1, len(ds) // args.accum),
        save_total_limit=6,
        seed=args.seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=merged, args=cfg, train_dataset=ds, peft_config=lora,
        processing_class=tok,
    )
    print("DPO 开训…")
    res = trainer.train()
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(f"完成: loss={res.training_loss:.3f} 峰值显存={mem:.2f} GiB")
    if not args.smoke:
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        Path(args.out, "dpo_stats.json").write_text(json.dumps(
            {"rows": len(rows), "per_axis": per_axis, "beta": args.beta,
             "lr": args.lr, "epochs": args.epochs, "init": args.init,
             "loss": res.training_loss, "peak_mem_gib": round(mem, 2)},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"adapter 已存到 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
