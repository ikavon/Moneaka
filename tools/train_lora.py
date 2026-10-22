"""LoRA 微调润色器（单机单卡，RTX 5060 Laptop 8GB）。

    # 冒烟：30 步只验证链路
    python tools/train_lora.py --smoke

    # 正式
    python tools/train_lora.py --epochs 3 --lr 2e-4 --lora-r 16

显存账（实测语料单轮中位 35 字 → 序列 512 足够）：
- Qwen3-1.7B bf16 权重 ~3.4 GB
- LoRA 参数 + 优化器状态：r=16 时几十 MB 级
- 激活（梯度检查点开启）：seq 512 × batch 1 → 约 2-3 GB
合计 7 GB 上下，8 GB 能跑但不宽裕；不够就 --q（4bit 基座）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "data" / "models"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moneaka import dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--pairs", default="data/pairs/plan.jsonl")
    ap.add_argument("--out", default="data/adapters/run1")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条（调试用）")
    ap.add_argument("--held-out", type=int, default=20)
    ap.add_argument("--q", action="store_true", help="4bit 基座（QLoRA），省 ~2GB")
    ap.add_argument("--save-steps", type=int, default=None,
                    help="每隔多少步存一次；默认每 epoch 一次（用来离线做 duel 选点）")
    ap.add_argument("--save-total", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="30 步冒烟，只验证链路")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    samples, stats = dataset.build(args.pairs, limit=None if not args.smoke else 40)
    print(f"数据: {stats}")
    if not samples:
        print("没有可用样本"); return 1
    train_s, eval_s = dataset.held_out(samples, n=args.held_out, seed=args.seed)
    print(f"训练 {len(train_s)} 条 / 留出 {len(eval_s)} 条")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {"device_map": "cuda", "trust_remote_code": True}
    quant = None
    if args.q:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs["quantization_config"] = quant
    else:
        kwargs["dtype"] = torch.bfloat16

    print(f"载入基座 {args.model}（{'4bit' if args.q else 'bf16'}）…")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    steps = 30 if args.smoke else -1
    # 本环境 trl 1.16 的 SFTConfig 没有 warmup_ratio，只有 warmup_steps（绝对步数）
    n_steps = steps if steps > 0 else max(1, int(math.ceil(len(train_s) / args.accum) * args.epochs))
    warmup = max(1, int(n_steps * args.warmup_ratio))
    save_steps = args.save_steps or max(1, len(train_s) // args.accum)   # ≈每 epoch 一次
    if args.smoke:
        save_steps = 1000
    print(f"总步数≈{n_steps} warmup={warmup} 每 {save_steps} 步存点")
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=0.5 if args.smoke else args.epochs,
        max_steps=steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup,
        max_grad_norm=1.0,
        weight_decay=args.weight_decay,
        max_length=args.max_len,
        assistant_only_loss=True,          # 只对 assistant 段算 loss
        packing=False,                     # 短样本别拼，拼了会学到跨样本续写
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps" if not args.smoke else "no",
        eval_steps=25,
        save_total_limit=args.save_total,
        seed=args.seed,
        report_to=[],
        optim="paged_adamw_8bit" if args.q else "adamw_torch",
    )

    from datasets import Dataset as HFDataset
    train_ds = HFDataset.from_list([{"messages": s.as_messages()} for s in train_s])
    eval_ds = HFDataset.from_list([{"messages": s.as_messages()} for s in eval_s])

    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds,
        peft_config=lora, processing_class=tok,
    )

    print("开训…")
    res = trainer.train()
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(f"完成: train_loss={res.training_loss:.3f} 峰值显存={mem:.2f} GiB")

    if not args.smoke:
        # 注意：不要用 eval_loss 选点。实测 eval_loss 从 1.07 涨到 1.89 的同时，
        # 盲评胜率反而从 0.30 升到 0.38——这个任务上记忆≠退化，损失不是好代理。
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        print(f"adapter 已存到 {args.out}")
        # 训练后立刻在留出集上做一次「照抄率」自检：这是阶段 0 翻车的头号指标
        from moneaka.evaluate import copy_rate_samples
        cr = copy_rate_samples(train_s[:0] + eval_s)
        print(f"留出集照抄率参考: {cr}")
        Path(args.out, "train_stats.json").write_text(
            json.dumps({"stats": stats, "train_n": len(train_s), "eval_n": len(eval_s),
                        "loss": res.training_loss, "peak_mem_gib": round(mem, 2)},
                       ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
