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

from moneaka import dataset, negatives, prompts  # noqa: E402


def merge_sft(base: str, adapter: str, out_dir: str) -> str:
    """SFT adapter 合并成独立基座（DPOTrainer 不允许 PeftModel + peft_config 并存）。

    两重保护，都是被真实事故逼出来的：

    1. **落了 marker 才复用**。原先只判 `config.json` 存在就返回，结果
       `data/models/Qwen3-1.7B-sft-merged` 是前一天用 **run2** 合出来的旧货，
       后面四组 DPO 全在它上面训，标签遵从率从 99% 崩到 69%，而报告里
       一个字都没体现——因为没人去查基座是谁。
    2. **合完自检**。拿一层 q_proj 算 `B@A × (alpha/r)`，跟 ``merged - base``
       比相关系数与尺度。相关 <0.9 或尺度偏离 >20% 直接报错，不往下跑。
    """
    tgt = str(Path(out_dir).resolve())
    marker = Path(tgt) / ".merge_source.json"
    want = {"base": str(Path(base).resolve()), "adapter": str(Path(adapter).resolve()),
            "adapter_mtime": round(Path(adapter, "adapter_model.safetensors").stat().st_mtime, 0)}
    if (Path(tgt) / "config.json").exists():
        got = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
        if got == want:
            print(f"合并基座已存在且来源一致: {tgt}")
            return tgt
        raise SystemExit(
            f"合并目录已存在但来源不符，拒绝复用：\n  {tgt}\n  期望 {want}\n  实际 {got}\n"
            f"请先删除该目录（或换 --merged）。")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"合并 {adapter} → {tgt}")
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
    m = PeftModel.from_pretrained(m, adapter)
    m = m.merge_and_unload()
    _verify_merge(m, adapter)
    m.save_pretrained(tgt)
    AutoTokenizer.from_pretrained(base).save_pretrained(tgt)
    marker.write_text(json.dumps(want, ensure_ascii=False, indent=1), encoding="utf-8")
    del m
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return tgt


def _load_base_tensor(base_dir: str, key: str):
    """从磁盘读基座的某一个张量（兼容单文件与分片两种格式）。"""
    from safetensors.torch import load_file

    d = Path(base_dir)
    one = d / "model.safetensors"
    if one.is_file():
        w = load_file(str(one), device="cpu")
        return w.get(key)
    idx = d / "model.safetensors.index.json"
    if not idx.is_file():
        return None
    import json as _json

    fname = _json.loads(idx.read_text(encoding="utf-8"))["weight_map"].get(key)
    if not fname:
        return None
    return load_file(str(d / fname), device="cpu").get(key)


def _verify_merge(model, adapter: str) -> None:
    """合并自检：``merged - base`` 应当 ≈ ``B@A × (alpha/r)`` 且高度相关。

    bf16 下小增量会被舍入吃掉一部分，所以相关不可能到 1，实测正常值 ≳0.95；
    拿错 adapter 时会掉到 0.1 量级（被真实事故验证过）。
    """
    import torch
    from safetensors.torch import load_file

    name = "model.layers.0.self_attn.q_proj"
    cfg = json.loads(Path(adapter, "adapter_config.json").read_text(encoding="utf-8"))
    scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    w = load_file(str(Path(adapter, "adapter_model.safetensors")), device="cpu")
    ak = f"base_model.model.{name}.lora_A.weight"
    bk = f"base_model.model.{name}.lora_B.weight"
    base_w = _load_base_tensor(cfg["base_model_name_or_path"], f"{name}.weight")
    if ak not in w or base_w is None:
        print("  ! 合并自检跳过（adapter 或基座权重取不到）")
        return
    delta = (w[bk].float() @ w[ak].float()).flatten()
    merged = model.state_dict()[f"{name}.weight"].float().cpu().flatten()
    d = merged - base_w.float().flatten()
    s = float((d * delta).sum() / (delta * delta).sum())
    corr = float(torch.corrcoef(torch.stack([d, delta]))[0, 1])
    ok = corr > 0.9 and abs(s / scale - 1) < 0.2
    print(f"  合并自检: 尺度 {s:.3f} (期望 {scale:.2f})  相关 {corr:.3f}  "
          f"{'✓' if ok else '✗ 合并结果与 adapter 不符'}")
    if not ok:
        raise SystemExit("合并自检失败，基座不可信，中止")


def build_pref_rows(samples: list[dataset.Sample], axes: list[str], seed: int = 0):
    """每条样本 × 每条轴 = 一个偏序对。prompt 与 SFT 完全一致（短 system）。"""
    rows = []
    for si, s in enumerate(samples):
        msgs = s.as_messages()
        prompt = msgs[:-1]    # system + user(对话上下文 + 表达指令)，与 SFT 逐字一致
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
    ap.add_argument("--pref-file", default=None,
                    help="直接用挖好的负例对（tools/mine_drift.py 的产出），忽略 --axes")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=896)
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
    if args.pref_file:
        # on-policy 模式：负例是模型自己生成、经语义评委判为漂移的输出。
        # 这里的 prompt 必须与 SFT 训练时逐字一致（system + 表达指令），
        # 否则模型会先学「换了输入格式」再学「别改事实」。
        sys_prompt = prompts.restyle_system(few_shot=False)
        rows = []
        for r in map(json.loads, open(args.pref_file, encoding="utf-8")):
            rows.append({"prompt": [{"role": "system", "content": sys_prompt},
                                    {"role": "user", "content": r["plan"]}],
                         "chosen": [{"role": "assistant", "content": r["chosen"]}],
                         "rejected": [{"role": "assistant", "content": r["rejected"]}],
                         "axis": "semantic", "turn_id": r["turn_id"]})
        for r in rows:
            if not r["chosen"][0]["content"] or not r["rejected"][0]["content"]:
                raise SystemExit("chosen/rejected 有空串，检查 pref-file")
    else:
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
        # 本版 trl 只支持 truncation_mode="keep_start"，也就是**从尾部截**：
        # 序列超长会砍掉 assistant 回复的结尾，而风格恰恰在结尾的语气词/标签上。
        # 所以 max_length 要给足（实测单条 prompt+回复 ≤ ~700 token），宁可用小 batch。
        max_length=args.max_len,
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
    # 自己载入并搬到 cuda，**不要用 device_map**：DPOTrainer 给 ref model 建 PEFT
    # "ref" adapter 时，accelerate 的分片加载会让一部分权重停在 meta device 上，
    # 反向传播直接炸 "expected device meta but got cuda:0"（实测冒烟复现）。
    from transformers import AutoModelForCausalLM
    loaded = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.bfloat16,
                                                  trust_remote_code=True).to("cuda")
    loaded.config.use_cache = False
    trainer = DPOTrainer(
        model=loaded, args=cfg, train_dataset=ds, peft_config=lora,
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
