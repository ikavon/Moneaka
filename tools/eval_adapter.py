"""在**留出集**上跑训好的 adapter，产出可直接喂给 duel 的 arms.jsonl。

为什么单独写一个脚本而不用 `moneaka polish`：那条命令跑的是文件前 N 条，
而训练时是从 226 条里随机抽 20 条留出的——用前 N 条评测等于**拿训练集考试**，
分数会假高。这里用同一个 seed 复现同一个切分，保证这 20 条模型没见过。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "models"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(ROOT / "src"))

from moneaka import dataset, prompts  # noqa: E402
from moneaka.contract import check  # noqa: E402
from moneaka.evaluate import copy_ratio  # noqa: E402
from moneaka.localhf import GenConfig, get_model  # noqa: E402
from moneaka.normalize import normalize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/plan.jsonl")
    ap.add_argument("--base", default="data/models/Qwen3-1.7B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", required=True, help="runs/ 下的输出名")
    ap.add_argument("--held-out", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42, help="必须与训练时一致")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--few-shot", action="store_true",
                    help="强行挂 few-shot 示例（默认关，与训练分布一致）")
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    samples, stats = dataset.build(args.pairs)
    _, eval_s = dataset.held_out(samples, n=args.held_out, seed=args.seed)
    print(f"留出集 {len(eval_s)} 条（训练 {len(samples) - len(eval_s)} 条）")

    model = get_model(args.base, adapter=args.adapter)
    print(f"基座 {args.base} + adapter {args.adapter}")

    def one(s: dataset.Sample) -> dict:
        msgs = s.as_messages()
        if not args.few_shot:
            # 训练用的就是短 system；换成长 few-shot 反而分布不一致
            msgs[0]["content"] = prompts.restyle_system(few_shot=False)
        raw = model.chat([dict(m) for m in msgs],
                         GenConfig(temperature=args.temperature,
                                   max_new_tokens=args.max_new))
        text, nst = (raw, None) if args.no_normalize else normalize(raw)
        c = check(text)
        return {
            "turn_id": s.turn_id, "user": s.user_msg, "draft": s.plan,
            "original": s.target, "polish": text, "polish_raw": raw,
            "contract_ok": c.ok, "violations": [str(v) for v in c.violations],
            "copy": round(copy_ratio(text, s.plan), 3),
            "norm": nst.as_dict() if nst else None,
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, eval_s))
    print(f"推理 {len(rows)} 条 / {time.time() - t0:.0f}s")

    out_dir = ROOT / "data" / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "arms.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    ok = sum(r["contract_ok"] for r in rows)
    sticky = sum(r["copy"] > 0.9 for r in rows)
    empty = sum(not (r["polish"] or "").strip() for r in rows)
    print(f"契约通过率 {ok}/{n} = {ok / n:.0%}")
    print(f"照抄（>0.9 与指令内容同字序）{sticky}/{n} = {sticky / n:.0%}  ← 阶段0的病根，必须低")
    print(f"空输出 {empty}/{n}")
    print("\n样例（左=模型输出，右=官方真实）:")
    for r in rows[:8]:
        print(f"  指令: {r['draft'][:70].replace(chr(10), ' / ')}")
        print(f"    出: {r['polish']}")
        print(f"    真: {r['original']}")
        print(f"    copy={r['copy']} ok={r['contract_ok']} {r['violations']}")
    print(f"\n写入 {out_dir / 'arms.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
