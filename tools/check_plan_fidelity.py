"""量表达指令（plan）本身对原句的忠实度。

数据侧改法的前提假设是：**内容行没把原句说清楚，模型只能在Pass模糊处发挥**。
但这个假设从没被量过。如果 plan 自身就有 20% 漏掉了原句的否定/数量/指向，
那不管润色器多听话都救不回来；如果 plan 本身忠实（sem 4 居多），
那瓶颈在润色器的执行力，改写法是白费功夫。

判据与润色器评测同一套（`evaluate.judge_semantic_rows`），只是把
「草稿 → 润色」换成「原句 → 表达指令的内容行」。

    python tools/check_plan_fidelity.py --pairs data/pairs/tagged_ds.jsonl
    python tools/check_plan_fidelity.py --pairs data/pairs/tagged_official.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka.destyle_tags import parse_tagged_plan  # noqa: E402
from moneaka.llm import LLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/tagged_ds.jsonl")
    ap.add_argument("--out", default=None, help="逐条结果落盘（jsonl）")
    ap.add_argument("--show", type=int, default=12, help="打印最差的几条")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    rows = []
    for r in map(json.loads, open(ROOT / args.pairs, encoding="utf-8")):
        try:
            _, contents = parse_tagged_plan(r.get("plan") or "")
        except Exception:  # noqa: BLE001
            continue
        rows.append({"turn_id": r["turn_id"],
                     "draft": r["original"],          # 评委的「草稿」= 官方原句
                     "polish": "；".join(contents)})  # 评委的「润色」= 内容行
    print(f"可解析的 plan {len(rows)} 条")

    sem = evaluate.judge_semantic_rows(LLM(), rows, "polish", batch_size=args.batch)
    vals = [sem[r["turn_id"]] for r in rows if r["turn_id"] in sem]
    print(f"语义均值 {statistics.mean(vals):.2f}  分布 {sorted(Counter(vals).items())}")
    print(f"≤2 分（漏意思）{sum(1 for v in vals if v <= 2)}/{len(vals)} = "
          f"{sum(1 for v in vals if v <= 2) / len(vals):.1%}")

    if args.out:
        with open(ROOT / args.out, "w", encoding="utf-8") as fh:
            for r in rows:
                if r["turn_id"] in sem:
                    fh.write(json.dumps({**r, "sem": sem[r["turn_id"]]},
                                        ensure_ascii=False) + "\n")
        print(f"逐条写入 {args.out}")

    worst = sorted((r for r in rows if r["turn_id"] in sem),
                   key=lambda r: sem[r["turn_id"]])[: args.show]
    for r in worst:
        print(f"\n[{sem[r['turn_id']]}] {r['turn_id']}")
        print(f"  原句: {r['draft'][:100]}")
        print(f"  内容: {r['polish'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())