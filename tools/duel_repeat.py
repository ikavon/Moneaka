"""同一份测量重跑 N 次，量出**评委本身的抖动**。

发现过程：dpo-b 的 win 在 `pick_checkpoint` 里量到 0.410，在 `win_breakdown` 里
同一批权重、同一批样本、同一顺序再量一次却得到 0.386。评委是 temperature=0，
所以这不是采样，是服务端不完全确定。**抖动 0.024 和我们要分辨的效应同量级**，
不先把这个数钉住，任何 ±0.05 以内的比较都不成立。

    python tools/duel_repeat.py --run pickg210_dpo-b_final --k 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka.duel import duel  # noqa: E402
from moneaka.llm import LLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pickg210_dpo-b_final")
    ap.add_argument("--against", default="original")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / "data" / "runs" / args.run / "arms.jsonl",
                                        encoding="utf-8")]
    items = [(r.get("user", ""), r["polish"], r[args.against]) for r in rows]
    llm = LLM()
    wins = []
    for i in range(args.k):
        r = duel(llm, items)
        wins.append(round(r.win_rate, 3))
        print(f"  第 {i + 1} 次: {wins[-1]}  (未解析 {r.unparsed})")
    print(f"\n{args.run}: 中位 {statistics.median(wins):.3f}  "
          f"极差 {max(wins) - min(wins):.3f}  全部 {wins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())