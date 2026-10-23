"""把盲评胜率拆开看：赢的输的分别是哪一类。

均值 0.410 这个数藏了很多东西。这个脚本按三个切片重算胜率：

1. **输出与官方原答逐字相同** vs 不同 —— 相同的时候评委只能瞎猜（≈0.5），
   这批样本会把均值往 0.5 拽，不是「质量」的一部分
2. **语义 4/3 分（没改意思）** vs **≤2 分（改了意思）** —— 「说得像」和「说得对」是两根轴
3. **多段（≥3 标签）** vs 单段 —— 小模型在多段指令上是不是明显吃力

    python tools/win_breakdown.py --run pickg210_dpo-b_final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import contract  # noqa: E402
from moneaka.duel import duel  # noqa: E402
from moneaka.llm import LLM  # noqa: E402


def rate(detail: list[tuple[int, int, int]], idxs: set[int]) -> str:
    a = sum(wa for i, wa, _ in detail if i in idxs)
    b = sum(wb for i, _, wb in detail if i in idxs)
    if not a + b:
        return "—"
    p = a / (a + b)
    return f"{p:.3f} (n={len(idxs)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pickg210_dpo-b_final")
    ap.add_argument("--against", default="original", help="对手字段（默认官方原答）")
    args = ap.parse_args()

    run = ROOT / "data" / "runs" / args.run
    rows = [json.loads(l) for l in open(run / "arms.jsonl", encoding="utf-8")]
    sem = {}
    sp = run / "sem.jsonl"
    if sp.is_file():
        sem = {r["turn_id"]: r["sem"] for r in map(json.loads, open(sp, encoding="utf-8"))}

    items = [(r.get("user", ""), r["polish"], r[args.against]) for r in rows]
    res = duel(LLM(), items)
    lo, hi = res.ci95()
    print(f"{args.run}  vs {args.against}:  {res.win_rate:.3f}  CI [{lo:.2f}, {hi:.2f}]"
          f"  未解析 {res.unparsed}\n")

    def squash(s: str) -> str:
        return "".join(c for c in contract.TAG_RE.sub("", s) if not c.isspace())

    same = {i for i, r in enumerate(rows) if squash(r["polish"]) == squash(r[args.against])}
    diff = set(range(len(rows))) - same
    good = {i for i, r in enumerate(rows) if sem.get(r["turn_id"], 0) >= 3}
    bad = {i for i, r in enumerate(rows) if sem.get(r["turn_id"], 9) <= 2}
    multi = {i for i, r in enumerate(rows) if r["polish"].count("[") >= 3}
    single = {i for i, r in enumerate(rows) if r["polish"].count("[") <= 2}

    print(f"{'切片':<26s} 胜率")
    print(f"{'全部':<26s} {rate(res.detail, set(range(len(rows))))}")
    print(f"{'与真货逐字相同':<24s} {rate(res.detail, same)}")
    print(f"{'与真货不同':<26s} {rate(res.detail, diff)}")
    print(f"{'语义 3~4 分':<26s} {rate(res.detail, good)}")
    print(f"{'语义 ≤2 分（改意思了）':<22s} {rate(res.detail, bad)}")
    print(f"{'≥3 段':<25s} {rate(res.detail, multi)}")
    print(f"{'1~2 段':<25s} {rate(res.detail, single)}")

    names = {0: "两问皆输", 1: "一胜一负", 2: "两问皆胜"}
    hist: dict[str, int] = {}
    for i, wa, wb in res.detail:
        k = names[2 if wa == 2 else (1 if wa == 1 else 0)]
        hist[k] = hist.get(k, 0) + 1
    print(f"\n逐条: {hist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())