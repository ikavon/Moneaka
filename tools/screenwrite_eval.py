"""量编剧：有没有「先例检索」差多少。

对照组 = 现在的编剧（只有角色设定 + 规范）；实验组 = 多给 4 条相似局面的 (语境 → plan)。

两个指标：
1. **标签序列吻合**（机械）：和真人 plan 的标签序列完全一致的比例
2. **内容要点覆盖**（评委）：把真人的内容行当草稿、模型产出的内容行当润色，
   问「意思对不对得上」——沿用润色器那把尺子，保证同一标准

    python tools/screenwrite_eval.py --n 60          # 先小样本看方向
    python tools/screenwrite_eval.py --n 206         # 全量
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import dataset, evaluate, screenwriter  # noqa: E402
from moneaka.destyle_tags import parse_tagged_plan  # noqa: E402
from moneaka.llm import LLM  # noqa: E402
from moneaka.planbank import PlanBank  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/split_holdout210.jsonl")
    ap.add_argument("--train", default="data/pairs/split_train_run8.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=4, help="检索几条先例")
    ap.add_argument("--tight", action="store_true", help="启用硬长度预算（15~45 字 / 1~3 段）")
    ap.add_argument("--out", default="data/runs/screenwrite_bank.jsonl")
    args = ap.parse_args()

    # 手动读：要带 context（编剧看得到历史，留出集的 tagged 文件里没有 context 字段）
    rows_in = [json.loads(l) for l in open(ROOT / args.pairs, encoding="utf-8")]
    pick = rows_in[: args.n]
    if not pick[0].get("context"):
        print("⚠️ 这个文件没有 context，编剧等于盲猜，评测无意义")
    bank = PlanBank.from_pairs(ROOT / args.train)
    print(f"范例库 {len(bank.docs)} 条；评测 {len(pick)} 条；对照组 k=0，实验组 k={args.k}")

    llm = LLM()
    rows: list[dict] = []
    for i, s in enumerate(pick, 1):
        human_tags, human_contents = parse_tagged_plan(s["plan"])
        history = s.get("context") or []
        rec = {"turn_id": s["turn_id"], "user": s["user"],
               "human_tags": human_tags, "human_contents": human_contents}
        modes = (("base", 0), ("bank", args.k))
        for mode, k in modes:
            ex = bank.search(s["user"], k=k) if k else None
            try:
                got = screenwriter.make_plan(llm, s["user"], history=history,
                                             examples=ex, tight=args.tight, retries=2)
                rec[mode] = {"tags": got["tags"], "contents": got["contents"],
                             "tag_ok": got["tags"] == human_tags}
            except Exception as exc:  # noqa: BLE001
                rec[mode] = {"tags": [], "contents": [], "tag_ok": False,
                             "error": str(exc)[:120]}
        rows.append(rec)
        if i % 10 == 0:
            print(f"  {i}/{len(pick)}  标签吻合 base "
                  f"{sum(1 for r in rows if r['base']['tag_ok'])} / bank "
                  f"{sum(1 for r in rows if r['bank']['tag_ok'])}")

    # 内容覆盖：真人的内容行当草稿，模型的内容行当润色
    probe_a = [{"turn_id": r["turn_id"], "draft": "；".join(r["human_contents"]),
                "polish": "；".join(r["base"]["contents"])} for r in rows if r["base"]["contents"]]
    probe_b = [{"turn_id": r["turn_id"], "draft": "；".join(r["human_contents"]),
                "polish": "；".join(r["bank"]["contents"])} for r in rows if r["bank"]["contents"]]
    sem_a = evaluate.judge_semantic_rows(llm, probe_a, "polish", batch_size=8)
    sem_b = evaluate.judge_semantic_rows(llm, probe_b, "polish", batch_size=8)

    out = ROOT / args.out
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "sem_base": sem_a.get(r["turn_id"]),
                                 "sem_bank": sem_b.get(r["turn_id"])},
                                ensure_ascii=False) + "\n")

    def rate(key: str) -> str:
        ok = sum(1 for r in rows if r[key]["tag_ok"])
        return f"{ok}/{len(rows)} = {ok / len(rows):.1%}"

    print("\n=== 结果 ===")
    print(f"标签序列完全吻合 base {rate('base')}   bank {rate('bank')}")

    def softer(key: str) -> str:
        from collections import Counter
        first = sum(1 for r in rows if r[key]["tags"] and r["human_tags"]
                    and r[key]["tags"][0] == r["human_tags"][0])
        jac = []
        for r in rows:
            a, b = Counter(r[key]["tags"]), Counter(r["human_tags"])
            inter = sum((a & b).values()); union = sum((a | b).values())
            if union:
                jac.append(inter / union)
        avg = sum(jac) / max(1, len(jac))
        return f"首标签吻合 {first}/{len(rows)} = {first/len(rows):.0%}，标签重合度 {avg:.2f}"

    print(f"  base: {softer('base')}")
    print(f"  bank: {softer('bank')}")
    for name, sem in (("base", sem_a), ("bank", sem_b)):
        v = list(sem.values())
        if v:
            print(f"内容覆盖 sem   {name}: 均值 {statistics.mean(v):.2f}  "
                  f"≤2 分 {sum(1 for x in v if x <= 2)}/{len(v)}")
    print(f"写入 {out}")

    print("\n样例（第一条最有代表性的差异）:")
    for r in rows:
        if r["bank"]["tag_ok"] and not r["base"]["tag_ok"]:
            print(f"  玩家: {r['user'][:60]}")
            print(f"  真人: {' > '.join(r['human_tags'])}")
            print(f"  基线: {' > '.join(r['base']['tags'])}  {'；'.join(r['base']['contents'])[:60]}")
            print(f"  先例: {' > '.join(r['bank']['tags'])}  {'；'.join(r['bank']['contents'])[:60]}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())