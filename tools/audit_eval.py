"""量旁路审查：B（台词复核）的查全/查准，C（plan 审查）的误报率。

B 的基准答案是已有的语义分：206 条留出集里 sem≤2 的 17 条算「真有问题」。
C 没有现成的基准，改用「真人 plan」当正样本——真人写的东西不应该被拦，
所以在 206 条真人 plan 上的拦截率就是**误报率**。

    python tools/audit_eval.py --b-n 57 --k 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import audit, dataset, screenwriter  # noqa: E402
from moneaka.destyle_tags import parse_tagged_plan  # noqa: E402
from moneaka.llm import LLM  # noqa: E402


def eval_b(llm, n_good: int, seed: int = 0) -> None:
    run = ROOT / "data" / "runs" / "pickg210_run8_final"
    rows = {r["turn_id"]: r for r in map(json.loads, open(run / "arms.jsonl", encoding="utf-8"))}
    sem = {r["turn_id"]: r["sem"] for r in map(json.loads, open(run / "sem.jsonl", encoding="utf-8"))}
    bad = [t for t, v in sem.items() if v <= 2]
    good = [t for t, v in sem.items() if v >= 4]
    rng = random.Random(seed)
    good = rng.sample(good, min(n_good, len(good)))
    print(f"\n=== B 台词复核：坏例 {len(bad)} 条 + 好例 {len(good)} 条 ===")

    res = {"bad": [], "good": []}
    for label, ids in (("bad", bad), ("good", good)):
        for tid in ids:
            r = rows[tid]
            v = audit.audit_polish(llm, r["draft"], r["polish"])
            res[label].append({"turn_id": tid, "sem": sem[tid], "ok": v.ok,
                               "kind": v.kind, "note": v.note})

    caught = sum(1 for x in res["bad"] if not x["ok"])
    fp = sum(1 for x in res["good"] if not x["ok"])
    print(f"  查全（坏例被拦）  {caught}/{len(bad)} = {caught / max(1, len(bad)):.0%}")
    print(f"  误报（好例被拦）  {fp}/{len(good)} = {fp / max(1, len(good)):.0%}")
    for x in res["bad"]:
        if not x["ok"]:
            print(f"    ✓ {x['turn_id']:20s} {x['kind']:4s} {x['note'][:36]}")
    for x in res["good"]:
        if not x["ok"]:
            print(f"    ✗误报 {x['turn_id']:16s} {x['kind']:4s} {x['note'][:36]}")
    (ROOT / "data" / "runs" / "audit_b.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def eval_c(llm, n_llm: int, seed: int = 0) -> None:
    """C：机械部分在真人 plan 上的误报率；模型部分抽 n 条看。"""
    samples, _ = dataset.build("data/pairs/split_holdout210.jsonl")
    print(f"\n=== C plan 审查：真人 plan {len(samples)} 条（正样本） ===")
    flags = []
    for s in samples:
        bad = audit.check_plan_mechanical(s.plan)
        if bad:
            flags.append((s.turn_id, bad))
    print(f"  机械拦截 {len(flags)}/{len(samples)} = {len(flags) / len(samples):.1%}"
          f"  ← 这些是误报（真人写的）")
    from collections import Counter
    c = Counter(p.split("（")[0].split("「")[0][:12] for _, ps in flags for p in ps)
    if c:
        print(f"  误报类型: {c.most_common()}")

    rng = random.Random(seed)
    pick = rng.sample(samples, min(n_llm, len(samples)))
    canon = "\n".join(screenwriter.load_canon())
    ok = 0
    for s in pick:
        v = audit.audit_plan(llm, s.plan, s.user_msg, canon=canon)
        ok += v.ok
        if not v.ok:
            print(f"    ✗误报 {s.turn_id:18s} {v.note[:50]}")
    print(f"  模型审查（{len(pick)} 条真人 plan）判 ok {ok}/{len(pick)} = {ok / len(pick):.0%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="bc", choices=["b", "c", "bc"])
    ap.add_argument("--b-n", type=int, default=40, help="好例抽多少条")
    ap.add_argument("--c-n", type=int, default=20, help="C 的模型审查抽多少条")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    llm = LLM()
    if "b" in args.which:
        eval_b(llm, args.b_n, args.seed)
    if "c" in args.which:
        eval_c(llm, args.c_n, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())