"""全链路 C → 润色 → A → B，量最终交付质量。

在任务书里的位置：C 在渲染前拦 plan，A 在生成后做机械对账，B 在交付前做语义复核。
本脚本按这个顺序串起来，并且**把每一层的动作都记下来**（触发几次、救回几条、
几条退化成直译），因为只看最终均值会掩盖「靠退化换来的分数」。

退化策略：A/B 都过不了的，退回 plan 直译（标签 + 内容原样拼接）。宁可平淡，不能改意思。

    python tools/chain_eval.py --n 206
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import audit, contract, dataset, evaluate  # noqa: E402
from moneaka.destyle_tags import parse_tagged_plan  # noqa: E402
from moneaka.duel import duel  # noqa: E402
from moneaka.guarded import STRICT, check as _guard  # noqa: E402  (见 tools/guarded_polish.py)
from moneaka.llm import LLM  # noqa: E402
from moneaka.localhf import GenConfig, get_model  # noqa: E402
from moneaka.normalize import normalize  # noqa: E402


def literal(plan: str) -> str:
    """退化：标签和内容原样拼，不加任何风格。"""
    tags, contents = parse_tagged_plan(plan)
    return " ".join(f"[{t}]{c}" for t, c in zip(tags, contents))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/split_holdout210.jsonl")
    ap.add_argument("--base", default="data/models/Qwen3-1.7B")
    ap.add_argument("--adapter", default="data/adapters/run8")
    ap.add_argument("--n", type=int, default=206)
    ap.add_argument("--k", type=int, default=3, help="每层最多重采几次")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out", default="data/runs/chain.jsonl")
    args = ap.parse_args()

    samples, _ = dataset.build(args.pairs)
    pick = samples[: args.n]
    model = get_model(args.base, adapter=args.adapter)
    llm = LLM()
    greedy = GenConfig(temperature=0.0, max_new_tokens=200)
    samp = GenConfig(temperature=args.temperature, max_new_tokens=200)

    stats = {"c_flag": 0, "a_fail": 0, "b_trigger": 0, "b_drift": 0,
             "rescued": 0, "fallback": 0, "gen": 0}
    rows: list[dict] = []

    for i, s in enumerate(pick, 1):
        tags, contents = parse_tagged_plan(s.plan)
        c_flags = audit.check_plan_mechanical(s.plan)
        if c_flags:
            stats["c_flag"] += 1

        best: dict | None = None
        for attempt in range(args.k + 1):
            cfg = greedy if attempt == 0 else samp
            raw = model.chat([dict(m) for m in s.as_messages()], cfg)
            stats["gen"] += 1
            text, _ = normalize(raw)
            a_bad = [p for p in _guard(s.plan, text)
                     if p.startswith(STRICT)]  # 只信高精度两项，见 guarded.STRICT
            why = audit.needs_polish_audit(s.plan, text)
            b_bad = []
            if why:
                stats["b_trigger"] += 1
                v = audit.audit_polish(llm, s.plan, text)
                if not v.ok:
                    b_bad = [f"{v.kind}: {v.note}"]
                    stats["b_drift"] += 1
            score = len(a_bad) + len(b_bad)
            if best is None or score < best["score"]:
                best = {"polish": text, "a": a_bad, "b": b_bad, "score": score,
                        "attempts": attempt}
            if score == 0:
                break

        assert best is not None
        if best["score"]:
            stats["fallback"] += 1
            best["polish"] = literal(s.plan)
            best["degraded"] = True
        if best["attempts"] > 0 and not best.get("degraded"):
            stats["rescued"] += 1
        rows.append({"turn_id": s.turn_id, "plan": s.plan, "original": s.target,
                     **best, "c_flags": c_flags})
        if i % 20 == 0:
            print(f"  {i}/{len(pick)}  B触发{stats['b_trigger']} 退化{stats['fallback']}")

    out = ROOT / args.out
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 语义（和基线同一把尺子：草稿 = plan）
    sem = evaluate.judge_semantic_rows(
        llm, [{"turn_id": r["turn_id"], "draft": r["plan"], "polish": r["polish"]}
              for r in rows], "polish", batch_size=8)
    vals = [sem[r["turn_id"]] for r in rows if r["turn_id"] in sem]

    # 标签序列 + 抄写率
    def tag_ok(r: dict) -> bool:
        got = [m.group(1) for m in contract.TAG_RE.finditer(
            contract.PLACEHOLDER_RE.sub("\x00", r["polish"]))]
        return got == parse_tagged_plan(r["plan"])[0]

    def copy_ratio(r: dict) -> float:
        import difflib
        import re as _re
        n = lambda s: _re.sub(r"[\s,.!?~…'\"、，。！？]", "",
                              contract.TAG_RE.sub("", s))
        return difflib.SequenceMatcher(None, n(r["polish"]), n(r["plan"])).ratio()

    n_tag = sum(1 for r in rows if tag_ok(r))
    cp = statistics.median(copy_ratio(r) for r in rows)

    print("\n=== 全链路结果（206 条留出，草稿 = 真人 plan） ===")
    print(f"语义均值 {statistics.mean(vals):.2f}   ≤2 分 {sum(1 for v in vals if v <= 2)}/{len(vals)}")
    print(f"标签序列一致 {n_tag}/{len(rows)} = {n_tag/len(rows):.1%}")
    print(f"抄 plan 相似度中位 {cp:.2f}")
    print(f"C 机械拦截 {stats['c_flag']} 条 | A 判定失败 {stats['a_fail']} | "
          f"B 触发 {stats['b_trigger']} 次判漂移 {stats['b_drift']} 次 | "
          f"重采救回 {stats['rescued']} | 退化直译 {stats['fallback']} | "
          f"总生成 {stats['gen']} 次（{stats['gen']/len(rows):.2f}×）")

    base = {json.loads(l)["turn_id"]: json.loads(l)["sem"]
            for l in open(ROOT / "data/runs/pickg210_run8_final/sem.jsonl", encoding="utf-8")}
    chg = [sem[t] - base[t] for t in sem if t in base]
    print(f"逐条对比基线: 变好 {sum(1 for c in chg if c>0)}  不变 {sum(1 for c in chg if c==0)}  "
          f"变差 {sum(1 for c in chg if c<0)}")

    items = [(r["plan"], r["polish"], r["original"]) for r in rows]
    d = duel(llm, items)
    lo, hi = d.ci95()
    print(f"盲评 vs 官方原答 {d.win_rate:.3f}  CI [{lo:.2f}, {hi:.2f}]")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())