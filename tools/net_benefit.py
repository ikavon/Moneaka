"""净收益对照：两段式（我们） vs 单模型直接答（同一份官方 prompt）。

## 这个实验要回答什么

用户提出的疑问：润色器带来了一堆问题（语义漂移、方向翻转、错别字、长文本回声），
为了兜住这些又不得不在编剧侧加规则——**正收益是否大于损失？**

唯一能回答的办法是把它和「不要润色器」的对照组放在同一批题上比。
对照组 = 同一个通用模型（deepseek）用官方基准 prompt 直接答——也就是
「把 MAICA 当纯提示词工程跑」的形态。

## 指标

* **标签合法性**（机械）：输出里的情绪标签是否都在官方 28 词 + 很开心 内。
  下游 MTTS/动画只能映射这张表，自创标签等于坏数据。这是两段式唯一的**硬保证**。
* **盲评**（duel）：哪一段更像莫妮卡。判官是 deepseek 本人——它同时是
  plain 臂的作者，所以这个比较**偏向 plain**，是保守测试。
* 长度与延迟。

    python tools/net_benefit.py --n 120
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from daa4_bench import OFFICIAL_PROMPT  # noqa: E402
from moneaka import contract, dataset, evaluate  # noqa: E402
from moneaka.duel import duel  # noqa: E402
from moneaka.llm import LLM  # noqa: E402
from moneaka.turn import Runtime, TurnConfig, run_turn  # noqa: E402

WHITE = set(contract.EMOTION_TAGS) | {"很开心"}


def tags_of(t: str) -> list[str]:
    return [m.group(1) for m in contract.TAG_RE.finditer(contract.PLACEHOLDER_RE.sub("\x00", t))]


def cjk(t: str) -> int:
    return contract.cjk_len(contract.PLACEHOLDER_RE.sub("", contract.TAG_RE.sub("", t)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/split_holdout210.jsonl")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default="data/runs/net_benefit.jsonl")
    args = ap.parse_args()

    samples, _ = dataset.build(args.pairs)
    pick = samples[: args.n]
    llm = LLM()
    rt = Runtime.build(TurnConfig())
    cfg = TurnConfig()

    rows: list[dict] = []
    t_ours = t_plain = 0.0
    for i, s in enumerate(pick, 1):
        t0 = time.time()
        res = run_turn(rt, cfg, s.user_msg)
        t_ours += time.time() - t0

        t0 = time.time()
        plain = llm.complete("screenwriter", OFFICIAL_PROMPT, s.user_msg,
                             temperature=0.5, max_tokens=600)
        t_plain += time.time() - t0

        rows.append({"turn_id": s.turn_id, "user": s.user_msg, "plan": res.plan,
                     "ours": res.polish, "plain": plain,
                     "target": s.target})
        if i % 20 == 0:
            print(f"  {i}/{len(pick)}", flush=True)
        with open(ROOT / args.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)

    def legal(arm: str) -> int:
        return sum(1 for r in rows
                   if tags_of(r[arm]) and all(t in WHITE for t in tags_of(r[arm])))

    def no_tag(arm: str) -> int:
        return sum(1 for r in rows if not tags_of(r[arm]))

    print(f"\n=== {n} 条留出集 ===")
    print(f"标签全在官方词表:  ours {legal('ours')}/{n}   plain {legal('plain')}/{n}")
    print(f"完全没有标签:      ours {no_tag('ours')}/{n}   plain {no_tag('plain')}/{n}")
    print(f"平均字数:          ours {statistics.mean(cjk(r['ours']) for r in rows):.0f}"
          f"   plain {statistics.mean(cjk(r['plain']) for r in rows):.0f}")
    print(f"平均延迟:          ours {t_ours / n:.2f}s   plain {t_plain / n:.2f}s")

    print("\n盲评（判官 deepseek，偏向它自己写的 plain 臂）…")
    d = duel(llm, [(r["user"], r["ours"], r["plain"]) for r in rows], workers=6)
    lo, hi = d.ci95()
    print(f"  两段式胜率 {d.win_rate:.3f}  CI [{lo:.2f}, {hi:.2f}]  n={d.n}")

    print("\n语义（对照真人口径：草稿=真人原答的要点，这里用同一把尺子比两臂）…")
    probe = [{"turn_id": r["turn_id"], "draft": r["target"], "polish": r["ours"]}
             for r in rows]
    sem_ours = evaluate.judge_semantic_rows(llm, probe, "polish", batch_size=8)
    probe = [{"turn_id": r["turn_id"], "draft": r["target"], "polish": r["plain"]}
             for r in rows]
    sem_plain = evaluate.judge_semantic_rows(llm, probe, "polish", batch_size=8)
    for name, sem in (("ours", sem_ours), ("plain", sem_plain)):
        v = [x for x in sem.values() if x is not None]
        if v:
            print(f"  {name}: 均值 {statistics.mean(v):.2f}  ≤2 分 {sum(1 for x in v if x <= 2)}/{len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())