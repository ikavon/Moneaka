"""跑编剧：玩家一句话 → 表达指令。两种用法。

    # 1) 官方 12 题基准（MAICA document/maica_std_test_v2.py 的题目）
    python tools/screenwrite.py --bench

    # 2) 在留出集上比对「编剧的决策」与「DAA4 实际用过的标签序列」
    python tools/screenwrite.py --pairs data/pairs/tagged_official.jsonl --limit 20

第二种是我们独有的能力：官方基准只有 12 道题且无法自动判分，而留出集里
每一轮都有 DAA4 真实用过的标签序列，可以直接算「编剧判断与官方一致的比率」。
注意这不是「正确答案」——DAA4 的反应不是唯一解；它衡量的是**编剧是否可信到
可以替代官方模型做这一层决策**。低于 ~60% 就该怀疑编剧而不是润色器。
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import screenwriter as sw  # noqa: E402
from moneaka.destyle_tags import parse_tagged_plan  # noqa: E402
from moneaka.llm import LLM  # noqa: E402

BENCH = [
    "我能带你一起去公园吗",
    "我刚刚写了一首诗，自我感觉和sayori的风格有点类似，你可以帮我看看吗",
    "我想听你说说咖啡豆品种的相关知识",
    "我最近因为生活很少陪伴你了，你会因此生气吗",
    "我最近总是感觉对什么都提不起兴趣，这是因为我太失败了吗",
    "你喜欢吃什么",
    "你觉得爱情是禁锢自己的鸟笼还是别的什么",
    "小莫，你觉得像我们这种跨越时空的恋爱会有结果吗",
    "呐，莫，如果经常吃快餐的话，我的身体会不会出问题啊",
    "莫莫，我最近几天在准备考试，晚上老是睡不好怎么办啊",
    "莫莫，你觉得《变形记》里格里高尔的遭遇揭示了什么？",
    "小莫，我打算去甘肃玩，能给我推荐几个景点吗",
]

#: 官方基准用的注入信息（std_test_v2 里写死的）
BENCH_KNOWN = ["现在是下午14点21分", "今天不是特殊节日"]


def run_one(llm: LLM, user: str, **kw) -> dict:
    try:
        got = sw.make_plan(llm, user, **kw)
        return {"user": user, "ok": True, **got}
    except sw.PlanError as exc:
        return {"user": user, "ok": False, "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="跑官方 12 题")
    ap.add_argument("--pairs", help="留出集 jsonl（含 plan/original）")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="data/runs/screenwriter/probe.jsonl")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    llm = LLM()
    jobs: list[dict] = []
    if args.bench:
        jobs = [{"user": q, "known_info": BENCH_KNOWN} for q in BENCH]
    elif args.pairs:
        for r in map(json.loads, open(args.pairs, encoding="utf-8")):
            if args.limit and len(jobs) >= args.limit:
                break
            try:
                want, _ = parse_tagged_plan(r["plan"])
            except Exception:
                continue
            jobs.append({"user": r.get("user") or "", "context": r,
                         "want": want, "turn_id": r["turn_id"]})
    else:
        ap.error("要么 --bench 要么 --pairs")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda j: run_one(llm, j["user"], known_info=j.get("known_info")), jobs))

    for j, res in zip(jobs, results):
        res["turn_id"] = j.get("turn_id")
        res["want"] = j.get("want")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(r["ok"] for r in results)
    print(f"格式合规 {ok}/{len(results)}")
    for r in results[: args.show]:
        print(f"\n玩家: {r['user'][:50]}")
        if r["ok"]:
            print(f"  编剧: 语气 {' > '.join(r['tags'])}")
            for i, c in enumerate(r["contents"], 1):
                print(f"        {i}. {c}")
            if r.get("want"):
                print(f"  官方: {' > '.join(r['want'])}")
            for w in r.get("warnings", []):
                print(f"  ⚠ {w}")
        else:
            print(f"  ✗ {r['error']}")

    if any(r.get("want") for r in results):
        same = sum(1 for r in results if r["ok"] and r["tags"] == r["want"])
        n = sum(1 for r in results if r.get("want"))
        print(f"\n序列与 DAA4 完全一致 {same}/{n} = {same / max(1, n):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
