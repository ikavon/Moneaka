"""推理侧兜底：生成后机械校验，不合规就重采样。

## 为什么先做这个

把 206 条留出集的润色失手（sem≤2 共 17 条）逐条对回 plan 保真度，**只有 1 条能归因到
plan**，其余 16 条 plan 都是忠实的、模型自己说歪了。所以改 plan 写法不是杠杆
（v2 实验已证否，见 docs/STAGE_C_PLAN_WORDING.md）。

同时失败形态高度机械：数字被换、否定被删、内容段整个丢掉。
这类东西**不需要模型判断，可以硬查**。

策略：greedy 生成 → 过闸门 → 不过就换温度采样重试，最多 k 次。
闸门只看可机械验证的东西，不评价「好不好」——那归评委和人。

    python tools/guarded_polish.py --adapter data/adapters/run8 --k 3
    python tools/guarded_polish.py --adapter data/adapters/run8 --only-bad   # 只跑 17 条失手
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import contract, dataset  # noqa: E402
from moneaka.destyle_tags import parse_tagged_plan  # noqa: E402
from moneaka.localhf import GenConfig, get_model  # noqa: E402
from moneaka.normalize import normalize  # noqa: E402

from moneaka.guarded import check, nums  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/models/Qwen3-1.7B")
    ap.add_argument("--adapter", default="data/adapters/run8")
    ap.add_argument("--pairs", default="data/pairs/split_holdout210.jsonl")
    ap.add_argument("--baseline-run", default="pickg210_run8_final",
                    help="用哪次评测的 sem.jsonl 判定「失手」")
    ap.add_argument("--only-bad", action="store_true")
    ap.add_argument("--k", type=int, default=3, help="不过闸门时最多重采样几次")
    ap.add_argument("--loose", action="store_true",
                    help="启用 3-gram 段落覆盖检查（触发率 54%%，多数是误报）")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out", default="data/runs/guarded.jsonl")
    args = ap.parse_args()

    samples, _ = dataset.build(args.pairs)
    by_id = {s.turn_id: s for s in samples}

    want: list[str] | None = None
    if args.only_bad:
        f = ROOT / "data" / "runs" / args.baseline_run / "sem.jsonl"
        sem = {r["turn_id"]: r["sem"] for r in map(json.loads, open(f, encoding="utf-8"))}
        want = [t for t, v in sem.items() if v <= 2]
        print(f"只跑失手样本 {len(want)} 条")
    pick = [by_id[t] for t in want] if want else samples

    model = get_model(args.base, adapter=args.adapter)
    greedy = GenConfig(temperature=0.0, max_new_tokens=200)
    rows: list[dict] = []
    fixed = 0
    for i, s in enumerate(pick, 1):
        raw = model.chat([dict(m) for m in s.as_messages()], greedy)
        text, _ = normalize(raw)
        problems = check(s.plan, text, args.loose)
        attempts = 0
        while problems and attempts < args.k:
            attempts += 1
            cfg = GenConfig(temperature=args.temperature, max_new_tokens=200)
            raw2 = model.chat([dict(m) for m in s.as_messages()], cfg)
            t2, _ = normalize(raw2)
            p2 = check(s.plan, t2, args.loose)
            if len(p2) < len(problems):
                text, problems = t2, p2
        rows.append({"turn_id": s.turn_id, "plan": s.plan, "original": s.target,
                     "greedy": text if attempts == 0 else None,
                     "polish": text, "attempts": attempts, "problems": problems})
        if attempts and not problems:
            fixed += 1
        if i % 20 == 0:
            print(f"  {i}/{len(pick)}  重采过 {sum(1 for r in rows if r['attempts'])}  救回 {fixed}")

    out = ROOT / args.out
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_bad_before = sum(1 for r in rows if r["attempts"] == 0 and r["problems"])
    print(f"\n共 {len(rows)} 条")
    print(f"  greedy 直接通过 {len(rows) - sum(1 for r in rows if r['attempts'])} 条")
    print(f"  触发重采样 {sum(1 for r in rows if r['attempts'])} 条，其中救回 {fixed} 条")
    print(f"  最终仍有问题 {sum(1 for r in rows if r['problems'])} 条")
    print(f"写入 {out}")
    for r in rows:
        if r["problems"]:
            print(f"  ✗ {r['turn_id']}: {r['problems']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())