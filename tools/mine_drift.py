"""挖 on-policy 负例：让**模型自己**产出语义漂移的输出，再把它们当 DPO 的 rejected。

为什么不用人工构造：手头的 18 例会漂成什么样，是事后归纳的。模型真实的错误分布比
我们的想象更窄也更怪（`百分之一`→`万分之一`、`我需要问你`→`我现在还不好吗`）。
用模型自己的失败当负例，DPO 学到的偏序才和它实际的错误一一对应。

**关键筛选**：rejected 必须**通过契约校验**。否则负例同时带格式错，模型会去学格式，
把「不许改事实」这个信号稀释掉——那就变成了另一个任务。

留出集永久冻结：本工具只在训练集上挖。`--pairs` 指向任何包含留出样本的文件都会被拒绝。

    python tools/mine_drift.py --pairs data/pairs/split_train_run8.jsonl \\
        --adapter data/adapters/run8 --out data/pairs/dpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import contract, dataset, evaluate  # noqa: E402
from moneaka.llm import LLM  # noqa: E402
from moneaka.localhf import GenConfig, get_model  # noqa: E402
from moneaka.normalize import normalize  # noqa: E402

FROZEN = ROOT / "data" / "pairs" / "split_holdout210.jsonl"


def frozen_ids() -> set[str]:
    if not FROZEN.is_file():
        return set()
    return {json.loads(l)["turn_id"] for l in open(FROZEN, encoding="utf-8") if l.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/split_train_run8.jsonl")
    ap.add_argument("--adapter", default="data/adapters/run8")
    ap.add_argument("--base", default="data/models/Qwen3-1.7B")
    ap.add_argument("--out", default="data/pairs/dpo_pairs.jsonl")
    ap.add_argument("--cache", default="data/pairs/dpo_mined_raw.jsonl",
                    help="生成 + 语义分的原始结果（断点续跑用）")
    ap.add_argument("--sem-max", type=int, default=2, help="判定为漂移的语义分上限")
    ap.add_argument("--k", type=int, default=1, help="每条采几个候选（>1 时用采样生成）")
    ap.add_argument("--temperature", type=float, default=0.8, help="k>1 时的采样温度")
    ap.add_argument("--max-sim", type=float, default=0.95,
                    help="与 chosen 的字序相似度超过此值视为「其实没变」——评委判过头了，"
                         "拿它当负例会教模型「别用『么』」，是错的信号")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()

    frozen = frozen_ids()
    samples, stats = dataset.build(args.pairs)
    samples = [s for s in samples if s.turn_id not in frozen]
    print(f"训练样本 {len(samples)}（{stats['dropped']}）；已排除留出集 {len(frozen)} 条")

    cache = Path(args.cache)
    done: dict[str, dict] = {}
    if cache.is_file():
        done = {r["key"]: r for r in map(json.loads, open(cache, encoding="utf-8"))}
        print(f"续跑：已有 {len(done)} 条候选")

    # 缓存键带候选序号：k=1 时是 `tid#0`（贪心），k>1 时是 `tid#0..k-1`（采样）。
    # 为什么需要多候选：贪心只给 29 对负例，且其中过半是评委判过头的琐碎差异。
    # 采样能把「模型偶尔会怎么错」的分布拓开，pair 数才能支撑 DPO。
    todo: list[tuple[dataset.Sample, int]] = []
    for s in samples:
        for i in range(args.k):
            if f"{s.turn_id}#{i}" not in done:
                todo.append((s, i))
    if todo:
        model = get_model(args.base, adapter=args.adapter)
        greedy = args.k == 1
        cfg = GenConfig(temperature=0.0 if greedy else args.temperature, max_new_tokens=200)
        print(f"生成 {len(todo)} 条候选"
              f"（{'greedy' if greedy else f'sampling T={args.temperature}'}）…")

        def one(item) -> dict:
            s, i = item
            raw = model.chat([dict(m) for m in s.as_messages()], cfg)
            text, _ = normalize(raw)
            return {"key": f"{s.turn_id}#{i}", "turn_id": s.turn_id, "idx": i,
                    "plan": s.plan, "chosen": s.target,
                    "rejected_raw": raw, "rejected": text,
                    "contract_ok": contract.check(text).ok}

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rec in ex.map(one, todo):
                done[rec["key"]] = rec
                if len(done) % 100 == 0:
                    print(f"  {len(done)}/{len(samples) * args.k}")
        with open(cache, "w", encoding="utf-8") as fh:
            for r in done.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"候选写入 {cache}")

    # 语义评委
    rows = [{"turn_id": r["key"], "draft": r["plan"], "polish": r["rejected"]}
            for r in done.values()]
    llm = LLM()
    print(f"语义评分 {len(rows)} 条（batch {args.batch}）…")
    sem = evaluate.judge_semantic_rows(llm, rows, "polish", batch_size=args.batch)
    print(f"  拿到分数 {len(sem)}/{len(rows)}")

    import difflib
    import re

    def _norm(s: str) -> str:
        return re.sub(r"[\s,.!?~…'\"、，。！？]", "", contract.TAG_RE.sub("", s))

    kept: list[dict] = []
    seen_pair: set[tuple[str, str]] = set()
    drop = {"no_sem": 0, "not_drift": 0, "contract": 0, "same": 0, "too_similar": 0, "dup": 0}
    for key, r in done.items():
        v = sem.get(key)
        if v is None:
            drop["no_sem"] += 1
            continue
        if v > args.sem_max:
            drop["not_drift"] += 1
            continue
        if not r["contract_ok"] or not contract.check(r["chosen"]).ok:
            drop["contract"] += 1
            continue
        if r["rejected"].strip() == r["chosen"].strip():
            drop["same"] += 1
            continue
        sim = difflib.SequenceMatcher(None, _norm(r["chosen"]), _norm(r["rejected"])).ratio()
        if sim > args.max_sim:
            drop["too_similar"] += 1
            continue
        sig = (r["plan"], r["rejected"])
        if sig in seen_pair:
            drop["dup"] += 1
            continue
        seen_pair.add(sig)
        kept.append({"turn_id": r["turn_id"], "key": key, "plan": r["plan"],
                     "chosen": r["chosen"], "rejected": r["rejected"],
                     "sem": v, "sim": round(sim, 2), "source": "on_policy"})

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"\n负例对 {len(kept)} 条 → {out}")
    print(f"  语义分分布 {dict(sorted(Counter(r['sem'] for r in kept).items()))}")
    print(f"  丢弃 {drop}")
    for r in kept[:6]:
        print(f"\n[{r['sem']}] 指令 {r['plan'].splitlines()[1][:40]}")
        print(f"     真货: {r['chosen'][:60]}")
        print(f"     负例: {r['rejected'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())