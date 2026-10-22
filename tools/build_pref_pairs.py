"""从挖负例的候选缓存里，构造**同输入**偏好对（chosen 与 rejected 都来自模型自己）。

## 为什么需要这一版

`dpo_pairs_k3.jsonl` 里 `chosen` 是官方原答。用它做 DPO 会同时教两件事：

1. 「别改事实」——我们想要的
2. 「更像官方原答的措辞」——我们**不**想要的（那本来就是 SFT 在做的事，
   而且会把润色器往「照抄指令内容」的方向拽，实测抄指令相似度 0.49→0.53、
   盲评 0.447→0.383 就是后果）

正确构造：同一个表达指令、同一个模型，一个候选语义干净（sem>=3）、一个候选漂移
（sem<=2）。两者风格都是模型自己的，**唯一差别就是有没有改意思**。这才是单轴负例。

    python tools/build_pref_pairs.py --cache data/pairs/dpo_mined_k3.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import contract, evaluate  # noqa: E402
from moneaka.llm import LLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/pairs/dpo_mined_k3.jsonl")
    ap.add_argument("--sem-cache", default="data/pairs/dpo_sem_all.jsonl")
    ap.add_argument("--out-own", default="data/pairs/dpo_pairs_own.jsonl")
    ap.add_argument("--out-official", default="data/pairs/dpo_pairs_k3.jsonl")
    ap.add_argument("--clean-min", type=int, default=3, help="视为「干净」的语义分下限")
    ap.add_argument("--drift-max", type=int, default=2)
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.cache, encoding="utf-8")]
    print(f"候选 {len(rows)} 条")

    sem_path = Path(args.sem_cache)
    sem: dict[str, int] = {}
    if sem_path.is_file():
        sem = {r["key"]: r["sem"] for r in map(json.loads, open(sem_path, encoding="utf-8"))}
        print(f"续跑：已有语义分 {len(sem)}")
    todo = [r for r in rows if r["key"] not in sem]
    if todo:
        # 注意：语义评委的输入是 (表达指令的内容行, 候选输出)。评委只看语义，
        # 不看风格——这是它能同时判「干净」和「漂移」的前提。
        probe = [{"turn_id": r["key"], "draft": r["plan"], "polish": r["rejected"]}
                 for r in todo]
        got = evaluate.judge_semantic_rows(LLM(), probe, "polish", batch_size=args.batch)
        sem.update(got)
        with open(sem_path, "w", encoding="utf-8") as fh:
            for k, v in sem.items():
                fh.write(json.dumps({"key": k, "sem": v}, ensure_ascii=False) + "\n")
        print(f"语义分写入 {sem_path}（{len(sem)}/{len(rows)}）")

    by_turn: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        v = sem.get(r["key"])
        if v is None:
            continue
        r = dict(r, sem=v)
        by_turn[r["turn_id"]].append(r)

    own: list[dict] = []
    for tid, cands in by_turn.items():
        clean = [c for c in cands if c["sem"] >= args.clean_min
                 and c["contract_ok"] and contract.check(c["chosen"]).ok]
        drift = [c for c in cands if c["sem"] <= args.drift_max
                 and c["contract_ok"]]
        if not clean or not drift:
            continue
        # 一条 clean 配一条 drift（取语义最好的 clean 和最差的 drift）
        c = max(clean, key=lambda x: x["sem"])
        d = min(drift, key=lambda x: x["sem"])
        if c["rejected"].strip() == d["rejected"].strip():
            continue
        own.append({"turn_id": tid, "plan": c["plan"],
                    "chosen": c["rejected"],          # ← 模型自己的干净输出
                    "rejected": d["rejected"],        # ← 模型自己的漂移输出
                    "sem_chosen": c["sem"], "sem_rejected": d["sem"],
                    "source": "on_policy_same_prompt"})

    out = Path(args.out_own)
    with open(out, "w", encoding="utf-8") as fh:
        for r in own:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n同输入偏好对 {len(own)} 条 → {out}")
    print(f"  （覆盖 {len(by_turn)} 个 turn 里既有干净又有漂移的那些）")

    # 同时把「chosen=官方原答」那版按同样的过滤条件重出一份，方便对比
    off: list[dict] = []
    for tid, cands in by_turn.items():
        for c in cands:
            if c["sem"] > args.drift_max or not c["contract_ok"]:
                continue
            if c["rejected"].strip() == c["chosen"].strip():
                continue
            off.append({"turn_id": tid, "plan": c["plan"], "chosen": c["chosen"],
                        "rejected": c["rejected"], "sem": c["sem"],
                        "source": "official_chosen"})
    p2 = Path(args.out_official)
    with open(p2, "w", encoding="utf-8") as fh:
        for r in off:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"官方-chosen 版 {len(off)} 条 → {p2}")

    for r in own[:5]:
        print(f"\n[{r['sem_chosen']} vs {r['sem_rejected']}] 指令 "
              f"{r['plan'].splitlines()[1][:36]}")
        print(f"   chosen: {r['chosen'][:66]}")
        print(f"   reject: {r['rejected'][:66]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())