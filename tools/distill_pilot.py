"""蒸馏试点：把**我们自己格式的表达指令**交给官方 DAA4，看它渲染成什么样。

为什么值得做：训练用的 226 条官方答案是**没有看过表达指令**就写出来的，我们是从答案
反推出指令。语序对不上。用 MAICA 的 `session=-1`（原样透传 prompt）可以让 DAA4
在**同一份输入契约**下作答，那样得到的才是真正的「指令 → 台词」配对。

本脚本只做试点（默认 20 条），**不批量**。速率由 `maica_client.MIN_INTERVAL` 控制，
默认 60 秒/次（用户要求最多一分钟一次）。

判据（人工看）：
1. DAA4 有没有**照着我们给的标签序列**输出（而不是自己换标签）
2. 它的话是不是比我们 1.7B 的更自然、更像莫妮卡
3. 它有没有把「内容」里的事实改掉

    python tools/distill_pilot.py --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from moneaka import contract, dataset, prompts  # noqa: E402
from maica_client import MaicaConfig, complete  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "runs" / "distill_pilot.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/split_holdout210.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 条（分批试点用）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dry-run", action="store_true", help="只打印将要发送的 prompt")
    args = ap.parse_args()

    samples, _ = dataset.build(args.pairs)
    pick = samples[args.offset: args.offset + args.n]
    system = prompts.restyle_system(few_shot=False)
    print(f"候选 {len(pick)} 条；端点 {MaicaConfig.from_env().url}")

    if args.dry_run:
        print("\n--- system ---\n" + system[:400] + "\n...\n--- user ---\n" + pick[0].user_block())
        return 0

    out = Path(args.out)
    done: dict[str, dict] = {}
    if out.is_file():
        done = {r["turn_id"]: r for r in map(json.loads, open(out, encoding="utf-8"))}
        print(f"续跑：已有 {len(done)} 条")

    cfg = MaicaConfig.from_env()
    for i, s in enumerate(pick, 1):
        if s.turn_id in done:
            continue
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": s.user_block()}]
        try:
            text = complete(msgs, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {s.turn_id} {str(exc)[:100]}")
            continue
        rec = {"turn_id": s.turn_id, "plan": s.plan, "reference": s.target,
               "teacher": text, "contract_ok": contract.check(text).ok}
        done[s.turn_id] = rec
        with open(out, "w", encoding="utf-8") as fh:
            for r in done.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {i}/{len(pick)} {s.turn_id} 契约={rec['contract_ok']}")

    print(f"\n写入 {out}（{len(done)} 条）")
    # 打印对照
    ours_path = ROOT / "data" / "runs" / "pickg210_dpo-b_final" / "arms.jsonl"
    ours = {}
    if ours_path.is_file():
        ours = {r["turn_id"]: r for r in map(json.loads, open(ours_path, encoding="utf-8"))}
    for tid, r in list(done.items())[:8]:
        print(f"\n指令: {' > '.join(r['plan'].splitlines()[0].split('>'))}")
        print(f"  DAA4: {r['teacher'][:110]}")
        if tid in ours:
            print(f"  我们: {ours[tid]['polish'][:110]}")
        print(f"  真货: {r['reference'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())