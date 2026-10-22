"""离线重新归一化一次旧的评测 run（不调模型）。

用途：`normalize.py` 的标签别名表是从真实失败里长出来的，每补一批别名，
只想看新效果，不想重跑模型（润色本地免费，但 direct 臂是花钱的）。
做法：用 arms.jsonl 里存的 `<arm>_raw` 重新归一化，写回 `<arm>`。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moneaka import contract
from moneaka.normalize import normalize

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    path = ROOT / "data" / "runs" / args.run / "arms.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    changed = 0
    before = after = 0
    for r in rows:
        r["norm"] = []
        for arm in ("polish", "direct"):
            raw = r.get(f"{arm}_raw")
            if not raw:
                continue
            fixed, st = normalize(raw)
            if fixed != r.get(arm):
                changed += 1
            before += contract.check(raw).ok
            r[arm] = fixed
            r["norm"].append({arm: st.as_dict()})
    for r in rows:
        for arm in ("polish", "direct"):
            if r.get(arm):
                after += contract.check(r[arm]).ok

    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows) * 2
    print(f"重新归一化 {path.name}：改动了 {changed} 个输出")
    print(f"契约通过（原始 -> 归一化）：{before}/{n} -> {after}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())