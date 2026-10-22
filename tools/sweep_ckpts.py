"""逐个 checkpoint 做「留出集生成 + 盲评胜率」，用来选点。

为什么不用 eval_loss 选：实测 run2 的 eval_loss 从 1.07 一路涨到 1.89，
而同一批 checkpoint 的盲评胜率却从 0.30 升到 0.38。这个任务里
「记住训练分布」和「损失变高」是同时发生的，损失不是风格的代理。

用法：
    python tools/sweep_ckpts.py --adapter-dir data/adapters/run3 --epochs 10 --sample 16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--base", default="data/models/Qwen3-1.7B")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--sample", type=int, default=16, help="每个点参与盲评的条数")
    ap.add_argument("--arms", default="polish,reference")
    ap.add_argument("--judge-workers", type=int, default=3)
    args = ap.parse_args()

    adir = ROOT / args.adapter_dir
    cks = sorted((p for p in adir.glob("checkpoint-*")),
                 key=lambda p: int(p.name.split("-")[1]))
    if not cks:
        print(f"{adir} 下没有 checkpoint-*")
        return 1

    rows = []
    for ck in cks:
        name = f"sweep_{ck.name.replace('checkpoint-', 'ck')}"
        cmd = [sys.executable, str(ROOT / "tools" / "eval_adapter.py"),
               "--adapter", str(ck), "--base", args.base, "--name", name]
        print(f"\n=== {ck.name} → 生成留出集输出 ===")
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        tail = [l for l in (r.stdout or "").splitlines()
                if l.startswith(("契约通过率", "空输出", "留出集"))]
        print("\n".join(tail) or (r.stderr or "")[-300:])
        if r.returncode:
            print(f"  跳过（生成失败）: {(r.stderr or '')[-200:]}")
            continue
        cmd2 = [sys.executable, "-m", "moneaka.cli", "duel", "--name", name,
                "--arms", args.arms, "--sample", str(args.sample)]
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1",
                   PYTHONIOENCODING="utf-8")
        r2 = subprocess.run(cmd2, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", env=env)
        win = None
        for line in (r2.stdout or "").splitlines():
            if line.startswith("polish"):
                nums = [float(x) for x in line.split()[1:] if x.replace(".", "").isdigit()]
                if nums:
                    win = nums[-1]      # 对 reference 的胜率
        print(f"  盲评 vs reference: {win if win is not None else '解析失败'}")
        rows.append({"checkpoint": ck.name, "win_vs_reference": win,
                     "gen": " ".join(tail)})

    print("\n=== 选点表 ===")
    for r in rows:
        w = r["win_vs_reference"]
        mark = ""
        if w is not None:
            mark = " ← 最佳" if w == max(x["win_vs_reference"] for x in rows
                                        if x["win_vs_reference"] is not None) else ""
        print(f"  {r['checkpoint']:>18s}  {w}{mark}")
    out = ROOT / "data" / "sweep" / (adir.name + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
