"""在固定留出集上，对多个 checkpoint 同时测「像不像」与「有没有改意思」。

为什么需要这个工具：`eval_loss` 在这个任务上是**反向**信号（run2 实测 loss 从 1.07 涨到
1.89，盲评胜率却从 0.30 升到 0.38）。所以选点必须靠生成 + 盲评。而 run7 又暴露了第二个
问题：**风格变好的同时语义保真会退步**。只看盲评会选到「说得很像但把事实改了」的点。

于是这里并排给两个数：

* `win`： vs 官方原答的盲评胜率（成对、正反各一次抵消位置偏差）
* `sem`： 语义保持均值（0~4，4=只剩表达差异）

选点规则（写死，不能事后改）：**先卡语义下限，再比风格**。
语义均分低于全部候选最高值 -0.15 的，无论风格多好都淘汰——「说错话」比「说得不像」更伤。

用法::

    python tools/pick_checkpoint.py --adapter-dir data/adapters/run8 \\
        --pairs data/pairs/split_sweep60.jsonl --workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import evaluate  # noqa: E402
from moneaka.duel import duel  # noqa: E402
from moneaka.llm import LLM  # noqa: E402


def gen(adapter: Path | str, base: str, name: str, pairs: str, workers: int,
        temperature: float = 0.7) -> bool:
    cmd = [sys.executable, str(ROOT / "tools" / "eval_adapter.py"),
           "--adapter", str(adapter), "--base", base, "--name", name,
           "--pairs", pairs, "--held-out", "0", "--workers", str(workers),
           "--temperature", str(temperature)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1",
               PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", env=env)
    ok = r.returncode == 0
    if not ok:
        print(f"  ! 生成失败: {(r.stderr or '')[-300:]}")
    return ok


def measure(name: str, llm: LLM, sample: int | None) -> dict:
    rows = [json.loads(l) for l in open(ROOT / "data" / "runs" / name / "arms.jsonl",
                                        encoding="utf-8")]
    if sample:
        rows = rows[:sample]
    ok = sum(r["contract_ok"] for r in rows)
    items = [(r["user"], r["polish"], r["original"]) for r in rows if r["polish"].strip()]
    d = duel(llm, items)
    sem = evaluate.judge_semantic_rows(llm, rows, "polish", batch_size=6)
    sem_vals = list(sem.values())
    return {
        "run": name,
        "n": len(rows),
        "contract": ok / max(1, len(rows)),
        "win": round(d.win_rate, 3) if d.win_rate is not None else None,
        "win_ci": [round(x, 2) for x in d.ci95()],
        "sem": round(sum(sem_vals) / len(sem_vals), 2) if sem_vals else None,
        "sem_low": sum(1 for v in sem_vals if v <= 2),
        "sem_n": len(sem_vals),
        "unparsed": d.unparsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--base", default="data/models/Qwen3-1.7B")
    ap.add_argument("--pairs", default="data/pairs/split_sweep60.jsonl")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sample", type=int, default=0, help="0=用全部留出题")
    ap.add_argument("--final-only", action="store_true", help="只测 adapter 目录本身（不测中间点）")
    ap.add_argument("--reuse", action="store_true", help="已有 runs/ 输出就跳过生成")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="0 = 贪心解码（确定性）。评测必须用 0，否则采样噪声会淹没差异")
    ap.add_argument("--tag", default="", help="输出名后缀，避免覆盖旧结果")
    args = ap.parse_args()

    adir = ROOT / args.adapter_dir
    targets: list[tuple[str, Path]] = []
    if not args.final_only:
        for ck in sorted(adir.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1])):
            targets.append((f"pick{args.tag}_{adir.name}_{ck.name}", ck))
    targets.append((f"pick{args.tag}_{adir.name}_final", adir))

    llm = LLM()
    rows: list[dict] = []
    for name, path in targets:
        have = (ROOT / "data" / "runs" / name / "arms.jsonl").is_file()
        if not (args.reuse and have):
            print(f"\n=== 生成 {path.name} ===")
            if not gen(path, args.base, name, args.pairs, args.workers, args.temperature):
                continue
        m = measure(name, llm, args.sample or None)
        print(f"  {path.name:>18s}  win={m['win']} CI{m['win_ci']}  "
              f"sem={m['sem']} (≤2分 {m['sem_low']}条)  契约 {m['contract']:.0%}")
        m["checkpoint"] = path.name
        rows.append(m)

    if not rows:
        print("没有可用结果")
        return 1

    # 选点：先卡语义下限（最高分 -0.15），再比风格
    best_sem = max(r["sem"] for r in rows if r["sem"] is not None)
    floor = round(best_sem - 0.15, 2)
    passed = [r for r in rows if r["sem"] is not None and r["sem"] >= floor]
    best = max(passed, key=lambda r: r["win"])

    print("\n=== 选点表 ===")
    print(f"{'checkpoint':>20s}  {'win':>5s}  {'sem':>5s}  判定")
    for r in rows:
        verdict = []
        if r["sem"] is not None and r["sem"] < floor:
            verdict.append(f"语义淘汰(<{floor})")
        if r is best:
            verdict.append("← 选中")
        print(f"{r['checkpoint']:>20s}  {str(r['win']):>5s}  {str(r['sem']):>5s}  {' '.join(verdict)}")
    print(f"\n语义下限 {floor}（最高 {best_sem} - 0.15）；选中 {best['checkpoint']}")

    out = ROOT / "data" / "sweep" / f"{adir.name}_joint{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "best": best["checkpoint"],
                               "sem_floor": floor}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())