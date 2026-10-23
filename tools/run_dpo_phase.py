"""阶段 A 驱动：三个 DPO 配置依次训练，再用冻结协议逐个评测。

为什么要串成一个脚本而不是手动敲：DPO 单轮 30~50 分钟，中间任何一步漏了
「评测用 greedy / 用 210 条留出集 / 报 win+sem 两个数」，结论就不可比。
把顺序和参数钉进代码，避免半夜手动操作时手滑。

    python tools/run_dpo_phase.py            # 全部
    python tools/run_dpo_phase.py --skip-train   # 只评测已有的 adapter
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 三个候选配置（见 docs/PLAN_NEXT.md A.2）
CONFIGS = [
    {"name": "dpo-a", "beta": 0.1, "lr": 5e-6, "epochs": 2},
    {"name": "dpo-b", "beta": 0.3, "lr": 5e-6, "epochs": 2},
    {"name": "dpo-c", "beta": 0.1, "lr": 1e-5, "epochs": 3},
    # dpo-d 与 dpo-b 超参**完全相同**，唯一变量是 chosen 的来源：
    # dpo-b 用官方原答当 chosen（会连带教「学官方措辞」），
    # dpo-d 用模型自己的干净输出（同输入偏好对，只教「别改意思」）。
    {"name": "dpo-d", "beta": 0.3, "lr": 5e-6, "epochs": 2,
     "pairs": "data/pairs/dpo_pairs_own.jsonl"},
]

MERGED = "data/models/Qwen3-1.7B-sft-merged"


def dump_samples(run: str, out_md: Path, n: int = 12) -> None:
    """把若干条 `指令 → 输出 → 官方原答` 摊成 markdown（不调 API）。"""
    import random

    f = ROOT / "data" / "runs" / run / "arms.jsonl"
    if not f.is_file():
        return
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    pick = random.Random(0).sample(rows, min(n, len(rows)))
    lines = [f"# {run} 随机 {len(pick)} 条
"]
    for r in pick:
        lines.append(f"### {r['turn_id']}")
        lines.append("```")
        lines.append("指令: " + " | ".join(x.strip() for x in r["draft"].splitlines()))
        lines.append("我们: " + r["polish"])
        lines.append("真货: " + r["original"])
        lines.append("```")
    out_md.write_text("
".join(lines), encoding="utf-8")
    print(f"  → 样本 {out_md}")
    for r in pick[:3]:
        print(f"     {r['polish'][:60]}")


def seq_rate(adapter_dir: str) -> float | None:
    """从 tag_compliance 写出的 json 里读序列遵从率。"""
    f = ROOT / "data" / "sweep" / f"tagcompliance_{Path(adapter_dir).name}g210.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text(encoding="utf-8")).get("exact_rate")


def run(cmd: list[str], log: Path) -> int:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1",
               PYTHONIOENCODING="utf-8")
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/dpo_pairs_k3.jsonl")
    ap.add_argument("--init", default="data/adapters/run8")
    ap.add_argument("--eval-pairs", default="data/pairs/split_holdout210.jsonl")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--only", default=None, help="只跑某些配置名，逗号分隔")
    args = ap.parse_args()

    py = sys.executable
    for cfg in CONFIGS:
        if args.only and cfg["name"] not in [x.strip() for x in args.only.split(",")]:
            continue
        out = f"data/adapters/{cfg['name']}"
        log = ROOT / "data" / f"{cfg['name']}.log"
        if not args.skip_train:
            print(f"\n=== 训练 {cfg['name']} {cfg} ===")
            pairs = cfg.get("pairs", args.pairs)
            print(f"  负例: {pairs}")
            rc = run([py, "tools/train_dpo.py", "--init", args.init,
                      "--pref-file", pairs, "--out", out,
                      "--merged", MERGED,
                      "--beta", str(cfg["beta"]), "--lr", str(cfg["lr"]),
                      "--epochs", str(cfg["epochs"])], log)
            print(f"  训练返回码 {rc}")
            if rc:
                print(f"  跳过评测（见 {log}）")
                continue
        print(f"=== 评测 {cfg['name']} ===")
        rc = run([py, "tools/pick_checkpoint.py", "--adapter-dir", out,
                  "--base", MERGED, "--pairs", args.eval_pairs,
                  "--final-only", "--temperature", "0", "--tag", "g210",
                  "--workers", "2"], log)
        # 不要用 --reuse：它会复用同名 runs 目录里的旧 arms.jsonl。
        # 事故教训——换了基座重训，评测却读到上一版基座的生成结果，
        # 数字一模一样（连 142/206 都没变），看起来像「模型没变」。
        print(f"  评测返回码 {rc}")
        # 标签序列遵从率必须单独查：`contract.check()` 只看格式合不合法，
        # 不看标签对不对。事故教训——四组 DPO 的「契约 100%」全过，
        # 而标签遵从率从 99% 崩到 69%，闸门没拦住。
        rc2 = run([py, "tools/tag_compliance.py",
                   "--run", f"pickg210_{Path(out).name}_final",
                   "--pairs", args.eval_pairs,
                   "--json-out", str(ROOT / "data" / "sweep"
                                     / f"tagcompliance_{Path(out).name}g210.json")], log)
        print(f"  标签遵从检查返回码 {rc2}")
        # 强制摊输出：不花 API，直接把 12 条样本写进日志和文件。
        # 事故教训——只报数字不摊输出，「标签全换掉」这种问题在均值里看不出来。
        dump_samples(f"pickg210_{Path(out).name}_final", log.parent / f"{cfg['name']}_samples.md")

    # 汇总：基线 + 三个 DPO
    base = ROOT / "data" / "sweep" / "run8_jointg210.json"
    rows = []
    if base.is_file():
        b = json.loads(base.read_text(encoding="utf-8"))["rows"][0]
        rows.append({"name": "baseline(run8)", "win": b["win"], "sem": b["sem"],
                     "ci": b["win_ci"], "contract": b.get("contract")})
    for cfg in CONFIGS:
        f = ROOT / "data" / "sweep" / f"{cfg['name']}_jointg210.json"
        if f.is_file():
            r = json.loads(f.read_text(encoding="utf-8"))["rows"][0]
            rows.append({"name": cfg["name"], "win": r["win"], "sem": r["sem"],
                         "ci": r["win_ci"], "contract": r.get("contract"),
                         "sem_low": r.get("sem_low"), "seq": seq_rate(out)})

    print("\n=== 阶段 A 汇总（206 条留出，greedy）===")
    print(f"{'name':>15s} {'win':>6s} {'CI':>14s} {'sem':>5s} {'<=2':>4s} {'标签序列':>8s}")
    for r in rows:
        print(f"{r['name']:>15s} {str(r['win']):>6s} {str(r.get('ci')):>14s} "
              f"{str(r['sem']):>5s} {str(r.get('sem_low','-')):>4s} "
              f"{str(r.get('seq','-')):>8s}")

    out = ROOT / "data" / "sweep" / "dpo_phase_summary.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {out}")

    # 按计划 A.3 判定
    if base.is_file() and len(rows) > 1:
        b = rows[0]
        print("\n=== 按 docs/PLAN_NEXT.md A.3 判定 ===")
        for r in rows[1:]:
            checks = [
                (f"sem {r['sem']} >= {round(b['sem'] + 0.05, 2)}", r["sem"] >= b["sem"] + 0.05),
                (f"win {r['win']} >= 0.40", r["win"] >= 0.40),
                (f"contract {r.get('contract')}", (r.get("contract") or 0) >= 0.99),
                (f"标签序列遵从 {r.get('seq')}", (r.get("seq") or 0) >= 0.99),
            ]
            ok = all(c[1] for c in checks)
            print(f"  {r['name']}: {'通过 ✅' if ok else '未通过'}")
            for label, passed in checks:
                print(f"      {'✓' if passed else '✗'} {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())