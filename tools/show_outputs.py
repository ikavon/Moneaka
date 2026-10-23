"""看输出：把评测结果按语义分档排出来，人眼直接读。

自动评委只说「均值 3.70」，看不出它在**哪**失手。这个脚本把每条摊开：
表达指令 → 模型台词 → 官方原答，附语义分与标签序列对比。

语义分按 turn_id 缓存在 `runs/<run>/sem.jsonl`，同一 run 看第二次不再花 API。

    python tools/show_outputs.py --run pickg210_dpo-b_final --n 25
    python tools/show_outputs.py --run pickg210_dpo-b_final --level bad   # 只看失手
    python tools/show_outputs.py --run pickg210_dpo-b_final --out 报告.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import contract, evaluate  # noqa: E402
from moneaka.destyle_tags import parse_tagged_plan  # noqa: E402
from moneaka.llm import LLM  # noqa: E402

#: 五类漂移的肉眼特征，用来在报告里打标（纯提示，不是判定）
HINTS = [
    ("数字", ("百分之", "万分之", "一个人", "两次", "三", "四", "五")),
    ("人称", ("你", "我", "他", "她")),
]


def tags_of(text: str) -> list[str]:
    return [m.group(1) for m in contract.TAG_RE.finditer(contract.PLACEHOLDER_RE.sub("\x00", text))]


def load_sem(run_dir: Path, rows: list[dict]) -> dict[str, int]:
    cache = run_dir / "sem.jsonl"
    sem: dict[str, int] = {}
    if cache.is_file():
        sem = {r["turn_id"]: r["sem"] for r in map(json.loads, open(cache, encoding="utf-8"))}
    todo = [r for r in rows if r["turn_id"] not in sem]
    if todo:
        print(f"语义评分 {len(todo)} 条…", file=sys.stderr)
        probe = [{"turn_id": r["turn_id"], "draft": r["draft"], "polish": r["polish"]}
                 for r in todo]
        sem.update(evaluate.judge_semantic_rows(LLM(), probe, "polish", batch_size=6))
        with open(cache, "w", encoding="utf-8") as fh:
            for k, v in sem.items():
                fh.write(json.dumps({"turn_id": k, "sem": v}, ensure_ascii=False) + "\n")
    return sem


def render(rows: list[dict], sem: dict[str, int]) -> list[str]:
    out: list[str] = []
    for r in rows:
        plan = r["draft"]
        want, contents = parse_tagged_plan(plan)
        got = tags_of(r["polish"])
        seq_ok = want == got
        v = sem.get(r["turn_id"])
        out.append(f"### {r['turn_id']}  语义 {v}/4")
        out.append(f"**语气** {' > '.join(want)}" + ("" if seq_ok else f"　⚠️ 实际 {' > '.join(got)}"))
        for i, c in enumerate(contents, 1):
            out.append(f"> {i}. {c}")
        out.append("")
        out.append(f"- 🟢 我们: {r['polish']}")
        out.append(f"- ⚪ 真货: {r['original']}")
        if not r["contract_ok"]:
            out.append(f"- ❌ 契约: {r['violations']}")
        out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pickg210_dpo-b_final")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--level", default="all", choices=["all", "good", "mid", "bad"])
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0, help="打乱顺序，避免总看开头几条")
    args = ap.parse_args()

    run_dir = ROOT / "data" / "runs" / args.run
    rows = [json.loads(l) for l in open(run_dir / "arms.jsonl", encoding="utf-8")]
    sem = load_sem(run_dir, rows)

    band = {"good": (4, 4), "mid": (3, 3), "bad": (0, 2)}.get(args.level)
    if band:
        rows = [r for r in rows if band[0] <= sem.get(r["turn_id"], -1) <= band[1]]
    if args.seed:
        import random
        random.Random(args.seed).shuffle(rows)
    pick = rows[args.offset: args.offset + args.n]

    lines = [f"# {args.run}　{args.level}　{len(pick)}/{len(rows)} 条", ""] + render(pick, sem)
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"写入 {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())