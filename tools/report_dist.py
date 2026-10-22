"""把一次评测的分布和原始记录摊平，便于人看。

用法：
    python tools/report_dist.py z17t0
    python tools/report_dist.py z17t0 --out docs/RUN_NOTES.md
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from moneaka import contract

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("reference", "direct", "polish")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def hist(values: list[int], maxv: int = 4) -> str:
    c = collections.Counter(v for v in values if v is not None)
    return "  ".join(f"{i}:{c.get(i, 0):>3}" for i in range(maxv + 1))


def bar(values: list[float], lo: float, hi: float, buckets: int = 10) -> str:
    if not values:
        return "(无)"
    step = (hi - lo) / buckets
    c = collections.Counter(min(int((v - lo) / step), buckets - 1) for v in values)
    return "\n".join(
        f"    {lo + i * step:5.1f}-{lo + (i + 1) * step:5.1f} | {'█' * c.get(i, 0)} {c.get(i, 0)}"
        for i in range(buckets) if c.get(i)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pairs", default=str(ROOT / "data" / "pairs" / "pairs60.jsonl"))
    args = ap.parse_args()

    run = ROOT / "data" / "runs" / args.run
    scored = load_jsonl(run / "scored.jsonl")
    arms = {r["turn_id"]: r for r in load_jsonl(run / "arms.jsonl")}
    L: list[str] = []
    add = L.append

    add(f"# 评测原始记录：`{args.run}`\n")
    add("## 文件\n")
    for f in sorted(run.glob("*")):
        add(f"- `data/runs/{args.run}/{f.name}` — {f.stat().st_size / 1024:.0f} KiB")
    add("")

    # ---- 分布 ----
    add("## 分数分布（评委抽样 30 条）\n")
    add("| arm | semantic 0..4 | style 0..4 |")
    add("|---|---|---|")
    for arm in ARMS:
        rows = [s for s in scored if s["arm"] == arm]
        add(f"| {arm} | {hist([s['semantic'] for s in rows])} | {hist([s['style'] for s in rows])} |")
    add("")
    for arm in ARMS:
        rows = [s for s in scored if s["arm"] == arm and s["style"] is not None]
        if not rows:
            continue
        st = [s["style"] for s in rows]
        sm = [s["semantic"] for s in rows if s["semantic"] is not None]
        add(f"**{arm}** style 均值 {statistics.mean(st):.2f} / 中位 {statistics.median(st)} / "
            f"标准差 {statistics.pstdev(st):.2f}；semantic 均值 {statistics.mean(sm):.2f}")
        if len(st) > 1:
            add(f"  style 与 semantic 的相关性: {_corr(st, sm):+.2f}")
        add("")

    # ---- 契约 ----
    add("## 契约校验（全量，不经评委）\n")
    add("| arm | 通过率 | 违规码统计 | 标签数分布 | 平均字数 |")
    add("|---|---|---|---|---|")
    for arm in ARMS:
        rows = [s for s in scored if s["arm"] == arm]
        n = len(rows)
        ok = sum(1 for s in rows if s["contract_ok"])
        codes = collections.Counter(v.split(":", 1)[0] for s in rows for v in s["violations"])
        ntags = collections.Counter(contract.check(s["text"]).n_tags for s in rows)
        lens = [contract.cjk_len(s["text"]) for s in rows if s["text"]]
        add(f"| {arm} | {ok}/{n} = {ok / n:.0%} | {dict(codes) or '—'} | "
            f"{dict(sorted(ntags.items()))} | {statistics.mean(lens):.0f} |")
    add("")

    # ---- 归一化统计 ----
    norm_rows = [r for r in arms.values() if r.get("norm")]
    if norm_rows:
        mapped, unmapped, ins, split = collections.Counter(), collections.Counter(), 0, 0
        for r in norm_rows:
            for d in r["norm"]:
                for _, st in d.items():
                    for k, v in st["tags_mapped"].items():
                        mapped[f"{k}→{v}"] += 1
                    unmapped.update(st["tags_unmapped"])
                    ins += st["tags_inserted"]
                    split += st["sentences_split"]
        add("## 归一化层做了什么\n")
        add(f"- 补插标签 {ins} 次 · 拆分长句 {split} 次")
        add(f"- 标签改写: {dict(mapped) if mapped else '无'}")
        add(f"- 表外无别名（落了默认值）: {dict(unmapped) if unmapped else '无'}")
        add("")

    # ---- 逐条表 ----
    add("## 逐条明细（60 条 × 3 臂；分数仅抽样 30 条有值）\n")
    add("| turn | style | sem | 契约 | polish 字数 |")
    add("|---|---|---|---|---|")
    by_turn: dict[str, dict] = collections.defaultdict(dict)
    for s in scored:
        by_turn[s["turn_id"]][s["arm"]] = s
    for tid in sorted(by_turn, key=lambda t: tuple(int(x) for x in t.split("-"))):
        p = by_turn[tid].get("polish")
        if not p:
            continue
        add(f"| {tid} | {_dash(p['style'])} | {_dash(p['semantic'])} | "
            f"{'✅' if p['contract_ok'] else '❌'} | {contract.cjk_len(p['text'])} |")
    add("")

    # ---- 文本量分布 ----
    add("## polish 输出字符数分布\n")
    add(bar([contract.cjk_len(by_turn[t]["polish"]["text"]) for t in by_turn if by_turn[t].get("polish")], 0, 120))
    add("")

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"写入 {args.out}")
    else:
        print(text)
    return 0


def _dash(v) -> str:
    return "·" if v is None else str(v)


def _corr(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


if __name__ == "__main__":
    raise SystemExit(main())