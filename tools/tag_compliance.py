"""标签序列遵从率：受控 schema 下润色器的**核心硬指标**。

为什么需要它：语气行从「自由描述」改成「官方标签序列」之后，「该选哪个标签」这个
判断已经**由编剧完成**，不再是模型要学的东西。于是旧的「标签吻合率」会虚高到接近
100%（模型在抄输入），失去意义。真正要测的是：

* 序列是否**完全照做**（顺序、个数、每个标签）
* 做不到时错在哪（换标签 / 并段 / 漏段 / 多段）

用法::

    python tools/tag_compliance.py --run s1_run6 --pairs data/pairs/tagged_official.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import contract  # noqa: E402
from moneaka.destyle_tags import SegmentError, parse_tagged_plan  # noqa: E402


def out_tags(text: str) -> list[str]:
    """从输出里按出现次序取标签。先屏蔽 [player]，否则会被当成标签。"""
    masked = contract.PLACEHOLDER_RE.sub("\x00", text or "")
    return [m.group(1) for m in contract.TAG_RE.finditer(masked)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--json-out", default=None, help="把结果写一份 json，供流水线闸门读取")
    args = ap.parse_args()

    rows = {r["turn_id"]: r for r in map(json.loads, open(
        ROOT / "data" / "runs" / args.run / "arms.jsonl", encoding="utf-8"))}
    src = {r["turn_id"]: r for r in map(json.loads, open(args.pairs, encoding="utf-8"))}

    stats = Counter()
    wrong_pos = 0
    total_pos = 0
    samples = []
    for tid, row in rows.items():
        p = src.get(tid)
        if not p:
            stats["no_pair"] += 1
            continue
        try:
            want, _ = parse_tagged_plan(p.get("plan") or p.get("draft") or "")
        except SegmentError:
            stats["input_not_tagged"] += 1
            continue
        got = out_tags(row.get("polish") or "")
        stats["n"] += 1
        total_pos += len(want)
        if got == want:
            stats["exact"] += 1
            continue
        if len(got) == len(want):
            stats["same_len_wrong_tags"] += 1
            wrong_pos += sum(1 for a, b in zip(want, got) if a != b)
        elif len(got) < len(want):
            stats["fewer_segments"] += 1
        else:
            stats["more_segments"] += 1
        if samples.__len__() < 6:
            samples.append((tid, want, got))

    n = stats["n"]
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps({
            "run": args.run, "n": n,
            "exact": stats["exact"], "exact_rate": round(stats["exact"] / n, 4),
            "wrong_tags": stats["same_len_wrong_tags"], "fewer": stats["fewer_segments"],
            "more": stats["more_segments"],
            "pos_acc": round((total_pos - wrong_pos) / total_pos, 4) if total_pos else None,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"run={args.run}  有效样本 {n}")
    if not n:
        print("没有可比对的样本（输入不是受控 schema？）")
        return 1
    print(f"  序列完全一致 {stats['exact']}/{n} = {stats['exact'] / n:.1%}")
    print(f"  个数对但标签错 {stats['same_len_wrong_tags']}/{n} = "
          f"{stats['same_len_wrong_tags'] / n:.1%}")
    print(f"  段数偏少（并段/漏段） {stats['fewer_segments']}/{n} = "
          f"{stats['fewer_segments'] / n:.1%}")
    print(f"  段数偏多 {stats['more_segments']}/{n} = {stats['more_segments'] / n:.1%}")
    if total_pos:
        print(f"  逐位标签准确 {total_pos - wrong_pos}/{total_pos} = "
              f"{(total_pos - wrong_pos) / total_pos:.1%}（仅在同段数样本上统计）")
    for tid, want, got in samples:
        print(f"    {tid}  期望 {' > '.join(want)}    得到 {' > '.join(got) or '(无标签)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())