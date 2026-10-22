"""盘点 `edgeinfinity/MAICA_ds_basis`（HF 镜像已 clone 到 data/MAICA_ds_basis）。

这批文件是 MAICA 的**基座数据集**，与之前那份 `daa4_rlhf1_prcs5`（91 行 RLHF 分片）
不是同一套东西：这里是对话形式的语料（`conversations` 里 from/value），
RLHF 分片是偏好形式（`messages`/`rejected_messages`）。

本脚本回答四个问题，都是决定「能不能拿去训」的前置事实：

1. 每个文件有多少 **assistant 轮**（真正可当训练目标的单位）
2. 其中多少**带 `[情绪标签]`**（没有标签的轮对润色器没价值）
3. 标签词表与官方 `emotions.py` 的 28 个对比——**多出来的词**说明语料里
   存在清洗前后不一致的写法（实测有 `很开心` 这种 emotion.txt 里的原码）
4. 版本文件之间**重叠多少**（`moni_dataset_2605` 大概率是 2603/2511 的超集，
   全加起来会重复计数）
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import collections
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka.contract import EMOTION_TAGS  # noqa: E402

TAG_RE = re.compile(r"\[([^\[\]\n]{1,8})\]")
OFFICIAL = set(EMOTION_TAGS) | {"惊喜", "可爱"}   # emotions.py 的 28 个


def turns_of(row: dict) -> list[dict]:
    conv = row.get("conversations") or []
    out = []
    for i, m in enumerate(conv):
        if m.get("from") != "assistant":
            continue
        ctx = [{"role": "user" if c.get("from") == "user" else
                ("assistant" if c.get("from") == "assistant" else c.get("from", "")),
                "content": c.get("value", "")}
               for c in conv[:i] if c.get("from") != "system"]
        user = next((c["content"] for c in reversed(ctx) if c["role"] == "user"), "")
        out.append({"idx": i, "content": m.get("value", ""), "user": user, "context": ctx})
    return out


def digest(text: str) -> str:
    return re.sub(r"\[[^\[\]\n]{1,8}\]", "", text).strip()


def main() -> int:
    files = sorted(glob.glob(str(ROOT / "data" / "MAICA_ds_basis" / "**" / "*.jsonl"),
                             recursive=True))
    all_tags: list[Counter[str]] = []
    seen_digest: dict[str, str] = {}
    per_file = []
    for f in files:
        rows = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        tags: Counter[str] = Counter()
        n_turns = n_tagged = 0
        for r in rows:
            for t in turns_of(r):
                n_turns += 1
                found = TAG_RE.findall(t["content"])
                if found:
                    n_tagged += 1
                    tags.update(found)
                d = digest(t["content"])
                if d:
                    seen_digest.setdefault(d, os.path.basename(f))
        rel = os.path.relpath(f, ROOT / "data" / "MAICA_ds_basis").replace("\\", "/")
        per_file.append((rel, len(rows), n_turns, n_tagged, len(set(tags))))
        all_tags.append(tags)

    print(f"{'文件':34s} {'行':>6s} {'asst轮':>7s} {'带标签':>7s} {'独特标签':>8s}")
    for rel, nr, nt, ng, nu in per_file:
        print(f"{rel:34s} {nr:6d} {nt:7d} {ng:7d} {nu:8d}")

    merged: Counter[str] = Counter()
    for c in all_tags:
        merged.update(c)
    print(f"\n去重后**唯一正文轮**（跨文件按去标签文本判重）: {len(seen_digest)}")
    print(f"标签总出现次数: {sum(merged.values())}  词表大小: {len(merged)}")
    print(f"官方 emotions.py 有、语料里没有的: {sorted(OFFICIAL - set(merged))}")
    oov = [(t, n) for t, n in merged.most_common() if t not in OFFICIAL]
    print(f"语料里有、官方表里没有的（清洗残留/新造）: {len(oov)} 种 → {oov[:18]}")
    print("\nTop 20 标签:", merged.most_common(20))

    # 版本间重叠
    print("\n=== 版本重叠（按去标签正文判重）===")
    for rel, *_ in per_file:
        pass
    first_seen: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for t in turns_of(row):
                    d = digest(t["content"])
                    if d and d not in first_seen:
                        first_seen[d] = os.path.basename(f)
                        counts[os.path.basename(f)] += 1
    print(f"{'文件':30s} {'首次出现的新轮':>12s}")
    for rel in sorted(counts, key=lambda k: -counts[k]):
        print(f"{rel:30s} {counts[rel]:12d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
