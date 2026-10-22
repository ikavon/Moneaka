"""第0步：真实数据的「标签 vs 句子」形态统计（只读，不调 API 不动 GPU）。

要回答的唯一问题：`语气: 标签A > 标签B > 标签C` 这种「标签序列 ↔ 逐句标签」
的受控 schema，在官方真实输出里**成立吗**？不成立的话偏差长什么样。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import contract  # noqa: E402
from moneaka.corpus import chosen_turns  # noqa: E402

TAG = contract.TAG_RE
PH = contract.PLACEHOLDER_RE


def mask(t: str) -> str:
    return PH.sub("\x00", t)


def analyse(text: str) -> dict:
    m = mask(text)
    # 每个标签在原文中的位置、前面是否就是句子起点
    tags = [x.group(1) for x in TAG.finditer(m)]
    # 一个「标签单元」= 标签开头到下一个标签开头
    units = contract.split_sentences(text)
    lead = [re.match(r"^\s*\[([^\[\]\n]{1,8})\]", mask(u)) for u in units]
    lead_tags = [x.group(1) if x else None for x in lead]
    # 句中标签（不在句首的）
    mid = 0
    for u in units:
        mm = mask(u)
        lead_one = re.match(r"^\s*\[[^\[\]\n]{1,8}\]", mm)
        body = mm[lead_one.end():] if lead_one else mm
        mid += len(TAG.findall(body))
    # 连续重复标签（[笑][笑]）
    dup = len(re.findall(r"\]\s*\[", m))
    bodies = [contract.sentence_body(u) for u in units]
    return {
        "n_tags": len(tags),
        "n_units": len(units),
        "units_no_leading_tag": sum(1 for t in lead_tags if not t),
        "mid_tags": mid,
        "adjacent_tag_pairs": dup,
        "multi_sentence_units": sum(1 for b in bodies if len(contract.sentence_parts(b)) > 1),
        "tags": tags,
        "unit_lens": [contract.cjk_len(b) for b in bodies],
        "raw": text,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="语料 jsonl（RLHF/sharegpt），或 pairs jsonl（读 original 字段）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.source)
    if src.name.startswith(("plan", "pairs")):
        # 原始 RLHF 分片已不在仓库里；pairs 里的 `original` 就是那 226 轮官方真实输出
        texts = [r["original"] for r in map(json.loads, open(src, encoding="utf-8"))
                 if r.get("original")]
    else:
        texts = [t.chosen for t in chosen_turns(path=args.source)]
    if args.limit:
        texts = texts[: args.limit]
    rows = texts
    A = [analyse(x) for x in texts]
    n = len(A)
    print(f"语料: {args.source}\n轮数: {n}\n")

    # ---- 1. 标签数 vs 单元数 ----
    eq = sum(1 for a in A if a["n_tags"] == a["n_units"])
    print("【1】标签数 == 标签单元数（严格 1:1）")
    print(f"    {eq}/{n} = {eq/n:.1%}")
    diff = collections.Counter(a["n_tags"] - a["n_units"] for a in A)
    print(f"    差值分布(标签数-单元数): {dict(sorted(diff.items()))}")

    # ---- 2. 有没有单元不以标签开头 ----
    no_lead = sum(1 for a in A if a["units_no_leading_tag"])
    raw_no_lead = sum(a["units_no_leading_tag"] for a in A)
    print("\n【2】存在「单元不以标签开头」的轮次")
    print(f"    {no_lead}/{n} = {no_lead/n:.1%}（涉及 {raw_no_lead} 个单元）")
    # 这些单元长什么样
    samples = []
    for t, a in zip(rows, A):
        for u in contract.split_sentences(t):
            if not re.match(r"^\s*\[", mask(u)):
                samples.append(u)
    for s in samples[: args.examples]:
        print(f"      · {s[:70]}")

    # ---- 3. 句中标签（标签不在句首）----
    mid = sum(1 for a in A if a["mid_tags"])
    raw_mid = sum(a["mid_tags"] for a in A)
    print("\n【3】存在「句中标签」的轮次（标签不在句首）")
    print(f"    {mid}/{n} = {mid/n:.1%}（涉及 {raw_mid} 个标签）")

    # ---- 4. 相邻标签 [笑][笑] ----
    dup = sum(1 for a in A if a["adjacent_tag_pairs"])
    print("\n【4】存在相邻标签（[A][B] 中间无正文）的轮次")
    print(f"    {dup}/{n} = {dup/n:.1%}")

    # ---- 5. 一个标签单元里套几句 ----
    multi = sum(1 for a in A if a["multi_sentence_units"])
    allu = [u for a in A for u in range(a["multi_sentence_units"])]
    print("\n【5】存在「一个标签套多句」的轮次")
    print(f"    {multi}/{n} = {multi/n:.1%}")

    # ---- 6. 每个标签平均覆盖几个单元 / 频次 ----
    print("\n【6】每轮标签序列（前 12 条示例）")
    for t, a in list(zip(rows, A))[:12]:
        print(f"    {a['n_units']}单元/ {a['n_tags']}标签  {' > '.join(a['tags'])}")

    # ---- 7. 单元正文长度 ----
    L = [x for a in A for x in a["unit_lens"]]
    L.sort()
    print(f"\n【7】单元正文字数 n={len(L)} 中位={statistics.median(L):.0f} "
          f"p90={L[int(.9*len(L))]} p99={L[int(.99*len(L))]} max={max(L)}")
    print(f"    超过 60 字的单元: {sum(1 for x in L if x > 60)}/{len(L)} = "
          f"{sum(1 for x in L if x > 60)/len(L):.1%}")

    # ---- 8. 结论判定 ----
    ok = (eq / n) > 0.95 and no_lead / n < 0.02 and mid / n < 0.02
    print("\n" + "=" * 60)
    print(f"严格 1:1 schema {'成立 ✅' if ok else '不成立 ❌'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
