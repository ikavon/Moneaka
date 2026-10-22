"""平行语料体检 + 训练集过滤。

全量 226 对跑完后发现的几类问题，都在这里统一处理：

- **退化草稿**：原稿只有 `mua~` 时草稿可能是空壳（实测出现过草稿就是 `[player]` 一条），
  这种样本会让润色器学到「短草稿可以随便生成」。
- **极短草稿**（< 8 汉字）：不是错误，但正是阶段 0 里编造率最高的一类，
  训练时单独标记、可配置是否保留。
- **模型自行补出的 `[player]`**：语义无害（把「你」换成占位符），但会让润色器
  以为「草稿没提玩家时也该加」，可选剔除。

用法：
    python tools/audit_pairs.py                      # 只体检
    python tools/audit_pairs.py --write-clean        # 另存 pairs.clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from moneaka import contract

DEFAULT = Path(__file__).resolve().parents[1] / "data" / "pairs" / "pairs.jsonl"

#: 低于这个汉字数视为退化草稿（不含占位符）
MIN_DRAFT_CJK = 4
#: 低于这个数算「极短」，单独统计
SHORT_DRAFT_CJK = 8


def cjk(text: str) -> int:
    return contract.cjk_len(text)


def classify(row: dict) -> list[str]:
    """返回该样本命中的问题标签（空列表 = 干净）。"""
    flags: list[str] = []
    draft = row.get("draft") or ""
    original = row.get("original") or ""
    n = cjk(draft)
    if n < MIN_DRAFT_CJK:
        flags.append("degenerate")          # 空壳草稿
    elif n < SHORT_DRAFT_CJK:
        flags.append("short")               # 极短，编造高发
    if draft.strip() == original.strip():
        flags.append("identical")           # 没做去风格化
    if contract.TAG_RE.findall(draft):
        flags.append("tag_left")            # 残留方括号内容（非 player）
    if row.get("draft_player_synth"):
        flags.append("player_synth")        # 占位符是模型自己补的
    if not contract.check(original).ok:
        flags.append("bad_target")          # 训练目标本身不合契约
    return [f for f in flags if f != "tag_left" or [t for t in contract.TAG_RE.findall(draft) if t.lower() != "player"]]


def audit(rows: list[dict]) -> dict:
    from collections import Counter
    flags = Counter()
    for r in rows:
        for f in classify(r):
            flags[f] += 1
    ratios = [cjk(r["draft"]) / max(cjk(r["original"]), 1) for r in rows]
    dl = sorted(cjk(r["draft"]) for r in rows)
    return {
        "n": len(rows),
        "draft_cjk": {"min": dl[0], "median": dl[len(dl) // 2], "p90": dl[int(len(dl) * 0.9)], "max": dl[-1]},
        "draft_over_original": {"median": round(statistics.median(ratios), 2),
                                "mean": round(statistics.mean(ratios), 2)},
        "flags": dict(flags),
    }


def clean(rows: list[dict], *, keep_short: bool = False, drop_synth_player: bool = False) -> list[dict]:
    out = []
    for r in rows:
        f = classify(r)
        if "degenerate" in f or "identical" in f or "bad_target" in f or "tag_left" in f:
            continue
        if "short" in f and not keep_short:
            continue
        if drop_synth_player and "player_synth" in f:
            continue
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DEFAULT))
    ap.add_argument("--write-clean", action="store_true")
    ap.add_argument("--keep-short", action="store_true", help="保留极短草稿（默认剔除）")
    ap.add_argument("--drop-synth-player", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    rep = audit(rows)
    print(json.dumps(rep, ensure_ascii=False, indent=1))

    if args.write_clean:
        kept = clean(rows, keep_short=args.keep_short, drop_synth_player=args.drop_synth_player)
        out = Path(args.pairs).with_name("pairs.clean.jsonl")
        with open(out, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n写入 {out}（{len(kept)}/{len(rows)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())