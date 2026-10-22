"""不花一次 API，把 pairs 里错的 `context`/`user` 重新导出。

背景：`corpus._context_before` 有个边界 bug（已修），导致每个样本的历史里
**包含了自己要生成的那一句回答**，而且 `user` 取到的是下一句提问。
只有 `context`/`user` 两列受影响——`draft`(表达指令) 和 `original`(目标)
都是从 `chosen` 直接来的，没有泄漏，所以**不需要重跑 destyle**，
按 turn_id 从原始官方 jsonl 重新切一遍就行。

    python tools/rederive_context.py            # 只检查
    python tools/rederive_context.py --write    # 就地修（先备份 .leaky.bak）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import corpus  # noqa: E402
from moneaka.contract import TAG_RE  # noqa: E402


def _cjk_only(s: str) -> str:
    return "".join(ch for ch in s if "\u4e00" <= ch <= "\u9fff")


def has_leak(original: str, old_ctx_json: str, min_len: int = 8) -> bool:
    """旧上下文里是否出现了目标句的连续正文（判定「答案在输入里」）。

    两个归一化缺一不可：
    1. 先 `TAG_RE.sub` —— 存进 pairs 的历史是**带 `[情绪标签]`** 的原文，
       而目标要取正文；不去标签，比对的开头就是标签文字，永远匹配不上。
    2. 再只留汉字 —— 原文里夹着半角标点与空格，跨标点就不是子串了。
    （这两步我各错过一次，两次都报告「0 条泄漏」，差点当没事发生。）
    """
    body = _cjk_only(TAG_RE.sub("", original))[:min_len]
    norm = _cjk_only(TAG_RE.sub("", old_ctx_json))
    return bool(body) and body in norm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/plan.jsonl")
    ap.add_argument("--source", default=None, help="官方 jsonl（默认取 corpus 里配好的路径）")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    turns = {t.turn_id: t for t in corpus.chosen_turns(path=args.source)}
    rows = [json.loads(l) for l in open(ROOT / args.pairs, encoding="utf-8")]

    changed = leak = user_wrong = 0
    out = []
    for r in rows:
        t = turns.get(r["turn_id"])
        if t is None:
            print(f"  ! 找不到 {r['turn_id']}，原样保留")
            out.append(r)
            continue
        old_ctx = json.dumps(r.get("context") or [], ensure_ascii=False)
        new_ctx = json.dumps(t.context, ensure_ascii=False)
        if old_ctx != new_ctx:
            changed += 1
        if has_leak(r["original"], old_ctx):
            leak += 1
        if r.get("user", "") != t.user:
            user_wrong += 1
        out.append(dict(r, context=t.context, user=t.user))

    print(f"{len(rows)} 条：context/user 有变动 {changed} 条")
    print(f"  旧上下文含目标正文（答案泄漏）: {leak} 条")
    print(f"  旧 user 取错句的: {user_wrong} 条")
    if args.write:
        src = ROOT / args.pairs
        shutil.copy2(src, src.with_suffix(".leaky.bak"))
        with open(src, "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"已写入 {src}（旧版备份 {src.name}.leaky.bak）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
