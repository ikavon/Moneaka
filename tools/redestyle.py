"""数据侧改造：用「最小改动」规则重造表达指令，并过机械闸门。

## 为什么

旧版 `DESTYLE_TAG_SYSTEM` 第 4 条要求「不照抄原句用词, 去原句痕迹」，本意是防止
下游润色器只会复制粘贴。实测代价：

| 语料 | plan 对原句的语义分 | ≤2 分占比 |
|---|---|---|
| ds（925 条） | 3.44 | 5.1% |
| official（226 条） | 3.59 | 4.9% |

`爱你哦` 被写成「我感谢你；我向[player]问好」——**意思已经变了**，而这类损失
在评测时会被记成「润色器的语义漂移」，于是我们一直在给数据的问题调模型。

新版规则：只删风格外壳，**禁止换同义词、禁止动否定/数量/人称**。

## 机械闸门（不花 API，先筛掉明显错的）

* 数字集合必须一致（`百分之一` vs `万分之一` 这类直接拦掉）
* 否定词个数不得减少（不/没/别/无/未/莫）
* 占位符个数一致
* 汉字数不得缩水到原段的 55% 以下（防「合并/省略」）

闸门不过就重试一次，再不过就记下来人工看，**不静默丢弃**。

    python tools/redestyle.py --src data/pairs/tagged_official.jsonl --out data/pairs/tagged_official_v2.jsonl
    python tools/redestyle.py --src data/pairs/tagged_ds.jsonl --out data/pairs/tagged_ds_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import contract, prompts  # noqa: E402
from moneaka.destyle_tags import (  # noqa: E402
    _SEG_LINE, MAX_SEGMENTS, SegmentError, neutralize, render_plan, split_tagged,
)
from moneaka.llm import LLM  # noqa: E402

_SENTINEL = "《player》"
_SENTINEL_BACK = re.compile(r"《\s*player\s*》", re.I)
_BARE_PLAYER = re.compile(r"(?<!\[)\b(?:[Pp]layer|玩家)\b(?!\])")
_NEG = re.compile(r"[不没别无未莫]")
# 只认真实数量：阿拉伯数字，以及「百分之三十」这种明确的分量表达。
# 早先版本把中文数词全抓了，结果「一起」「一样」「一些」里的「一」被当成数量，
# 制造了一堆假失败（实测 17 条官方失败里 4 条是这个假阳性）。
_NUM = re.compile(r"\d+|[零一二两三四五六七八九十百千万亿]+分之[零一二三四五六七八九十百千]+")
_DIGITS = re.compile(r"\d")


def numbers(text: str) -> list[str]:
    return sorted(_NUM.findall(text))


def gate(orig: str, content: str) -> list[str]:
    """机械校验：不过就返回问题列表（空 = 通过）。"""
    bad: list[str] = []
    if numbers(orig) != numbers(content):
        bad.append(f"数字 {numbers(orig)} → {numbers(content)}")
    if len(_NEG.findall(orig)) > len(_NEG.findall(content)):
        bad.append("否定词变少")
    if orig.count("[player]") != content.count("[player]"):
        bad.append(f"占位符 {orig.count('[player]')} → {content.count('[player]')}")
    if contract.cjk_len(content) < 0.55 * max(1, contract.cjk_len(orig)):
        bad.append("字数缩水")
    return bad


def make_one(llm: LLM, original: str) -> dict:
    tags, bodies = split_tagged(original)
    n = len(tags)
    if n > MAX_SEGMENTS:
        raise SegmentError(f"段数 {n} 超上限")
    listing = "\n".join(f"{i}. [{t}] {b}" for i, (t, b) in enumerate(zip(tags, bodies), 1))
    user = prompts.DESTYLE_TAG_USER.format(n=n, segments=listing)

    last: list[str] = []
    for _ in range(2):
        raw = llm.complete("destyle", prompts.DESTYLE_TAG_SYSTEM_V2.format(n=n), user)
        contents: list[str] = []
        for line in raw.splitlines():
            m = _SEG_LINE.match(line)
            if m:
                contents.append(m.group(2).strip().strip("。").strip())
        if len(contents) != n:
            last = [f"段数 {len(contents)} != {n}"]
            continue
        contents = [_SENTINEL_BACK.sub("[player]", c) for c in contents]
        for i, c in enumerate(contents):
            if _SENTINEL in bodies[i] and "[player]" not in c.lower():
                c = _BARE_PLAYER.sub("[player]", c)
            contents[i] = neutralize(c)
        problems: list[str] = []
        for i, (b, c) in enumerate(zip(bodies, contents)):
            if not contract._CJK_RE.search(c):
                problems.append(f"第{i+1}段无中文")
                continue
            if contract.cjk_len(c) > contract.SENTENCE_MAX_CHARS:
                problems.append(f"第{i+1}段超长")
            # bodies 里的占位符是哨兵 `《player》`，比对前先还原，否则会误判丢占位符
            problems += [f"第{i+1}段 {p}"
                         for p in gate(b.replace(_SENTINEL, "[player]"), c)]
        if problems:
            last = problems
            continue
        return {"tags": tags, "contents": contents, "plan": render_plan(tags, contents),
                "original": original, "n_tags": n, "gate": "ok"}
    raise SegmentError("；".join(last) or "未知失败")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fail-out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = [json.loads(l) for l in open(ROOT / args.src, encoding="utf-8") if l.strip()]
    if args.limit:
        src = src[: args.limit]
    out, fail = Path(ROOT / args.out), Path(ROOT / (args.fail_out or args.out + ".fail"))
    done: dict[str, dict] = {}
    if out.is_file():
        done = {r["turn_id"]: r for r in map(json.loads, open(out, encoding="utf-8"))}
        print(f"续跑：已有 {len(done)}")
    todo = [r for r in src if r["turn_id"] not in done]
    print(f"待处理 {len(todo)} / {len(src)}")

    llm = LLM()
    fails: list[dict] = []

    def one(r: dict) -> dict | None:
        try:
            d = make_one(llm, r["original"])
        except Exception as exc:  # noqa: BLE001
            fails.append({"turn_id": r["turn_id"], "original": r["original"],
                          "error": str(exc)[:200]})
            return None
        return {**r, **d}

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, rec in enumerate(ex.map(one, todo), 1):
                if rec:
                    done[rec["turn_id"]] = rec
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}  成功 {len(done)}  失败 {len(fails)}")
        with open(out, "w", encoding="utf-8") as fh:
            for r in done.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(fail, "w", encoding="utf-8") as fh:
            for r in fails:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成 {len(done)}/{len(src)}  失败 {len(fails)} → {out}")
    if fails:
        print(f"失败样本写入 {fail}（不要静默丢弃，人工看过再决定）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())