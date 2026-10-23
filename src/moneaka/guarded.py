"""A 层：生成后的机械闸门（纯正则，零成本）。

抽成包内模块是因为它现在有两个调用方：`tools/guarded_polish.py`（单跑评测）
和 `tools/chain_eval.py`（全链路）。两边必须用**同一套**判据，否则
「闸门说通过」和「链路说通过」会打架。

判据只留可机械验证的事实：标签序列、数字、否定词、占位符。
另有一条 3-gram「段落是否落地」，触发率 54% 且多为换语序造成的误报，
默认关闭（`loose=True` 才开）。
"""

from __future__ import annotations

import re

from . import contract
from .destyle_tags import parse_tagged_plan

_DIGITS = re.compile(r"\d+")
_NUMWORD = re.compile(r"[零一二两三四五六七八九十百千万亿]+分之[零一二三四五六七八九十百千]+")
_NEG = re.compile(r"[不没别无未莫]")


def nums(text: str) -> set[str]:
    return set(_DIGITS.findall(text)) | set(_NUMWORD.findall(text))


#: 只留「机械可判且几乎不会误报」的两项。**否定词与占位符不再进闸门**：
#: 实测 36 条退化样本里 29 条是这两项误报（plan 有 [player] 而她说「你」、
#: 反问句里的「不」被改写成别的说法）——那不是错误，是正常措辞。
#: 否定/方向/近义交给 B（强模型复核），它有 88% 查全。
STRICT = ("标签", "数字")

def check(plan: str, out: str, loose: bool = False) -> list[str]:
    """机械闸门。返回问题列表，空 = 通过。"""
    bad: list[str] = []
    try:
        tags, contents = parse_tagged_plan(plan)
    except Exception:  # noqa: BLE001
        return ["plan 不可解析"]
    got = [m.group(1) for m in contract.TAG_RE.finditer(contract.PLACEHOLDER_RE.sub("\x00", out))]
    if got != tags:
        bad.append(f"标签 {tags} → {got}")
    body = contract.PLACEHOLDER_RE.sub("[player]", out)
    body = contract.TAG_RE.sub("", body)
    plan_body = "；".join(contents)

    # 双向比对：少了是丢信息，多了是编数字。只查「少」会漏掉「凭空加一个数」。
    if not nums(plan_body) <= nums(body):
        bad.append(f"数字丢失 {sorted(nums(plan_body) - nums(body))}")
    if not nums(body) <= nums(plan_body):
        # 消息必须以「数字」开头：调用方用 startswith(STRICT) 过滤，前缀不对就漏拦
        bad.append(f"数字凭空多出 {sorted(nums(body) - nums(plan_body))}")
    if len(_NEG.findall(body)) < len(_NEG.findall(plan_body)):
        bad.append("否定词变少")
    if out.count("[player]") < len(re.findall(r"\[player\]", plan)):
        bad.append("占位符丢失")
    if not loose:
        return bad
    # 每段内容至少要有一部分字面落地：用 3-gram 覆盖检查（噪声大，默认关）
    for i, c in enumerate(contents, 1):
        grams = {c[j:j + 3] for j in range(0, max(1, len(c) - 2)) if contract._CJK_RE.search(c[j:j + 3])}
        if grams and not (grams & {body[j:j + 3] for j in range(0, max(1, len(body) - 2))}):
            bad.append(f"第{i}段内容完全没落地")
    return bad


