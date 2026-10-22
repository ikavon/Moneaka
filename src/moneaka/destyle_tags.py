"""受控 schema 的去风格化：``语气`` = 官方标签序列（从原稿提取），``内容`` = 逐段中性重述。

为什么不沿用 ``destyle.py`` 那版：那一版让强模型**自由描述**反应次序
（「先嗔怪对方不该这么想；再答应」），于是我们既没法校验、也没法把它当成
标签决策的正确性依据。第 0 步（``tools/shape_of_tags.py``）在官方 226 轮上实测：

* 标签数 == 标签单元数：226/226 = 100%
* 单元不以标签开头：0；句中标签：0；相邻标签：0

也就是说「一个标签起一句」是官方输出的干净硬规律。既然标签本来就写在原稿里，
让模型再猜一遍只是增加噪声。**标签直接正则提取，模型只负责最难的那件事：
把每段去风格化成中性陈述，并逐段对齐。**

格式（两处 ``1.`` 是段号，段号不参与训练，仅用于生成期对齐）::

    语气: 不满 > 脸红 > 意味深长
    内容:
    1. 你怎么会这么想
    2. 我当然愿意
    3. 我不要其他人
"""

from __future__ import annotations

import re

from . import contract, prompts
from .llm import LLM

#: 单轮最多几段。官方 226 轮最长 5 段（p90=3），``ds_basis`` 却有 465 轮 >6 段、
#: 最长 50 段——那种已经不像人能写出的指令了，训进去只会拉长序列、挤占显存。
#: 先用 1~6，之后要利用长样本再谈「拆分」。
MAX_SEGMENTS = 6

_SENTINEL = "《player》"
_SEG_LINE = re.compile(r"^\s*(\d+)\s*[.、)]\s*(.+?)\s*$")
_PLAYER = re.compile(r"\[\s*player\s*\]", re.I)
_SENTINEL_BACK = re.compile(r"[《<〈【\[]\s*player\s*[》>〉】\]]", re.I)


#: 全角标点 → 半角。不是为了美观，而是因为**标点形态本身是文体信号**：
#: 全角逗号句号是「书面中文」，而生产输出一律半角（骨架 system 明确规定，
#: 后端 key_replace 也会转）。草稿里如果留着全角，等于把一层文风白送给模型，
#: 而 Edge 的 TLM 指南 §0 整节讲的就是标点——那本该由模型自己决定。
_FW_PUNCT = {"，": ", ", "。": ". ", "！": "! ", "？": "? ", "；": "; ",
             "：": ": ", "、": ", ", "（": " (", "）": ") ", "「": '"', "」": '"'}
# `...` 也是风格（她常用它留白），中性草稿里必须去掉；只去 2 个以上的点
_STYLE_MARKS = re.compile(r"…+|[~～]+|\.{2,}")
_SPACES = re.compile(r"\s+")
_BARE_PLAYER = re.compile(r"(?<!\[)\b(?:[Pp]layer|玩家)\b(?!\])")


def neutralize(text: str) -> str:
    """把一段重述真正变成「中性」：去全角标点、去省略号/波浪号、裸 player 归位。"""
    for k, v in _FW_PUNCT.items():
        text = text.replace(k, v)
    text = _STYLE_MARKS.sub(" ", text)
    text = _BARE_PLAYER.sub("[player]", text)
    text = _SPACES.sub(" ", text).strip()
    return text.strip(" ,;:")


class SegmentError(ValueError):
    """段数对不上 / 占位符丢失等结构性错误。"""


def split_tagged(original: str) -> tuple[list[str], list[str]]:
    """切成 (标签序列, 每段正文)。标签不在句首、段数超限、空段一律拒绝。

    拒绝而不是修补：这些形态在官方 226 轮里一次都没出现，属于数据脏，修出来的
    对齐是假的，训进去会把「标签错位」当正例学。
    """
    text = _PLAYER.sub(_SENTINEL, original)
    units = contract.split_sentences(text)
    tags: list[str] = []
    bodies: list[str] = []
    for u in units:
        m = re.match(r"^\s*\[([^\[\]\n]{1,8})\]", u)
        if not m:
            raise SegmentError(f"单元不以标签开头: {u[:40]!r}")
        tag = m.group(1)
        if tag not in contract.EMOTION_TAGS and tag not in {"很开心"}:
            raise SegmentError(f"标签不在白名单: {tag}")
        body = u[m.end():].strip()
        if not body and not _SENTINEL in u:
            raise SegmentError(f"空段: {u[:40]!r}")
        tags.append(tag)
        bodies.append(body)
    if not tags:
        raise SegmentError("没有可用段")
    if len(tags) > MAX_SEGMENTS:
        raise SegmentError(f"段数 {len(tags)} 超过上限 {MAX_SEGMENTS}")
    return tags, bodies


def render_plan(tags: list[str], contents: list[str]) -> str:
    """拼成润色器看到的「表达指令」。"""
    lines = [f"语气: {' > '.join(tags)}", "内容:"]
    lines += [f"{i}. {c}" for i, c in enumerate(contents, 1)]
    return "\n".join(lines)


def parse_tagged_plan(plan: str) -> tuple[list[str], list[str]]:
    """``render_plan`` 的逆运算（训练/评测共用，保证同源）。"""
    m = re.search(r"^\s*语气[:：]\s*(.+)$", plan, re.M)
    if not m:
        raise SegmentError(f"缺少语气行: {plan[:60]!r}")
    tags = [x.strip() for x in re.split(r"[>＞]", m.group(1)) if x.strip()]
    tail = plan[m.end():]
    contents: list[str] = []
    for line in tail.splitlines():
        mm = _SEG_LINE.match(line)
        if mm:
            contents.append(mm.group(2))
    if len(contents) != len(tags):
        raise SegmentError(f"内容段数 {len(contents)} != 标签数 {len(tags)}")
    return tags, contents


def destyle_one(llm: LLM, original: str) -> dict:
    """一段官方原稿 → 一条受控格式的平行样本。

    返回 ``{"tags", "contents", "plan", "original", "n_tags"}``。
    """
    tags, bodies = split_tagged(original)
    n = len(tags)
    listing = "\n".join(f"{i}. [{t}] {b}" for i, (t, b) in enumerate(zip(tags, bodies), 1))
    raw = llm.complete(
        "destyle",
        prompts.DESTYLE_TAG_SYSTEM.format(n=n),
        prompts.DESTYLE_TAG_USER.format(n=n, segments=listing),
    )
    contents: list[str] = []
    for line in raw.splitlines():
        m = _SEG_LINE.match(line)
        if m:
            contents.append(m.group(2).strip().strip("。").strip())
    if len(contents) != n:
        raise SegmentError(f"模型输出段数 {len(contents)} != {n} | 原始输出: {raw[:120]!r}")

    # 顺序很重要，写成两步：先还原哨兵，再中性化，最后再还原一次。
    # 反过来会把哨兵包成 `《《player》》`——因为 `_BARE_PLAYER` 会把《player》里的
    # player 再包一层（那一步本来只为抓模型自己编的裸 `Player`）。
    contents = [_SENTINEL_BACK.sub("[player]", c) for c in contents]

    # 占位符：逐段校验，缺了先尝试把「player/玩家」改回来（与 destyle.py 同一策略）
    for i, (b, c) in enumerate(zip(bodies, contents)):
        need = _SENTINEL in b
        if need and "[player]" not in c.lower():
            fixed = _BARE_PLAYER.sub("[player]", c)
            if "[player]" not in fixed.lower():
                raise SegmentError(f"第 {i+1} 段占位符丢失: {c!r}")
            c = fixed
        contents[i] = c

    # 中性化：标点、省略号、裸 player（实测约 5% 的样本会自己编个「Player」称呼）
    contents = [neutralize(c) for c in contents]
    contents = [_SENTINEL_BACK.sub("[player]", c) for c in contents]

    for i, c in enumerate(contents):
        if not contract._CJK_RE.search(c):
            raise SegmentError(f"第 {i+1} 段无中文: {c!r}")
        if contract.cjk_len(c) > contract.SENTENCE_MAX_CHARS:
            raise SegmentError(f"第 {i+1} 段超 {contract.SENTENCE_MAX_CHARS} 字: {c!r}")

    return {
        "tags": tags,
        "contents": contents,
        "plan": render_plan(tags, contents),
        "original": original,
        "n_tags": n,
    }