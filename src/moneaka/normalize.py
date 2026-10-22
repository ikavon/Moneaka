"""输出归一化：把模型的「大致合规」压成「严格合规」。

为什么不指望模型自己守格式：1.7B 哪怕给了 18 词白名单 + 4 个 few-shot，
仍会自创标签（实测 23/60 条）。而标签是下游硬契约——MAICA 的情绪要映射到
MTTS 语气和 YSM 表情动画，一个表外标签就是一次映射失败。

所以分层：
- **模型**负责「怎么说」（措辞、语气、分寸）——这没法用代码写；
- **代码**负责「格式」（标签白名单、逐句前置、句长、标点）——这不该让模型记。

归一化做的四件事都是可逆/可审计的，不做语义改写。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .contract import (
    EMOTION_TAGS,
    PLACEHOLDER_RE,
    SENTENCE_MAX_CHARS,
    TAG_GROUPS,
    TAG_RE,
    cjk_len,
    split_sentences,
)

#: 常见自创标签 → 白名单。按**效价 + 唤起度**归类：
#: 负向高唤起→不满/担心, 负向低唤起→沉重/尴尬, 正向高→开心/笑, 正向低→微笑/宠爱。
#: 没收录的落到最常用的 `微笑`（她的默认底色），并计入 stats.unmapped ——
#: 这个列表是从真实失败里长出来的，所以每次 eval 后看 unmapped 再补。
TAG_ALIASES: dict[str, str] = {
    # 思考系
    "沉思": "思考", "思索": "思考", "疑惑": "思考", "困惑": "思考", "好奇": "思考",
    "回忆": "思考", "认真": "思考", "考虑": "思考", "琢磨": "思考",
    # 不满系（负向高唤起）
    "嫌弃": "不满", "蹙眉": "不满", "生气": "不满", "抱怨": "不满", "傲娇": "不满",
    "嘟嘴": "不满", "委屈": "不满", "恼火": "不满", "严肃": "不满",
    # 担心系（负向 + 关切）
    "担忧": "担心", "忧虑": "担心", "害怕": "担心", "不安": "担心", "焦虑": "担心",
    "关心": "担心", "关切": "担心", "care": "担心", "心疼": "担心", "揪心": "担心",
    # 开心系（正向高唤起）
    "兴奋": "开心", "激动": "开心", "惊喜": "开心", "高兴": "开心", "大笑": "笑",
    "调皮": "笑", "打趣": "笑", "逗趣": "笑", "大笑不止": "笑", "感慨": "笑",
    "调皮地": "笑", "轻声": "微笑", "微笑着": "微笑", "微笑地": "微笑",
    # 宠爱系（正向低唤起 + 亲近）
    "甜蜜": "宠爱", "温柔": "宠爱", "溺爱": "宠爱", "宠溺": "宠爱", "关爱": "宠爱",
    "温柔地": "宠爱", "期待": "憧憬", "向往": "憧憬", "遥想": "憧憬",
    # 脸红系
    "羞涩": "脸红", "羞怯": "害羞", "不好意思": "脸红", "害羞地": "害羞",
    # 沉重系（负向低唤起）
    "悲伤": "沉重", "难过": "沉重", "失落": "沉重", "沉默": "沉重", "暗淡": "沉重",
    # 尴尬/无奈（负面但轻）
    "无奈": "尴尬", "苦笑": "尴尬", "尬笑": "尴尬", "尴尬地": "尴尬", "无语": "尴尬",
    # 自信/得意
    "自信": "得意", "骄傲": "得意", "得意地": "得意", "炫耀": "得意",
    # 其它
    "凝视着": "凝视", "注视": "凝视", "盯着": "凝视",
    "顽皮": "眨眼", "俏皮": "眨眼", "眨眼笑": "眨眼", "眨眼间": "眨眼",
    "放松": "轻松", "欣慰": "轻松", "安心": "轻松", "舒心": "轻松", "释然": "轻松",
    "感动地": "感动", "触动": "感动",
    "惊讶": "开心", "震惊": "开心", "吃惊": "开心",
    "不满地": "不满", "思考着": "思考", "脸红着": "脸红",
    # 补：跑完全量后从 stats.tags_unmapped 里长出来的（这 11 个原采都落到了默认值微笑）
    "幸福": "开心", "拥抱": "宠爱", "轻抚": "宠爱", "抚摸": "宠爱",
    "坚定": "思考", "想知道": "思考", "轻柔": "微笑", "握手": "微笑",
    "小心": "担心", "紧张": "担心",
    "暧昧": "脸红", "调情": "脸红",
    "讽刺": "不满", "嘲笑": "笑", "危笑": "笑",
}

#: 中文标点 → 半角。后端 key_replace 本来也会转，但我们在本地转掉，
#: 这样一个润色结果在 MAS / MC / QQ bot 任何前端都长得一样。
_PUNCT_MAP = str.maketrans({
    "，": ", ", "。": ". ", "！": "!", "？": "?", "；": "; ",
    "：": ": ", "、": ", ", "（": "(", "）": ")", "～": "~",
})

_SPLIT_AT = re.compile(r"(?<=[,、;:!?])\s*")
_TAG_HEAD = re.compile(r"^\s*\[([^\[\]\n]{1,8})\]\s*")

DEFAULT_TAG = "微笑"


@dataclass
class NormStats:
    tags_mapped: dict[str, str] = field(default_factory=dict)   # 原名 -> 改成
    tags_unmapped: list[str] = field(default_factory=list)      # 表外且无别名
    tags_inserted: int = 0
    sentences_split: int = 0
    punct_converted: int = 0

    def as_dict(self) -> dict:
        return {
            "tags_mapped": self.tags_mapped,
            "tags_unmapped": self.tags_unmapped,
            "tags_inserted": self.tags_inserted,
            "sentences_split": self.sentences_split,
            "punct_converted": self.punct_converted,
        }


def canonical_tag(tag: str, stats: NormStats) -> str:
    """把一个标签压到白名单内。"""
    t = tag.strip()
    if t in EMOTION_TAGS:
        return t
    if t.lower() == "player":
        return t  # 占位符不是标签，原样保留（由调用方处理）
    if t in TAG_ALIASES:
        stats.tags_mapped[t] = TAG_ALIASES[t]
        return TAG_ALIASES[t]
    for group in TAG_GROUPS:
        if t in group:
            return t
    # 表外无别名：找最长公共子串式的近似（「温柔的微笑」→ 微笑）
    for known in EMOTION_TAGS:
        if known in t:
            stats.tags_mapped[t] = known
            return known
    stats.tags_unmapped.append(t)
    stats.tags_mapped[t] = DEFAULT_TAG
    return DEFAULT_TAG


def _fix_tags(text: str, stats: NormStats) -> str:
    masked_tags: dict[str, str] = {}

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name.lower() == "player":
            return m.group(0)
        fixed = canonical_tag(name, stats)
        masked_tags[fixed] = fixed
        return f"[{fixed}]"

    return TAG_RE.sub(repl, text)


def _fix_punct(text: str, stats: NormStats) -> str:
    out = text.translate(_PUNCT_MAP)
    stats.punct_converted = sum(1 for ch in text if ch in "，。！？；：、（）～")
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def _split_long(body: str) -> list[str]:
    """把超长正文在逗号/顿号处切开，每块 ≤ SENTENCE_MAX_CHARS。切不动就硬切。"""
    if cjk_len(body) <= SENTENCE_MAX_CHARS:
        return [body]
    pieces = [p for p in _SPLIT_AT.split(body) if p.strip()]
    out: list[str] = []
    buf = ""
    for p in pieces:
        cand = (buf + p).strip()
        if buf and cjk_len(cand) > SENTENCE_MAX_CHARS:
            out.append(buf.strip())
            buf = p
        else:
            buf = cand
    if buf.strip():
        out.append(buf.strip())
    # 单块仍超长 → 按字符硬切（宁可断句难看，也不能违反下游硬约束）
    final: list[str] = []
    for chunk in out:
        while cjk_len(chunk) > SENTENCE_MAX_CHARS:
            final.append(chunk[:SENTENCE_MAX_CHARS])
            chunk = chunk[SENTENCE_MAX_CHARS:]
        if chunk.strip():
            final.append(chunk)
    return final


def normalize(text: str, *, stats: NormStats | None = None) -> tuple[str, NormStats]:
    """返回 (归一化后的文本, 统计)。幂等：对已合规文本再跑不会变。"""
    st = stats or NormStats()
    text = (text or "").strip()
    if not text:
        return text, st
    text = _fix_tags(text, st)
    text = _fix_punct(text, st)

    prev_tag = DEFAULT_TAG
    parts: list[str] = []
    for unit in split_sentences(text):
        m = _TAG_HEAD.match(unit)
        if m and m.group(1).lower() != "player":
            tag = m.group(1)
            body = unit[m.end():].strip()
        elif m:  # 句首是 [player]，不是标签
            tag = prev_tag
            body = unit.strip()
            st.tags_inserted += 1
        else:
            tag = prev_tag
            body = unit.strip()
            st.tags_inserted += 1
        if not body:
            continue
        chunks = _split_long(body)
        if len(chunks) > 1:
            st.sentences_split += len(chunks) - 1
        for ch in chunks:
            parts.append(f"[{tag}]{ch}")
            prev_tag = tag

    # 结尾补句号（真数据几乎都终结在 . ! ? ~ …）
    out = " ".join(parts)
    if out and out[-1] not in ".!?~…":
        out += "."
    return out, st