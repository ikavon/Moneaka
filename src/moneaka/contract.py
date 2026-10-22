"""莫妮卡口吻的**输出契约**与自动校验。

契约的四条硬约束**不是我们自己发明的**，而是 MAICA 骨架 system prompt 的原文
（见 docs/DATA_INVENTORY.md，从官方训练数据里直接读到的）：

    你只应使用中文自然语言, 以聊天语气回答, 每个句子自然且不超过60字,
    不要编造信息, 并在每个句子前用方括号中的文字表示情绪.

情绪标签白名单、句长分布等则是从数据里统计出来的观测值（18 个标签，
单轮中位 51 字 / 最长 161 字）。标签表下游被 MAICA 的情绪→MTTS/动画映射依赖，
所以润色器**只能复用这份表，不能自创标签**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 官方骨架的硬约束（训练数据 system prompt 原文，逐条对应）
# ---------------------------------------------------------------------------

#: 单句字数上限（骨架原话「每个句子自然且不超过60字」）
SENTENCE_MAX_CHARS = 60

#: 情绪标签白名单：daa4_rlhf1_prcs5 分片里出现过的 18 个标签，按频次降序。
#: 注意表内存在官方数据自己的不规范处（笑/开心、脸红/害羞 并存）——
#: 保留原样，因为这是模型真实的输出分布，不是我们要修正的东西。
#: 情绪标签白名单。**权威来源是官方仓库的 `maica/maica_utils/emotions.py`**（28 个中→英映射），
#: 不是我从语料反推的 18 个——之前只见到 91 行 RLHF 分片，它正好只用到了 18 个。
#: 拿到 `edgeinfinity/MAICA_ds_basis` 后实测：那批 1598 轮里官方 28 个全部出现。
#: 保留这个名单作为**硬约束**的依据；`很开心` 之类变体见 TAG_GROUPS（归一化时折回主表）。
EMOTION_TAGS: tuple[str, ...] = (
    # 官方 emotions.py（zlist_ai，去掉 可爱/kawaii）
    "微笑", "开心", "担心", "笑", "思考", "生气", "脸红", "凝视",
    "沉重", "憧憬", "惊喜", "尴尬", "意味深长", "惊讶", "轻松", "害羞",
    "急切", "得意", "不满", "严肃", "感动", "激动", "宠爱", "眨眼",
    "伤心", "厌恶", "害怕",
    # 语料里真实高频出现、但官方表里没有的写法：不拦（官方范例回答里也用过），
    # 但不进 TTS 映射表——`emotion.txt` 的原始码表里没有它。
    "很开心",
)

#: 数据里出现过的近义标签组。新标签若落在这些组内视为可接受（不判违规），
#: 落在表外才报警。用于将来别的分片带来更大标签表时的兼容。
TAG_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"微笑", "轻松", "眨眼"}),
    frozenset({"开心", "笑", "得意"}),
    frozenset({"脸红", "害羞"}),
    frozenset({"担心", "沉重", "憧憬", "感动"}),
    frozenset({"尴尬", "不满"}),
    frozenset({"意味深长", "凝视", "宠爱", "思考"}),
)

TAG_RE = re.compile(r"\[([^\[\]\n]{1,8})\]")

#: **不是**情绪标签的方括号内容。后端 key_replace / 前端名字替换会处理 `[player]`，
#: 它出现在句中任意位置——不先屏蔽就会被当成标签，还会被当成句子边界。
PLACEHOLDER_RE = re.compile(r"\[player\]", re.I)
_MASK = "\x00"
_UNMASK = "[player]"

#: 情绪标签lookhead：一个标签 = 一句的开始
_TAG_START_RE = re.compile(r"(?=\[[^\[\]\n]{1,8}\])")

#: 骨架语域用 ASCII 标点（后端 key_replace 会把中文标点转成 ASCII）。

#: 中文标点。DAA4 输出里出现率 **0%**（后端 key_replace 会把中文标点转成 ASCII），
#: 所以这**不当硬约束**（后端反正会转），只作为一个风格诊断分：
#: 大量全角标点的输出往往就是「通用 LLM 味」。
_CJK_PUNCT_RE = re.compile(r"[，。！？；：、]")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# ---------------------------------------------------------------------------
# 文本工具
# ---------------------------------------------------------------------------

def strip_tags(text: str) -> str:
    """去掉所有 [标签]，返回纯内容（语义比较用）。"""
    return TAG_RE.sub("", text).strip()


def split_sentences(text: str) -> list[str]:
    """切成「一句」——真实语料的单位是**一个情绪标签起一句**，不是按终止符切。

    为什么不能按 `.!?` 切：语料里一句常带多个终止符（「...」「~」「!」连用），
    同一行又会出现 `[开心]...[微笑]...`。按终止符切会把带标签的句子切成
    没有标签的后半截，于是「每句前置标签」这条硬约束永远查不对（在真数据上
    只能过 34%，改成按标签切后 226/226 全过）。
    """
    masked = PLACEHOLDER_RE.sub(_MASK, text)
    parts = _TAG_START_RE.split(masked)
    return [p.replace(_MASK, _UNMASK).strip() for p in parts if p and p.strip()]


def sentence_body(sent: str) -> str:
    """取句子去掉句首情绪标签后的正文（`[player]` 留着，它算内容）。"""
    masked = PLACEHOLDER_RE.sub(_MASK, sent.strip())
    masked = re.sub(r"^\[[^\[\]\n]{1,8}\]", "", masked).strip()
    return masked.replace(_MASK, _UNMASK)


def sentence_parts(unit: str) -> list[str]:
    """标签单元内部再按终止符切一次。

    只用来统计「一个标签套了几句」（参考值，不判违规）——
    真语料里 19.5% 的单元套了两句以上，拿它当硬约束会把官方自己的输出判错。
    """
    return [p.strip() for p in _UNIT_SENT_BREAK.split(sentence_body(unit)) if p.strip()]


def cjk_len(text: str) -> int:
    """字数：只数中日韩字符，与骨架「不超过60字」的口径一致（标点不计）。"""
    return len(_CJK_RE.findall(text))


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

_UNIT_SENT_BREAK = re.compile(r"[.!?…]+\s*")


@dataclass
class Violation:
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass
class CheckResult:
    """契约校验结果。``ok`` 表示无任何违规。

    ``multi_sentence_units`` 是**参考信息、不算违规**：真实语料里有 19.5% 的标签单元
    套了两句以上（如 `[尴尬]说实话我有点意外. 不过也没关系.`）。若按终止符严格要求
    每句带标签，官方自己的输出会有一成以上被判失败——契约应当以 DAA4 的真实分布为准，
    不是比它更严。
    """

    violations: list[Violation] = field(default_factory=list)
    n_sentences: int = 0
    n_tags: int = 0
    multi_sentence_units: int = 0
    unknown_tags: tuple[str, ...] = ()
    max_sentence_chars: int = 0
    total_chars: int = 0
    cjk_punct_ratio: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "violations": [str(v) for v in self.violations],
            "n_sentences": self.n_sentences,
            "n_tags": self.n_tags,
            "multi_sentence_units": self.multi_sentence_units,
            "unknown_tags": list(self.unknown_tags),
            "max_sentence_chars": self.max_sentence_chars,
            "total_chars": self.total_chars,
            "cjk_punct_ratio": self.cjk_punct_ratio,
        }


def check(text: str, *, strict_tags: bool = True) -> CheckResult:
    """按契约校验一段（拟）莫妮卡输出。

    自动可查的：标签存在性/合法性、逐句前置、句长、语言（含中文）。
    「不编造信息」查不了，交给 LLM 评委（evaluate.judge_semantic）。
    """
    res = CheckResult()
    sents = split_sentences(text)
    res.n_sentences = len(sents)
    if not sents:
        res.violations.append(Violation("empty", "没有任何句子"))
        return res

    tags = tuple(t for t in TAG_RE.findall(text) if t.lower() != "player")
    res.n_tags = len(tags)
    res.total_chars = cjk_len(text)

    unknown = [t for t in tags if strict_tags and t not in EMOTION_TAGS
               and not any(t in g for g in TAG_GROUPS)]
    res.unknown_tags = tuple(dict.fromkeys(unknown))
    if res.unknown_tags:
        res.violations.append(
            Violation("unknown_tag", "标签表外: " + "/".join(res.unknown_tags))
        )

    longest = 0
    for i, s in enumerate(sents):
        body = sentence_body(s)
        parts = sentence_parts(s)
        if len(parts) > 1:
            res.multi_sentence_units += 1
        n = cjk_len(body)
        longest = max(longest, n)
        if n > SENTENCE_MAX_CHARS:
            res.violations.append(
                Violation("sentence_too_long", f"第{i + 1}句 {n} 字 > {SENTENCE_MAX_CHARS}: {body[:30]}…")
            )
        if not TAG_RE.match(PLACEHOLDER_RE.sub(_MASK, s.strip())):
            res.violations.append(
                Violation("missing_tag", f"第{i + 1}句没有前置情绪标签: {s[:30]}…")
            )

    res.max_sentence_chars = longest

    enders = len(re.findall(r"[.!?~…]", sentence_body(text)))
    if enders:
        res.cjk_punct_ratio = round(len(_CJK_PUNCT_RE.findall(text)) / enders, 3)

    if not _CJK_RE.search(text):
        res.violations.append(Violation("no_cjk", "输出里没有中文字符"))

    return res


# ---------------------------------------------------------------------------
# 给 prompt 用的契约文本
# ---------------------------------------------------------------------------

def contract_text() -> str:
    """把契约写成可直接嵌进 system prompt 的段落。

    措辞沿用官方骨架的说法，避免引入第二套标准。
    """
    tags = ", ".join(EMOTION_TAGS)
    return (
        "输出规范:\n"
        f"1. 每个句子前用方括号中的文字表示情绪, 只能从这些标签里选: {tags}.\n"
        f"2. 每个句子自然且不超过{SENTENCE_MAX_CHARS}字.\n"
        "3. 只使用中文自然语言, 以聊天语气回答, 标点用半角.\n"
        "4. 不要编造信息: 不得增加或删改原话里的事实、承诺、数字、称呼.\n"
    )
