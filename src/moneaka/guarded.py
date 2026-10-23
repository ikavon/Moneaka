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


#: 只留「机械可判且几乎不会误报」的项。**否定词与占位符不再进闸门**：
#: 实测 36 条退化样本里 29 条是这两项误报（plan 有 [player] 而她说「你」、
#: 反问句里的「不」被改写成别的说法）——那不是错误，是正常措辞。
#: 否定/近义交给 B（强模型复核），它有 88% 查全。
#:
#: 「方向」是后加的：实测 Q2 把「我帮你看看」说成「帮帮我看看」、
#: Q10 把「陪你」说成「帮你」，B 层两次都没拦住。
STRICT = ("标签", "数字", "方向", "语言")

#: 中英混杂：真实对话里模型把整句末段写成了英文
#: （「聊什么都行, 我 just want to hear your words.」），A/B 都没拦住。
_EN_RUN = re.compile(r"[A-Za-z]+(?:[\s'’-]+[A-Za-z]+){2,}")   # ≥3 个连续英文词
_LETTER = re.compile(r"[A-Za-z]")
_CJKCH = re.compile(r"[一-鿿]")


def language_flags(plan_body: str, out_body: str) -> list[str]:
    """输出里出现成句的英文 / 英文占比异常。单词级外来词（apple、词源）不算。"""
    bad: list[str] = []
    # 必须先把占位符抹掉：`[player_nickname]` 里就有 15 个字母，
    # 不处理会把正常台词误判成「英文占比过高」（实测 206 条里误报 5 条）。
    # 注意：`PLACEHOLDER_RE` 只认 `[player]`，`[player_nickname]` 会漏过去，
    # 于是它的 14 个字母被当成英文（实测误报 5 条）。这里连通用占位符一起抹。
    out_body = re.sub(r"\[[A-Za-z_]+\]", "\x00", out_body)
    if _EN_RUN.search(out_body):
        bad.append(f"语言 出现英文句: {_EN_RUN.search(out_body).group(0)[:24]}")
    n_en, n_cjk = len(_LETTER.findall(out_body)), len(_CJKCH.findall(out_body))
    if n_en >= 10 and n_en > 0.3 * max(1, n_cjk):
        bad.append(f"语言 英文占比过高 {n_en}/{n_cjk}")
    return bad


#: 施受关系：动词 + 受事人称。「我帮你」→「帮帮我」是翻转，「陪你」→「帮你」是换人做事。
_ROLE = re.compile(r"(帮|给|替|陪|教|送|带|让|跟)(你|您|我)")


def _roles(text: str) -> set[tuple[str, str]]:
    return set(_ROLE.findall(text))


def direction_flags(plan_body: str, out_body: str) -> list[str]:
    """施受方向被改写。只报**确凿的矛盾**，不做“两边不完全一致”这种粗比较。"""
    bad: list[str] = []
    a, b = _roles(plan_body), _roles(out_body)
    # (1) 同动词换了受事：帮你 → 帮我
    #     前提是指令里**本来只有一个方向**。指令里两个方向都写了（「讲给你听可以…
    #     你把看到的讲给我听」），模型只留一个方向不是翻转——实测 Q12 甘肃这题
    #     就是这么被误判、整句掉成直译的。
    for verb, obj in a:
        other = "我" if obj != "我" else "你"
        if (verb, other) in b and (verb, obj) not in b and (verb, other) not in a:
            bad.append(f"方向 {verb}{obj} → {verb}{other}")
    # (2) 换了个动词把事情揽到自己身上 / 反之：陪你 ↔ 帮你
    #     「帮/替/给」是替对方做事，「陪」是待在他身边——两者都不是同义写法，互换就是改意思。
    DO, BE = {"帮", "替", "给"}, {"陪"}
    for obj in ("你", "您"):
        a_do = {v for v in DO if (v, obj) in a}
        b_do = {v for v in DO if (v, obj) in b}
        a_be, b_be = ("陪", obj) in a, ("陪", obj) in b
        # 同样要求「指令里本来只有一个方向」
        if a_do and not b_do and b_be and not a_be:
            bad.append(f"方向 {'/'.join(sorted(a_do))}{obj} → 陪{obj}")
        if a_be and not b_be and b_do and not a_do:
            bad.append(f"方向 陪{obj} → {'/'.join(sorted(b_do))}{obj}")
    return bad

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
    bad += direction_flags(plan_body, body)
    bad += language_flags(plan_body, body)
    if not loose:
        return bad
    # 每段内容至少要有一部分字面落地：用 3-gram 覆盖检查（噪声大，默认关）
    for i, c in enumerate(contents, 1):
        grams = {c[j:j + 3] for j in range(0, max(1, len(c) - 2)) if contract._CJK_RE.search(c[j:j + 3])}
        if grams and not (grams & {body[j:j + 3] for j in range(0, max(1, len(body) - 2))}):
            bad.append(f"第{i}段内容完全没落地")
    return bad




# ---------------------------------------------------------------------------
# 标签修复：内容对、标签错时按位重贴，而不是退化直译
# ---------------------------------------------------------------------------
# 事故：真实对话 10 轮里 2 轮（20%）因为「标签个数/内容不符」被判失败，重采也救不回来，
# 最后退化成「直译」——plan 原话 + 标签原样拼，既没标点也没风格。用户看到的就是
# deepseek 的原文。但那些句子的**内容是对的**，错只在标签，完全可以机械修。
#
# 做法：把模型输出里的标签全部剥掉，按句切分，再按 plan 的标签序列逐句贴回去。
# 句子数与标签数不一致时：句子多了就相邻合并，少了就放弃（不猜）。

#: 按**标点**切句（`contract.split_sentences` 是按标签切的，剥掉标签后只剩一段，
#: 所以这里必须自己切）。
_SENT = re.compile(r"[^.。!！?？~～…]*(?:[.。!！?？~～…]+|$)")


def _split_by_punct(text: str) -> list[str]:
    out = []
    for m in _SENT.finditer(text or ""):
        s = m.group(0).strip()
        if s and contract._CJK_RE.search(s):
            out.append(s)
    return out


def retag(plan: str, out: str) -> str | None:
    """把 plan 的标签序列按位重贴到 out 的句子上。成功返回新台词，失败返回 None。

    只在「内容可用、标签不对」时用；不改一个字的内容，只换标签与分段。
    """
    try:
        want, _ = parse_tagged_plan(plan)
    except Exception:  # noqa: BLE001
        return None
    if not want:
        return None

    # 1) 剥掉输出里所有标签，按句切
    body = contract.TAG_RE.sub("", contract.PLACEHOLDER_RE.sub("\x00", out))
    body = body.replace("\x00", "[player]")
    sentences = _split_by_punct(body)
    if not sentences:
        return None

    # 2) 句子数与标签数对齐：多了相邻合并，少了放弃
    n = len(want)
    if len(sentences) > n:
        # 均匀合并成 n 组，保持顺序
        groups: list[list[str]] = [[] for _ in range(n)]
        for i, s in enumerate(sentences):
            groups[min(n - 1, i * n // len(sentences))].append(s)
        sentences = [" ".join(g) for g in groups]
    elif len(sentences) < n:
        return None

    line = " ".join(f"[{t}]{s}" for t, s in zip(want, sentences))

    # 3) 修完必须过闸门：标签对、数字对、方向对、格式合法
    if [m.group(1) for m in contract.TAG_RE.finditer(
            contract.PLACEHOLDER_RE.sub("\x00", line))] != want:
        return None
    if [p for p in check(plan, line) if p.startswith(STRICT)]:
        return None
    if not contract.check(line).ok:
        return None
    return line


def repair_tags(plan: str, out: str) -> tuple[str, bool]:
    """给调用方的便利函数：能修就修，修不了原样返回。"""
    fixed = retag(plan, out)
    return (fixed, True) if fixed else (out, False)
