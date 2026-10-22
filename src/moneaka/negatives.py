"""受控负例：与真实输出**同内容**，但只在一条风格轴上错。

为什么不用官方 rejected：实测 chosen 与 rejected 去标签后的字序相似度中位只有
0.18（见 docs/DATA_INVENTORY.md），两者说的根本不是同一件事。拿它做 DPO，
log-prob 差距会被「内容对不对」吃干净，风格轴几乎拿不到梯度。

四条轴各自对应一个已经量出来的真实缺陷：

| 轴 | 针对的缺陷 | 证据 |
|---|---|---|
| ``tags``   | 情绪标签选错（爱用「微笑」兜底） | run1 逐位吻合 65%，错例集中在 微笑↔开心/尴尬/担心 |
| ``order``  | 不按「语气」行的反应次序演 | 阶段 0 病根；duel 里输 reference 主要输这 |
| ``merge``  | 句子不拆、一句塞多个意思 | 真实中位 15 字/句，契约上限 60 |
| ``flatten``| 语气外壳被抹平、说得太书面 | 出 0.95 vs 真 1.25 个语气词 |

每条负例只动一条轴，其余尽量与真实输出逐字相同 → DPO 的梯度就是这一条轴的偏序。
负例**不要求**符合契约（它本来就是反面教材），但也不能顺带违反别的轴，
否则一个负例同时压低好几条轴，学到的偏序就不干净了。
"""

from __future__ import annotations

import random
import re

from .contract import EMOTION_TAGS

#: 只认真正的情绪标签。`[player]` 也长得像标签，但它是**句内占位符**，
#: 拿 `\[[^\]]+\]` 切句会把句子从中间劈开（实测踩过）。
_TAG_ALT = "|".join(re.escape(t) for t in EMOTION_TAGS)
_TAG_AT = re.compile(rf"\[({_TAG_ALT})\]")


def split_sentences(text: str) -> list[tuple[str, str]]:
    """→ ``[(标签, 正文), ...]``；只按白名单标签切句。"""
    text = text.strip()
    hits = list(_TAG_AT.finditer(text))
    if not hits:
        return []
    out = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        out.append((m.group(0), text[m.end():end].strip()))
    return out


def join(parts: list[tuple[str, str]], sep: str = " ") -> str:
    return sep.join(f"{t}{b}" for t, b in parts if b).strip()


def tags(text: str, rng: random.Random | None = None) -> str:
    """标签整体错位一位，正文一字不动。

    错位而不是随机替换：随机替换有概率换出等价的合法标签（微笑/开心经常通用），
    那种「假负例」会往 DPO 里灌噪声。标签全相同时错位无效，才退回换标签。
    """
    rng = rng or random.Random(0)
    parts = split_sentences(text)
    if len(parts) < 2:
        return ""
    labs = [t for t, _ in parts]
    if len(set(labs)) > 1:
        rot = labs[1:] + labs[:1]
        return join([(r, b) for r, (_, b) in zip(rot, parts)])
    cand = [f"[{t}]" for t in EMOTION_TAGS if f"[{t}]" != labs[0]]
    if not cand:
        return ""
    return join([(rng.choice(cand), b) for _, b in parts])


def order(text: str) -> str:
    """句子次序反转——内容一致，反应次序错了。"""
    parts = split_sentences(text)
    if len(parts) < 2:
        return ""
    return join(list(reversed(parts)))


def merge(text: str) -> str:
    """合成一句、只留第一个标签——违反「一句一个意思」，可能超 60 字（正是要罚的）。"""
    parts = split_sentences(text)
    if len(parts) < 2:
        return ""
    head, first = parts[0]
    rest = "".join(b for _, b in parts[1:])
    return f"{head}{first}{rest}".strip()


def flatten(text: str) -> str:
    """抹掉全部语气外壳：去标签、去语气词/波浪号，保留中文标点原样。

    标点不动是刻意的：顺手把 `。` 改成 `.` 会让这条负例**额外**违反半角标点轴，
    梯度就不干净了。
    """
    parts = split_sentences(text)
    if not parts:
        return ""
    body = " ".join(b for _, b in parts if b)
    body = re.sub(r"[~～]", "", body)
    body = re.sub(r"(啦|嘛|哦|呢|呀|呗|嘿嘿)[,，]?", "", body)
    return re.sub(r"\s+", " ", body).strip()


AXES = {"tags": tags, "order": order, "merge": merge, "flatten": flatten}


def negatives(text: str, seed: int = 0) -> dict[str, str]:
    """返回 ``{轴: 负例}``；退化（为空或与原文相同）的轴自动丢。"""
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for name, fn in AXES.items():
        try:
            neg = fn(text, rng) if name == "tags" else fn(text)
        except Exception:
            neg = ""
        neg = (neg or "").strip()
        if neg and neg != text.strip() and len(neg) >= 2:
            out[name] = neg
    return out
