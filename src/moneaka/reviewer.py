"""台词复核：agent 自审润色器的产物。**尚未接入 `turn.py`。**

## 架构（用户定的，别改）

agent 的身份是**连续的**，不是三个独立角色：

    同一个 agent：写表达指令 → 交给润色器 → 检查润色器交回的台词 → 定稿输出

所以这个复核**必须拿到它自己刚写的那条指令**，判断的是「润色器有没有忠实执行我的意图」，
而不是「这句话写得好不好」。少了 plan，复核就变成从零重写，等于把润色器架空——
那也就没法回答「润色器这一层值不值」。

## 为什么要强模型

见 `docs/JUDGE_POLICY.md`：角色扮演专用模型只有「扮演莫妮卡」一个用途，
它审不出自己的人称错位（那是它自己的说话习惯）。

## 这个模块**不是**机械闸门

`guarded.py` 那套正则闸门只管四类高精度硬错（标签/数字/方向/语言），
故意窄。**正则枚举式审查做不成**——实测把 206 条官方原答过一遍，
命中 39%（「聊聊」「说说」被当成叠字复读、`？` 后半句被当成缺标签）。
真货被判有病，说明这类检查没有判断力。证据在 `tools/scan_defects.py`。
"""

from __future__ import annotations

import re

from . import contract

REVIEW_SYSTEM = """你是这台莫妮卡对话系统的**编剧本人**。流程是这样的：

1. 你读完玩家的话，写下了一条「表达指令」（语气标签序列 + 内容要点）。
2. 一个润色器把这条指令写成了一句台词，交回给你。
3. 你现在要做的是：**检查它有没有忠实执行你的指令**，然后定稿。

## 检查项（按严重度）

1. **人称与身份错位**：莫妮卡用「我」说自己、用「你」/`[player]` 说玩家。
   把玩家说成「我的女朋友」、用「他/她」指玩家、用第三人称说自己、
   突然变成长辈或客服口吻——都算错。
2. **没执行的意图**：指令里的要点被漏掉、被改意思、被加进你没写的内容。
3. **错别字与病句**：写错的字、重复虚词（我我、的的）、缺主语的断句、
   标签后直接跟标点。
4. **情绪标签**：每句以方括号开头，只能用下面这张表；表外的（温柔、认真、好奇）
   换成最接近的表内词。**标签数量与位置保持与指令一致**。

## 情绪标签表（唯一合法集合）

{tags}

## 输出

只输出定稿的台词，一整段。不需要改就原样输出。
"""


def build_system() -> str:
    return REVIEW_SYSTEM.format(tags=" / ".join(contract.EMOTION_TAGS))


def _first_tag(text: str) -> str | None:
    m = contract.TAG_RE.search(text or "")
    return m.group(1) if m else None


def review(llm, user_msg: str, plan: str, polisher_out: str, *,
           canon: list[str] | None = None, temperature: float = 0.0,
           max_tokens: int = 500, profile: str = "screenwriter") -> dict:
    """以「编剧本人」的身份复核润色器输出。

    ``plan`` **必传**：这是判断「忠实与否」的唯一依据，也是身份连续性的来源。
    ``canon`` 可选，给了就防止把设定写反（比如她不能有真实双手）。
    """
    if not (plan or "").strip():
        raise ValueError("复核必须带 plan：没有它就不是复核，是重写")
    system = build_system()
    if canon:
        system += "\n## 角色事实（越界即错）\n" + "\n".join(f"- {x}" for x in canon[:40])
    prompt = (f"玩家: {user_msg}\n\n你的表达指令:\n{plan}\n\n"
              f"润色器交回的台词:\n{polisher_out}\n\n按规则复核后输出定稿。")
    out = _strip_wrapper((llm.complete(profile, system, prompt, temperature=temperature,
                                       max_tokens=max_tokens) or "").strip())
    notes: list[str] = []
    if out and not contract.check(out).ok:
        notes.append("定稿仍不满足契约")
    if out and _first_tag(out) != _first_tag(polisher_out):
        notes.append("首标签被改，需人看一眼")
    if not out:
        notes.append("复核返回空，保留润色器原稿")
        out = polisher_out
    return {"text": out, "changed": out != polisher_out, "notes": notes}


def _strip_wrapper(s: str) -> str:
    s = re.sub(r"^```[\w]*\s*", "", s.strip())
    s = re.sub(r"\s*```$", "", s)
    return re.sub(r"^(?:定稿[:：]?|复核后[:：]?|输出[:：]?)\s*", "", s).strip()