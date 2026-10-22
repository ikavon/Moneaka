"""步骤 2：两种出话方式，A/B 对照。

- ``polish`` —— 润色器路线（本项目主角）：拿到平实草稿，只改说法不改内容
- ``direct`` —— 对照组：通用模型按官方骨架 prompt 直接扮演莫妮卡

阶段 0 的核心问题就是「polish 比 direct 像多少、语义丢多少」。
两个函数都接收同一个模型对象，所以同一端点也能同时充当两者。
"""

from __future__ import annotations

from . import corpus, prompts
from .llm import LLM
from .contract import TAG_RE


def polish(llm: LLM, draft: str, *, profile: str = "restyle", model: str | None = None) -> str:
    text = llm.complete(profile, prompts.restyle_system(),
                        prompts.RESTYLE_USER.format(draft=draft),
                        **( {"model": model} if model else {}))
    # 润色器不许输出外壳：去掉可能的「改写：」前缀与包裹引号
    text = text.strip().strip('"\'')
    return text


def direct(llm: LLM, user_msg: str, context: list[dict] | None = None,
           *, profile: str = "restyle", model: str | None = None,
           system: str | None = None) -> str:
    """按官方骨架 prompt 直接扮演（对照组）。system 默认取数据里的骨架原文。

    ``profile`` 决定是谁在演：本地小模型（衡量「润色器 vs 小模型自己演」）
    或强模型（衡量「润色器 vs 通用大模型直接上」）——后者才是真正的竞品。
    """
    sys_txt = system if system is not None else corpus.skeleton_prompt()
    msgs = [{"role": "system", "content": sys_txt}]
    for m in (context or []):
        if m["role"] in ("user", "assistant") and m["content"] != user_msg:
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": user_msg})
    return llm.chat(profile, msgs, **( {"model": model} if model else {}))


def strip_tagged(text: str) -> str:
    """去掉情绪标签，用于语义比较（标签是风格，不该进 embedding 对比）。"""
    return TAG_RE.sub("", text).strip()
