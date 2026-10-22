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


def polish(llm: LLM, plan: str, *, profile: str = "restyle", model: str | None = None,
           adapter: str | None = None, few_shot: bool | None = None,
           context: list | None = None, user: str | None = None) -> str:
    """把表达指令改写成莫妮卡口吻。

    ``context``/``user`` 是对话历史。标签选择是上下文的函数，不喂就只能退回
    众数（实测错例清一色选「微笑」）。训练样本用的就是带历史的输入，
    推理不一致会直接掉性能。

    ``few_shot`` 默认自动：基座模型需要例子引导，挂了 adapter 就不需要。
    **训练与推理必须走同一个分支**，否则分布不一致。
    """
    if few_shot is None:
        few_shot = adapter is None
    text = llm.complete(profile, prompts.restyle_system(few_shot=few_shot),
                        prompts.RESTYLE_USER_BODY.format(
                            context=prompts.render_context(context, user), plan=plan),
                        **({"model": model} if model else {}),
                        **({"adapter": adapter} if adapter else {}))
    # 润色器不许输出外壳：去掉可能的「改写：」前缀与包裹引号
    return text.strip().strip('"\'')


def direct(llm: LLM, user_msg: str, context: list[dict] | None = None,
           *, profile: str = "restyle", model: str | None = None,
           system: str | None = None, temperature: float = 0.0) -> str:
    """按官方骨架 prompt 直接扮演（对照组）。system 默认取数据里的骨架原文。

    ``profile`` 决定是谁在演：本地小模型（衡量「润色器 vs 小模型自己演」）
    或强模型（衡量「润色器 vs 通用大模型直接上」）——后者才是真正的竞品。

    ``temperature`` 默认 0：对照组是**基准线**，两次生成不同会让 style 分
    飘 0.5 左右（实测 2.2 vs 2.7，0/60 条相同），那就没法比较了。
    """
    sys_txt = system if system is not None else corpus.skeleton_prompt()
    msgs = [{"role": "system", "content": sys_txt}]
    for m in (context or []):
        if m["role"] in ("user", "assistant") and m["content"] != user_msg:
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": user_msg})
    overrides = {"temperature": temperature}
    if model:
        overrides["model"] = model
    return llm.chat(profile, msgs, **overrides)


def strip_tagged(text: str) -> str:
    """去掉情绪标签，用于语义比较（标签是风格，不该进 embedding 对比）。"""
    return TAG_RE.sub("", text).strip()
