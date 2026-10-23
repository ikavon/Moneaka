"""一轮对话的编排：编剧 → C → 润色 → A → B。

为什么单独抽一个模块：CLI 和 Streamlit 界面必须是**同一段逻辑**。任何一处手抄
都会让「界面上看起来好的」和「评测里量到的」变成两套系统——这正是之前踩过的坑
（合并基座读错、评测复用旧输出）。

调用方只负责准备 `Runtime`（模型和 LLM 客户端都可以缓存），本模块不持有状态。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import audit, screenwriter
from .destyle_tags import parse_tagged_plan
from .guarded import STRICT, check as guard, retag
from .localhf import GenConfig, get_model
from .normalize import normalize


@dataclass
class TurnConfig:
    base: str = "data/models/Qwen3-1.7B"
    adapter: str = "data/adapters/run8"
    use_c: bool = True          # plan 前置审查（机械，免费）
    use_c_llm: bool = False     # plan 前置审查（强模型，看历史；每轮多一次云调用）
    use_a: bool = True          # 机械闸门
    use_b: bool = True          # 台词语义复核（触发式）
    use_bank: bool = True       # 编剧先例检索
    tight: bool = True          # 编剧长度纪律
    nsfw: bool = False
    k: int = 3                  # 最多重采几次
    temperature: float = 0.6
    bank_k: int = 4
    max_new_tokens: int = 200
    #: 额外注入给编剧的「相关信息」（如 OpenAI 外壳里调用方自己的 system prompt）。
    #: 放在 known_info 里当补充材料，不顶掉编剧规范——标签白名单和长度预算必须由我们控制。
    extra_info: list[str] = field(default_factory=list)
    #: 一轮里最多花几次 B 审查（每次都是一次云调用）。不封顶的话，
    #: 「重采 4 次」会变成 4 次云端往返，交互延迟从 4 秒涨到 15 秒。
    max_audits: int = 2
    #: 重采仍失败时的兜底：``literal`` = 标签+内容原样拼（绝对忠实、没有风格）。
    #:
    #: 这里**不再回编剧**（重写 plan 或直出台词）。理由：编剧的职责是「决定说什么」，
    #: 一旦让它看到下游的失败再改，权责就乱了——它会开始迎合润色器的口味，
    #: 而且每轮多一次不可控的云调用。要约束编剧，应该在**它输出之前**约束，
    #: 而不是事后让它改。
    fallback: str = "literal"


@dataclass
class Runtime:
    """重资源。CLI/UI 各自缓存它，不要每轮重建。"""
    model: Any = None
    llm: Any = None
    bank: Any = None

    @classmethod
    def build(cls, cfg: TurnConfig, *, with_bank: bool = True) -> "Runtime":
        from .llm import LLM

        rt = cls()
        rt.model = get_model(cfg.base, adapter=cfg.adapter)
        rt.llm = LLM()
        if with_bank and cfg.use_bank:
            from .planbank import PlanBank

            rt.bank = PlanBank.from_pairs("data/pairs/split_train_run8.jsonl")
        return rt


@dataclass
class TurnResult:
    user: str
    plan: str = ""
    tags: list[str] = field(default_factory=list)
    contents: list[str] = field(default_factory=list)
    c_flags: list[str] = field(default_factory=list)
    c_repaired: bool = False
    polish: str = ""
    attempts: int = 0
    a_flags: list[str] = field(default_factory=list)
    b_verdict: str = ""
    b_note: str = ""
    b_trigger: list[str] = field(default_factory=list)
    degraded: bool = False
    candidates: list[str] = field(default_factory=list)
    #: 重采也救不回来 → 已退化为直译（仅供观测）
    degraded_to_literal: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    error: str = ""
    #: 最终这一句是谁写的：polisher / literal
    final_source: str = "polisher"
    #: 这一轮的内部记录，供下一轮交给编剧（agent 化的连续历史）
    log_line: str = ""
    #: 是否靠「按位重贴标签」救回来的（说明模型标签不稳，但内容可用）
    retagged: bool = False
    #: 模型吐出空串的次数（解码异常/被截断）。>0 说明本地服务可能有问题
    empty_outputs: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.polish) and not self.error


def literal(plan: str) -> str:
    """退化：标签与内容原样拼，不加风格。宁可平淡，不能改意思。"""
    tags, contents = parse_tagged_plan(plan)
    return " ".join(f"[{t}]{c}" for t, c in zip(tags, contents))


def _plan_stage(rt: Runtime, cfg: TurnConfig, user: str,
                history: list[dict] | None, res: TurnResult,
                log: list[str] | None = None) -> None:
    """编剧 + C。C 拦下的重写一次（把拦截原因回灌给编剧）。"""
    t0 = time.time()
    examples = rt.bank.search(user, k=cfg.bank_k) if rt.bank else None
    hint = ""
    for round_ in (1, 2):
        try:
            got = screenwriter.make_plan(
                rt.llm, user, history=history, examples=examples, log=log,
                known_info=cfg.extra_info or None,
                tight=cfg.tight, nsfw=cfg.nsfw, retries=1)
        except Exception as exc:  # noqa: BLE001
            res.error = f"编剧失败: {str(exc)[:160]}"
            res.timings["编剧"] = time.time() - t0
            return
        res.plan, res.tags, res.contents = got["plan"], got["tags"], got["contents"]
        res.c_repaired = res.c_repaired or got.get("repaired", False)
        # 必须把上一轮她说过的话传进去，否则「复读」那条检查永远不触发（曾是这样）
        prev = _last_assistant(history)
        flags = audit.check_plan_mechanical(res.plan, prev_reply=prev) if cfg.use_c else []
        if cfg.use_c_llm and not flags:
            # 机械查不出的（答非所问、承接错对象、把别人的事当成自己的）交给模型，
            # 但必须带历史——只给最后一句时实测真人 plan 误报 20%。
            v = audit.audit_plan(rt.llm, res.plan, user, history=history,
                                 canon=chr(10).join(screenwriter.load_canon()))
            if not v.ok:
                flags = [f"审查: {v.note}"]
        res.c_flags = flags
        if not flags or round_ == 2:
            break
        hint = "；".join(flags)
        print(f"  C 拦下: {hint} → 让编剧重写")
        # 带着拦截原因重来一次：把它塞进用户消息，比改 system 更不容易被忽略
        user = user + f"\n\n（上一条表达指令被驳回：{hint}。请重写，规避该问题。）"
    res.timings["编剧"] = time.time() - t0


def _log_line(res: TurnResult) -> str:
    """一轮的压缩记录（一行），下一轮交给编剧看自己做过什么。"""
    if res.error:
        return f"- 上一轮「{res.user[:14]}」失败: {res.error[:30]}"
    tags = " > ".join(res.tags) if res.tags else "?"
    tail = []
    if res.c_flags:
        tail.append("审查驳回: " + res.c_flags[0][:24])
    if res.degraded:
        tail.append("退化为直译")
    elif res.attempts:
        tail.append(f"重写{res.attempts}次")
    return (f"- 上一轮「{res.user[:14]}」你写的是 语气 {tags}"
            f"（{len(res.contents)}段）；" + ("；".join(tail) if tail else "一次通过"))


def _last_assistant(history: list[dict] | None) -> str | None:
    for m in reversed(history or []):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            return m["content"]
    return None


def _polish_stage(rt: Runtime, cfg: TurnConfig, res: TurnResult) -> None:
    """润色 + A + B。失败时**把审查理由回灌**再重写，而不是盲目重采。

    为什么改：原先只是用温度 0.6 再抽一次，B 辛辛苦苦判出来的「帮你→帮我」被丢掉了。
    盲重采能救回偶发错误，救不了**系统性**错误（同一个 plan 每次都错在同一个地方）。
    把理由写回 prompt 才是对症：和编剧那边的自修复是同一套思路。

    仍然失败时退化为直译（``res.final_source = "literal"``），**不回编剧**。
    两次都渲染不出来说明 plan 有问题，但改 plan 必须在编剧输出**之前**做，
    不能让它看完下游结果再改（权责会乱，且多一次不可控的云调用）。
    """
    t0 = time.time()
    from .prompts import restyle_system

    base_user = f"表达指令:\n{res.plan}\n\n莫妮卡会说:"
    sys_msg = restyle_system(few_shot=False)
    greedy = GenConfig(temperature=0.0, max_new_tokens=cfg.max_new_tokens)
    samp = GenConfig(temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens)

    best: dict | None = None
    best_seen = ""
    audits = 0
    hint = ""
    for attempt in range(cfg.k + 1):
        user = base_user if not hint else (
            base_user + f"\n\n（上一版有问题：{hint}。请重写这条台词，"
                        f"意思严格照指令，不要改动谁给谁做事。）\n莫妮卡会说:")
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": user}]
        raw = rt.model.chat(msgs, greedy if attempt == 0 else samp)
        text, _ = normalize(raw)
        if not text.strip():
            # 模型偶尔吐空串（截断/解码异常）。必须显式记一笔，否则会一路走到
            # 「退化直译」甚至返回空台词，调用方看到的是「她没说话」。
            res.empty_outputs += 1
            continue
        a_flags = [p for p in guard(res.plan, text) if p.startswith(STRICT)] if cfg.use_a else []
        if a_flags and all(p.startswith("标签") for p in a_flags):
            # 内容对、只是标签错 → 按位重贴，不要浪费一次重采，更不要退化直译
            fixed = retag(res.plan, text)
            if fixed:
                text, a_flags = fixed, []
                res.retagged = True
        trigger = audit.needs_polish_audit(res.plan, text) if cfg.use_b else []
        if trigger and audits >= cfg.max_audits:
            trigger = []          # 预算用完：本轮不再送审，只留机械检查

        b_flags: list[str] = []
        audited = False
        if trigger:
            audits += 1
            audited = True
            v = audit.audit_polish(rt.llm, res.plan, text)
            if not v.ok:
                b_flags = [f"{v.kind}: {v.note}"]
            else:
                best_seen = "ok"
        res.candidates.append(text)
        score = len(a_flags) + len(b_flags)
        if best is None or score < best["score"]:
            best = {"polish": text, "a": a_flags, "b": b_flags, "score": score,
                    "attempts": attempt, "trigger": trigger}
        if score == 0:
            break
        # 把这一版的失败原因带进下一版（机械问题优先，它更确定）
        hint = "；".join(a_flags + b_flags)[:80]

    # 关键：**交付前一定要审一次**。max_audits 的预算很容易被前面的废候选吃光，
    # 结果真正交付的那个候选一次都没审过——真实对话第 5 轮就这么漏掉了内容丢失。
    if best is not None and cfg.use_b and not best.get("audited"):
        trigger = audit.needs_polish_audit(res.plan, best["polish"])
        if trigger:
            v = audit.audit_polish(rt.llm, res.plan, best["polish"])
            if not v.ok:
                best["b"] = [f"{v.kind}: {v.note}"]
                best["score"] = len(best["a"]) + 1
            else:
                best_seen = "ok"
            best["audited"] = True

    if best is None:      # 所有尝试都是空串
        res.error = f"润色器连续 {cfg.k + 1} 次输出空串"
        res.polish = literal(res.plan)
        res.degraded = True
        res.final_source = "literal"
        res.timings["润色+A+B"] = time.time() - t0
        return
    res.polish, res.a_flags, res.attempts = best["polish"], best["a"], best["attempts"]
    res.b_trigger = best.get("trigger") or []
    res.b_verdict = best_seen if not best["b"] else res.b_verdict
    if best["b"]:
        res.b_verdict, res.b_note = best["b"][0].split(":", 1)[0], best["b"][0].split(":", 1)[-1].strip()
    if best["score"]:
        # 重采也救不回来 → 不是措辞问题，是 plan 让下游无法忠实表达
        res.degraded_to_literal = True
        res.polish = literal(res.plan)
        res.degraded = True
        res.final_source = "literal"
    res.timings["润色+A+B"] = time.time() - t0


def run_turn(rt: Runtime, cfg: TurnConfig, user: str,
             history: list[dict] | None = None, on_stage=None,
             log: list[str] | None = None) -> TurnResult:
    """走完一轮。``history`` 只给编剧看，润色器始终无状态。

    ``on_stage(name)`` 是给界面用的进度回调（CLI 不用），在每层开始时调用一次。
    """
    def step(name: str) -> None:
        if on_stage:
            on_stage(name)

    res = TurnResult(user=user)
    if not user.strip():
        res.error = "空输入"
        return res
    step("编剧")
    _plan_stage(rt, cfg, user, history, res, log=log)
    if res.error:
        return res
    step("润色 + 审查")
    _polish_stage(rt, cfg, res)
    res.log_line = _log_line(res)

    return res