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
from .guarded import STRICT, check as guard
from .localhf import GenConfig, get_model
from .normalize import normalize


@dataclass
class TurnConfig:
    base: str = "data/models/Qwen3-1.7B"
    adapter: str = "data/adapters/run8"
    use_c: bool = True          # plan 前置审查
    use_a: bool = True          # 机械闸门
    use_b: bool = True          # 台词语义复核（触发式）
    use_bank: bool = True       # 编剧先例检索
    tight: bool = True          # 编剧长度纪律
    nsfw: bool = False
    k: int = 3                  # 最多重采几次
    temperature: float = 0.6
    bank_k: int = 4
    max_new_tokens: int = 200
    #: 一轮里最多花几次 B 审查（每次都是一次云调用）。不封顶的话，
    #: 「重采 4 次」会变成 4 次云端往返，交互延迟从 4 秒涨到 15 秒。
    max_audits: int = 2
    #: 重采仍失败时是否回编剧重写 plan（一次）。关掉就退回纯直译。
    escalate_to_plan: bool = True
    #: 最后一道兜底由谁来出：``literal``（直译，无风格）/ ``screenwriter``（强模型直出）。
    #: 直译保证意思，但没有一丝风格；强模型直出有风格，但会偏离专用润色器。
    final_writer: str = "screenwriter"


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
    #: 重采也救不回来 → 建议回编剧重写 plan（Q10「陪你收尾」那类就是这个原因）
    need_new_plan: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    error: str = ""
    escalated: bool = False
    #: 最终这一句是谁写的：polisher / screenwriter / literal
    final_source: str = "polisher"

    @property
    def ok(self) -> bool:
        return bool(self.polish) and not self.error


def literal(plan: str) -> str:
    """退化：标签与内容原样拼，不加风格。宁可平淡，不能改意思。"""
    tags, contents = parse_tagged_plan(plan)
    return " ".join(f"[{t}]{c}" for t, c in zip(tags, contents))


def _plan_stage(rt: Runtime, cfg: TurnConfig, user: str,
                history: list[dict] | None, res: TurnResult) -> None:
    """编剧 + C。C 拦下的重写一次（把拦截原因回灌给编剧）。"""
    t0 = time.time()
    examples = rt.bank.search(user, k=cfg.bank_k) if rt.bank else None
    hint = ""
    for round_ in (1, 2):
        try:
            got = screenwriter.make_plan(
                rt.llm, user, history=history, examples=examples,
                tight=cfg.tight, nsfw=cfg.nsfw, retries=1)
        except Exception as exc:  # noqa: BLE001
            res.error = f"编剧失败: {str(exc)[:160]}"
            res.timings["编剧"] = time.time() - t0
            return
        res.plan, res.tags, res.contents = got["plan"], got["tags"], got["contents"]
        res.c_repaired = res.c_repaired or got.get("repaired", False)
        flags = audit.check_plan_mechanical(res.plan) if cfg.use_c else []
        res.c_flags = flags
        if not flags or round_ == 2:
            break
        hint = "；".join(flags)
        print(f"  C 拦下: {hint} → 让编剧重写")
        # 带着拦截原因重来一次：把它塞进用户消息，比改 system 更不容易被忽略
        user = user + f"\n\n（上一条表达指令被驳回：{hint}。请重写，规避该问题。）"
    res.timings["编剧"] = time.time() - t0


def _polish_stage(rt: Runtime, cfg: TurnConfig, res: TurnResult) -> None:
    """润色 + A + B。失败时**把审查理由回灌**再重写，而不是盲目重采。

    为什么改：原先只是用温度 0.6 再抽一次，B 辛辛苦苦判出来的「帮你→帮我」被丢掉了。
    盲重采能救回偶发错误，救不了**系统性**错误（同一个 plan 每次都错在同一个地方）。
    把理由写回 prompt 才是对症：和编剧那边的自修复是同一套思路。

    仍然失败时置 ``res.need_new_plan``，让调用方回编剧重写指令——
    两次都渲染不出来，说明问题在 plan，不在措辞。
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
        a_flags = [p for p in guard(res.plan, text) if p.startswith(STRICT)] if cfg.use_a else []
        trigger = audit.needs_polish_audit(res.plan, text) if cfg.use_b else []
        if trigger and audits >= cfg.max_audits:
            trigger = []          # 预算用完：本轮不再送审，只留机械检查

        b_flags: list[str] = []
        if trigger:
            audits += 1
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

    assert best is not None
    res.polish, res.a_flags, res.attempts = best["polish"], best["a"], best["attempts"]
    res.b_trigger = best.get("trigger") or []
    res.b_verdict = best_seen if not best["b"] else res.b_verdict
    if best["b"]:
        res.b_verdict, res.b_note = best["b"][0].split(":", 1)[0], best["b"][0].split(":", 1)[-1].strip()
    if best["score"]:
        # 重采也救不回来 → 不是措辞问题，是 plan 让下游无法忠实表达
        res.need_new_plan = True
        res.polish = literal(res.plan)
        res.degraded = True
    res.timings["润色+A+B"] = time.time() - t0


def run_turn(rt: Runtime, cfg: TurnConfig, user: str,
             history: list[dict] | None = None, on_stage=None) -> TurnResult:
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
    _plan_stage(rt, cfg, user, history, res)
    if res.error:
        return res
    step("润色 + 审查")
    _polish_stage(rt, cfg, res)

    # 渲染两次都失败 → 回编剧重写一次。只重写一次，避免来回打转。
    if res.need_new_plan and cfg.escalate_to_plan:
        step("编剧重写")
        hint = "；".join(res.a_flags + [f"{res.b_verdict} {res.b_note}".strip()]).strip()
        user2 = user + (f"\n\n（上一条表达指令下游无法忠实表达，问题：{hint}。"
                        f"请把它改成更好说出口的说法，尤其是谁给谁做事要写清楚。）")
        res.need_new_plan = False
        res.escalated = True
        before = res.plan
        _plan_stage(rt, cfg, user2, history, res)
        if res.error:
            res.plan, res.tags, res.contents = before, [], []
            res.error = ""
            return res
        step("润色 + 审查")
        _polish_stage(rt, cfg, res)

    # 仍然失败 → 最后一道兜底。默认让强模型直出（有风格），可选退回直译（绝对忠实）。
    if res.degraded and cfg.final_writer == "screenwriter":
        step("编剧直出")
        try:
            text = screenwriter.render_direct(rt.llm, res.plan)
            text, _ = normalize(text)
            flags = [p for p in guard(res.plan, text) if p.startswith(STRICT)]
            if not flags:
                res.polish = text
                res.degraded = False
                res.final_source = "screenwriter"
        except Exception:  # noqa: BLE001  兜底失败就保持直译
            pass
    return res