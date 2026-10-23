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
    timings: dict[str, float] = field(default_factory=dict)
    error: str = ""

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
    """润色 + A + B，失败就重采，最后退化直译。"""
    t0 = time.time()
    msgs = [{"role": "user", "content": f"表达指令:\n{res.plan}\n\n莫妮卡会说:"}]
    from .prompts import restyle_system

    msgs = [{"role": "system", "content": restyle_system(few_shot=False)}, *msgs]
    greedy = GenConfig(temperature=0.0, max_new_tokens=cfg.max_new_tokens)
    samp = GenConfig(temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens)

    best: dict | None = None
    best_seen = ""
    audits = 0
    for attempt in range(cfg.k + 1):
        raw = rt.model.chat([dict(m) for m in msgs], greedy if attempt == 0 else samp)
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

    assert best is not None
    res.polish, res.a_flags, res.attempts = best["polish"], best["a"], best["attempts"]
    res.b_trigger = best.get("trigger") or []
    res.b_verdict = best_seen if not best["b"] else res.b_verdict
    if best["b"]:
        res.b_verdict, res.b_note = best["b"][0].split(":", 1)[0], best["b"][0].split(":", 1)[-1].strip()
    if best["score"]:
        res.degraded = True
        res.polish = literal(res.plan)
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
    return res