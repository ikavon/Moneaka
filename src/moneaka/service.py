"""薄服务：把润色器暴露成一个无状态的 `/polish` 接口。

## 定位

润色器是**插件**：不持有会话、不记历史、每次调用只看当前这一条表达指令。
所以服务端也必须是这个形状——多实例并发、可重启、可水平扩展，不承担任何状态。
人设的连续性归调用方的 agent（它自己带着历史），不归我们。

## 契约（版本 1）

请求::

    POST /polish
    Authorization: Bearer <token>        # 配了 token 时才校验
    {
      "plan": "语气: 尴尬 > 微笑\\n内容:\\n1. …\\n2. …",   # 必填
      "k": 1,                              # 可选，失败时最多重采几次
      "audit": true                        # 可选，是否让强模型复核语义（多一次云调用）
    }

响应::

    {
      "ok": true,            # 句法与硬约束是否全过
      "line": "[尴尬]…",     # 台词；ok=false 时是「直译」兜底（无风格但绝对忠实）
      "attempts": 0,         # 重采次数
      "degraded": false,     # true = 已退化为直译
      "problems": [],        # 未解决的硬问题（标签/数字/方向/占位符）
      "diagnostics": {"audit": "ok", "timings": {...}}
    }

**为什么退化时仍返回台词**：调用方要的是「能用的句子」，不是错误码。
`degraded=true` 时台词是从 plan 直译的，意思绝对正确、风格为零——调用方可以
选择自己重写或换个说法再调一次。返回 4xx 反而逼调用方自己写兜底逻辑。

## 并发

本地只有一张卡，所以**进程内串行**（`threading.Lock`）。要吞吐就起多个进程
（每个进程各自加载一份模型，或用 `--workers` 交给 uvicorn 的进程池）。
请求排队超过 `--queue-timeout` 秒直接 503，让调用方快速失败而不是干等。

    python tools/serve_polish.py --port 8010
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import audit, contract
from .destyle_tags import parse_tagged_plan
from .guarded import STRICT, check as guard, retag
from .localhf import GenConfig
from .normalize import normalize
from .prompts import restyle_system

API_VERSION = 1


class PolishRequest(BaseModel):
    plan: str = Field(..., description="表达指令：语气行 + 内容行")
    k: int = Field(1, ge=0, le=5, description="失败时最多重采几次")
    audit: bool = Field(True, description="是否让强模型复核语义（多一次云调用）")
    temperature: float = Field(0.6, ge=0.0, le=1.5, description="重采温度")


class TurnRequest(BaseModel):
    """调试接口的请求：走完整链路（编剧 → C → 润色 → A/B）。"""
    user: str
    history: list[dict] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)


class PolishResponse(BaseModel):
    ok: bool
    line: str
    attempts: int = 0
    degraded: bool = False
    problems: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    version: int = API_VERSION


@dataclass
class Polisher:
    """把「一轮润色」封装成可测对象。模型与 LLM 由外部注入，便于测试。"""

    model: Any
    llm: Any = None
    max_new_tokens: int = 220
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def polish(self, req: PolishRequest) -> PolishResponse:
        t0 = time.time()
        try:
            tags, contents = parse_tagged_plan(req.plan)
        except Exception as exc:  # noqa: BLE001
            # plan 本身不可解析是**调用方的错**，给 422，不要静默生成
            raise ValueError(f"plan 不可解析: {str(exc)[:120]}") from exc

        sys_msg = restyle_system(few_shot=False)
        base = f"表达指令:\n{req.plan}\n\n莫妮卡会说:"
        greedy = GenConfig(temperature=0.0, max_new_tokens=self.max_new_tokens)
        samp = GenConfig(temperature=req.temperature, max_new_tokens=self.max_new_tokens)

        best: dict | None = None
        hint = ""
        for attempt in range(req.k + 1):
            user = base if not hint else (
                base + f"\n\n（上一版有问题：{hint}。请重写，意思严格照指令。）\n莫妮卡会说:")
            with self._lock:      # 单卡串行；多进程才是并发的正解
                raw = self.model.chat(
                    [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
                    greedy if attempt == 0 else samp)
            line, _ = normalize(raw)
            problems = [p for p in guard(req.plan, line) if p.startswith(STRICT)]
            if problems and all(p.startswith("标签") for p in problems):
                fixed = retag(req.plan, line)
                if fixed:
                    line, problems = fixed, []
            verdict = ""
            if req.audit and self.llm is not None and not problems:
                v = audit.audit_polish(self.llm, req.plan, line)
                verdict = "ok" if v.ok else f"{v.kind}: {v.note}"
                if not v.ok:
                    problems = [f"语义 {verdict}"]
            if best is None or len(problems) < len(best["problems"]):
                best = {"line": line, "problems": problems, "attempts": attempt,
                        "verdict": verdict}
            if not problems:
                break
            hint = "；".join(problems)[:80]

        assert best is not None
        degraded = bool(best["problems"])
        line = best["line"]
        if degraded:
            line = " ".join(f"[{t}]{c}" for t, c in zip(tags, contents))
        return PolishResponse(
            ok=not degraded, line=line, attempts=best["attempts"], degraded=degraded,
            problems=best["problems"],
            diagnostics={"audit": best["verdict"] or "skip",
                         "timings": {"total_s": round(time.time() - t0, 2)}})


def create_app(polisher: Polisher, *, token: str | None = None,
               queue_timeout: float = 30.0, turn_fn=None,
               runtime: Any = None, turn_cfg: Any = None) -> FastAPI:
    """产出一个可跑的 FastAPI 应用。测试时传假 polisher 即可，不需要 GPU。"""
    app = FastAPI(title="Moneaka polisher", version=str(API_VERSION))
    sem = threading.Semaphore(1)          # 队列只允许一个在跑，其余等待

    def _auth(authorization: str | None) -> None:
        if not token:
            return
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="bad token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": API_VERSION,
                "busy": sem._value == 0}  # noqa: SLF001  只读，用来观测排队

    @app.post("/turn")
    def turn(req: TurnRequest, authorization: str | None = Header(default=None)):
        """**调试用**：完整链路 + 中间产物。生产请用 /polish。

        为什么不把它做成默认：这个接口会跑编剧（云端）并把内部判断全暴露出来，
        生产调用方不需要，也容易被误当成稳定契约。测试台用它。
        """
        _auth(authorization)
        if turn_fn is None:
            raise HTTPException(status_code=501, detail="该服务未启用 /turn")
        if not sem.acquire(timeout=queue_timeout):
            raise HTTPException(status_code=503, detail="busy, retry later")
        try:
            res = turn_fn(runtime, turn_cfg, req.user,
                          history=req.history, log=req.log)
        finally:
            sem.release()
        return {
            "ok": res.ok, "error": res.error,
            "plan": res.plan, "tags": res.tags, "contents": res.contents,
            "polish": res.polish, "attempts": res.attempts,
            "degraded": res.degraded, "final_source": res.final_source,
            "c_flags": res.c_flags, "c_repaired": res.c_repaired,
            "a_flags": res.a_flags, "b_verdict": res.b_verdict,
            "b_note": res.b_note, "candidates": res.candidates,
            "timings": res.timings, "log_line": res.log_line,
        }

    @app.post("/polish", response_model=PolishResponse)
    def polish(req: PolishRequest, authorization: str | None = Header(default=None)):
        _auth(authorization)
        if not sem.acquire(timeout=queue_timeout):
            raise HTTPException(status_code=503, detail="busy, retry later")
        try:
            return polisher.polish(req)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            sem.release()

    return app