"""OpenAI 兼容外壳：让现成的聊天客户端把 Moneaka 当成一个「模型」用。

## 为什么需要它

真实场景测试的障碍不是润色器，是**没有 agent**。而几乎所有聊天客户端
（QQ 机器人框架、SillyTavern、各种 MAS 桥接、LM Studio 风格的工具）都讲
OpenAI 协议。把服务伪装成一个模型，就不必先写一个 agent：

    base_url = http://127.0.0.1:8010/v1
    model    = moneaka
    api_key  = <token>

## 为什么这在架构上不违和

OpenAI 的 chat 请求**每次都带完整 messages**——也就是说「历史」本来就在调用方手里。
我们的服务端仍然是零状态，只是把 messages 里最后一句当玩家输入、前面的当历史。
这跟 `docs/PIPELINE_SINGLE_TURN.md` 里「记忆归编剧」是一致的，没有偷偷存会话。

## 一点妥协

调用方自己的 system prompt（人设）我们不会丢，但也不让它顶掉我们的编剧规范——
它会被放进「相关信息」里当补充材料。真正决定「说什么」的仍然是我们这边的编剧，
因为它要吃事实卡、长度预算、标签白名单。想完全接管的话应该直接调 `/polish`。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "moneaka"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


def _split(messages: list[ChatMessage]) -> tuple[str, list[dict], str]:
    """拆成 (玩家最后一句, 历史, 调用方的人设)。"""
    system = "\n".join(m.content for m in messages if m.role == "system").strip()
    dialogue = [{"role": m.role, "content": m.content}
                for m in messages if m.role in ("user", "assistant")]
    user = ""
    for m in reversed(dialogue):
        if m["role"] == "user":
            user = m["content"]
            break
    history = dialogue[:-1] if user and dialogue[-1]["role"] == "user" else dialogue
    return user, history, system


def add_openai_routes(app: FastAPI, rt, cfg, *, token: str | None = None,
                      run_turn=None) -> None:
    """把 /v1/models 与 /v1/chat/completions 挂到已有 app 上。

    ``rt`` / ``cfg`` 是 `moneaka.turn` 的运行时与基准配置；``run_turn`` 可注入
    （测试时用假的，不需要 GPU）。
    """
    import dataclasses

    if run_turn is None:
        from .turn import run_turn as run_turn_impl

        run_turn = run_turn_impl

    def _auth(authorization: str | None) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="bad token")

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)):
        _auth(authorization)
        return {"object": "list", "data": [
            {"id": "moneaka", "object": "model", "owned_by": "moneaka",
             "created": int(time.time())}]}

    def _one(req: ChatRequest, authorization: str | None) -> tuple[str, dict]:
        _auth(authorization)
        user, history, system = _split(req.messages)
        if not user:
            raise HTTPException(status_code=422, detail="messages 里没有 user 内容")
        use_cfg = dataclasses.replace(cfg, extra_info=[system]) if system else cfg
        res = run_turn(rt, use_cfg, user, history=history)
        if res.error:
            raise HTTPException(status_code=500, detail=res.error)
        meta = {"degraded": res.degraded, "attempts": res.attempts,
                "c_repaired": res.c_repaired, "final_source": res.final_source}
        return res.polish, meta

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        line, meta = _one(req, authorization)
        created = int(time.time())

        if not req.stream:
            return {
                "id": rid, "object": "chat.completion", "created": created,
                "model": req.model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": line}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
                "moneaka": meta,
            }

        def gen() -> Iterator[str]:
            head = {"id": rid, "object": "chat.completion.chunk", "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"},
                                 "finish_reason": None}]}
            yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n"
            # 按 8 个字符切块：客户端期望的是流式，我们这边是一次生成完，
            # 但切成块能让打字机效果正常显示，也避免一次性吐出长文本卡住。
            for i in range(0, len(line), 8):
                chunk = {"id": rid, "object": "chat.completion.chunk",
                         "created": created, "model": req.model,
                         "choices": [{"index": 0, "finish_reason": None,
                                      "delta": {"content": line[i:i + 8]}}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            tail = {"id": rid, "object": "chat.completion.chunk", "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "moneaka": meta}
            yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")