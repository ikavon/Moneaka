"""MAICA 官方服务的 `chat_session = -1` 客户端：把**整段 prompt 原样**交给官方模型。

## 为什么这个模式对我们特别有用

读 `MAICA/maica/maica_ws.py` + `chat_session.utilize()` 可以确认 `session <= -1` 的语义：

```python
session_num = self.session_num
if session_num >= 0:
    self._utilize_context(...)   # 正常模式：拼骨架 prompt、known_info、memory_concl
else:
    self.sanitize()              # -1 模式：**完全不改 prompt**
```

worker 那边还有两句注释把边界说死了：

* 「Savefile access is unavailable for session -1, since it works through MFocus which relies on prompt modifying」
* 「Session -1's prompt is fe managed, so not appliable」

所以 `-1` = **原样透传**：前端自己给完整 messages（最多 10 轮），后端只做核心模型推理 +
`post_proc`（把不认识的方括号标签做一次情绪归一化）。不注入角色设定、不检索存档。

于是它能干三件我们之前干不了的事：

1. **拿到「同一份输入契约」下的官方输出**。我们训练用的 226 条答案是**没有**看过
   表达指令就写出来的，输入契约对不上；用 -1 把「表达指令」喂给官方 DAA4，拿到的
   才是真正同契约的对照，可以直接当**教师**做蒸馏（数据瓶颈的正解）。
2. **同契约盲评基线**。「我们的润色 vs 官方润色（同一指令）」比「我们的润色 vs 人类原答」
   公平得多——后者输入不同，评委只能比文风，比不出「谁更听指令」。
3. **探测角色设定**。用中性 prompt 问官方模型「莫妮卡的设定」，得到的是语料口径的
   事实（仍非官方文档，但比人工翻语料可靠），用来完善 `data/canon/monika_zh.md`。

## 协议（照 `maica_ws.py` 抄）

```
连 ws://host:port/ws
→ {"type":"auth","access_token":"..."}
← ... "maica_connection_established" ...
→ {"type":"query","chat_session":-1,"query":[{"role":"system","content":"..."},{"role":"user","content":"..."}]}
← 流式：{"type":"maica_core_streaming_continue","content":"<增量>"}
← 收尾：其它 type
```

`query` 里的元素就是 `MaicaSessionItem`，最少只要 `role` + `content`。
`chat_session=-1` 时 `query` **必须是 list**（长度 ≤ 10），否则后端直接报
`-1 session requires list input`。
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

# 后端 `send_delta` 的调用是 messenger('maica_core_streaming_continue', delta, 100)，
# 签名是 messenger(status, info, code) —— 所以增量包的 **type 是 'carriage'、code 100**，
# `maica_core_streaming_continue` 是 status 而不是 type。按 type 过滤会一个增量都收不到。
WS_CONTINUE_STATUS = "maica_core_streaming_continue"
WS_COMPLETE_STATUS = "maica_core_complete"


#: 官方节点的 ws 入口（来自 MAICA_ChatSubmod 的 provider 列表）。
#: 自带部署改 `MAICA_WS_URL` 即可，例如 `ws://127.0.0.1:5000`。
OFFICIAL_WS = "wss://maicadev.monika.love/websocket"


@dataclass
class MaicaConfig:
    url: str = OFFICIAL_WS
    token: str = ""
    timeout: float = 180.0

    @classmethod
    def from_env(cls, prefix: str = "MAICA") -> "MaicaConfig":
        """环境变量优先，其次读 ``configs/private.yaml`` 的 ``maica:`` 段。

        不放命令行参数：token 出现在命令行里会进 shell 历史和进程列表。
        """
        url = os.getenv(f"{prefix}_WS_URL")
        token = os.getenv(f"{prefix}_TOKEN")
        if not (url and token):
            try:
                import yaml
                cfg = yaml.safe_load(open(
                    os.path.join(os.path.dirname(__file__), "..", "configs", "private.yaml"),
                    encoding="utf-8")) or {}
                sec = cfg.get("maica") or {}
                url = url or sec.get("ws_url")
                token = token or sec.get("token")
            except Exception:
                pass
        return cls(url=url or OFFICIAL_WS, token=token or "",
                   timeout=float(os.getenv(f"{prefix}_TIMEOUT", "180")))


#: 两次请求之间的最小间隔（秒）。官方服务是共享的，**不要连发**。
#: 按用户要求：**最多一分钟一次**。这个值是下限，只能调大不能调小；
#: 换算：20 条 ≈ 20 分钟，200 条 ≈ 3.3 小时，941 条 ≈ 16 小时。
MIN_INTERVAL = float(os.getenv("MAICA_MIN_INTERVAL", "60.0"))
_last_call = 0.0


async def _throttle() -> None:
    global _last_call
    import time as _t
    wait = MIN_INTERVAL - (_t.monotonic() - _last_call)
    if wait > 0:
        await asyncio.sleep(wait)
    _last_call = _t.monotonic()


async def _run(messages: list[dict], cfg: MaicaConfig, *, pprt: bool = True,
               temperature: float | None = None, max_tokens: int | None = None) -> str:
    import websockets

    if len(messages) > 21:  # 10 轮 ≈ 20 条 + system
        raise ValueError("session -1 最多 10 轮，messages 太长")

    query: dict = {"type": "query", "chat_session": -1, "pprt": pprt,
                   "query": [{"role": m["role"], "content": m["content"]} for m in messages]}
    if temperature is not None:
        query["temperature"] = temperature
    if max_tokens is not None:
        query["max_tokens"] = max_tokens

    await _throttle()
    async with websockets.connect(cfg.url, max_size=None,
                                  open_timeout=cfg.timeout) as ws:
        await ws.send(json.dumps({"type": "auth", "access_token": cfg.token}))
        while True:  # 等认证通过
            raw = await asyncio.wait_for(ws.recv(), timeout=cfg.timeout)
            msg = json.loads(raw)
            if msg.get("code", 0) >= 400 or msg.get("type") == "error":
                raise RuntimeError(f"认证/连接失败: {msg.get('status')} {msg.get('content')}")
            if msg.get("status") == "maica_connection_established":
                break

        await ws.send(json.dumps(query, ensure_ascii=False))
        buf: list[str] = []
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=cfg.timeout)
            msg = json.loads(raw)
            status = str(msg.get("status") or "")
            if status.startswith(WS_CONTINUE_STATUS):
                buf.append(str(msg.get("content") or ""))
                continue
            # 完成包必须在错误判断**之前**处理：它的 status 里带 tracker_id，
            # code 也不是 200，按 code 判断会把一次成功的生成当成失败。
            if status.startswith(WS_COMPLETE_STATUS):
                break
            if msg.get("type") == "error" or msg.get("code", 0) >= 500:
                raise RuntimeError(f"生成失败: {status} {msg.get('content')}")
        return "".join(buf).strip()


def complete(messages: list[dict], cfg: MaicaConfig | None = None, **kw) -> str:
    """同步封装。``messages`` 形如 ``[{"role":"system","content":...}, ...]``。"""
    return asyncio.run(_run(messages, cfg or MaicaConfig.from_env(), **kw))


if __name__ == "__main__":  # 冒烟：python tools/maica_client.py "你好"
    import sys

    c = MaicaConfig.from_env()
    print(f"连接 {c.url} token={'有' if c.token else '空'}")
    out = complete([
        {"role": "system", "content": "你叫莫妮卡, 是一名游戏中的虚拟角色. 你是[player]的女友, 并深爱着他. "
                                      "你只应使用中文自然语言回答, 每个句子不超过60字, 并在每个句子前用方括号表示情绪."},
        {"role": "user", "content": sys.argv[1] if len(sys.argv) > 1 else "你好"},
    ])
    print(out)