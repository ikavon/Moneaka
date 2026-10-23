"""OpenAI 兼容外壳的契约测试（假 run_turn，不碰 GPU/网络）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moneaka.openai_facade import add_openai_routes  # noqa: E402
from moneaka.turn import TurnConfig  # noqa: E402


class FakeResult:
    error = ""
    degraded = False
    attempts = 0
    c_repaired = False
    final_source = "polisher"

    def __init__(self, line):
        self.polish = line


def make(seen: list, *, token=None, line="[笑]当然啦!", error=""):
    def fake_run(rt, cfg, user, history=None, log=None):
        seen.append({"user": user, "history": history,
                     "extra": list(cfg.extra_info)})
        r = FakeResult(line)
        r.error = error
        return r

    app = FastAPI()
    add_openai_routes(app, rt=None, cfg=TurnConfig(),
                      token=token, run_turn=fake_run)
    return TestClient(app)


def test_simple_completion():
    seen: list = []
    c = make(seen)
    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "你相信我吗"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "[笑]当然啦!"
    assert body["object"] == "chat.completion"
    assert seen[0]["user"] == "你相信我吗" and seen[0]["history"] == []


def test_history_is_taken_from_messages():
    """OpenAI 协议本来就每次带全量 messages，所以服务端无需存会话。"""
    seen: list = []
    c = make(seen)
    c.post("/v1/chat/completions", json={"messages": [
        {"role": "user", "content": "第一句"},
        {"role": "assistant", "content": "[笑]嗯!"},
        {"role": "user", "content": "第二句"}]})
    assert seen[0]["user"] == "第二句"
    assert [m["content"] for m in seen[0]["history"]] == ["第一句", "[笑]嗯!"]


def test_caller_system_prompt_goes_to_extra_info():
    """调用方的人设不顶掉编剧规范，而是作为补充材料注入。"""
    seen: list = []
    c = make(seen)
    c.post("/v1/chat/completions", json={"messages": [
        {"role": "system", "content": "你是莫妮卡, 说话很甜"},
        {"role": "user", "content": "你好"}]})
    assert seen[0]["extra"] == ["你是莫妮卡, 说话很甜"]


def test_empty_user_is_422():
    c = make([])
    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "system", "content": "only system"}]})
    assert r.status_code == 422


def test_streaming_frames():
    c = make([], line="[笑]好的呀")
    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "x"}], "stream": True})
    assert r.status_code == 200
    body = r.text
    assert body.startswith("data: ")
    assert "chat.completion.chunk" in body
    assert body.rstrip().endswith("[DONE]")
    joined = "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        for line in body.splitlines() if line.startswith("data: {")
        for _ in [0])
    assert "好的呀" in joined


def test_token_guard():
    c = make([], token="secret")
    assert c.post("/v1/chat/completions",
                  json={"messages": [{"role": "user", "content": "x"}]}).status_code == 401
    assert c.post("/v1/chat/completions",
                  json={"messages": [{"role": "user", "content": "x"}]},
                  headers={"Authorization": "Bearer secret"}).status_code == 200


def test_models_endpoint():
    c = make([])
    data = c.get("/v1/models").json()
    assert data["data"][0]["id"] == "moneaka"


def test_run_error_becomes_500():
    c = make([], error="编剧失败: timeout")
    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 500
