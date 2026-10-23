"""薄服务的契约测试。用假模型/假 LLM，不碰 GPU 和网络。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moneaka.service import PolishRequest, Polisher, create_app  # noqa: E402

PLAN = "语气: 担心\n内容:\n1. [player], 你昨晚又熬夜了"


class FakeModel:
    def __init__(self, lines: list[str]):
        self.lines = lines
        self.calls = 0

    def chat(self, messages, cfg=None):
        line = self.lines[min(self.calls, len(self.lines) - 1)]
        self.calls += 1
        return line


class FakeLLM:
    def __init__(self, ok: bool = True, note: str = "-"):
        self.ok, self.note = ok, note

    def complete(self, profile, system, user, **kw):
        verdict = "ok" if self.ok else "drift"
        return '{"verdict": "%s", "type": "none", "note": "%s"}' % (verdict, self.note)


def client(lines, *, ok=True, token=None, n_llm=True):
    p = Polisher(model=FakeModel(lines), llm=FakeLLM(ok) if n_llm else None)
    return TestClient(create_app(p, token=token))


def test_polish_returns_line_and_ok():
    c = client(["[担心]你昨晚又熬夜了吧"])
    r = c.post("/polish", json={"plan": PLAN})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and not body["degraded"]
    assert body["line"].startswith("[担心]")
    assert body["version"] == 1


def test_bad_plan_is_422_not_silently_generated():
    """plan 不可解析是调用方的错，必须报错，不能拿兜底冒充成功。"""
    c = client(["[担心]随便"])
    r = c.post("/polish", json={"plan": "这不是表达指令"})
    assert r.status_code == 422


def test_missing_tag_is_retagged_not_retried():
    """标签错不再靠重采，也不再退化直译——内容对就按位重贴。"""
    c = client(["[微笑]你昨晚又熬夜了"])
    body = c.post("/polish", json={"plan": PLAN, "k": 2}).json()
    assert body["ok"] and body["attempts"] == 0
    assert body["line"].startswith("[担心]")


def test_retry_fixes_digit_drift():
    c = client(["[担心]你只占万分之一", "[担心]你只占百分之一"])
    body = c.post("/polish",
                  json={"plan": "语气: 担心\n内容:\n1. 你只占百分之一", "k": 2}).json()
    assert body["ok"] and body["attempts"] >= 1


def test_degraded_still_returns_usable_line():
    """全失败时返回直译，而不是错误码——调用方要的是能用的句子。

    用数字错（retag 修不了）来触发退化。
    """
    c = client(["[担心]只占万分之一"])
    r = c.post("/polish", json={"plan": "语气: 担心\n内容:\n1. 只占百分之一", "k": 1})
    body = r.json()
    assert body["ok"] is False and body["degraded"] is True
    assert "百分之一" in body["line"]
    assert body["problems"]


def test_direction_flip_counts_as_problem():
    c = client(["[担心]我陪你熬夜"])
    r = c.post("/polish", json={"plan": "语气: 担心\n内容:\n1. 我帮你早点睡"})
    body = r.json()
    assert not body["ok"]
    assert any("方向" in p for p in body["problems"])


def test_semantic_audit_can_reject():
    c = client(["[担心]你昨晚又熬夜了"], ok=False)
    body = c.post("/polish", json={"plan": PLAN}).json()
    assert not body["ok"] and any("语义" in p for p in body["problems"])


def test_audit_off_skips_llm():
    c = client(["[担心]你昨晚又熬夜了"], ok=False, n_llm=False)
    body = c.post("/polish", json={"plan": PLAN, "audit": False}).json()
    assert body["ok"] and body["diagnostics"]["audit"] == "skip"


def test_token_required_when_configured():
    c = client(["[担心]你昨晚又熬夜了"], token="secret")
    assert c.post("/polish", json={"plan": PLAN}).status_code == 401
    assert c.post("/polish", json={"plan": PLAN},
                  headers={"Authorization": "Bearer secret"}).status_code == 200


def test_health():
    c = client(["[担心]你昨晚又熬夜了"])
    assert c.get("/health").json()["ok"] is True


def test_request_validation_bounds():
    c = client(["[担心]你昨晚又熬夜了"])
    assert c.post("/polish", json={"plan": PLAN, "k": 99}).status_code == 422
