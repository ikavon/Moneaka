"""OpenAI 兼容客户端的小封装。

阶段 0 有三类调用方，各自可以是不同的端点：

- ``destyle`` / ``judge`` —— 强模型（任何 OpenAI 兼容 API，模型由用户配置）
- ``restyle`` —— 候选小模型（本地 llama.cpp / LM Studio / Ollama 的 ``/v1`` 同样适用）

所以这里不写死供应商，只认 profile → (base_url, api_key, model) 三元组。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

try:  # 延迟导入：没装 openai 时也能用 contract/corpus
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "default.yaml",
)


def _expand(value: Any) -> Any:
    """支持 ``env:VAR`` 取值，以及 ``env:VAR:default`` 兜底。"""
    if isinstance(value, str) and value.startswith("env:"):
        rest = value[4:]
        var, _, default = rest.partition(":")
        return os.environ.get(var, default or "")
    return value


@dataclass
class Profile:
    name: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 512
    backend: str = "openai"
    load_in_4bit: bool = False
    enable_thinking: bool = False
    extra: dict = field(default_factory=dict)


class LLM:
    """按 profile 名取用的聊天客户端。"""

    def __init__(self, config_path: str | None = None):
        self.config = self._load_config(config_path or DEFAULT_CONFIG_PATH)
        self._profiles: dict[str, Profile] = {}
        self._clients: dict[str, Any] = {}

    # -- 配置 --------------------------------------------------------------

    @staticmethod
    def _load_config(path: str) -> dict:
        if not os.path.isfile(path):
            return {}
        try:
            import yaml
        except Exception:
            return {}
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def profile(self, name: str) -> Profile:
        if name in self._profiles:
            return self._profiles[name]
        eps = (self.config.get("endpoints") or {})
        roles = (self.config.get("roles") or {})
        target = roles.get(name, name)  # 角色名 → 端点名
        raw = dict(eps.get(target) or {})
        if not raw:
            raise KeyError(
                f"没有名为 {name!r} 的 endpoint（roles 也没映射到它）。"
                f"检查 configs/default.yaml 或环境变量。"
            )
        prof = Profile(
            name=target,
            base_url=_expand(raw.get("base_url", "http://localhost:1234/v1")) or "http://localhost:1234/v1",
            api_key=_expand(raw.get("api_key", "EMPTY")) or "EMPTY",
            model=_expand(raw.get("model", "")) or "",
            temperature=float(_expand(raw.get("temperature", 0.7))),
            max_tokens=int(_expand(raw.get("max_tokens", 512))),
            backend=_expand(raw.get("backend", "openai")) or "openai",
            load_in_4bit=bool(_expand(raw.get("load_in_4bit", False))),
            enable_thinking=bool(_expand(raw.get("enable_thinking", False))),
            extra={k: _expand(v) for k, v in raw.items() if k not in (
                "base_url", "api_key", "model", "temperature", "max_tokens",
                "backend", "load_in_4bit", "enable_thinking")},
        )
        self._profiles[name] = prof
        return prof

    # -- 调用 --------------------------------------------------------------

    def _client(self, prof: Profile):
        if prof.name not in self._clients:
            if OpenAI is None:
                raise RuntimeError("未安装 openai 包：pip install openai")
            self._clients[prof.name] = OpenAI(base_url=prof.base_url, api_key=prof.api_key)
        return self._clients[prof.name]

    def chat(
        self,
        profile: str,
        messages: list[dict[str, str]],
        **overrides: Any,
    ) -> str:
        """一次非流式补全。``overrides`` 可覆盖 model/temperature/max_tokens。"""
        prof = self.profile(profile)
        if prof.backend == "hf":
            return self._chat_hf(prof, messages, **overrides)

        kw: dict[str, Any] = dict(
            model=overrides.pop("model", None) or prof.model,
            messages=messages,
            temperature=overrides.pop("temperature", prof.temperature),
            max_tokens=overrides.pop("max_tokens", prof.max_tokens),
        )
        kw.update(prof.extra)
        kw.update(overrides)
        kw = {k: v for k, v in kw.items() if v not in (None, "")}

        client = self._client(prof)
        attempts = int(kw.pop("attempts", 3))
        last: Exception | None = None
        for i in range(attempts):
            try:
                resp = client.chat.completions.create(**kw)
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # 网络/限流/服务端抖动
                last = exc
                if i + 1 < attempts:
                    time.sleep(1.5 * (i + 1))
        raise RuntimeError(f"[{profile}] 调用失败: {last}") from last

    def _chat_hf(self, prof: Profile, messages: list[dict[str, str]],
                 **overrides: Any) -> str:
        """本地 HF 后端：不联网、进程内单例加载。"""
        from .localhf import GenConfig, get_model

        model_id = overrides.pop("model", None) or prof.model
        if not model_id:
            raise ValueError(f"endpoint {prof.name!r} 是 hf 后端但没配 model")
        cfg = GenConfig(
            max_new_tokens=int(overrides.pop("max_tokens", prof.max_tokens)),
            temperature=float(overrides.pop("temperature", prof.temperature)),
        )
        model = get_model(model_id, load_in_4bit=prof.load_in_4bit)
        return model.chat(messages, cfg=cfg, enable_thinking=prof.enable_thinking)

    def complete(self, profile: str, system: str, user: str, **overrides: Any) -> str:
        return self.chat(profile, [{"role": "system", "content": system},
                                    {"role": "user", "content": user}], **overrides)

    def score(self, profile: str, system: str, user: str) -> int | None:
        """评委专用：从回复里抠出第一个 0-4 的整数。"""
        text = self.complete(profile, system, user, temperature=0.0, max_tokens=8)
        for ch in text:
            if ch.isdigit() and int(ch) <= 4:
                return int(ch)
        return None


def dump_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
