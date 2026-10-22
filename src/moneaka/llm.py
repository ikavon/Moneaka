"""OpenAI 兼容客户端的小封装。

阶段 0 有三类调用方，各自可以是不同的端点：

- ``destyle`` / ``judge`` —— 强模型（任何 OpenAI 兼容 API，模型由用户配置）
- ``restyle`` —— 候选小模型（本地 llama.cpp / LM Studio / Ollama 的 ``/v1`` 同样适用）

所以这里不写死供应商，只认 profile → (base_url, api_key, model) 三元组。
"""

from __future__ import annotations

import json
import os
import threading
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

#: 本地私密覆盖（不进 git）。只需改这个文件就能换端点/key/model。
PRIVATE_CONFIG_PATH = os.path.join(os.path.dirname(DEFAULT_CONFIG_PATH), "private.yaml")


def _expand(value: Any) -> Any:
    """支持 ``env:VAR`` 取值，以及 ``env:VAR:default`` 兜底。"""
    if isinstance(value, str) and value.startswith("env:"):
        rest = value[4:]
        var, _, default = rest.partition(":")
        return os.environ.get(var, default or "")
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 里的端点击中 base 时按键合并，不整块替换。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class EmptyResponse(RuntimeError):
    """网关返回 200 但 content 为空。"""


class TruncatedResponse(EmptyResponse):
    """空回复且 finish_reason=length——思考型模型把 max_tokens 烧在推理上了。

    这种不能无脑重试同一个上限，得把 max_tokens 提上去重试；
    换非思考模型能直接避开（destyle/judge 这类机械任务不需要推理）。
    """


_THROTTLE_LOCK = threading.Lock()
_LAST_CALL: dict[str, float] = {}


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
    #: LoRA adapter 目录（只有 backend=hf 有意义）
    adapter: str = ""
    #: 同一端点的最小调用间隔（秒）。限流严格的网关靠它把速率压下来。
    min_interval: float = 0.0
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
        """default.yaml 打底，private.yaml 覆盖（私密文件不进 git）。"""
        try:
            import yaml
        except Exception:
            return {}

        def read(p: str) -> dict:
            if not os.path.isfile(p):
                return {}
            with open(p, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}

        cfg = read(path)
        private = PRIVATE_CONFIG_PATH
        if os.path.abspath(private) != os.path.abspath(path):
            cfg = _deep_merge(cfg, read(private))
        return cfg

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
            adapter=_expand(raw.get("adapter", "")) or "",
            min_interval=float(_expand(raw.get("min_interval", 0.0)) or 0.0),
            extra={k: _expand(v) for k, v in raw.items() if k not in (
                "base_url", "api_key", "model", "temperature", "max_tokens",
                "backend", "load_in_4bit", "enable_thinking", "min_interval", "adapter")},
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
        attempts = int(kw.pop("attempts", 5))
        last: Exception | None = None
        for i in range(attempts):
            self._throttle(prof)
            try:
                resp = client.chat.completions.create(**self._with_body(kw))
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    if resp.choices[0].finish_reason == "length":
                        raise TruncatedResponse(
                            f"max_tokens={kw.get('max_tokens')} 被推理吃光")
                    raise EmptyResponse(
                        f"空回复 (finish_reason={resp.choices[0].finish_reason})")
                return text
            except TruncatedResponse as exc:
                last = exc
                cur = int(kw.get("max_tokens") or 512)
                kw["max_tokens"] = min(int(cur * 2), 8192)
                if i + 1 < attempts:
                    time.sleep(min(2 * (i + 1), 10))
            except EmptyResponse as exc:
                last = exc
                if i + 1 < attempts:
                    time.sleep(min(3 * 2 ** i, 30))
            except Exception as exc:  # 网络/限流/服务端抖动
                last = exc
                if i + 1 < attempts:
                    time.sleep(min(self._backoff(exc, i), 60))
        raise RuntimeError(f"[{profile}] 调用失败: {last}") from last

    @staticmethod
    def _with_body(kw: dict) -> dict:
        """把 profile 里的 ``extra_body`` 子字典拆成 SDK 的 extra_body 参数。"""
        body = kw.pop("extra_body", None)
        if body:
            kw["extra_body"] = body
        return kw

    def _throttle(self, prof: Profile) -> None:
        """同一端点的最小调用间隔（跨线程共享）。"""
        if prof.min_interval <= 0:
            return
        with _THROTTLE_LOCK:
            wait = prof.min_interval - (time.monotonic() - _LAST_CALL.get(prof.name, 0.0))
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL[prof.name] = time.monotonic()

    @staticmethod
    def _backoff(exc: Exception, attempt: int) -> float:
        """限流退避明显拉长，其它错误用常规指数退避。"""
        text = str(exc).lower()
        rate_limited = "429" in text or "rate limit" in text or "tpm" in text or "rpm" in text
        return (15 if rate_limited else 1.5) * (2 ** attempt)

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
        model = get_model(model_id, load_in_4bit=prof.load_in_4bit,
                          adapter=overrides.pop("adapter", None) or prof.adapter or None)
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
