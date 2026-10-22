"""本地 HF 推理后端：直接跑 transformers 权重，不做服务化。

为什么不用 LM Studio / Ollama：
- 多装一个运行时 = 多一份格式漂移（模板、停止符、thinking 开关各走一套）；
- 训练和推理共用同一套 transformers 代码，评测看到的行为就是训练时的行为；
- 插件形态最终要能打包成「装上就能用」，依赖越少越好。

Qwen3 的 thinking 默认是开的，会先吐 ``...`` 再给答案——润色任务必须关掉，
否则输出里混推理、还会吃满 ``max_tokens``。这里统一走
``apply_chat_template(enable_thinking=False)``，并对没这个参数的模板回退到拒绝。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

_MODEL_CACHE: dict[tuple[str, str], "LocalModel"] = {}
_LOCK = threading.Lock()


def hf_home() -> str:
    return os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "data", "models"),
    )


def resolve_model_id(model_id: str) -> str:
    """优先用 ``data/models/<name>`` 下的本地权重，否则回退到 HF repo id。

    本地优先的理由：离线可用、不受网络抖动影响，插件分发时用户也是这么放的。
    """
    for base in (hf_home(), os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "data", "models")):
        cand = os.path.join(base, model_id.replace("/", "--"))
        for p in (cand, os.path.join(base, model_id.split("/")[-1])):
            if os.path.isfile(os.path.join(p, "config.json")):
                return p
    return model_id


@dataclass
class GenConfig:
    max_new_tokens: int = 300
    temperature: float = 0.9
    top_p: float = 0.95
    repetition_penalty: float = 1.05


class LocalModel:
    """一个已加载的本地因果语言模型。"""

    def __init__(self, model_id: str, dtype: str = "auto", device: str = "auto",
                 load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HOME", hf_home())

        self.model_id = model_id
        self.torch = torch
        self.path = resolve_model_id(model_id)
        self.tok = AutoTokenizer.from_pretrained(self.path)

        kwargs: dict = {"device_map": device}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["dtype"] = torch.bfloat16 if dtype == "auto" else getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, **kwargs).eval()
        self.device = next(self.model.parameters()).device

    # -- 生成 --------------------------------------------------------------

    def _messages(self, messages: list[dict]) -> list[dict]:
        # 本地模型没有 system 角色限制，原样传；Qwen 模板原生支持
        return messages

    def render(self, messages: list[dict], *, enable_thinking: bool = False) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tok.apply_chat_template(
                self._messages(messages), enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            # 模板不支持 enable_thinking：Qwen3 系列可用软开关 /no_think 兜底
            msgs = list(messages)
            if not enable_thinking and msgs and msgs[0]["role"] == "system":
                msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n/no_think"}
            return self.tok.apply_chat_template(msgs, **kwargs)

    def chat(self, messages: list[dict], cfg: GenConfig | None = None,
             enable_thinking: bool = False, stop: list[str] | None = None) -> str:
        cfg = cfg or GenConfig()
        torch = self.torch
        text = self.render(messages, enable_thinking=enable_thinking)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        gen_kwargs: dict = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        if cfg.temperature > 0:
            gen_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        reply = self.tok.decode(new_ids, skip_special_tokens=True).strip()
        if stop:
            for s in stop:
                if s and s in reply:
                    reply = reply.split(s)[0].strip()
        return reply


def get_model(model_id: str, **kwargs) -> LocalModel:
    """进程内单例：同一 (model_id, 量化设置) 只加载一次。"""
    key = (model_id, f"{kwargs.get('load_in_4bit', False)}-{kwargs.get('dtype', 'auto')}")
    with _LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = LocalModel(model_id, **kwargs)
        return _MODEL_CACHE[key]


def unload_all() -> None:
    """释放显存（评测多模型时逐个跑完就放掉，8GB 塞不下两个）。"""
    import gc
    for m in _MODEL_CACHE.values():
        del m.model
    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass