"""Moneaka —— 莫妮卡口吻润色器。

通用 LLM 负责「说什么」，Moneaka 负责「怎么说」。
"""

from .contract import EMOTION_TAGS, SENTENCE_MAX_CHARS, check, contract_text

__all__ = ["EMOTION_TAGS", "SENTENCE_MAX_CHARS", "check", "contract_text"]
__version__ = "0.0.1"
