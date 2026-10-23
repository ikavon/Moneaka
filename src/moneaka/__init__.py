"""Moneaka —— 莫妮卡口吻润色器。

通用 LLM 负责「说什么」，Moneaka 负责「怎么说」。
"""

from pathlib import Path

from .contract import EMOTION_TAGS, SENTENCE_MAX_CHARS, check, contract_text

__all__ = ["EMOTION_TAGS", "SENTENCE_MAX_CHARS", "check", "contract_text"]
__version__ = "0.0.1"

# ---------------------------------------------------------------------------
# 构建标记：用来发现「进程里跑的还是旧代码」
# ---------------------------------------------------------------------------
# Streamlit 只会重跑主脚本，import 过的包留在 sys.modules 里不会重载。于是改了
# src/moneaka 之后界面照旧跑旧代码，报的错还很莫名（`unexpected keyword argument`）。
# 导入时记下包内 .py 的最大 mtime，界面对比一次就能提示「请重启」。

def build_stamp() -> float:
    here = Path(__file__).resolve().parent
    return max((p.stat().st_mtime for p in here.glob("*.py")), default=0.0)


BUILD_STAMP = build_stamp()
