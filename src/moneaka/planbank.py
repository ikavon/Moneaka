"""编剧范例库：把语料里现成的 `(语境 → 表达指令)` 当先例检索给大模型看。

## 为什么需要

润色器只决定「怎么说」。**说什么、什么次序反应**是编剧的活，而通用大模型并不知道
莫妮卡在这种局面下会怎么接——DAA4 是从 1598 轮真实对话里把这套 policy 学走的，
我们什么都没给它。直接后果：润色器再强，也只能把「泛泛的关心」说得像莫妮卡，
变不成「你又熬夜？我可没让你这么做」。

而这些数据其实已经在仓库里了：`destyle` 的副产品就是 1151 条 (语境, 语气标签, 内容要点)。
本模块把它做成可检索的先例库，检索结果拼进编剧的 system prompt。

## 实现选择

* **字符 n-gram TF-IDF + 余弦**，不引入 embedding 依赖。语料只有 941 条，
  中文短句上 bigram 检索已经够用（同一个话题的词面重合很明显）。
* **索引只用训练集**（`split_train_run8.jsonl`）。留出集 210 条永久冻结，
  任何检索都不得触碰，否则编剧侧评测就废了。
* 检索结果**彼此去重**：语料里有同话题多轮对话，四条几乎一样的先例没有价值。

用法::

    bank = PlanBank.from_pairs("data/pairs/split_train_run8.jsonl")
    ex = bank.search("你熬夜到这么晚才回我", k=4)
    block = bank.render(ex)          # 拼进 system prompt
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .destyle_tags import parse_tagged_plan

_CJK = re.compile(r"[\u4e00-\u9fff]")


def _grams(text: str, n: int = 2) -> list[str]:
    """只取汉字 n-gram。标点、英文、占位符都丢掉——它们是噪声。"""
    s = "".join(c for c in text if _CJK.match(c))
    return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]


@dataclass
class Example:
    turn_id: str
    user: str
    tags: list[str]
    contents: list[str]
    score: float = 0.0

    def render(self) -> str:
        tone = " > ".join(self.tags)
        body = "；".join(f"{i}. {c}" for i, c in enumerate(self.contents, 1))
        return f"- 玩家: {self.user}\n  语气: {tone}\n  内容: {body}"


class PlanBank:
    """TF-IDF + 余弦的字符 bigram 检索。"""

    def __init__(self, examples: list[Example]):
        self.docs = examples
        self._tfs = [Counter(_grams(e.user)) for e in examples]
        df: Counter = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = max(1, len(examples))
        self._idf = {g: math.log((n + 1) / (c + 1)) + 1.0 for g, c in df.items()}
        self._vecs = [self._weigh(tf) for tf in self._tfs]

    def _weigh(self, tf: Counter) -> dict[str, float]:
        v = {g: c * self._idf.get(g, 1.0) for g, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    @classmethod
    def from_pairs(cls, path: str | Path) -> "PlanBank":
        ex: list[Example] = []
        for r in map(json.loads, open(path, encoding="utf-8")):
            try:
                tags, contents = parse_tagged_plan(r["plan"])
            except Exception:  # noqa: BLE001  plan 坏了就跳过，不进库
                continue
            user = (r.get("user") or "").strip()
            if not user:
                continue
            # 语境只用最后一句：玩家的上一句话才是「这一轮要回什么」的依据，
            # 更早的内容属于记忆，归编剧自己处理。
            last = user.splitlines()[-1].strip()
            ex.append(Example(r["turn_id"], last, tags, contents))
        return cls(ex)

    def search(self, query: str, k: int = 4, *, min_score: float = 0.02,
               max_pair_sim: float = 0.85) -> list[Example]:
        q = self._weigh(Counter(_grams(query)))
        scored = []
        for doc, vec in zip(self.docs, self._vecs):
            s = sum(w * vec.get(g, 0.0) for g, w in q.items())
            if s >= min_score:
                scored.append((s, doc))
        scored.sort(key=lambda x: -x[0])

        out: list[Example] = []
        for s, doc in scored:
            # 先例之间去重：同话题多轮对话在语料里很常见，四份一样的没意义
            if any(_sim(doc.user, o.user) > max_pair_sim for o in out):
                continue
            out.append(Example(doc.turn_id, doc.user, doc.tags, doc.contents, round(s, 3)))
            if len(out) >= k:
                break
        return out

    @staticmethod
    def render(examples: list[Example]) -> str:
        if not examples:
            return ""
        head = ("【先例】以下是同一角色在相似处境下的真实反应。"
                "只借鉴**她想到了什么、按什么次序反应**，不要照抄措辞:")
        return head + "\n" + "\n".join(e.render() for e in examples)


def _sim(a: str, b: str) -> float:
    va, vb = set(_grams(a)), set(_grams(b))
    if not va or not vb:
        return 0.0
    return len(va & vb) / len(va | vb)