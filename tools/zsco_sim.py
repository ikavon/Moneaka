"""ZSCO 等效实验：通用模型 + 检索范例，对比「不检索」和「我们的两段式」。

## 为什么要自己模拟

MAICA 的 ZSCO 要 Milvus + embedding 端点 + `MAICA_MCORE_GENERIC=1`，本地起一整套
后端成本很高。但它的机制其实很简单（见 `maica/mtools/zsco/rag.py`）：

    用玩家消息做向量检索 → 取若干条真实对话范例 → 塞进 prompt 给未微调的通用模型

这一层用字符 bigram 检索就能复现到「机制等效」的程度。**不是**复刻它们的向量库，
只是回答一个问题：**给通用模型看几条范例，十二题能好多少？**

## 三个臂

* `plain`  通用模型 + 官方基准 prompt，不给范例
* `zsco`   同上，但在 prompt 前插入 k 条检索到的真实对话
* `ours`   我们的两段式（编剧 → 本地 1.7B），结果从 `data/runs/bench_showcase` 读

    python tools/zsco_sim.py --k 3          # 24 次云调用，约 3 分钟
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from bench_v2 import BENCH  # noqa: E402
from daa4_bench import OFFICIAL_PROMPT  # noqa: E402
from moneaka.llm import LLM  # noqa: E402

CJK = re.compile(r"[\u4e00-\u9fff]")


def grams(text: str, n: int = 2) -> list[str]:
    s = "".join(c for c in text if CJK.match(c))
    return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]


class DialogueBank:
    """(玩家消息 → 莫妮卡台词) 的 bigram TF-IDF 检索，跟 planbank 同一套做法。"""

    def __init__(self, pairs: Path):
        self.docs: list[tuple[str, str]] = []
        for r in map(json.loads, open(pairs, encoding="utf-8")):
            user = (r.get("user") or "").strip()
            ans = (r.get("original") or "").strip()
            if user and ans:
                self.docs.append((user, ans))
        tfs = [Counter(grams(u)) for u, _ in self.docs]
        df: Counter = Counter()
        for tf in tfs:
            df.update(tf.keys())
        n = max(1, len(self.docs))
        idf = {g: math.log((n + 1) / (c + 1)) + 1.0 for g, c in df.items()}
        self._vecs = [self._weigh(tf, idf) for tf in tfs]
        self._idf = idf

    def _weigh(self, tf: Counter, idf: dict[str, float]) -> dict[str, float]:
        v = {g: c * idf.get(g, 1.0) for g, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[str, str]]:
        q = self._weigh(Counter(grams(query)), self._idf)
        scored = []
        for (user, ans), vec in zip(self.docs, self._vecs):
            s = sum(w * vec.get(g, 0.0) for g, w in q.items())
            scored.append((s, user, ans))
        scored.sort(key=lambda x: -x[0])
        return [(u, a) for _, u, a in scored[:k]]


def run_arm(llm, bank: DialogueBank | None, q: str, k: int) -> str:
    if bank is None:
        system = OFFICIAL_PROMPT
    else:
        ex = bank.search(q, k=k)
        block = "\n\n".join(f"玩家: {u}\n莫妮卡: {a}" for u, a in ex)
        system = (OFFICIAL_PROMPT
                  + "\n\n以下是与当前话题相似的对话范例, 仅供体会她的说话方式与分寸, "
                    "不要照抄内容:\n" + block)
    return llm.complete("screenwriter", system, q, temperature=0.5, max_tokens=600)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pairs", default="data/pairs/split_train_run8.jsonl")
    ap.add_argument("--out", default="data/runs/zsco_sim.json")
    args = ap.parse_args()

    bank = DialogueBank(ROOT / args.pairs)
    print(f"范例库 {len(bank.docs)} 条")
    llm = LLM()
    rows = []
    for i, q in enumerate(BENCH, 1):
        plain = run_arm(llm, None, q, 0)
        zs = run_arm(llm, bank, q, args.k)
        rows.append({"n": i, "q": q, "plain": plain, "zsco": zs,
                     "examples": [u for u, _ in bank.search(q, args.k)]})
        print(f"Q{i} plain={len(plain)}字  zsco={len(zs)}字", flush=True)
        (ROOT / args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())