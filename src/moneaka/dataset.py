"""训练数据：把 (表达指令 → 莫妮卡原话) 做成对话样本。

样本结构（trl 的 conversational 格式，配 `assistant_only_loss=True`）：

    system   = restyle_system(few_shot=False)   ← 训练与推理必须同一分支
    user     = 表达指令（语气 + 内容两行）
    assistant = 真实 DAA4 输出                   ← 只对这段算 loss

为什么只对 assistant 算 loss：system/user 是输入条件，让它们参与拟合会让模型
去背指令文本，白白吃掉容量。

为什么 few-shot 不进训练样本：风格进权重之后，例子就是纯噪声 + 上下文开销
（示例部分 ~600 token，而真实输出中位 35 字）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import contract, prompts
from .llm import load_jsonl

#: 数据里反复出现的「语气: 语气:」重复前缀（模型自续导致的，实测 221/226）
_DUP_PREFIX = re.compile(r"^(?:\s*语气\s*[:：]\s*)+")


def clean_plan(plan: str) -> str:
    plan = _DUP_PREFIX.sub("语气: ", (plan or "").strip())
    if not plan.startswith("语气"):
        plan = "语气: " + plan
    return plan


@dataclass
class Sample:
    turn_id: str
    plan: str
    target: str
    user_msg: str = ""

    def as_messages(self) -> list[dict]:
        return [
            {"role": "system", "content": prompts.restyle_system(few_shot=False)},
            {"role": "user", "content": self.plan},
            {"role": "assistant", "content": self.target},
        ]


def build(pairs_path: str, *, require_target_contract: bool = True,
          max_target_chars: int = 200, limit: int | None = None) -> tuple[list[Sample], dict]:
    """从 pairs jsonl 构造样本，并返回被剔除的统计（透明比静默重要）。"""
    rows = load_jsonl(pairs_path)
    kept: list[Sample] = []
    dropped: dict[str, int] = {}

    def drop(code: str) -> None:
        dropped[code] = dropped.get(code, 0) + 1

    for r in rows:
        plan = clean_plan(r.get("draft", ""))
        target = (r.get("original") or "").strip()
        content = ""
        for marker in ("内容:", "内容："):
            if marker in plan:
                content = plan.split(marker, 1)[1].strip()
                break
        if not plan.startswith("语气:") or "内容" not in plan:
            drop("plan_malformed"); continue
        if not target:
            drop("no_target"); continue
        if contract.cjk_len(content) < 1 and not content:
            drop("empty_content"); continue
        if len(target) > max_target_chars:
            drop("target_too_long"); continue
        if require_target_contract and not contract.check(target).ok:
            drop("target_breaks_contract"); continue
        if target.strip() == content.strip():
            drop("identical"); continue
        kept.append(Sample(turn_id=r["turn_id"], plan=plan, target=target,
                           user_msg=r.get("user", "")))
        if limit and len(kept) >= limit:
            break

    stats = {"rows": len(rows), "kept": len(kept), "dropped": dropped}
    return kept, stats


def to_hf_dataset(samples: list[Sample]):
    """转成 datasets.Dataset（trl 直接吃 conversational 格式）。"""
    from datasets import Dataset

    return Dataset.from_list([{"messages": s.as_messages()} for s in samples])


def held_out(samples: list[Sample], n: int = 20, seed: int = 0) -> tuple[list[Sample], list[Sample]]:
    """留出集：训练时不看，用来生成对比样例。"""
    import random

    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    hold = {i for i in idx[:n]}
    train = [s for i, s in enumerate(samples) if i not in hold]
    evals = [s for i, s in enumerate(samples) if i in hold]
    return train, evals