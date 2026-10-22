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
from dataclasses import dataclass, replace

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
    context: list | None = None
    #: 呷进输入的历史句数。训练时按多个宽度造变体（见 expand），
    #: 但**评测永远用 canonical=4**，否则跳 run 的数字不可比。
    last_n: int = 4

    def user_block(self) -> str:
        # 不传 context：润色器无状态（见 prompts.RESTYLE_USER_BODY 注释）。
        # `context` 字段留着只是为了旧 pairs 文件能读，**不进输入**。
        return prompts.RESTYLE_USER_BODY.format(plan=self.plan)

    def as_messages(self) -> list[dict]:
        return [
            {"role": "system", "content": prompts.restyle_system(few_shot=False)},
            {"role": "user", "content": self.user_block()},
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
        # 两种语料格式：旧版（draft = 自由描述的「语气/内容」）与
        # 受控版（plan = 「语气: 标签A > 标签B」+ 逐段内容，见 destyle_tags）。
        raw_plan = (r.get("plan") or "").strip()
        plan = raw_plan or clean_plan(r.get("draft", ""))
        tagged = bool(raw_plan)
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
        if tagged:
            from .destyle_tags import MAX_SEGMENTS, SegmentError, parse_tagged_plan
            try:
                tags, _ = parse_tagged_plan(plan)
            except SegmentError as exc:
                drop(f"tagged_{str(exc).split(':')[0][:20]}"); continue
            if len(tags) > MAX_SEGMENTS:
                drop("tagged_too_many_segments"); continue
        elif target.strip() == content.strip():
            drop("identical"); continue
        kept.append(Sample(turn_id=r["turn_id"], plan=plan, target=target,
                           user_msg=r.get("user", ""), context=r.get("context") or []))
        if limit and len(kept) >= limit:
            break

    stats = {"rows": len(rows), "kept": len(kept), "dropped": dropped}
    return kept, stats


def to_hf_dataset(samples: list[Sample]):
    """转成 datasets.Dataset（trl 直接吃 conversational 格式）。"""
    from datasets import Dataset

    return Dataset.from_list([{"messages": s.as_messages()} for s in samples])


def expand(samples: list[Sample], last_ns: tuple[int, ...] = (2, 4, 6)) -> list[Sample]:
    """同一条 (指令→真话) 按不同历史长度出多个变体（只复制输入，不造新目标）。

    目的是**缓解过拟合 + 不把模型锁死在「历史恰好 4 句」上**。
    真实调用时 agent 给的上下文长短不一（MC 里可能只有一句，MAS 里可能十句），
    固定宽度训练会让模型把它当成输入格式的一部分去记。

    注意：这是**输入侧增广**，不是新信息——目标完全相同，所以它不能突破
    226 条真实风格样本的上限，只能缓解过拟合。不要拿它当「数据翻倍」汇报。
    """
    out: list[Sample] = []
    for s in samples:
        seen: set[str] = set()
        for n in last_ns:
            v = replace(s, last_n=n)
            key = v.user_block()
            # 历史本来就比 n 短的，不同 n 会渲染成**逐字相同**的输入。
            # 不去重就会白拿到若干个完全一样的样本：loss 呷得更低，但不注入任何新东西。
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
    return out


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