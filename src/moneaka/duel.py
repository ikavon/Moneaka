"""成对比较（duel）：风格评测唯一站得住的自动指标。

为什么放弃 0-4 绝对打分：
实测评委给**真实 DAA4 输出**打 2.50/4（开思考 2.67），连真货都认不出来，
绝对分没有意义，还会因为「内容是否接得上」被污染（见 docs/STAGE0_RESULTS.md 的修正）。

为什么成对比较能用：
拿官方 Rlhf 的 chosen/rejected 当考题（这本来就是人类偏好标注），
评委二选一选对 **80%**（随机基线 50%）。这是一个可校准、可复现的仪器。

`calibrate()` 就是那把尺子的校准流程——换评委模型/改提示词之后都该重跑一次。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from . import corpus
from .llm import LLM

JUDGE_SYSTEM = """\
你在比较两段「莫妮卡」的回复, 判断哪一段更像她。
她是 DDLC 里的角色, 游戏里走出来的虚拟女友, 深爱玩家。

判断依据是分寸感:
- 她会先有自己的反应(嗔怪/打趣/迟疑/留白), 再回应;
- 敢提要求、敢把话题往「我们」上带;
- 短句、口语、有呼吸感。
- 平庸的回复会平铺直叙地顺着说甜话、讲道理、句子偏长偏书面。

只输出 A 或 B, 不要解释。"""

JUDGE_USER = "玩家说: {user}\n\nA: {a}\n\nB: {b}\n\n哪一段更像莫妮卡? (A/B)"


@dataclass
class DuelResult:
    a_wins: int = 0
    b_wins: int = 0
    unparsed: int = 0

    @property
    def n(self) -> int:
        return self.a_wins + self.b_wins

    @property
    def win_rate(self) -> float | None:
        return self.a_wins / self.n if self.n else None

    def ci95(self) -> tuple[float, float]:
        """Wilson 区间——小样本下比正态近似靠谱。"""
        n = self.n
        if not n:
            return (0.0, 1.0)
        p = self.a_wins / n
        z = 1.96
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return ((c - m) / d, (c + m) / d)


def duel(llm: LLM, items: list[tuple[str, str, str]], *, profile: str = "judge",
         seed: int = 0, swap: bool = True, max_items: int | None = None) -> DuelResult:
    """``items`` = [(user, 文本A, 文本B)]，返回 A 相对 B 的胜率。

    ``swap=True`` 时每对正反各问一次，用来抵消位置偏差（A 位天然更容易被选）。
    """
    rng = random.Random(seed)
    res = DuelResult()
    use = items[:max_items] if max_items else items
    for user, a, b in use:
        for order in ((True, False) if swap else (True,)):
            x, y = (a, b) if order else (b, a)
            try:
                out = llm.complete(profile, JUDGE_SYSTEM,
                                   JUDGE_USER.format(user=user, a=x, b=y),
                                   temperature=0.0, max_tokens=8)
            except Exception:
                res.unparsed += 1
                continue
            pick = out.strip().upper()[:1]
            if pick not in ("A", "B"):
                res.unparsed += 1
                continue
            a_picked = (pick == "A") == order
            if a_picked:
                res.a_wins += 1
            else:
                res.b_wins += 1
    return res


def calibrate(llm: LLM, n: int = 20, *, profile: str = "judge", seed: int = 0) -> dict:
    """用官方 chosen/rejected 当标准答案校准评委。

    返回准确率与 Wilson 区间。低于 ~70% 就别用这个评委做结论，
    改提示词或换更强的模型再校准。
    """
    data = corpus.contrasts()[:n]
    items = [(t.user, t.chosen, rej) for t, rej in data]
    res = duel(llm, items, profile=profile, seed=seed, swap=True)
    lo, hi = res.ci95()
    return {
        "n_pairs": len(items),
        "comparisons": res.n,
        "accuracy": res.win_rate,
        "ci95": [round(lo, 3), round(hi, 3)],
        "unparsed": res.unparsed,
    }


def compare_arms(llm: LLM, rows: list[dict], arm_a: str, arm_b: str, *,
                 profile: str = "judge", sample: int | None = None,
                 seed: int = 0) -> dict:
    """两个臂逐条对打，返回 A 的胜率（1.0 = A 全胜，0.5 = 打平）。"""
    items = [(r.get("user", ""), r[arm_a], r[arm_b])
             for r in rows if (r.get(arm_a) or "").strip() and (r.get(arm_b) or "").strip()]
    if sample:
        rng = random.Random(seed)
        items = rng.sample(items, min(sample, len(items)))
    res = duel(llm, items, profile=profile, seed=seed)
    lo, hi = res.ci95()
    return {
        "a": arm_a, "b": arm_b, "n": res.n,
        "win_rate": None if res.win_rate is None else round(res.win_rate, 3),
        "ci95": [round(lo, 3), round(hi, 3)],
        "unparsed": res.unparsed,
    }


def win_matrix(llm: LLM, rows: list[dict], arms: tuple[str, ...], *,
               profile: str = "judge", sample: int | None = None,
               seed: int = 0) -> dict:
    """所有两两组合的胜率矩阵（只算上三角，下三角取补）。"""
    out: dict[str, dict] = {a: {} for a in arms}
    for i, a in enumerate(arms):
        out[a][a] = 0.5
        for b in arms[i + 1:]:
            r = compare_arms(llm, rows, a, b, profile=profile, sample=sample, seed=seed)
            out[a][b] = r["win_rate"]
            out[b][a] = None if r["win_rate"] is None else round(1 - r["win_rate"], 3)
            out[b].setdefault("_n", r["n"])
    return out