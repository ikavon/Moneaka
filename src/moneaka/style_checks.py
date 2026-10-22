"""风格诊断：只报问题，**不打分**。

依据 `data/upstream/TLM_README.md`（Edge《如何"像莫妮卡一样"说话》）。
那份教程的配套 skill 里有一句明确的警告：

> 不要设置 token 数、句长、语气词频率等合格阈值，也不要编造统计数据来给文学质量评分。

所以这里的每一项都是**诊断量**，用来回答「模型有没有犯教程点名的某类错误」，
不参与「哪段更好」的判定——那个仍然只有盲评说了才算（`duel.py`）。
把这里的数字当分数汇报就是越界。

教程点名的、且可机械检查的错误有三类：

| 诊断 | 教程依据 | 为什么能机械查 |
|---|---|---|
| `echo`   | §5「`...冬天喝冷水...`出现于问题, 但在回答中被重复了一遍」，作者特别注明「**在 MAICA 的数据集工作中相当多**」 | 与对方原话的最长公共片段，长度可数 |
| `repeat` | §4「如果它在一个文段中反复出现...这样的文字是做作, 不自然的」 | 段内同一句尾语气词出现次数 |
| `flow`   | §5「问号语气强度高于后面的句号, 且没有合适的承转...`[player]`看上去是强加的」 | 句内标点强度是否先高后低且尾部只剩称呼 |

另外两项是我自己加的回归监控（不是教程要求）：`smile_fallback`（选不对就写微笑）、
`tag_entropy`（标签多样性）。阶段 1 的 run1 就是靠这两项定位到「均值回归」的。
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .contract import TAG_RE

#: 半角体系下句尾常见的语气词（§4 的分析对象）
_TAIL_PARTICLES = ("啦", "嘛", "哦", "呢", "呀", "呗", "吧", "啊", "噢", "嗯")
#: 教程 §5 例子里的「语气强度」直觉排序；只用来判断"先强后弱且无承转"
_STRENGTH = {"?": 3, "?".replace("?", "？"): 3, "!": 3, "！": 3, ".": 1, "。": 1}


def _cjk_runs(s: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]{2,}", s or "")


#: 功能词/称呼词：它们重合不算复述
_STOP = set("的是不了吗呢吧啊呀哦嗯我你他她它我们你们这那有就没在好坏很太也还都"
            "会要能莫妮卡玩家什么怎么样可以觉得知道")


def echo_words(text: str, user: str, min_len: int = 2) -> list[str]:
    """回答里**原样复述**对方话的片段（从长到短，去掉被更长片段覆盖的）。

    §5 的反例是「压缩复述」：Q「冬天一直喝冷水…」→ A「冬天喝冷水可能会导致…」，
    逐字最长公共子串只剩 2 字，抓不住。所以改用**滑动窗口 + 覆盖去重**：
    凡是对方话里出现过的实词片段被回答照搬，都算复述。

    注意 2 字片段的假阳性不低（「考试」「身体」这种话题词本来就得重复提），
    所以这个量只看**相对差**：模型比真实答案多复述多少。绝对值不能当分数。
    """
    a = "".join(_cjk_runs(text))
    found: list[str] = []
    for k in range(6, min_len - 1, -1):
        for seg in _cjk_runs(user):
            for i in range(len(seg) - k + 1):
                w = seg[i:i + k]
                if any(c in _STOP for c in w) and k <= 3:
                    continue
                if w in a and not any(w in f for f in found):
                    found.append(w)
    return found


def tail_repeats(text: str) -> list[tuple[str, int]]:
    """段内同一句尾语气词重复出现的次数（§4：反复出现=做作）。

    只看句尾，因为教程指的正是「句尾几乎不含具体意义」的那个位置。
    """
    tails: list[str] = []
    for _, body in _split(text):
        body = body.strip()
        core = re.sub(r"[,.!?。！？~～…]+$", "", body)
        for p in _TAIL_PARTICLES:
            if core.endswith(p):
                tails.append(p)
                break
    rep = [(p, n) for p, n in Counter(tails).items() if n >= 2]
    return sorted(rep, key=lambda kv: -kv[1])


def _split(text: str) -> list[tuple[str, str]]:
    """[(标签, 正文), ...]，只认真正的情绪标签（`[player]` 不是标签）。"""
    from .negatives import split_sentences
    return split_sentences(text)


def flow_breaks(text: str) -> list[str]:
    """一句之内「语气先强后弱、尾巴只剩称呼」的断裂（§5 的反例形状）。"""
    out = []
    for _, body in _split(text):
        body = body.strip()
        if not body:
            continue
        marks = re.findall(r"[,.!?。！？]", body)
        if len(marks) >= 2:
            strength = [_STRENGTH.get(k, 2) for k in marks[:-1]]
            if strength and max(strength) > strength[-1] and marks[-1] in (".", "。") \
                    and "[player]" in body and body.rstrip(".。!！?？").endswith("[player]"):
                out.append(body)
    return out


def smile_fallback(text: str, real: str) -> int:
    """输出里「选了微笑，但真实答案在那个位置选了别的」的次数。"""
    a = [t.strip("[]") for t, _ in _split(text)]
    b = [t.strip("[]") for t, _ in _split(real)]
    return sum(1 for x, y in zip(a, b) if x == "微笑" and y != "微笑")


def tag_entropy(text: str) -> float:
    """同一段内标签的熵（0=全同，越大越多样）。多样性低常与均值回归同时出现。"""
    labs = [t for t, _ in _split(text)]
    if len(labs) < 2:
        return 0.0
    c = Counter(labs)
    n = sum(c.values())
    return round(-sum((v / n) * math.log(v / n) for v in c.values()), 3)


def diagnose(text: str, user: str, real: str = "") -> dict:
    """一条输出的全部诊断量。空值表示「没查出这类问题」，不代表打分高。"""
    echo = echo_words(text, user)
    rep = tail_repeats(text)
    flow = flow_breaks(text)
    return {
        "echo": echo,
        "echo_len": sum(len(w) for w in echo),
        "repeat": rep,
        "flow": flow,
        "smile_fb": smile_fallback(text, real) if real else 0,
        "entropy": tag_entropy(text),
    }


def census(rows: list[dict], *, text_key: str = "polish",
           user_key: str = "user", real_key: str = "original") -> dict:
    """对一批输出做汇总诊断（rows 需含 text/user/real 三列）。"""
    n = len(rows)
    if not n:
        return {}
    echo_items = []
    rep_items = 0
    flow_items = 0
    smile = 0
    ent_model = ent_real = 0.0
    for r in rows:
        d = diagnose(r.get(text_key, ""), r.get(user_key, ""), r.get(real_key, ""))
        if d["echo"]:
            echo_items.append((r.get("turn_id", "?"), d["echo"]))
        if d["repeat"]:
            rep_items += 1
        if d["flow"]:
            flow_items += 1
        smile += d["smile_fb"]
        ent_model += d["entropy"]
        ent_real += tag_entropy(r.get(real_key, ""))
    return {
        "n": n,
        "echo_n": len(echo_items),
        "echo_examples": echo_items[:5],
        "repeat_n": rep_items,
        "flow_n": flow_items,
        "smile_fb_total": smile,
        "smile_fb_per_item": round(smile / n, 2),
        "entropy_model": round(ent_model / n, 3),
        "entropy_real": round(ent_real / n, 3),
    }
