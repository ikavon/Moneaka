"""步骤 1：去风格化，生成 (平实草稿, 莫妮卡原稿) 平行语料。

这一步由强 LLM 完成——官方偏好数据里的 rejected 也是莫妮卡风格（只是分寸感差），
不能当「平实版」用，所以必须自己造草稿。见 docs/DATA_INVENTORY.md 的分析。

产出 ``data/pairs/pairs.jsonl``：

    {"turn_id", "user", "draft", "original", "rejected", "injected_info"}

``draft`` 是后续一切的地基（润色器的输入、评测的语义基准），所以这里就做一次
往返自检：草稿去掉的语义不能再被润色器「合法地」加回来，评测时以 draft 为真值。
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor

from . import corpus, prompts
from .llm import LLM, dump_jsonl

PAIRS_PATH = "data/pairs/pairs.jsonl"


#: 占位符哨兵。直接把 `[player]` 交给模型不行——它会把方括号当情绪标签、把 player
#: 当昵称, 结果 27/60 的草稿直接丢了占位符（写成了「你」或干脆删掉, 还留下「，。」残痕）。
#: 换成书名号包裹的罕见形式后模型会原样照抄。
_SENTINEL = "《player》"
_SENTINEL_VARIANTS = re.compile(r"[《<〈【\[]\s*player\s*[》>〉】\]]", re.I)


def destyle_one(llm: LLM, original: str) -> str:
    protected = re.sub(r"\[\s*player\s*\]", _SENTINEL, original, flags=re.I)
    text = llm.complete("destyle", prompts.DESTYLE_SYSTEM,
                        prompts.DESTYLE_USER.format(original=protected))
    # 去风格化必须干净：残留的 [标签] 会让润色器误以为标签属于语义内容
    from .contract import TAG_RE
    text = TAG_RE.sub("", text).strip()
    text = _restore_placeholder(text, original)
    if len(text) < 4:
        raise ValueError(f"草稿过短或为空: {text!r}")
    return text


_PLAYER_WORD = re.compile(r"(?<![\[])\b(?:[Pp]layer|玩家)\b(?![\]])")


def destyle_one(llm: LLM, original: str) -> tuple[str, bool]:
    """返回 ``(草稿, 是否由模型自行补出 [player]``。"""
    protected = re.sub(r"\[\s*player\s*\]", _SENTINEL, original, flags=re.I)
    text = llm.complete("destyle", prompts.DESTYLE_SYSTEM,
                        prompts.DESTYLE_USER.format(original=protected))
    # 去风格化必须干净：残留的 [标签] 会让润色器误以为标签属于语义内容
    from .contract import TAG_RE
    text = TAG_RE.sub("", text).strip()
    text, synthesized = _restore_placeholder(text, original)
    if len(text) < 4:
        raise ValueError(f"草稿过短或为空: {text!r}")
    return text, synthesized


_PLAYER_WORD = re.compile(r"(?<![\[])\b(?:[Pp]layer|玩家)\b(?![\]])")


def _restore_placeholder(draft: str, original: str) -> tuple[str, bool]:
    """把哨兵还原成 ``[player]``。返回 ``(草稿, 是否额外出现占位符)``。

    两种情况：
    1. 原稿有 `[player]`、草稿没有 → 先试把 `player`/`玩家` 改回来；仍不行就报错。
       丢占位符是硬伤——真数据里它占 38.5%，润色器学不到就永远用不出。
    2. 原稿没有、草稿冒出 `[player]` → **不报错**。这是模型把「你」换成了占位符，
       指的还是同一个人，语义没变（只是把一点风格带进了草稿）。早期当成「编造信息」
       拦了 9/60 条，属于判过了；现在只打标记，后续可以在训练时筛掉。
    """
    had = bool(re.search(r"\[\s*player\s*\]", original, re.I))
    draft = _SENTINEL_VARIANTS.sub("[player]", draft)
    got = "[player]" in draft.lower()
    if had and not got:
        fixed = _PLAYER_WORD.sub("[player]", draft)
        if "[player]" not in fixed.lower():
            raise ValueError(f"占位符丢失且无法恢复: {draft!r}")
        return fixed, False
    return draft, (got and not had)


def run(profile_llm: LLM | None = None, limit: int | None = None,
        out_path: str = PAIRS_PATH, workers: int = 4, log=print,
        source: str | None = None) -> list[dict]:
    """造草稿。若 ``out_path`` 已有内容则**按 turn_id 续跑**（只补缺的）。

    续跑很重要：去风格化是强模型调用，失败重跑一次就是几十次配额。
    """
    llm = profile_llm or LLM()
    turns = list(corpus.iter_for_destyle(limit, path=source))
    # 同位置的 rejected（官方偏好数据的较差输出）一并存下，当事后参照
    rejected_by_id = {t.turn_id: r for t, r in corpus.contrasts(path=source)}

    existing: dict[str, dict] = {}
    if os.path.isfile(out_path):
        from .llm import load_jsonl
        existing = {r["turn_id"]: r for r in load_jsonl(out_path)}
        if existing:
            log(f"续跑：已有 {len(existing)} 条")
    todo = [t for t in turns if t.turn_id not in existing]
    log(f"待造草稿 {len(todo)}/{len(turns)} 条")
    if not todo:
        dump_jsonl(out_path, list(existing.values()))
        return list(existing.values())

    rows: list[dict] = [None] * len(todo)  # type: ignore[list-item]
    failed: list[tuple[str, str]] = []

    def work(i_t: tuple[int, corpus.Turn]):
        i, t = i_t
        try:
            draft, synthesized = destyle_one(llm, t.chosen)
        except Exception as exc:
            failed.append((t.turn_id, f"{exc} | 原稿: {t.chosen[:60]}"))
            return i, None
        return i, {
            "turn_id": t.turn_id,
            "user": t.user,
            "context": t.context,        # direct 臂需要多轮上下文
            "draft": draft,
            "original": t.chosen,
            "rejected": rejected_by_id.get(t.turn_id),
            "injected_info": t.injected_info,
            "draft_player_synth": synthesized,   # 草稿里的 [player] 是模型自己补的
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = 0
        for i, row in pool.map(work, enumerate(todo)):
            rows[i] = row
            done += 1
            if done % 10 == 0 or done == len(todo):
                log(f"  destyle {done}/{len(todo)}")

    rows = [r for r in rows if r]
    for tid, why in failed:
        log(f"  ! {tid} {why}")
    merged = list(existing.values()) + rows
    dump_jsonl(out_path, merged)
    log(f"写入 {out_path}（{len(merged)}/{len(turns)} 对）")
    if len(merged) < len(turns):
        log(f"⚠️ 缺 {len(turns) - len(merged)} 条（见上方 ! 行）；直接重跑即可续上")
    return merged
