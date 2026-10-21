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
from concurrent.futures import ThreadPoolExecutor

from . import corpus, prompts
from .llm import LLM, dump_jsonl

PAIRS_PATH = "data/pairs/pairs.jsonl"


def destyle_one(llm: LLM, original: str) -> str:
    text = llm.complete("destyle", prompts.DESTYLE_SYSTEM,
                        prompts.DESTYLE_USER.format(original=original))
    # 去风格化必须干净：残留的 [标签] 会让润色器误以为标签属于语义内容
    from .contract import TAG_RE
    return TAG_RE.sub("", text).strip()


def run(profile_llm: LLM | None = None, limit: int | None = None,
        out_path: str = PAIRS_PATH, workers: int = 4, log=print,
        source: str | None = None) -> list[dict]:
    llm = profile_llm or LLM()
    turns = list(corpus.iter_for_destyle(limit, path=source))
    # 同位置的 rejected（官方偏好数据的较差输出）一并存下，当事后参照
    rejected_by_id = {t.turn_id: r for t, r in corpus.contrasts(path=source)}
    rows: list[dict] = [None] * len(turns)  # type: ignore[list-item]

    def work(i_t: tuple[int, corpus.Turn]):
        i, t = i_t
        try:
            draft = destyle_one(llm, t.chosen)
        except Exception as exc:
            log(f"  ! {t.turn_id} 去风格化失败: {exc}")
            return i, None
        return i, {
            "turn_id": t.turn_id,
            "user": t.user,
            "context": t.context,        # direct 臂需要多轮上下文
            "draft": draft,
            "original": t.chosen,
            "rejected": rejected_by_id.get(t.turn_id),
            "injected_info": t.injected_info,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = 0
        for i, row in pool.map(work, enumerate(turns)):
            rows[i] = row
            done += 1
            if done % 10 == 0 or done == len(turns):
                log(f"  destyle {done}/{len(turns)}")

    rows = [r for r in rows if r]
    dump_jsonl(out_path, rows)
    log(f"写入 {out_path}（{len(rows)} 对）")
    return rows
