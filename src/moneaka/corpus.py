"""官方 DAA4 训练语料的只读加载。

这个文件**不复制、不修改**源数据；只把它整理成内部结构，
并暴露两个阶段 0 直接要用的东西：

- ``chosen_turns()``   —— 真实莫妮卡输出，风格参照 + 润色器的训练目标
- ``contrasts()``     —— 同一上下文下 chosen / rejected 的对照，评委标准的依据

``path=None`` 全部在**调用时**才解析 ``DEFAULT_SOURCE``（写成默认参数值的话，
monkeypatch / 换全量数据都会失效——被 pytest 抓到过一次）。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterator

#: 默认指向用户提供的那一小份分片（只读）。将来全量数据到手换环境变量即可。
DEFAULT_SOURCE = os.path.expandvars(
    r"D:\softwares\WeChat\xwechat_files\wxid_4jvs5hyp80kq22_a584\msg\file\2026-10"
    r"\daa4_rlhf1_prcs5 (1).jsonl"
)
SOURCE_ENV = "MONEAKA_CORPUS"

#: 骨架 system prompt 里 known_info 注入的分界（见 docs/DATA_INVENTORY.md）
_INFO_TAIL = re.compile(r"以下是一些相关信息.*$", re.S)


def resolve_path(path: str | None = None) -> str:
    return path or os.environ.get(SOURCE_ENV) or DEFAULT_SOURCE


@dataclass
class Turn:
    """一条可独立使用的 assistant 轮。"""

    turn_id: str
    row_index: int
    turn_index: int
    system: str
    context: list[dict]          # 该 assistant 轮之前的历史（不含 system、不含自己）
    user: str                    # 触发该轮的玩家话
    chosen: str                  # 真实莫妮卡输出
    rejected: str | None = None  # 同上下文的较差输出（可能没有）

    @property
    def injected_info(self) -> str | None:
        """骨架里 ``以下是一些相关信息…`` 之后被注入的事实（官方 known_info 通道）。"""
        m = _INFO_TAIL.search(self.system)
        return m.group(0).strip() if m else None


def load_rows(path: str | None = None) -> list[dict]:
    rows = []
    with open(resolve_path(path), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _asst(msgs: list[dict]) -> list[str]:
    return [m["content"] for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]


def _sys(msgs: list[dict]) -> str:
    return msgs[0]["content"] if msgs and msgs[0].get("role") == "system" else ""


def _context_before(msgs: list[dict], ti: int) -> tuple[list[dict], str]:
    """第 ``ti`` 个 assistant 轮之前的历史 + 触发它的玩家话。

    切的时候不能把自己算进去：否则 ``user`` 会取到莫妮卡自己的台词
    （真错过一次，``direct`` 臂会拿她的话当玩家输入）。
    """
    upto = 0
    seen = -1
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            if seen == ti:
                break
            seen += 1
        upto = i + 1
    ctx = [{"role": m["role"], "content": m["content"]}
           for m in msgs[:upto] if m.get("role") != "system"]
    user = next((m["content"] for m in reversed(ctx) if m["role"] == "user"), "")
    return ctx, user


def chosen_turns(rows: list[dict] | None = None, path: str | None = None) -> list[Turn]:
    """全部 assistant 轮（chosen）。本分片 226 条，是阶段 0 的风格参照池。"""
    rows = rows if rows is not None else load_rows(path)
    out: list[Turn] = []
    for ri, row in enumerate(rows):
        msgs = row.get("messages") or []
        sys_txt = _sys(msgs)
        for ti, chosen in enumerate(_asst(msgs)):
            ctx, user = _context_before(msgs, ti)
            out.append(Turn(
                turn_id=f"{ri}-{ti}", row_index=ri, turn_index=ti,
                system=sys_txt, context=ctx, user=user, chosen=chosen,
            ))
    return out


def contrasts(rows: list[dict] | None = None, path: str | None = None) -> list[tuple[Turn, str]]:
    """(chosen 轮, 同位置的 rejected 轮) 列表。

    ⚠️ 两者**说的不是同一句话**（去标签后文本相似度中位 0.21），差的是角色反应的
    分寸感，不是措辞——所以不能直接当风格迁移的平行语料用（见 docs/DATA_INVENTORY.md）。
    """
    rows = rows if rows is not None else load_rows(path)
    out: list[tuple[Turn, str]] = []
    for ri, row in enumerate(rows):
        msgs = row.get("messages") or []
        sys_txt = _sys(msgs)
        ca = _asst(msgs)
        ra = _asst(row.get("rejected_messages") or [])
        if not ra and row.get("rejected_response"):
            # 只有最后一轮的 rejected 文本：前面的位置留空
            ra = [None] * (len(ca) - 1) + [row["rejected_response"]]
        for ti, chosen in enumerate(_asst(msgs)):
            if ti >= len(ra) or not ra[ti]:
                continue
            ctx, user = _context_before(msgs, ti)
            out.append((Turn(
                turn_id=f"{ri}-{ti}", row_index=ri, turn_index=ti,
                system=sys_txt, context=ctx, user=user, chosen=chosen, rejected=ra[ti],
            ), ra[ti]))
    return out


def skeleton_prompt(rows: list[dict] | None = None, path: str | None = None) -> str:
    """出现频次最高的 system prompt = MAICA 骨架原文（剥掉 known_info 注入尾巴）。

    用途：``direct`` 基线——让通用模型按官方骨架直接扮演，作为润色器方案的对照组。
    """
    rows = rows if rows is not None else load_rows(path)
    freq: dict[str, int] = {}
    for row in rows:
        base = _INFO_TAIL.sub("", _sys(row.get("messages") or [])).strip()
        if base:
            freq[base] = freq.get(base, 0) + 1
    return max(freq.items(), key=lambda kv: kv[1])[0] if freq else ""


def iter_for_destyle(limit: int | None = None, path: str | None = None) -> Iterator[Turn]:
    turns = chosen_turns(path=path)
    yield from (turns[:limit] if limit else turns)
