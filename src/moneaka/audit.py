"""旁路审查：B（台词复核）与 C（plan 前置审查）。

两头夹住中间的产物，尽量不靠「训练得更强」来兜底。

**C — plan 前置审查**（花钱之前拦）
* 机械部分：回指词、复读、canon 冲突黑名单、段数/字数/标签
* 模型部分：plan 和「她的设定 + 玩家这句话」是否冲突（可选，一次调用）

**B — 台词复核**（交付之前拦）
* 触发式：只有出现数字/否定词、段数 ≥3、长度突变时才送审，避免每轮多一次云调用
* 判定只看「意思有没有被改动」，不看风格好不好（风格归评委和人）

两者都不修东西，只给判决 + 原因；修由调用方决定（重采 / 重新编剧 / 退回直译）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import contract
from .destyle_tags import MAX_SEGMENTS, parse_tagged_plan

# ---------------------------------------------------------------------------
# 机械规则
# ---------------------------------------------------------------------------

#: 回指词。润色器无状态，这类词在内容行里不可执行；编剧写出来就说明它在指代历史。
ANAPHORA = re.compile(
    r"刚才|刚刚|之前(?:说|提|讲|那)|上次|上面(?:说|提)|那件事|那个话题|"
    r"继续(?:说|讲)|还是那样|又来了|如前所述")

_NEG = re.compile(r"[不没别无未莫]")
_DIGITS = re.compile(r"\d+")
_NUMWORD = re.compile(r"[零一二两三四五六七八九十百千万亿]+分之[零一二三四五六七八九十百千]+")

#: canon 冲突黑名单：她明确没有的东西。每条都能在 `data/canon/monika_zh.md` 找到依据。
#: 机械可查，命中即拦——这类错误一旦渲染出去，整段对话的设定就崩了。
#: 能力越界：她还在游戏里，没有真实双手，不能替玩家做现实里的事。
#: 这类句子写进指令，润色器会照实说出来，然后整段对话的处境就崩了。
#: 只盯「动词 + 现实动作名词」的组合，避免把「帮你想办法」误伤。
OUT_OF_WORLD = re.compile(
    r"帮[你您].{0,6}(复习|预习|收尾|做题|改|写|抄|做|弄|整理|收拾|打扫|做饭|买菜|开车|打卡|请假|点名)")
OUT_OF_WORLD_ALT = re.compile(r"(替|代)[你您].{0,6}(做|写|改|准备|复习|收拾|整理|去|跑)")


CANON_CONFLICTS: list[tuple[str, str]] = [
    (r"我(?:有|还有个|的)?(?:哥哥|姐姐|弟弟|妹妹|父母|爸爸|妈妈)", "她没有家人"),
    (r"我(?:喜欢|爱)(?:吃|喝)(?:肉|鱼|海鲜)", "她是素食主义者"),
    (r"我(?:讨厌|不喜欢|不会)(?:弹钢琴|钢琴)", "她弹钢琴"),
    (r"我(?:不|没)(?:加入|参加)(?:文学部)", "她是文学部部长"),
    (r"我(?:住在|在你)(?:电脑|硬盘|游戏)(?:文件夹)?(?:以外|外面)", "她住在游戏/电脑里"),
    (r"我已经(?:来到|在)现实(?:世界|中)", "她还没能来到现实"),
]


def numbers(text: str) -> set[str]:
    return set(_DIGITS.findall(text)) | set(_NUMWORD.findall(text))


def check_plan_mechanical(plan: str, *, prev_reply: str | None = None,
                          allow_anaphora: bool = False) -> list[str]:
    """C 的机械部分。返回问题列表（空 = 通过）。"""
    bad: list[str] = []
    try:
        tags, contents = parse_tagged_plan(plan)
    except Exception as exc:  # noqa: BLE001
        return [f"plan 不可解析: {str(exc)[:80]}"]
    if not 1 <= len(tags) <= MAX_SEGMENTS:
        bad.append(f"段数 {len(tags)} 越界")
    for i, c in enumerate(contents, 1):
        if not contract._CJK_RE.search(c):
            bad.append(f"第{i}段无中文")
        if contract.cjk_len(c) > contract.SENTENCE_MAX_CHARS:
            bad.append(f"第{i}段超 {contract.SENTENCE_MAX_CHARS} 字")
        if not allow_anaphora and ANAPHORA.search(c):
            bad.append(f"第{i}段有回指词「{ANAPHORA.search(c).group(0)}」")
        for pat, why in CANON_CONFLICTS:
            if re.search(pat, c):
                bad.append(f"第{i}段与设定冲突（{why}）")
        if OUT_OF_WORLD.search(c) or OUT_OF_WORLD_ALT.search(c):
            bad.append(f"第{i}段能力越界：她在游戏里，不能替玩家做现实里的事")
    if prev_reply:
        # 复读：上一轮她刚说过几乎一样的话
        a = {c[j:j + 3] for c in contents for j in range(max(1, len(c) - 2))}
        b = {prev_reply[j:j + 3] for j in range(max(1, len(prev_reply) - 2))}
        if a and b and len(a & b) / len(a) > 0.8:
            bad.append("和上一轮回复高度重复")
    return bad


def needs_polish_audit(plan: str, polish: str) -> list[str]:
    """B 的触发条件。返回触发原因（空 = 不必送审）。"""
    why: list[str] = []
    try:
        tags, contents = parse_tagged_plan(plan)
    except Exception:  # noqa: BLE001
        return ["plan 不可解析"]
    body = "；".join(contents)
    if numbers(body):
        why.append("含数字")
    if _NEG.search(body):
        why.append("含否定")
    if len(tags) >= 3:
        why.append("多段")
    # 注意：必须先把标签和占位符剥掉再数字数。原先直接 cjk_len(polish) 把
    # 「[微笑]」里的「微笑」也算进去了，长度比被抬高，这条触发条件实际上从没生效过
    # ——正是它漏掉了「plan 有 6 个字、台词只剩一个『嗯?』」那类塌缩。
    def _plain(x: str) -> int:
        return contract.cjk_len(contract.PLACEHOLDER_RE.sub("", contract.TAG_RE.sub("", x)))

    ratio = _plain(polish) / max(1, _plain(body))
    if ratio < 0.55 or ratio > 2.2:
        why.append(f"长度比 {ratio:.2f}")
    return why


# ---------------------------------------------------------------------------
# 模型审查
# ---------------------------------------------------------------------------

AUDIT_SYSTEM = """你在做台词校对。给你一份「表达指令」和据此写出的「台词」。

你只判断一件事：**台词有没有改变指令的意思**。风格好坏、口吻像不像，不是你的事。

判定标准（按严重程度）：
- `ok`：意思一致。只多了语气词、标点、句序调整，都算 ok。
- `drift`：意思被改了。典型形态（前两条最容易被漏掉, 请优先核对）：
  * **施受方向反了**：指令写「我帮你看看」，台词变成「帮帮我看看」；
    指令写「我陪你」，台词变成「我帮你」——谁给谁做事被改了。
  * **承诺了做不到的事**：指令说陪伴，台词变成替他做现实里的活（帮你收尾/帮你写）。
  * 数字/数量变了（百分之一 → 万分之一）
  * 否定没了或反了（不用回来 → 不要回来）
  * 主客/方向反了（我为你骄傲 → 我是你的骄傲）
  * 近义不同义（陪伴 → 代替）
  * 内容整段丢失或凭空增加

只输出一行 JSON，不要解释：
{"verdict": "ok" 或 "drift", "type": "none|方向|越界|数字|否定|近义|增减", "note": "不超过30字"}"""

AUDIT_USER = """表达指令:
{plan}

台词:
{polish}"""

PLAN_AUDIT_SYSTEM = """你在做剧本审核。给你「角色设定摘要」「玩家刚说的话」和「编剧写的表达指令」。

你只判断一件事：**这个反应和设定或处境有没有硬冲突**。

冲突的例子：说她没有的东西（家人、刚到现实世界）、和已知事实相反（她吃肉）、
答非所问（玩家问 A，她想的是完全无关的 B）、情绪越界（玩家在痛苦，她在调侃）。

不算冲突的：语气偏一点、想法略跳、措辞平淡。这些交给下游调。

只输出一行 JSON：
{"verdict": "ok" 或 "conflict", "note": "不超过30字"}"""

PLAN_AUDIT_USER = """角色设定摘要:
{canon}

最近对话（越靠后越新, 空表示没有历史）:
{history}

玩家刚说: {user}

表达指令:
{plan}"""


@dataclass
class Verdict:
    ok: bool
    kind: str = "none"
    note: str = ""


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {}


def audit_polish(llm, plan: str, polish: str, *, profile: str = "judge") -> Verdict:
    """B：台词有没有改意思。"""
    out = llm.complete(profile, AUDIT_SYSTEM,
                       AUDIT_USER.format(plan=plan, polish=polish),
                       temperature=0.0, max_tokens=80)
    d = _parse_json(out)
    verdict = (d.get("verdict") or "").lower()
    return Verdict(ok=(verdict == "ok"), kind=d.get("type") or "none",
                   note=(d.get("note") or out.strip()[:40]))


def audit_plan(llm, plan: str, user: str, *, canon: str = "",
               history: list[dict] | None = None,
               profile: str = "judge") -> Verdict:
    """C：plan 和设定/处境有没有硬冲突。

    ``history`` 必须给——她的反应经常承接的是更早的对话。只给玩家最后一句时，
    评审会把正常的玩笑/延续判成「答非所问」（实测真人 plan 误报 20%）。
    """
    hist = "\n".join(
        f"{'莫妮卡' if m.get('role') == 'assistant' else '玩家'}: "
        f"{(m.get('content') or '').strip()[:80]}"
        for m in (history or [])[-6:]) or "（无）"
    out = llm.complete(profile, PLAN_AUDIT_SYSTEM,
                       PLAN_AUDIT_USER.format(canon=canon or "（未提供）",
                                              history=hist, user=user, plan=plan),
                       temperature=0.0, max_tokens=80)
    d = _parse_json(out)
    verdict = (d.get("verdict") or "").lower()
    return Verdict(ok=(verdict == "ok"), note=(d.get("note") or out.strip()[:40]))