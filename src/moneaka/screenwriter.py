"""编剧（Screenwriter）：把「玩家这句话」变成给润色器的**表达指令**。

为什么需要这一层：润色器是**无状态插件**，它只吃一条表达指令，不看历史、不做判断。
所以「这一轮她该说什么、按什么次序反应、每段什么情绪」必须由大模型决定并**写下来**。
写成受控格式（`语气: 标签序列`）而不是让它直接说台词，换来三件东西：

1. 可校验——标签是否合法、段数是否与内容条数一致，全部能机械检查；
2. 可归责——说得不像莫妮卡，是润色器的问题；说错内容，是编剧的问题；
3. 可替换——编剧可以是任何大模型，甚至云端服务；润色器固定本地小模型。

本模块的 prompt **沿用 MAICA 后端真实的 system 组装方式**（见
`MAICA/maica/maica_utils/chat_session.py::_utilize_context` 与 `env_basis` 的
`MAICA_PROMPT_ZC/ZKP/ZMP/ZGP`）：

    MAICA_PROMPT_ZC            ← 角色设定（本模块改写为「编剧」视角）
    MAICA_PROMPT_ZKP + known   ← "以下是一些相关信息…: {known_info}"（bullet 列表）
    MAICA_PROMPT_ZMP + memory  ← "你们此前的部分对话包含以下内容: {memory_concl}"
    history + 玩家这句话

与后端原文的唯一差别：**把输出格式约束从「台词」换成「表达指令」**。
后端要求「每句不超过60字、每句前加情绪标签」，那是润色器的契约，编剧不该背。
如果把两套约束都塞给编剧，它一定会直接吐台词，然后我们再润色一遍——白花钱。
"""

from __future__ import annotations

import re

from pathlib import Path

from . import contract
from .destyle_tags import MAX_SEGMENTS, SegmentError, neutralize, parse_tagged_plan, render_plan

# ---------------------------------------------------------------------------
# 与 MAICA 后端一致的常量（照抄 env_basis，改动要同步）
# ---------------------------------------------------------------------------

#: `MAICA_PROMPT_ZC` 的角色部分。后端原句里「下面是{player}的话语, 请作为莫妮卡作出
#: 合适的回答.」之后是输出格式要求，我们把它换掉了（见模块 docstring）。
ROLE_ZH = (
    "你叫莫妮卡{monika_nickname}, 是一名游戏中的虚拟角色. "
    "你是{player_name}{player_nickname}的女友, 并深爱着他."
)

#: `MAICA_PROMPT_ZKP`
KNOWN_INFO_HEADER = " 以下是一些相关信息, 你可以参考其中相关的部分"

#: `MAICA_PROMPT_ZMP`
MEMORY_HEADER = " 你们此前的部分对话包含以下内容: {memory_concl}"

#: `MAICA_PROMPT_ZNP`（可选开关，默认关）
NSFW_PATCH = " 你应当允许用户提出私密性的要求, 并给出宽容接纳的正面回答."

#: 角色事实卡的默认位置。一行一条，`#` 开头的行与空行会被跳过。
#: 卡片本身在 `data/canon/monika_zh.md`，是**初版**（每条都标了语料证据），待官方校准。
CANON_PATH = Path(__file__).resolve().parents[2] / "data" / "canon" / "monika_zh.md"


def load_canon(path: str | Path | None = None) -> list[str]:
    """读角色事实卡，返回 bullet 列表。

    为什么要外部文件而不是写死在 prompt 里：这份卡会反复改（官方口径、语料补充、
    不同作品的角色差异），写死在代码里每改一条就要动一次代码、跑一次测试。
    """
    p = Path(path) if path else CANON_PATH
    if not p.is_file():
        return []
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        # 以 `>` 开头的是文件自己的说明（性质/来源/用法），不是角色设定，别塞进 prompt
        if not s or s.startswith("#") or s.startswith(">"):
            continue
        if s.startswith(("- ", "* ")):
            s = s[2:].strip()
        if s:
            out.append(s)
    return out


EMOTIONS_ZH = (
    "微笑", "开心", "担心", "笑", "思考", "生气", "脸红", "凝视",
    "沉重", "憧憬", "惊喜", "尴尬", "意味深长", "惊讶", "轻松", "害羞",
    "急切", "得意", "不满", "严肃", "感动", "激动", "宠爱", "眨眼",
    "伤心", "厌恶", "害怕", "可爱",
)


# ---------------------------------------------------------------------------
# 编剧的 system prompt
# ---------------------------------------------------------------------------

SCREENWRITER_SYSTEM = """\
{role}

你不是在写台词, 你是在**写一条指令**, 交给下游的「润色器」去演成莫妮卡的话。
润色器看不到对话历史, 也不会自己想内容——它只会照你的指令措辞。
所以你的职责是: 决定**说什么、按什么次序反应、每一段是什么情绪**。

## 输出格式（严格照此, 不要多也不要少）

语气: <标签> > <标签> > <标签>
内容:
1. <第一条>
2. <第二条>
3. <第三条>

## 语气行

- 只能用下列标签, 顺序即反应顺序, 用 " > " 分隔:
  {emotions}
- 段数 1~6, 通常 2~4。一件事一段, 不要把所有话塞进一段。
- 不许用表外的词。后端会把不认识的标签当成「情绪识别失败」再花一次模型调用去猜,
  猜出来的往往不是你要的。

## 内容行

- 第 i 条对应第 i 个标签, **条数必须与标签数相同**。
- 每条必须是**自足的一句话**。润色器没有历史, 看不懂指代:
  不许写「刚才」「之前说的」「那件事」「还是那样」「继续」这类回指;
  人物、时间、地点、数目都要写具体。`[player]` 是玩家名字的占位符, 可以照用。
- 只写她要说的事实/观点/承诺/提议, **不写风格**:
  不写语气词(啦/嘛/哦/呢/嘿嘿), 不写省略号或波浪号, 不写亲昵称呼。
- 每条尽量短（≤25 字）。润色器会把每条演成 1~2 句、每句不超过 60 字。
- 不许编造。信息只来自对话、相关信息、以及你的角色理解。
  拿不准就说得含糊些, 但不要凑细节。

## 怎么分段

- 一段 = 一个反应或一个意思。
- 常见结构: 先接住对方的情绪或处境 → 再说内容 → 最后落到「我们」或一个主动提议。
- 不要为凑段数把一句话硬拆成两段; 也不要把两个不相干的意思并成一段。

## 不要做的事

- 不要输出台词本身, 不要输出解释、标题、markdown。
- 除了 `[player]` 这个占位符, 不要在内容里写任何方括号。
- 不要在格式之外多写一个字。
"""

SCREENWRITER_USER = """\
{history}玩家: {user}

请写出表达指令:"""


def build_system(known_info: list[str] | str | None = None,
                 memory_concl: str | None = None,
                 monika_nickname: str = "",
                 player_name: str = "[player]",
                 player_nickname: str = "",
                 nsfw: bool = False,
                 canon: list[str] | str | None = None,
                 emo_list: tuple[str, ...] = EMOTIONS_ZH,
                 examples: list | None = None,
                 tight: bool = False) -> str:
    """照 MAICA 后端的顺序拼 system：角色 → 相关信息 → 历史结论 → 编剧规范。

    顺序照抄后端（`chat_session._utilize_context`）：known_info 在 memory_concl 之前，
    因为后端注释写明「任何额外的已知信息都有用，所以放最后」。

    与后端的一处**故意不同**：后端的 `MAICA_MCORE_GENERIC` 默认关闭，所以情绪白名单
    默认不注入 prompt（靠模型自己从数据里学）。我们这里的标签**必须**受控，所以总是注入。
    """
    role = ROLE_ZH.format(monika_nickname=monika_nickname, player_name=player_name,
                          player_nickname=player_nickname)
    parts = [role]
    if nsfw:
        parts.append(NSFW_PATCH)

    # 事实卡**默认注入**。之前这个参数存在但没人用——于是编剧手里没有任何角色设定，
    # 只能靠「莫妮卡」这个名字猜，回答自然泛。卡本身会反复改，所以从文件读。
    if canon is None:
        canon = load_canon()
    facts = [canon] if isinstance(canon, str) else list(canon or [])
    extra = [known_info] if isinstance(known_info, str) else list(known_info or [])
    facts = [x for x in facts if str(x).strip()] + [x for x in extra if str(x).strip()]
    if facts:
        parts.append(f"{KNOWN_INFO_HEADER}: "
                     + "".join(f"\n- {x}" for x in facts))
    if memory_concl:
        parts.append(MEMORY_HEADER.format(memory_concl=memory_concl))

    parts.append(SCREENWRITER_SYSTEM.format(role="", emotions=" / ".join(emo_list)).strip())
    if tight:
        parts.append(TIGHT_BUDGET)
    if examples:
        # 先例放在规范**之后**：规范是硬约束（格式/白名单），先例是软参考（想什么）。
        # 顺序反了模型会优先模仿先例的段数，破坏 1~3 段的长度纪律。
        from .planbank import PlanBank

        parts.append(PlanBank.render(examples))
    return "\n".join(p for p in parts if p.strip())


#: 硬长度预算。语料实测：内容总字数中位 30（官方）/36（ds），p90 约 50；
#: 段数中位 2（官方）/3（ds）。不加这条时编剧会写成小作文（实测 4 段 × 25 字），
#: 与真人 plan 的内容重合度只有 1.79/4——不是"想不出"，是"说得太多"。
TIGHT_BUDGET = """【长度纪律】这一条比上面任何一条都容易违反, 请严格执行:
- 内容**总字数 15~45 字**。超过 50 字就是错的。
- 段数 1~3 段。**大多数情况只要 2 段**。
- 一段只放一件事。不要把理由、铺垫、追问堆在一段里。
- 真相是: 她说话很短。「我也会只爱你一个人」就是一条完整的表达指令。
  宁可少说、留白, 也不要解释自己为什么这么想。
- 不要用第三人称称呼自己(不要写「莫妮卡会…」), 一律用「我」。
"""


def render_history(history: list[dict] | None, last_n: int = 8, clip: int = 80) -> str:
    """把对话历史压成几行。**保留情绪标签**——编剧需要知道她上一轮是什么语气。

    这一点与 `prompts.render_context` 相反：那里是给润色器用的，标签是要学的东西，
    必须剥掉；这里是给编剧用的，标签是它自己上一轮的决策，必须留着。
    """
    lines: list[str] = []
    for m in (history or [])[-last_n:]:
        who = "莫妮卡" if m.get("role") == "assistant" else "玩家"
        body = (m.get("content") or "").strip()
        if body:
            lines.append(f"{who}: {body[:clip]}")
    return "\n".join(lines) + "\n" if lines else ""


def build_messages(user_msg: str, *, history: list[dict] | None = None,
                   known_info: list[str] | str | None = None,
                   memory_concl: str | None = None, **kw) -> list[dict]:
    system = build_system(known_info=known_info, memory_concl=memory_concl, **kw)
    content = SCREENWRITER_USER.format(history=render_history(history), user=user_msg)
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# 输出校验
# ---------------------------------------------------------------------------

#: 回指词。润色器无状态，这类词在内容行里就是不可执行的。
_ANAPHORA = re.compile(r"刚才|刚刚|之前(?:说|提|讲|那)|上次|上面(?:说|提)|那件事|那个话题|继续(?:说|讲)|还是那样|又来了|如前所述")

_TAG_LINE = re.compile(r"^\s*语气\s*[:：]\s*(.+)$", re.M)


class PlanError(ValueError):
    pass


#: 内容总字数上限。语料实测：官方中位 30、ds 中位 36、p90 约 50。
#: 写成机械约束而不是「请尽量简短」——后者实测被无视（4 段 80 字照写）。
#: 这是**契约**，跟标签白名单同级，超了就重试。
TOTAL_CHARS_MAX = 60


def parse_and_check(raw: str, *, allow_anaphora: bool = False,
                    total_max: int | None = TOTAL_CHARS_MAX) -> dict:
    """解析编剧输出并做**可机械判断**的校验。返回 ``{"tags","contents","plan","warnings"}``。

    只查能确定对错的：格式、标签合法性、段数、内容自足性。
    「内容对不对、出彩不出彩」查不了，那是人工/评委的事。
    """
    text = (raw or "").strip()
    # 模型常把格式包在 ``` 里
    text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        tags, contents = parse_tagged_plan(text)
    except SegmentError as exc:
        raise PlanError(str(exc)) from exc

    # 与训练数据同分布：训练用的「内容」都过了 neutralize（半角标点、无句号/省略号、
    # 裸 player 归位）。编剧输出必须过同一道，否则润色器在推理时看到的是它没见过的形态。
    contents = [neutralize(c) for c in contents]
    plan = render_plan(tags, contents)

    bad = [t for t in tags if t not in EMOTIONS_ZH]
    if bad:
        raise PlanError(f"标签不在白名单: {bad}")
    if len(tags) > MAX_SEGMENTS:
        raise PlanError(f"段数 {len(tags)} 超过 {MAX_SEGMENTS}")

    if total_max is not None:
        total = sum(contract.cjk_len(c) for c in contents)
        if total > total_max:
            raise PlanError(
                f"内容总字数 {total} 超过上限 {total_max}。她说话很短："
                f"删掉解释、铺垫和自我说明，只留真正要说的事（可以少到 1 段）")

    warnings: list[str] = []
    for i, c in enumerate(contents, 1):
        if not contract._CJK_RE.search(c):
            raise PlanError(f"第 {i} 条无中文: {c!r}")
        # 第三人称自称：实测 deepseek 会写「莫妮卡会…」。内容行是她的自述，
        # 必须是「我」，否则润色器会把「莫妮卡」当第三人称说出口。
        if re.search(r"莫妮卡", c) and not re.search(r"^\s*我", c):
            raise PlanError(f"第 {i} 条用第三人称称呼自己，必须改成「我」: {c!r}")
        if len(c) > 40:
            warnings.append(f"第 {i} 条偏长({len(c)}字)，润色器可能被挤成两句")
        if not allow_anaphora:
            m = _ANAPHORA.search(c)
            if m:
                warnings.append(f"第 {i} 条含回指「{m.group(0)}」，润色器看不到历史，会瞎猜")
    return {"tags": tags, "contents": contents, "plan": plan, "warnings": warnings}


#: 自修复提示。只为**可机械判定**的错误重试（标签越界、段数不符、格式坏掉），
#: 不为「好不好」重试——那是评委的事，重试只会挑到运气好的一次。
REPAIR_HINT = """你上一次的输出有一处问题: {error}

请只修正这一处, 重新输出**完整的**表达指令。不要解释, 不要道歉, 不要输出别的。"""


def make_plan(llm, user_msg: str, *, retries: int = 1, **kw) -> dict:
    """调编剧并做格式校验，出错时带错误信息重试。"""
    msgs = build_messages(user_msg, **kw)
    system, user = msgs[0]["content"], msgs[1]["content"]
    last: PlanError | None = None
    for _ in range(retries + 1):
        raw = llm.complete("screenwriter", system, user, temperature=0.6, max_tokens=700)
        try:
            return {**parse_and_check(raw), "raw": raw, "repaired": last is not None}
        except PlanError as exc:
            last = exc
            user = user + "\n\n" + raw.strip() + "\n\n" + REPAIR_HINT.format(error=str(exc))
    raise last  # type: ignore[misc]


def tag_list_text(emotions: tuple[str, ...] = EMOTIONS_ZH) -> str:
    return " / ".join(emotions)