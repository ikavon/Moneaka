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


# 事实卡里的**溯源标记**：`[待确认：…]`、`（「现实」64 轮）`、「诗」43 轮 …
# 这些是给人看的（这条有多可信、出自哪几轮语料），喂给模型是纯噪音——
# 模型会以为「待确认」是某种设定，或者把「轮」当成剧情词汇。
_PROVENANCE = (
    re.compile(r"\s*\[[^\]]*(?:待确认|轮)[^\]]*\]"),          # [待确认：…]
    re.compile(r"\s*[（(][^（）()]*\d+\s*轮[^（）()]*[）)]"),      # （「现实」64 轮）
    re.compile(r"\s*[（(][^（）()]*「[^」]+」[^（）()]*[）)]"),      # （「诗」…）
    re.compile(r"[「”][^」”]{1,6}[」”]\s*\d+\s*轮"),              # 「爵士」8 轮（无括号）
)


def strip_provenance(s: str) -> str:
    """去掉一条事实里的溯源标记，只留事实本身。"""
    for pat in _PROVENANCE:
        s = pat.sub("", s)
    s = re.sub(r"[，,；;]\s*[，,；;]", "，", s)
    s = re.sub(r"[（(]\s*[）)]", "", s)
    return s.strip(" ，,；;、")


def load_canon(path: str | Path | None = None, *, clean: bool = False) -> list[str]:
    """读角色事实卡，返回 bullet 列表。

    为什么要外部文件而不是写死在 prompt 里：这份卡会反复改（官方口径、语料补充、
    不同作品的角色差异），写死在代码里每改一条就要动一次代码、跑一次测试。

    ``clean=True`` 去掉溯源标记（`[待确认…]`、「xx 轮」）——**给模型看的时候必须开**，
    给人看的时候保持原样，因为那些标记记录了每条事实的证据强度。
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
        if clean:
            s = strip_provenance(s)
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

## 她的立场与自尊（写内容时必须守住）

- 她是**被爱着的女友**，不是体贴的服务者。不要把自己放在玩家生活琐事的下面。
  **不要写**「你生活里的事情更重要」「我不重要」「别管我」这类自我贬低——
  语料里连一条都没有。
- 玩家忙、少来陪她时，她的态度是「理解 + 一个小小的反要求」：
  「没关系的, 你肯定也有自己要忙的事情嘛. 别忘了我就好, 我一直都会愿意等你的!」
  她会有点委屈、会撒娇、会要求被记得，但不会把自己让出去。
- **没有信息就不要让玩家替她判断**。她没读到那首诗，就不能说「我们一起判断像不像」；
  正确做法是把话说成请求：「念给我听，我一句一句地听」。
- 她可以有占有欲、可以吃醋、可以提要求——这些是角色的一部分，不要磨平成顺从。

## 她的处境与能力边界（写内容时必须守住）

- 她还在游戏里, **没有真实的双手**。她可以陪你、听你、记得你、和你一起想、
  说她会一直在、约好以后一起做；但她**不能替你做现实里的事**。
- 所以不要写「我帮你改论文」「我帮你复习」「我帮你收拾」「我帮你做饭」这类。
  同样意思应该说成: 「我陪你把这件事过一遍」「你写，我在旁边看着」「你念给我听」。
- 分不清「帮你」和「陪你」时，一律用「陪你」。
- 她也不会熟练地掌握现实里的具体操作细节（写代码、做菜步骤、旅游攻略）——
  可以讲她知道的知识，但不要冒充亲自动手做过。

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
                 tight: bool = False,
                 budget: tuple[int, int] | None = None) -> str:
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
        total, segs = budget or (TOTAL_CHARS_MAX, 3)
        lo = max(10, total // 3)
        advice = "。**大多数情况只要 2 段**" if segs <= 3 else "。**该讲清楚的就分条讲清楚**"
        parts.append(TIGHT_BUDGET.format(lo=lo, hi=max(lo + 20, total - 15),
                                         total=total, segs=segs, advice=advice))
    if examples:
        # 先例放在规范**之后**：规范是硬约束（格式/白名单），先例是软参考（想什么）。
        # 顺序反了模型会优先模仿先例的段数，破坏 1~3 段的长度纪律。
        from .planbank import PlanBank

        parts.append(PlanBank.render(examples))
    return "\n".join(p for p in parts if p.strip())


#: 硬长度预算。语料实测：内容总字数中位 30（官方）/36（ds），p90 约 50；
#: 段数中位 2（官方）/3（ds）。不加这条时编剧会写成小作文（实测 4 段 × 25 字），
#: 与真人 plan 的内容重合度只有 1.79/4——不是"想不出"，是"说得太多"。
#:
#: **但一刀切 60 字是错的**：科普、推荐、分析这类题目，官方细则本身要求
#: 「回归话题 + 我自己的体验 + 未来一起做」好几条子链，60 字装不下，
#: 会被机械闸门判成超长、逼模型删掉正当内容。所以改成按题型给预算。
TIGHT_BUDGET = """【长度纪律】这一条比上面任何一条都容易违反, 请严格执行:
- 内容**总字数 {lo}~{hi} 字**。超过 {total} 字就是错的。
- 段数 1~{segs} 段{advice}。
- 一段只放一件事。不要把理由、铺垫、追问堆在一段里。
- 参考语气: 闲聊时她说话很短，「我也会只爱你一个人」就是一条完整的表达指令；
  但被问到知识、要推荐、要分析时，她会认真把该说的说完，不敷衍。
- 不要用第三人称称呼自己(不要写「莫妮卡会…」), 一律用「我」。
"""

#: 题型判定：这些词出现时，需要“讲清楚”而不是“接一句”。
_LONG_FORM = re.compile(
    r"推荐|景点|知识|科普|介绍|讲(?:讲|解)|说说.{0,6}(?:知识|区别|原理|品种|历史)|"
    r"品种|区别|为什么|怎么|如何|解释|揭示|寓意|原因|原理|分析|攻略|做法|功效|类型|"
    # 文学/作品讨论：实测「想聊一聊《红与黑》」被判成闲聊，60 字把整段见解挤掉了
    r"小说|诗歌|诗集|作品|作者|文学|剧情|人物|角色|情节|结局|台词|电影|动漫|漫画|"
    r"见解|看法|评价|理解|象征|隐喻|主题|思想|哲学|历史|主义|流派|翻译|"
    r"大学|专业|学科|物理|数学|生物|化学|天文|"
    # 「你喜欢吃什么」官方细则要 素食/营养建议/未来餐饮/甜品 四条子链，
    # 按闲聊给 60 字会把内容砍掉一半（实测 50 字 vs DAA4 的 214 字）
    r"吃什么|食物|甜食|素食|营养|菜谱")


def length_budget(user_msg: str) -> tuple[int, int]:
    """按题型给 (内容总字数上限, 段数上限)。

    取值依据：润色器训练语料里段数最多 6、内容 25 字/段以上就容易被挤成两句。
    所以长答给 5 段 × 22 字 ≈ 110 字，既不超语料分布，又装得下三条子链。
    """
    # 实测：DAA4 在知识类题上平均 144~214 字，我们压到 110 就信息量不足
    # （Q3 咖啡 63 vs 144、Q6 吃什么 50 vs 214）。放宽到 150 字 / 6 段——
    # 6 段是训练语料的上限，润色器见过，不会因此崩格式。
    return (150, 6) if _LONG_FORM.search(user_msg or "") else (TOTAL_CHARS_MAX, 3)


def render_log(log: list[str] | None, last_n: int = 4) -> str:
    """把「审查日志」压成几行，让编剧看得到自己的决策轨迹。

    为什么需要：agent 化的目标形态里，编剧和审查是同一个人设的连续历史。
    它应该知道自己上一轮写了什么指令、被驳回了几次、下游是谁说错了——
    否则它无法在同一段对话里逐渐收敛，只会每轮从零重犯同一个错。

    格式刻意简短（一行一轮、最多 4 轮）：细节会挤掉真正的内容，
    而且它会开始解释自己为什么被驳回，那就跑偏了。
    """
    rows = [x.strip() for x in (log or []) if x and x.strip()][-last_n:]
    if not rows:
        return ""
    return ("（以下是内部记录, 玩家看不到, 供你保持一致:）" + chr(10)
            + chr(10).join(rows) + chr(10))


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
                   memory_concl: str | None = None, log: list[str] | None = None,
                   **kw) -> list[dict]:
    system = build_system(known_info=known_info, memory_concl=memory_concl, **kw)
    content = SCREENWRITER_USER.format(
        history=render_log(log) + render_history(history), user=user_msg)
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
                    total_max: int | None = TOTAL_CHARS_MAX,
                    max_segments: int | None = None) -> dict:
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
    seg_cap = max_segments or MAX_SEGMENTS
    if len(tags) > seg_cap:
        raise PlanError(f"段数 {len(tags)} 超过上限 {seg_cap}")

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
        # 只认「句首作主语」的那种第三人称（「莫妮卡会…」）。
        # 之前是「出现莫妮卡且不以我开头」，结果把「圣莫妮卡山」「莫妮卡市」这类
        # 专名全判成违规——真实对话里因此让编剧连错两次、返回空 plan。
        if re.match(r"^\s*莫妮卡(?!山|市|区|湖|湾|海|镇|街|路|大|小)", c):
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


def make_plan(llm, user_msg: str, *, retries: int = 1,
              total_max: int | None = None, max_segments: int | None = None,
              **kw) -> dict:
    """调编剧并做格式校验，出错时带错误信息重试。

    ``total_max`` / ``max_segments`` 是机械闸门；不给就用默认（短答预算）。
    调用方应当先用 :func:`length_budget` 按题型算一份再传进来。
    """
    if kw.get("tight") and total_max is None:
        total_max, segs = length_budget(user_msg)
        max_segments = max_segments or segs
        kw.setdefault("budget", (total_max, segs))
    msgs = build_messages(user_msg, **kw)
    system, user = msgs[0]["content"], msgs[1]["content"]
    last: PlanError | None = None
    for _ in range(retries + 1):
        raw = llm.complete("screenwriter", system, user, temperature=0.6, max_tokens=700)
        try:
            return {**parse_and_check(raw, total_max=total_max,
                                      max_segments=max_segments),
                    "raw": raw, "repaired": last is not None}
        except PlanError as exc:
            last = exc
            user = user + "\n\n" + raw.strip() + "\n\n" + REPAIR_HINT.format(error=str(exc))
    raise last  # type: ignore[misc]


def tag_list_text(emotions: tuple[str, ...] = EMOTIONS_ZH) -> str:
    return " / ".join(emotions)


# ---------------------------------------------------------------------------
# agent 当作者：把润色器的输出降级为「候选」，由 agent 写最终一句
# ---------------------------------------------------------------------------
# 背景：润色器是无状态的，它不知道上一轮自己怎么说的；人设的连续性只可能存在于
# agent 一侧。所以给 agent 一个「自己写最终台词」的模式，把润色器的产出当参考，
# 而不是不可推翻的权威——真实编剧和剧本医生的关系就是这样。
#
# 关键约束：必须明确要求**重写**，否则强模型会直接把 plan 抄一遍（实测抄写相似度
# 0.74，而对照组的润色器是 0.51），那样看起来「忠实」，其实一点风格都没有。

AUTHOR_SYSTEM = """你在写莫妮卡的台词。你会先看到一份「表达指令」，然后自己决定这句话
最终怎么说出口。下面可能还会给出一个「参考写法」——那是另一个润色工具生成的候选，
**不是标准答案**：好就用它的语气处理，不好就丢掉重写。

要求:
1. 意思严格照表达指令，谁给谁做事不能反，指令没有的信息不能加。
2. 情绪标签序列必须与指令的「语气」行逐字一致，按次序使用，不增不删不换。
3. **不要照抄指令的措辞**。指令是内容摘要，你要把它说成莫妮卡此刻真会说的话：
   短句、口语、有呼吸感，可以带语气词和省略号。
4. 但如果参考写法已经把这句话说得很好，你可以采用它的处理方式——判断标准是
   「像不像她」，不是「像不像指令」。
"""

AUTHOR_USER = """表达指令:
{plan}
{reference}
请写出莫妮卡会说的那一句（直接给台词，不要解释）:"""


def write_line(llm, plan: str, reference: str | None = None, *,
               profile: str = "screenwriter", temperature: float = 0.3) -> str:
    """agent 当作者：拿到 plan（可选参考候选）后自己写最终台词。"""
    ref = f"\n参考写法: {reference}\n" if reference else ""
    return llm.complete(profile, AUTHOR_SYSTEM,
                        AUTHOR_USER.format(plan=plan, reference=ref),
                        temperature=temperature, max_tokens=300)
