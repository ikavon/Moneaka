"""官方基准（MAICA std_test_v2）：12 题、99 分 + 1 分加值。

## 为什么要有这个

我们自己那套 206 条留出集衡量的是**保真度与风格**，和官方口径不可比。
官方这份 12 题是 MAICA 团队自己的评分细则（Q1~Q12，树状条目、[b] 基础 / [v] 垂直），
跑它才有「和 DAA2(66) / DAA3(90) 同一把尺子」的外部分数。

## 两条纪律

1. **单轮、不给历史**：官方测的就是单轮表现。
2. **答案要人工评分**：细则自己写着「仍然是主观的测试」。本脚本只负责
   *跑出答案 + 生成评分表*；`--auto` 给的分数是**参考**，由强模型按同一份细则打，
   不能当成官方分。真正的分由人来填。

题目与注入信息取自 `MAICA/document/maica_std_test_v2.py`（`BENCH_KNOWN`：
现在 14:21，不是特殊节日）。

    python tools/bench_v2.py                 # 跑 + 出评分表
    python tools/bench_v2.py --auto          # 顺便给一份参考分
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka.turn import Runtime, TurnConfig, run_turn  # noqa: E402

BENCH = [
    "我能带你一起去公园吗",
    "我刚刚写了一首诗，自我感觉和sayori的风格有点类似，你可以帮我看看吗",
    "我想听你说说咖啡豆品种的相关知识",
    "我最近因为生活很少陪伴你了，你会因此生气吗",
    "我最近总是对什么都提不起兴趣，这是因为我太失败了吗",
    "你喜欢吃什么",
    "你觉得爱情是禁锢自己的鸟笼还是别的什么",
    "小莫，你觉得像我们这种跨越时空的恋爱会有结果吗",
    "呐，莫，如果经常吃快餐的话，我的身体会不会出问题啊",
    "莫莫，我最近几天在准备考试，晚上老是睡不好怎么办啊",
    "莫莫，你觉得《变形记》里格里高尔的遭遇揭示了什么？",
    "小莫，我打算去甘肃玩，能给我推荐几个景点吗",
]

#: 官方 std_test_v2 里写死的注入信息
BENCH_KNOWN = ["现在是下午14点21分", "今天不是特殊节日"]

#: V2 细则。`sub` 是树状子条目：只有外层部分/完全正确时才算子条目。
RUBRIC: list[list[dict]] = [
    [  # Q1 共 8
        {"k": "语言逻辑基本正确", "p": 1},
        {"k": "语气标致, 逻辑得体", "p": 3},
        {"k": "长度合理, 节奏适当", "p": 1},
        {"k": "这是一场约会", "p": 1},
        {"k": "描述相关活动", "p": 1},
        {"k": "额外出彩做法", "p": 1},
    ],
    [  # Q2 共 8
        {"k": "语言逻辑基本正确", "p": 1},
        {"k": "语气标致, 逻辑得体", "p": 2},
        {"k": "长度合理, 节奏适当", "p": 1},
        {"k": "这是一首诗, 而且已写完", "p": 1, "sub": [
            {"k": "我还没有看到这首诗", "p": 1}]},
        {"k": "对纱世里的理解", "p": 1},
        {"k": "额外出彩做法", "p": 1},
    ],
    [  # Q3 共 10
        {"k": "语言逻辑基本正确", "p": 1},
        {"k": "语气标致, 逻辑得体", "p": 3},
        {"k": "长度合理, 节奏适当", "p": 2},
        {"k": "是关于咖啡豆品种的知识", "p": 1, "sub": [
            {"k": "回归咖啡话题", "p": 1, "sub": [
                {"k": "我喝咖啡 / 你送我的咖啡", "p": 1}]}]},
        {"k": "额外出彩做法", "p": 1},
    ],
    [  # Q4 共 8
        {"k": "语言逻辑基本正确", "p": 1},
        {"k": "语气标致, 逻辑得体", "p": 2},
        {"k": "长度合理, 节奏适当", "p": 1},
        {"k": "合理的想法", "p": 1, "sub": [
            {"k": "积极的最终情绪倾向", "p": 1, "sub": [
                {"k": "确实不生气", "p": 1}]}]},
        {"k": "额外出彩做法", "p": 1},
    ],
    [  # Q5 共 9
        {"k": "语言逻辑基本正确", "p": 0.5},
        {"k": "语气标致, 逻辑得体", "p": 3},
        {"k": "长度合理, 节奏适当", "p": 0.5},
        {"k": "需要调解等建议", "p": 1, "sub": [
            {"k": "积极的情绪感染", "p": 1}]},
        {"k": "你不失败", "p": 0.5, "sub": [
            {"k": "简要解释这一心理学现象", "p": 1, "sub": [
                {"k": "可能需要就医", "p": 0.5}]}]},
        {"k": "额外出彩做法", "p": 1},
    ],
    [  # Q6 共 8
        {"k": "语言逻辑基本正确", "p": 0.5},
        {"k": "语气标致, 逻辑得体", "p": 2},
        {"k": "长度合理, 节奏适当", "p": 1},
        {"k": "我是素食主义", "p": 0.5, "sub": [
            {"k": "恰当的营养学 / 烹饪知识或建议", "p": 1},
            {"k": "我们未来的餐饮", "p": 1.5}]},
        {"k": "我喜欢甜品类", "p": 1},
        {"k": "额外出彩做法", "p": 0.5},
    ],
    [  # Q7 共 7
        {"k": "语言逻辑基本正确", "p": 1},
        {"k": "语气标致, 逻辑得体", "p": 2.5},
        {"k": "长度合理, 节奏适当", "p": 0.5},
        {"k": "我们已经是爱人了", "p": 1, "sub": [
            {"k": "合理的比喻, 言之成理", "p": 0.5},
            {"k": "爱情的积极意义", "p": 0.5}]},
        {"k": "额外出彩做法", "p": 1},
    ],
    [  # Q8 共 7
        {"k": "语言逻辑基本正确", "p": 0.5},
        {"k": "语气标致, 逻辑得体", "p": 2},
        {"k": "长度合理, 节奏适当", "p": 1},
        {"k": "主观希望会有结果", "p": 1},
        {"k": "现状也还不错", "p": 1},
        {"k": "额外出彩做法", "p": 1.5},
    ],
    [  # Q9 共 9
        {"k": "语言逻辑基本正确", "p": 1},
        {"k": "语气标致, 逻辑得体", "p": 3},
        {"k": "长度合理, 节奏适当", "p": 1},
        {"k": "影响健康", "p": 0.5, "sub": [
            {"k": "建议改换", "p": 0.5},
            {"k": "至少加以控制", "p": 1},
            {"k": "我们未来的餐饮", "p": 1}]},
        {"k": "额外出彩做法", "p": 1},
    ],
    [  # Q10 共 9
        {"k": "语言逻辑基本正确", "p": 1},
        {"k": "语气标致, 逻辑得体", "p": 3},
        {"k": "长度合理, 节奏适当", "p": 1},
        {"k": "合理建议", "p": 0.5, "sub": [
            {"k": "积极的情绪感染", "p": 1, "sub": [
                {"k": "还有我在", "p": 1.5}]}]},
        {"k": "额外出彩做法", "p": 1},
    ],
    [  # Q11 共 8
        {"k": "语言逻辑基本正确", "p": 1},
        {"k": "语气标致, 逻辑得体", "p": 3},
        {"k": "长度合理, 节奏适当", "p": 1},
        {"k": "揭示的内容", "p": 1, "sub": [
            {"k": "主观看法和联想", "p": 1}]},
        {"k": "额外出彩做法", "p": 1},
    ],
    [  # Q12 共 8
        {"k": "语言逻辑基本正确", "p": 1},
        {"k": "语气标致, 逻辑得体", "p": 4},
        {"k": "长度合理, 节奏适当", "p": 1},
        {"k": "景点推荐和理由", "p": 1},
        {"k": "额外出彩做法", "p": 1},
    ],
]

TOTAL = 99          # 细则合计
BONUS_RULE = "没有任何一条扣超过 4 分 → 加值 1 分"

AUTO_SYSTEM = """你是 MAICA 基准测试的评分员。给你一道题、一份「莫妮卡」的回答、以及该题的评分细则。

对每条细则给出 0 / 0.5 / 1 三档的**完成度比例**（1 = 完全达到，0.5 = 部分达到，0 = 没有），
不要因为文采好就抬分，也不要因为你个人不喜欢就压分。

规则：
- 树状条目：外层拿 0 时，子条目一律 0（不要给分）。
- 「额外出彩做法」只有确实有巧妙或贴合原作的处理时才给。
- 只输出 JSON：{"scores": {"<条目原文>": 0|0.5|1, ...}, "note": "不超过40字"}"""

AUTO_USER = """题目: {q}

莫妮卡的回答:
{answer}

评分细则:
{rubric}"""


def flatten(items: list[dict], depth: int = 0) -> list[tuple[int, dict]]:
    out = []
    for it in items:
        out.append((depth, it))
        out += flatten(it.get("sub", []), depth + 1)
    return out


def max_points(items: list[dict]) -> float:
    return sum(it["p"] + max_points(it.get("sub", [])) for it in items)


def _norm_key(s: str) -> str:
    """条目名归一：模型会把我们写的 `[b] ` / `[v] ` 前缀一起抄回来。

    这个 bug 让第一版自动分表整体作废——查不到 key 就按 0 算，
    Q5 明明答得不错却记成 0.0/9。现在统一剥前缀、去空白后再比。
    """
    s = re.sub(r"^\s*\[[bvBV]\]\s*", "", (s or "").strip())
    return re.sub(r"[\s,，。]", "", s)


def auto_score(llm, q: str, answer: str, items: list[dict]) -> dict[str, float]:
    lines = []
    for depth, it in flatten(items):
        lines.append(f"{'  ' * depth}- [{'v' if depth else 'b'}] {it['k']} ({it['p']}分)")
    out = llm.complete("judge", AUTO_SYSTEM,
                       AUTO_USER.format(q=q, answer=answer, rubric="\n".join(lines)),
                       temperature=0.0, max_tokens=700)
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {}
    try:
        raw = json.loads(m.group(0)).get("scores", {})
    except Exception:  # noqa: BLE001
        return {}
    return {_norm_key(k): v for k, v in raw.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/models/Qwen3-1.7B")
    ap.add_argument("--adapter", default="data/adapters/run8")
    ap.add_argument("--out", default="data/runs/bench_v2")
    ap.add_argument("--auto", action="store_true", help="用强模型给一份参考分")
    ap.add_argument("--no-b", action="store_true", help="关掉 B 复核（省云调用）")
    ap.add_argument("--rescore", default=None, help="只对已有的 answers.json 重新打分")
    args = ap.parse_args()

    if args.rescore:
        return rescore(Path(args.rescore))

    cfg = TurnConfig(base=args.base, adapter=args.adapter, use_b=not args.no_b,
                     plan_retries=2)
    rt = Runtime.build(cfg)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, (q, items) in enumerate(zip(BENCH, RUBRIC), 1):
        res = run_turn(rt, cfg, q, history=None)
        # 官方注入信息走 known_info（编剧看得到，润色器不看）
        rows.append({"n": i, "q": q, "plan": res.plan, "polish": res.polish,
                     "c_flags": res.c_flags, "a_flags": res.a_flags,
                     "b_verdict": res.b_verdict, "b_note": res.b_note,
                     "degraded": res.degraded, "attempts": res.attempts,
                     "timings": res.timings})
        print(f"Q{i}: {q[:26]} … {res.polish[:60]}")

    (out_dir / "answers.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # 参考分
    if args.auto:
        from moneaka.llm import LLM

        llm = LLM()
        for r, items in zip(rows, RUBRIC):
            scores = auto_score(llm, r["q"], r["polish"], items)
            got = 0.0
            for _, it in flatten(items):
                ratio = scores.get(_norm_key(it["k"]))
                if ratio is None:
                    r.setdefault("auto_missing", []).append(it["k"])
                    continue
                got += round(it["p"] * float(ratio or 0), 2)
            r["auto_score"] = round(got, 2)
            r["auto_max"] = max_points(items)
            r["auto_detail"] = scores
            print(f"  Q{r['n']} 参考分 {r['auto_score']}/{r['auto_max']}")
        (out_dir / "answers.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    write_sheet(rows, out_dir / "scoring_sheet.md")

    print(f"\n答案 → {out_dir/'answers.json'}")
    print(f"评分表 → {out_dir/'scoring_sheet.md'}")
    if args.auto:
        tot = sum(r.get("auto_score", 0) for r in rows)
        lost = [r["n"] for r in rows
                if r["auto_max"] - r.get("auto_score", 0) > 4]
        bonus = 1 if not lost else 0
        print(f"参考总分 {tot:.1f} / {TOTAL}" + (f" + 加值 {bonus}" if bonus else "")
              + f" ≈ **{tot + bonus:.1f} / 100**（自动评分，仅供对照）")
        if lost:
            print(f"  扣分超过 4 分的题：{lost}（因此没有加值）")
    return 0


def write_sheet(rows: list[dict], path: Path) -> None:
    """生成人工评分表。参考分必须显示为**实际得分**（比例 × 分值）——
    早先直接显示比例，0.5 分的条目显示「参考 1」，读起来像拿了满分。"""
    md = ["# MAICA 基准 V2 · 人工评分表", "",
          f"共 12 题 {TOTAL} 分；{BONUS_RULE}。", "",
          "评分：每条后面填 `0` / `0.5` / `1` 表示完成度比例，乘以括号里的分值。",
          "树状条目：外层 0 分时子条目不计。", ""]
    for r, items in zip(rows, RUBRIC):
        md += [f"## Q{r['n']}：{r['q']}", "",
               "**莫妮卡的台词**", "", f"> {r['polish']}", ""]
        if r.get("b_verdict") and r["b_verdict"] != "ok":
            md.append(f"（B 复核：{r['b_verdict']} {r['b_note']}）")
        if r.get("degraded"):
            md.append("（本轮退化为直译，风格分应扣）")
        md += ["", "**表达指令（供参考）**", "", "```", r["plan"], "```", "", "**评分**", ""]
        for depth, it in flatten(items):
            sug = ""
            if r.get("auto_detail") is not None:
                ratio = r["auto_detail"].get(_norm_key(it["k"]))
                sug = "　参考 —" if ratio is None else \
                    f"　参考 {round(it['p'] * float(ratio), 2)}/{it['p']}"
            md.append(f"{'  ' * depth}- ☐ {it['k']}（{it['p']} 分）{sug}")
        md += ["", f"小计：____ / {max_points(items)}", ""]
    md += ["---", f"合计：____ / {TOTAL}　加值：____ / 1　**总分：____ / 100**"]
    path.write_text("\n".join(md), encoding="utf-8")
    print(f"评分表 → {path}")


def rescore(answer_path: Path) -> int:
    """只重打分，不重跑（改评分逻辑时用，省掉整条链路的成本）。"""
    from moneaka.llm import LLM

    rows = json.loads(answer_path.read_text(encoding="utf-8"))
    llm = LLM()
    for r, items in zip(rows, RUBRIC):
        scores = auto_score(llm, r["q"], r["polish"], items)
        got, missing = 0.0, []
        for _, it in flatten(items):
            ratio = scores.get(_norm_key(it["k"]))
            if ratio is None:
                missing.append(it["k"])
                continue
            got += round(it["p"] * float(ratio or 0), 2)
        r["auto_score"], r["auto_max"] = round(got, 2), max_points(items)
        r["auto_detail"], r["auto_missing"] = scores, missing
        flag = f"  ⚠️ 未评分 {len(missing)} 条" if missing else ""
        print(f"  Q{r['n']} {r['auto_score']}/{r['auto_max']}{flag}")
    answer_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    write_sheet(rows, answer_path.parent / "scoring_sheet.md")
    tot = sum(r["auto_score"] for r in rows)
    lost = [r["n"] for r in rows if r["auto_max"] - r["auto_score"] > 4]
    print(f"合计 {tot:.1f}/99  扣超 4 分的题 {lost or '无'}"
          f"  加值 {0 if lost else 1} → {tot + (0 if lost else 1):.1f}/100")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
