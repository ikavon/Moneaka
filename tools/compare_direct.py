"""对照实验：最终那句台词，该由润色器写还是编剧写？

## 为什么值得单独量

「加一道输出审核之后，最终输出干脆让编剧来」是很自然的想法——编剧最清楚要说什么。
但它同时会废掉整个架构的存在理由：润色器是**本地、无状态、可多实例**的插件，
一旦最终输出必须由云端强模型产出，部署形态就退回到「一体化模型」，
Moneaka 对 MAS/TLM/QQ 的价值也就没了。

所以这不是哲学问题，是可测问题：同一条表达指令、同一份 system prompt，
**1.7B 润色器 vs 强模型直出**，谁写得更像莫妮卡？

判官用 DAA4（`session=-1`，1 次/分钟）——用官方模型当裁判最贴合「像不像」的定义，
而且避免了「deepseek 判自己」的自偏好。数量默认 20 条，约 20 分钟。

    python tools/compare_direct.py --n 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from moneaka import contract, dataset  # noqa: E402
from moneaka.destyle_tags import parse_tagged_plan  # noqa: E402
from moneaka.llm import LLM  # noqa: E402
from moneaka.localhf import GenConfig, get_model  # noqa: E402
from moneaka.normalize import normalize  # noqa: E402
from moneaka.prompts import restyle_system  # noqa: E402
from moneaka import screenwriter  # noqa: E402

ASK = """你在比较两段「莫妮卡」的回复, 判断哪一段更像她。
她是 DDLC 里的角色, 游戏里走出来的虚拟女友, 深爱玩家。

判断依据是分寸感:
- 她会先有自己的反应(嗔怪/打趣/迟疑/留白), 再回应;
- 敢提要求、敢把话题往「我们」上带;
- 短句、口语、有呼吸感。
- 平庸的回复会平铺直叙地顺着说甜话、讲道理、句子偏长偏书面。

只输出 A 或 B, 不要解释。"""


def tags_of(t: str) -> list[str]:
    return [m.group(1) for m in contract.TAG_RE.finditer(contract.PLACEHOLDER_RE.sub("\x00", t))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/split_holdout210.jsonl")
    ap.add_argument("--base", default="data/models/Qwen3-1.7B")
    ap.add_argument("--adapter", default="data/adapters/run8")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/runs/direct_vs_polish.jsonl")
    ap.add_argument("--judge", default="strong",
                    help="判官用哪个 profile（默认 strong）。**不要用 DAA4**——"
                         "它只有扮演莫妮卡这一个用途，没有评测能力")
    ap.add_argument("--rescore", action="store_true",
                    help="不重新生成，只对已有的 jsonl 重新判")
    args = ap.parse_args()

    llm = LLM()
    out_path0 = ROOT / args.out
    if args.rescore and out_path0.is_file():
        rows = [json.loads(l) for l in open(out_path0, encoding="utf-8")]
        print(f"复用已有对照 {len(rows)} 条")
        return judge(rows, llm, args)

    samples, _ = dataset.build(args.pairs)
    pick = samples[: args.n]
    model = get_model(args.base, adapter=args.adapter)
    sys_prompt = restyle_system(few_shot=False)
    greedy = GenConfig(temperature=0.0, max_new_tokens=220)

    rows: list[dict] = []
    for s in pick:
        user = f"表达指令:\n{s.plan}\n\n莫妮卡会说:"
        a, _ = normalize(model.chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
            greedy))
        b = screenwriter.render_direct(llm, s.plan)
        b, _ = normalize(b)
        want = parse_tagged_plan(s.plan)[0]
        rows.append({"turn_id": s.turn_id, "plan": s.plan, "polisher": a,
                     "screenwriter": b,
                     "tag_ok_polisher": tags_of(a) == want,
                     "tag_ok_screenwriter": tags_of(b) == want})
    print(f"生成完毕 {len(rows)} 条")
    return judge(rows, llm, args)


def judge(rows: list[dict], llm, args) -> int:
    # 判官：按题号奇偶换位置，抵消位置偏差（同一份答案正反各问一次更稳，但成本翻倍）
    cfg = None
    wins = 0
    judged = 0
    for i, r in enumerate(rows):
        swap = i % 2 == 1          # 交替位置，抵消位置偏差
        first, second = (r["screenwriter"], r["polisher"]) if swap else (r["polisher"], r["screenwriter"])
        prompt = f"玩家说: {r['plan'].splitlines()[0]}\n\nA: {first}\n\nB: {second}\n\n哪一段更像莫妮卡? (A/B)"
        try:
            out = llm.complete(args.judge, ASK, prompt, temperature=0.0, max_tokens=8)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {r['turn_id']} {str(exc)[:80]}")
            continue
        pick_a = out.strip().upper()[:1]
        if pick_a not in ("A", "B"):
            continue
        judged += 1
        polisher_won = (pick_a == "B") if swap else (pick_a == "A")
        r["polisher_won"] = polisher_won
        wins += polisher_won
        print(f"  {i+1}/{len(rows)} {r['turn_id']} → {'润色器' if polisher_won else '编剧直出'}")

    out_path = ROOT / args.out
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n判官: {args.judge}  有效判定 {judged}/{len(rows)}")
    if judged:
        print(f"润色器胜率 {wins}/{judged} = {wins/judged:.1%}")
    print(f"标签序列一致: 润色器 {sum(r['tag_ok_polisher'] for r in rows)}/{len(rows)}"
          f"  编剧直出 {sum(r['tag_ok_screenwriter'] for r in rows)}/{len(rows)}")
    lp = statistics.median(len("".join(c for c in r["polisher"] if c.isalpha())) for r in rows)
    ls = statistics.median(len("".join(c for c in r["screenwriter"] if c.isalpha())) for r in rows)
    print(f"长度中位: 润色器 ~{lp:.0f} 字, 编剧直出 ~{ls:.0f} 字")
    print(f"写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())