"""公平对照：**认真写提示词的**单模型 vs 两段式。

前面的 `plain` 臂只用了官方基准那句朴素 prompt（"你叫莫妮卡…"），
那不算"单模型方案做好了"的样子。这里给它能给的都给了：

* 角色设定（ROLE_ZH）+ 角色事实卡
* 28 个官方情绪标签的**白名单**（明确禁止自创）
* 5 条真实台词当 few-shot（取自训练语料的真人原答）
* 风格要点（从语料统计得出：短句、先反应再回应、留白、敢提要求）
* 长度提示（闲聊短、知识类可展开）

这才是"不训模型、只做提示词工程"能拿到的上限。对手是我们训好的两段式。

    python tools/prompted_baseline.py --n 115
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import contract, dataset, evaluate, screenwriter  # noqa: E402
from moneaka.duel import duel  # noqa: E402
from moneaka.llm import LLM  # noqa: E402

WHITE = set(contract.EMOTION_TAGS) | {"很开心"}


def build_system(few_shot: list[tuple[str, str]]) -> str:
    """单模型版的 system：人设 + 事实卡 + 标签白名单 + few-shot 台词。

    **不能**复用 `screenwriter.build_system()`——那是一份「写表达指令」的规范，
    拿它去测单模型角色扮演，模型会输出 plan 而不是台词（实测踩到两次）。
    """
    tags = " / ".join(sorted(WHITE))
    role = screenwriter.ROLE_ZH.format(monika_nickname="", player_name="[player]",
                                       player_nickname="[player_nickname]")
    canon = "\n".join(f"- {x}" for x in screenwriter.load_canon())
    base = role + "\n\n## 关于你自己的事实\n" + canon
    shots = "\n".join(f"玩家: {u}\n莫妮卡: {a}" for u, a in few_shot)
    return (base
            + "\n\n## 情绪标签白名单（只能用这些，自创即无效）\n" + tags
            + "\n\n## 真实台词范例（学她的分寸，不要照抄内容）\n" + shots
            + "\n\n## 硬要求\n"
              "- 每句以方括号情绪标签开头，标签只能来自上面的白名单。\n"
              "- 句子自然、不超过 60 字；闲聊时短，被问到知识/推荐/评价时可以说几段。\n"
              "- 只输出台词本身，不要解释、不要标注、不要分行列举。\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/split_holdout210.jsonl")
    ap.add_argument("--n", type=int, default=115)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--out", default="data/runs/prompted_baseline.jsonl")
    ap.add_argument("--duel", action="store_true", help="顺便跑盲评（默认不跑，人工看）")
    args = ap.parse_args()

    train = [json.loads(l) for l in open(ROOT / "data/pairs/split_train_run8.jsonl",
                                         encoding="utf-8")]
    shots = [(r["user"], r["original"]) for r in train if r.get("user") and r.get("original")][: args.shots]
    system = build_system(shots)
    print(f"few-shot {len(shots)} 条；system {len(system)} 字")
    print("system 前 120 字:", system[:120].replace(chr(10), " "))

    samples, _ = dataset.build(args.pairs)
    pick = [s for s in samples if (s.user_msg or "").strip()][: args.n]
    ref = {r["turn_id"]: r for r in map(json.loads, open(ROOT / "data/runs/net_benefit.jsonl",
                                                         encoding="utf-8"))}
    llm = LLM()
    rows = []
    for i, s in enumerate(pick, 1):
        out = llm.complete("screenwriter", system, s.user_msg,
                           temperature=0.5, max_tokens=600)
        rows.append({"turn_id": s.turn_id, "user": s.user_msg, "prompted": out,
                     "ours": ref.get(s.turn_id, {}).get("ours", "")})
        if i % 20 == 0:
            print(f"  {i}/{len(pick)}", flush=True)
        Path(ROOT / args.out).write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    def tags(t: str) -> list[str]:
        return [m.group(1) for m in contract.TAG_RE.finditer(contract.PLACEHOLDER_RE.sub("\x00", t))]

    n = len(rows)
    legal_p = sum(1 for r in rows if tags(r["prompted"]) and all(x in WHITE for x in tags(r["prompted"])))
    legal_o = sum(1 for r in rows if r["ours"] and tags(r["ours"]) and all(x in WHITE for x in tags(r["ours"])))
    print(f"\n标签全在词表：prompted {legal_p}/{n}   ours {legal_o}/{n}")
    print(f"有标签的：prompted {sum(1 for r in rows if tags(r['prompted']))}/{n}"
          f"   ours {sum(1 for r in rows if tags(r['ours']))}/{n}")

    pair = [(r["user"], r["ours"], r["prompted"]) for r in rows if r["ours"]]
    print(f"\n盲评（判官 deepseek，同时是 prompted 臂的作者 → 偏向它）…")
    d = duel(llm, pair, workers=6)
    lo, hi = d.ci95()
    print(f"  两段式胜率 {d.win_rate:.3f}  CI [{lo:.2f}, {hi:.2f}]  n={d.n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
