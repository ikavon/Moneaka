"""人类盲测：生成对照表 + 事后评分（含人机一致率）。

为什么需要：自动评委只有成对比较可信（官方 chosen/rejected 上 88%），
而「像不像莫妮卡」的最终标准在人这里。这份表做三件事：

1. 四臂（draft / direct / polish / reference）**随机打乱**成 A/B/C/D，人不看来源；
2. 标签与臂的对应、以及原文本存进 key 文件，填表期间不要看；
3. `--grade` 统计每个臂的人类得票，并用同一批题跑成对比较，**算人机一致率**
   ——一致率是验收那把尺子的，不是用来看谁赢。

用法：
    python tools/human_eval.py --name z17t0 --sample 20
    #   → docs/HUMAN_EVAL.md（填「最像:」「最不像:」）
    #   → docs/HUMAN_EVAL_KEY.json（别偷看）
    python tools/human_eval.py --grade docs/HUMAN_EVAL.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARMS = ("draft", "direct", "polish", "reference")
LABELS = "ABCD"
BEST_RE = re.compile(r"最像[:：]\s*([ABCD])", re.I)
WORST_RE = re.compile(r"最不像[:：]\s*([ABCD])", re.I)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_conversations(run: str, pairs_path: Path, min_turns: int = 3) -> list[dict]:
    """把同一个 row（一段连续对话）的多轮拼起来，每个臂生成一份完整台词本。

    为什么要连续对话而不是单句：上一版用单条玩家消息出题，人和自动评委都在真空里判，
    结果 direct（甜但没性格）反而拿了 50% 最像票——因为单句看不出她会不会「接住长线话题」。
    """
    arms_rows = {r["turn_id"]: r for r in load_jsonl(ROOT / "data" / "runs" / run / "arms.jsonl")}
    by_row: dict[int, list[dict]] = {}
    for p in load_jsonl(pairs_path):
        tid = p["turn_id"]
        if tid not in arms_rows:
            continue
        row = int(tid.split("-")[0])
        by_row.setdefault(row, []).append((int(tid.split("-")[1]), p, arms_rows[tid]))

    convos: list[dict] = []
    for row, items in sorted(by_row.items()):
        items.sort(key=lambda x: x[0])
        if len(items) < min_turns:
            continue
        if any(not (a.get("direct") and a.get("polish")) for _, _, a in items):
            continue
        cands = {
            "draft": "\n\n".join(f"[player] {p['user']}\nMonika {p['draft']}" for _, p, _ in items),
            "reference": "\n\n".join(f"[player] {p['user']}\nMonika {p['original']}" for _, p, _ in items),
            "direct": "\n\n".join(f"[player] {p['user']}\nMonika {a['direct']}" for _, p, a in items),
            "polish": "\n\n".join(f"[player] {p['user']}\nMonika {a['polish']}" for _, p, a in items),
        }
        convos.append({"row": row, "turns": len(items),
                       "tids": [ti for ti, _, _ in items], "cands": cands})
    return convos


def convo_sheet(convos: list[dict], seed: int) -> tuple[str, dict]:
    rng = random.Random(seed)
    key: dict[str, dict] = {}
    L = [
        "# 人类盲测表（连续对话版）\n",
        "每组是**同一段对话的四个版本**（玩家的话完全一样，只有莫妮卡的回复不同）。\n",
        "**怎么填**：读完四个版本后，在「最像:」后写一个字母，「最不像:」后写一个字母。\n",
        "判据只有「像不像她」——尤其注意：**她接不接得住长线话题、会不会把对话往「我们」上带、\n",
        "有没有自己的立场**。单句好看没用的，这里能看出来。\n",
        "四个版本来自不同系统（含真实 DAA4，也含完全没加工的平实草稿），顺序随机。\n",
        "---\n",
    ]
    for i, c in enumerate(convos, 1):
        order = list(ARMS)
        rng.shuffle(order)
        cid = f"row{c['row']}"
        key[cid] = {"labels": {LABELS[j]: arm for j, arm in enumerate(order)},
                    "turns": c["turns"], "tids": c["tids"], "cands": c["cands"]}
        L.append(f"## {i}  `{cid}`  （{c['turns']} 轮）\n")
        for j, arm in enumerate(order):
            L.append(f"### {LABELS[j]}\n")
            L.append("```")
            L.append(c["cands"][arm])
            L.append("```")
            L.append("")
        L.append("最像: 　　最不像: ")
        L.append("")
        L.append("---")
    return "\n".join(L) + "\n", key


def build_rows(run: str, pairs_path: Path) -> list[dict]:
    arms_rows = {r["turn_id"]: r for r in load_jsonl(ROOT / "data" / "runs" / run / "arms.jsonl")}
    pairs = {p["turn_id"]: p for p in load_jsonl(pairs_path)}
    rows = []
    for tid, a in arms_rows.items():
        p = pairs.get(tid, {})
        cands = {
            "draft": a.get("draft") or p.get("draft", ""),
            "direct": a.get("direct") or "",
            "polish": a.get("polish") or "",
            "reference": p.get("original", ""),
        }
        if any(not v for v in cands.values()):
            continue
        rows.append({"turn_id": tid, "user": p.get("user", ""), "cands": cands})
    return rows


def sheet(rows: list[dict], seed: int) -> tuple[str, dict]:
    rng = random.Random(seed)
    key: dict[str, dict] = {}
    L = [
        "# 人类盲测表\n",
        "**怎么填**：每组读完四个选项后，在「最像:」后写一个字母（最像莫妮卡的那句），",
        "在「最不像:」后写一个字母。四个选项来自不同系统，**顺序是随机打的**，",
        "其中包含真实 DAA4 的输出，也包含完全没加工过的平实草稿。\n",
        "判据只有「像不像她」——不看哪句更有道理、更体贴、更长、更像好回答。\n",
        "---\n",
    ]
    for i, r in enumerate(rows, 1):
        order = list(ARMS)
        rng.shuffle(order)
        key[r["turn_id"]] = {"labels": {LABELS[j]: arm for j, arm in enumerate(order)},
                             "user": r["user"], "cands": r["cands"]}
        L.append(f"## {i}  `{r['turn_id']}`\n")
        L.append(f"玩家说: {r['user']}\n")
        for j, arm in enumerate(order):
            L.append(f"- **{LABELS[j]}**. {r['cands'][arm]}")
        L.append("")
        L.append("最像: 　　最不像: ")
        L.append("")
        L.append("---")
    return "\n".join(L) + "\n", key


def parse_answers(md_path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    text = md_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n##\s+\d+\s+", "\n" + text)[1:]
    best: dict[str, str] = {}
    worst: dict[str, str] = {}
    tids: list[str] = []
    for block in blocks:
        m = re.match(r"`([^`]+)`", block)
        if not m:
            continue
        tid = m.group(1)
        tids.append(tid)
        b, w = BEST_RE.search(block), WORST_RE.search(block)
        if b:
            best[tid] = b.group(1).upper()
        if w:
            worst[tid] = w.group(1).upper()
    return best, worst, tids


def grade(md_path: Path, key_path: Path, *, want_agreement: bool, pairs: list[tuple[str, str]],
          sample: int) -> int:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    best_lab, worst_lab, tids = parse_answers(md_path)
    if not best_lab:
        print("还没填答案（或没匹配到题号）")
        return 1

    best: Counter[str] = Counter()
    worst: Counter[str] = Counter()
    # 人真正选过的臂，用于后续成对比较的题面
    used: list[tuple[str, dict]] = []
    for tid, lab in best_lab.items():
        arm = key.get(tid, {}).get("labels", {}).get(lab)
        if arm:
            best[arm] += 1
            used.append((key[tid]["user"], key[tid]["cands"]))
    for tid, lab in worst_lab.items():
        arm = key.get(tid, {}).get("labels", {}).get(lab)
        if arm:
            worst[arm] += 1

    filled = len(best_lab)
    left = [t for t in tids if t not in best_lab]
    print(f"已填 {filled}/{len(tids)} 组" + (f"，未填: {left[:8]}" if left else ""))

    print(f"\n{'臂':12s} {'最像得票':>8s} {'占比':>7s} {'最不像得票':>10s}")
    for arm in ARMS:
        print(f"{arm:12s} {best[arm]:>8d} {best[arm] / filled:>6.0%} {worst[arm]:>10d}")

    print("\n注：reference 是真实 DAA4 输出（天花板）；draft 是完全没加工的平实草稿（地板）。")
    if best["reference"]:
        print(f"    polish / reference 得票比 = {best['polish'] / best['reference']:.2f}"
              f"　（draft / reference = {best['draft'] / best['reference']:.2f}，"
              f"direct / reference = {best['direct'] / best['reference']:.2f}）")

    if want_agreement and used and all("\n" not in v for cands in used for v in cands[1].values()):
        from moneaka import duel
        from moneaka.llm import LLM

        llm = LLM()
        print("\n人机一致率（同一批题，每题正反各问一次）：")
        items_by_pair = {}
        for user, cands in used:
            for a, b in (("polish", "reference"), ("polish", "direct"), ("polish", "draft")):
                items_by_pair.setdefault((a, b), []).append((user, cands[a], cands[b]))
        for (a, b), items in items_by_pair.items():
            use = items[:sample]
            res = duel.duel(llm, use, swap=True)
            tot = best[a] + best[b]
            human = best[a] / tot if tot else float("nan")
            machine = res.win_rate
            if machine is None or tot == 0:
                print(f"  {a} vs {b}: 样本不足")
                continue
            same = (machine >= 0.5) == (human >= 0.5)
            print(f"  {a} vs {b}: 人判 {a} {human:.0%} | 评委判 {a} {machine:.0%} "
                  f"({'同向 ✅' if same else '分歧 ❌'}，n={len(use)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="z17t0")
    ap.add_argument("--pairs", default=str(ROOT / "data" / "pairs" / "pairs60.jsonl"))
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", choices=["single", "convo"], default="single",
                   help="single=逐句四选一（旧版，已被证明容易误判）；convo=连续对话四选一")
    ap.add_argument("--min-turns", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "docs" / "HUMAN_EVAL.md"))
    ap.add_argument("--key", default=str(ROOT / "docs" / "HUMAN_EVAL_KEY.json"))
    ap.add_argument("--grade", default=None, help="填完的表")
    ap.add_argument("--no-agreement", action="store_true", help="评分时不再调用评委")
    args = ap.parse_args()

    if args.grade:
        key_path = Path(args.key)
        return grade(Path(args.grade), key_path,
                     want_agreement=not args.no_agreement,
                     pairs=[], sample=args.sample)

    if args.mode == "convo":
        convos = build_conversations(args.name, Path(args.pairs), args.min_turns)
        rng = random.Random(args.seed)
        rng.shuffle(convos)
        convos = convos[: args.sample]
        if not convos:
            print(f"没有 ≥{args.min_turns} 轮且四臂齐全的对话；先跑 polish 补数据，或降低 --min-turns")
            return 1
        md, key = convo_sheet(convos, args.seed)
    else:
        rows = build_rows(args.name, Path(args.pairs))
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.sample]
        md, key = sheet(rows, args.seed)

    Path(args.out).write_text(md, encoding="utf-8")
    Path(args.key).write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {args.out}（{len(key)} 组，mode={args.mode}）")
    print(f"密钥 {args.key} —— 填表期间别看")
    print(f"填完跑: python tools/human_eval.py --mode {args.mode} --grade {args.out} --key {args.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

def export_preferences(md_path: Path, key_path: Path, out_path: Path,
                       pairs_path: Path) -> int:
    """把填好的表转成偏好对（格式对齐官方 RLHF：messages / rejected_messages）。

    人类的一次「最像 vs 最不像」= 一组 chosen/rejected。这份数据有两个用途：
    1. 阶段 1 的 DPO 偏好信号（比只用 chosen 的 SFT 信息量大）；
    2. 自动评委的**留出验证集**——评委在官方数据上 88%，但那是它见过的分布；
       对自己系统输出的判断（polish vs direct）它判反了，得用这份重校。
    """
    key = json.loads(key_path.read_text(encoding="utf-8"))
    pairs = {p["turn_id"]: p for p in load_jsonl(pairs_path)}
    best_lab, worst_lab, _ = parse_answers(md_path)
    rows = []
    for tid, bl in best_lab.items():
        wl = worst_lab.get(tid)
        info = key.get(tid)
        if not (info and wl):
            continue
        labels = info["labels"]
        chosen_arm, rejected_arm = labels.get(bl.upper()), labels.get(wl.upper())
        if not chosen_arm or not rejected_arm or chosen_arm == rejected_arm:
            continue
        cands = info["cands"]
        p = pairs.get(tid, {})
        user = info.get("user") or p.get("user", "")
        ctx = [m for m in p.get("context", []) if m.get("role") != "system"]
        rows.append({
            "turn_id": tid,
            "context": ctx,
            "messages": ctx + [{"role": "user", "content": user},
                               {"role": "assistant", "content": cands[chosen_arm]}],
            "rejected_messages": ctx + [{"role": "user", "content": user},
                                        {"role": "assistant", "content": cands[rejected_arm]}],
            "chosen_arm": chosen_arm,
            "rejected_arm": rejected_arm,
            "source": "human",
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写入 {out_path}（{len(rows)} 对偏好）")
    dist = Counter(f"{r['chosen_arm']}>{r['rejected_arm']}" for r in rows)
    print("  方向分布:", dict(dist))
    return 0
