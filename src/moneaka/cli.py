"""``moneaka`` 命令行。

    moneaka stats                     # 不联网：语料盘点 + 契约阈值自检
    moneaka destyle --limit 40        # 强 LLM 造平实草稿 → data/pairs/pairs.jsonl
    moneaka polish --name z1          # 润色 + 对照组 → data/runs/z1/arms.jsonl
    moneaka eval --name z1 --embed    # 打分出报告 → data/runs/z1/report.md
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from . import contract, corpus, destyle as destyle_mod, evaluate, prompts, restyle
from .llm import LLM, dump_jsonl, load_jsonl
from .normalize import NormStats, normalize

DATA = "data"
PAIRS = os.path.join(DATA, "pairs", "pairs.jsonl")


def _runs_dir(name: str) -> str:
    return os.path.join(DATA, "runs", name)


# ---------------------------------------------------------------------------
# stats —— 不需要任何 API，先跑这个
# ---------------------------------------------------------------------------

def cmd_stats(args: argparse.Namespace) -> int:
    rows = corpus.load_rows(args.source)
    turns = corpus.chosen_turns(rows)
    contrasts = corpus.contrasts(rows)
    print(f"文件: {args.source}")
    print(f"  行数(偏好对) {len(rows)} | assistant 轮 {len(turns)} | 有 rejected 对照的轮 {len(contrasts)}")

    lens = [contract.cjk_len(t.chosen) for t in turns]
    lens.sort()
    print(f"  chosen 单轮字数: min {lens[0]} / 中位 {lens[len(lens)//2]} / p90 {lens[int(len(lens)*0.9)]} / max {lens[-1]}")

    tagc = Counter()
    for t in turns:
        for tg in contract.TAG_RE.findall(t.chosen):
            if tg != "player":
                tagc[tg] += 1
    print(f"  情绪标签 {len(tagc)} 种: " + ", ".join(f"{k}×{v}" for k, v in tagc.most_common()))

    # ⭐ 阈值自检：拿真数据当考题。真数据触发的违规 = 我们的校验器写错了
    fails = [(t, contract.check(t.chosen)) for t in turns]
    bad = [(t, r) for t, r in fails if not r.ok]
    print(f"\n契约校验在真实 chosen 上的通过率: {1 - len(bad) / len(turns):.1%} "
          f"（{len(turns) - len(bad)}/{len(turns)}）")
    print("  注：reference 不过 = 我们的阈值/切句错了，不是模型错。逐条看下面的样例。")
    codes = Counter()
    for _, r in bad:
        for v in r.violations:
            codes[str(v).split(":", 1)[0]] += 1
    if codes:
        print("  违规码分布（若 reference 也大量中招，先修 contract.py 的阈值再谈模型）:")
        for c, n in codes.most_common():
            print(f"    {c}: {n}")
        print("\n  样例:")
        for t, r in bad[:3]:
            print(f"    [{t.turn_id}] {t.chosen[:70]}")
            for v in r.violations:
                print(f"       - {v}")

    inj = [t for t in turns if t.injected_info]
    print(f"\n  带 known_info 注入的轮: {len(inj)}")

    # 标签单元层面：契约的真实单位是「一个标签 = 一句」，套多句只是参考值
    units = [contract.split_sentences(t.chosen) for t in turns]
    n_units = sum(len(u) for u in units)
    multi = sum(1 for u in units for s in u if len(contract.sentence_parts(s)) > 1)
    ulens = sorted(contract.cjk_len(contract.sentence_body(s)) for u in units for s in u)
    print(f"  标签单元 {n_units} 个：套两句以上的 {multi} 个（{100 * multi / n_units:.1f}%，不判违规）")
    print(f"  单元字数: min {ulens[0]} / 中位 {ulens[len(ulens)//2]} / p90 {ulens[int(len(ulens)*0.9)]} / max {ulens[-1]}")
    print(f"  每轮句数: 中位 {sorted(len(u) for u in units)[len(units)//2]} / max {max(len(u) for u in units)}")
    if args.show_skeleton:
        print("\n骨架 prompt（出现最多的 system）:\n" + corpus.skeleton_prompt(rows))
    return 0


# ---------------------------------------------------------------------------
# destyle
# ---------------------------------------------------------------------------

def cmd_destyle(args: argparse.Namespace) -> int:
    llm = LLM()
    print(f"端点: destyle={llm.profile('destyle').base_url} model={llm.profile('destyle').model}")
    destyle_mod.run(llm, limit=args.limit, workers=args.workers,
                    out_path=args.out or PAIRS, source=args.source)
    return 0


# ---------------------------------------------------------------------------
# polish
# ---------------------------------------------------------------------------

def cmd_polish(args: argparse.Namespace) -> int:
    pairs = load_jsonl(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    llm = LLM()
    prof = llm.profile("restyle")
    model = args.model or prof.model
    print(f"润色端点: {prof.base_url} model={model}")

    out_dir = _runs_dir(args.name)
    os.makedirs(out_dir, exist_ok=True)
    direct_profile = args.direct_profile or "restyle"
    if direct_profile != "restyle":
        dp = llm.profile(direct_profile)
        print(f"direct 端点: {direct_profile} -> {dp.base_url} model={args.direct_model or dp.model}")

    arms_rows = []
    t0 = time.time()
    for i, p in enumerate(pairs, 1):
        rec = dict(turn_id=p["turn_id"], user=p["user"], draft=p["draft"],
                   original=p["original"], rejected=p.get("rejected"),
                   polish=None, direct=None, norm=[])
        try:
            if not args.skip_polish:
                rec["polish"] = restyle.polish(llm, p["draft"], model=model)
        except Exception as exc:
            print(f"  ! {p['turn_id']} polish 失败: {str(exc)[:80]}")
        try:
            if not args.skip_direct:
                rec["direct"] = restyle.direct(llm, p["user"], context=p.get("context"),
                                                profile=direct_profile, model=args.direct_model)
        except Exception as exc:
            print(f"  ! {p['turn_id']} direct 失败: {str(exc)[:80]}")
        if args.normalize:
            # 归一化同时作用于两臂：它是通用输出后处理，不是某个模型的专利
            for arm in ("polish", "direct"):
                raw = rec.get(arm)
                if raw:
                    fixed, st = normalize(raw)
                    rec[f"{arm}_raw"] = raw
                    rec[arm] = fixed
                    rec["norm"].append({arm: st.as_dict()})
        arms_rows.append(rec)
        if i % 10 == 0 or i == len(pairs):
            print(f"  polish {i}/{len(pairs)}  {time.time() - t0:.0f}s")
    dump_jsonl(os.path.join(out_dir, "arms.jsonl"), arms_rows)
    print(f"写入 {os.path.join(out_dir, 'arms.jsonl')}（{len(arms_rows)} 条）")
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def cmd_eval(args: argparse.Namespace) -> int:
    pairs = load_jsonl(args.pairs)
    arms_path = os.path.join(_runs_dir(args.name), "arms.jsonl")
    arms_rows = load_jsonl(arms_path)
    by_id = {p["turn_id"]: p for p in pairs}
    rows = []
    for r in arms_rows:
        base = by_id.get(r["turn_id"], {})
        rows.append({
            "turn_id": r["turn_id"],
            "user": r.get("user") or base.get("user", ""),
            "draft": r.get("draft") or base.get("draft", ""),
            "original": r.get("original") or base.get("original", ""),
            "reference": r.get("original") or base.get("original", ""),
            "direct": r.get("direct"),
            "polish": r.get("polish"),
            # draft 单独作为一个臂：它就「平实草稿」本身。
            # 润色器真正贡献的 style = polish - draft，而不是 polish - 0。
            "draft_arm": r.get("draft") or base.get("draft", ""),
        })

    llm = LLM()
    embedder = None
    if args.embed:
        try:
            embedder = evaluate.Embedder(args.embed_model)
        except Exception as exc:
            print(f"  ! embedding 不可用，跳过 cos_ref 列: {exc}")

    scored = []
    arm_field = {"draft": "draft_arm", "reference": "reference",
                 "direct": "direct", "polish": "polish"}
    for arm in evaluate.ARMS:
        if args.only_arms and arm not in args.only_arms.split(","):
            continue
        print(f"  评分: {arm}")
        view = [{"turn_id": r["turn_id"], "draft": r["draft"], "user": r.get("user", ""),
                 arm: r.get(arm_field[arm])} for r in rows]
        scored += evaluate.score_arm(llm, view, arm, embedder=embedder,
                                     judge=not args.no_judge,
                                     batch_size=args.judge_batch,
                                     sample=args.judge_sample)

    agg = evaluate.summarize(scored)
    ref_viol = evaluate.violation_census(scored)
    for arm, a in agg.items():
        print(f"  {arm:9s} n={a['n']} pass={a['contract_pass_rate']} "
              f"semantic={a['semantic_mean']} style={a['style_mean']} cos_ref={a['cos_ref_mean']}")
    path = evaluate.write_report(_runs_dir(args.name), scored, agg, ref_viol, pairs)
    print(f"报告: {path}")
    return 0


# ---------------------------------------------------------------------------
# prompt —— 把契约/模板打出来，方便人工核对与复制到别处
# ---------------------------------------------------------------------------

def cmd_duel(args: argparse.Namespace) -> int:
    """成对比较：风格评测唯一站得住的自动指标（绝对打分已被证伪）。"""
    from . import duel

    llm = LLM()
    arms_text = args.arms.split(",")

    if args.calibrate:
        info = duel.calibrate(llm, n=args.calibrate)
        print("评委校准（官方 chosen vs rejected）:")
        print(f"  {info['comparisons']} 次比较, 准确率 {info['accuracy']:.0%} "
              f"CI95={info['ci95']}, 未解析 {info['unparsed']}")
        if info["accuracy"] < 0.7:
            print("  ⚠️ 低于 70%，这个评委不能用来做结论")
        return 0

    pairs = load_jsonl(args.pairs)
    arms_rows = {r["turn_id"]: r for r in load_jsonl(os.path.join(_runs_dir(args.name), "arms.jsonl"))}
    rows = []
    for tid, r in arms_rows.items():
        base = next((p for p in pairs if p["turn_id"] == tid), {})
        rows.append({"turn_id": tid, "user": base.get("user", ""),
                     "draft": r.get("draft") or base.get("draft", ""),
                     "reference": base.get("original", ""),
                     "direct": r.get("direct"), "polish": r.get("polish")})

    mat = duel.win_matrix(llm, rows, tuple(arms_text), sample=args.sample)
    print("\n胜率矩阵（行 vs 列；0.5=打平）\n")
    head = "{:<11s}".format("") + "".join(f"{a:>11s}" for a in arms_text)
    print(head)
    for a in arms_text:
        line = "{:<11s}".format(a)
        for b in arms_text:
            v = mat[a].get(b)
            line += f"{('  —  ' if v is None else f'{v:.2f}'):>11s}"
        print(line)
    print(f"\n每对比较次数(正反各一次): {mat[arms_text[1]].get('_n', '?')}")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    if args.kind == "restyle":
        print(prompts.restyle_system())
    elif args.kind == "contract":
        print(contract.contract_text())
    elif args.kind == "tags":
        print(json.dumps(list(contract.EMOTION_TAGS), ensure_ascii=False))
    else:
        print(f"未知 kind: {args.kind}")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="moneaka", description="莫妮卡口吻润色器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats", help="语料盘点 + 契约阈值自检（不联网）")
    p.add_argument("--source", default=None)
    p.add_argument("--show-skeleton", action="store_true")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("destyle", help="莫妮卡稿 → 平实草稿")
    p.add_argument("--source", default=None, help="语料 jsonl（默认官方分片）")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out")
    p.set_defaults(func=cmd_destyle)

    p = sub.add_parser("polish", help="草稿 → 润色稿（+ 通用模型直接扮演对照）")
    p.add_argument("--name", required=True)
    p.add_argument("--pairs", default=PAIRS)
    p.add_argument("--model")
    p.add_argument("--direct-model")
    p.add_argument("--direct-profile", default=None,
                   help="用哪个端点出 direct 对照组（默认 restyle=本地；填 strong 才是真竞品）")
    p.add_argument("--normalize", action="store_true",
                   help="对两臂输出做确定性归一化（标签白名单/逐句前置/句长/半角标点）")
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-polish", action="store_true")
    p.add_argument("--skip-direct", action="store_true")
    p.set_defaults(func=cmd_polish)

    p = sub.add_parser("eval", help="三臂打分 + 出报告")
    p.add_argument("--name", required=True)
    p.add_argument("--pairs", default=PAIRS)
    p.add_argument("--embed", action="store_true")
    p.add_argument("--embed-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--judge-batch", type=int, default=8,
                   help="批量评委每批条数（默认 8；TPM 紧就调小）")
    p.add_argument("--judge-sample", type=int, default=None,
                   help="评委只跑 N 条抽样（契约校验仍跑全量）")
    p.add_argument("--only-arms", default="")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("duel", help="成对比较胜率（风格评测；绝对打分已证伪）")
    p.add_argument("--name", required=False)
    p.add_argument("--pairs", default=PAIRS)
    p.add_argument("--arms", default="draft,direct,polish,reference")
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--calibrate", type=int, default=0,
                   help="用官方 chosen/rejected 校准评委，传对数（建议 20）")
    p.set_defaults(func=cmd_duel)

    p = sub.add_parser("prompt", help="打印契约/模板")
    p.add_argument("kind", choices=["restyle", "contract", "tags"])
    p.set_defaults(func=cmd_prompt)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
