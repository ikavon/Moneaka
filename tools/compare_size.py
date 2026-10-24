"""同一批输入、同一个生成配置，比较两个尺寸的润色器。

用途：回答「参数变大值不值」。做法是拿 run8（1.7B + LoRA）已经跑过的那 206 条
留出输入，原样喂给另一个 adapter（例如 4B），逐条并列，产出文档给人看。

    # 1) 出 4B 的输出（4bit 推理，8GB 够用）
    python tools/compare_size.py --base data/models/Qwen3-4B \
        --adapter data/adapters/run9_4b --q --name pickg210_run9_4b --gen

    # 2) 生成对照文档（不打分，只并列）
    python tools/compare_size.py --base x --adapter y --name pickg210_run9_4b \
        --doc docs/COMPARE_SIZE_4B.md

注意：**不在这里算胜率**。尺寸差异该由人看输出判断；机器分只作辅助，
且判官若与大模型同源会偏向自己。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "models"))
sys.path.insert(0, str(ROOT / "src"))

from moneaka import prompts  # noqa: E402
from moneaka.contract import EMOTION_TAGS, TAG_RE, PLACEHOLDER_RE, cjk_len  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def gen(args: argparse.Namespace) -> int:
    from moneaka.localhf import GenConfig, get_model

    base_rows = load_rows(ROOT / args.baseline)
    rows = base_rows[: args.limit] if args.limit else base_rows
    print(f"输入 {len(rows)} 条（来自 {args.baseline}）")

    model = get_model(args.base, adapter=args.adapter, load_in_4bit=args.q)
    system = prompts.restyle_system(few_shot=False)

    def one(r: dict) -> dict:
        raw = model.chat([{"role": "system", "content": system},
                          {"role": "user", "content": r["input"]}],
                         GenConfig(temperature=args.temperature,
                                   max_new_tokens=args.max_new))
        return {"turn_id": r["turn_id"], "user": r.get("user", ""), "input": r["input"],
                "original": r.get("original", ""), "polish": (raw or "").strip(),
                "polish_raw": raw, "draft": r.get("draft", "")}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(one, rows))
    dt = time.time() - t0

    dst = ROOT / "data" / "runs" / args.name
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "arms.jsonl", "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    empty = sum(1 for r in out if not r["polish"])
    dt_avg = dt / max(1, len(out))
    print(f"推理 {len(out)} 条 / {dt:.0f}s（{dt_avg:.2f}s/条，{args.workers} 并发）")
    print(f"空输出 {empty}/{len(out)}")
    print(f"写入 {dst / 'arms.jsonl'}")
    return 0


def doc(args: argparse.Namespace) -> int:
    """并列两种尺寸的输出，按「差异最大」排序，方便人一眼看出有没有进步。"""
    base_rows = load_rows(ROOT / args.baseline)
    new_rows = load_rows(ROOT / "data" / "runs" / args.name / "arms.jsonl")
    new_by_id = {r["turn_id"]: r for r in new_rows}

    def has_tag(t: str) -> bool:
        t = PLACEHOLDER_RE.sub("\x00", t or "")
        return all(m.group(1) in EMOTION_TAGS for m in TAG_RE.finditer(t))

    pairs = [r for r in base_rows if r["turn_id"] in new_by_id]
    if args.limit:
        pairs = pairs[: args.limit]

    out = [f"# 尺寸对照：{args.base_name} vs {args.new_name}", "",
           "同 206 条留出输入、同 system、greedy 解码。**只列输出，不打分。**", "",
           f"- 左：{args.base_name}（`{args.baseline}`）",
           f"- 右：{args.new_name}（`data/runs/{args.name}/arms.jsonl`）", "", "---", ""]
    for i, r in enumerate(pairs, 1):
        n = new_by_id[r["turn_id"]]
        out += [f"## {i}. 玩家：{r.get('user', '')}", "",
                "**表达指令**", f"> {r['draft'].replace(chr(10), ' / ')}", "",
                f"**{args.base_name}**（{cjk_len(r['polish'])} 字"
                f"{'' if has_tag(r['polish']) else ' ⚠标签越界'}）",
                f"> {r['polish']}", "",
                f"**{args.new_name}**（{cjk_len(n['polish'])} 字"
                f"{'' if has_tag(n['polish']) else ' ⚠标签越界'}）",
                f"> {n['polish']}", "",
                f"<sub>官方原答：{r.get('original', '')}</sub>", "", "---", ""]

    dst = ROOT / args.doc
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out), encoding="utf-8")
    print(f"写入 {dst}（{len(pairs)} 条）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="data/runs/pickg210_run8_final/arms.jsonl")
    ap.add_argument("--base", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--q", action="store_true", help="4bit 加载（4B 全精度装不进 8GB）")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--gen", action="store_true", help="跑推理并落 arms.jsonl")
    ap.add_argument("--doc", default="", help="产出对照文档的路径")
    ap.add_argument("--base-name", default="1.7B (run8)")
    ap.add_argument("--new-name", default="4B (run9)")
    args = ap.parse_args()
    if args.gen:
        if not args.base:
            ap.error("--gen 需要 --base")
        return gen(args)
    if args.doc:
        return doc(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())