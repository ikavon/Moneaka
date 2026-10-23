"""按受控 schema 重造平行语料：``语气: 标签A > 标签B`` + 逐段 ``内容``。

用法::

    python tools/make_tagged_pairs.py --src data/pairs/plan.jsonl --out data/pairs/tagged_official.jsonl
    python tools/make_tagged_pairs.py --src data/pairs/pairs_ds.jsonl --out data/pairs/tagged_ds.jsonl --require-contract

断点续跑：``--out`` 已存在则按 ``turn_id`` 只补缺的（每条都是一次强模型调用）。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import contract  # noqa: E402
from moneaka.llm import LLM  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="含 original 字段的 pairs jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--require-contract", action="store_true",
                    help="只收本来符合生产契约的轮（新语料必开）")
    ap.add_argument("--dry-run", action="store_true", help="只统计可处理的条数，不调 API")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.src))
    keep, dropped = [], {}
    for r in rows:
        orig = r.get("original") or ""
        if not orig:
            dropped["no_original"] = dropped.get("no_original", 0) + 1
            continue
        if args.require_contract and not contract.check(orig).ok:
            dropped["contract"] = dropped.get("contract", 0) + 1
            continue
        try:
            tags, _ = split_tagged(orig)
        except SegmentError as exc:
            key = "seg_" + str(exc).split(":")[0][:24]
            dropped[key] = dropped.get(key, 0) + 1
            continue
        keep.append((r, tags))
    print(f"可处理 {len(keep)}/{len(rows)}；丢弃原因: {dropped}")
    if args.limit:
        keep = keep[: args.limit]
    if args.dry_run:
        import collections
        print("段数分布:", dict(sorted(collections.Counter(len(t) for _, t in keep).items())))
        return 0

    out = Path(args.out)
    existing: dict[str, dict] = {}
    if out.exists():
        existing = {r["turn_id"]: r for r in load_jsonl(out)}
        if existing:
            print(f"续跑：已有 {len(existing)} 条")
    todo = [(r, t) for r, t in keep if r["turn_id"] not in existing]
    print(f"待造 {len(todo)} 条")

    llm = LLM()
    prof = llm.profile("destyle")
    print(f"端点 {prof.base_url} model={prof.model}")

    lock = threading.Lock()
    done = [0]
    failed: list[tuple[str, str]] = []

    def work(item):
        r, tags = item
        try:
            got = destyle_one(llm, r["original"])
        except Exception as exc:  # noqa: BLE001
            with lock:
                failed.append((r["turn_id"], f"{exc}"))
            return None
        rec = {
            "turn_id": r["turn_id"],
            "user": r.get("user"),
            "tags": got["tags"],
            "contents": got["contents"],
            "plan": got["plan"],
            "original": r["original"],
            "n_tags": got["n_tags"],
            "source": Path(args.src).name,
        }
        with lock:
            done[0] += 1
            if done[0] % 25 == 0:
                _checkpoint(out, existing, rows_buf)
            if done[0] % 10 == 0:
                print(f"  {done[0]}/{len(todo)}")
        return rec

    rows_buf: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rec in pool.map(work, todo):
            if rec:
                rows_buf.append(rec)
    merged = list(existing.values()) + rows_buf
    with open(out, "w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写入 {out}（{len(merged)} 条）")
    for tid, why in failed[:12]:
        print(f"  ! {tid} {why[:110]}")
    if failed:
        print(f"  共失败 {len(failed)} 条（重跑即可续）")
    return 0


def _checkpoint(out: Path, existing: dict, buf: list[dict]) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in list(existing.values()) + buf:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)


if __name__ == "__main__":
    raise SystemExit(main())