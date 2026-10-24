"""量「正则能看出来的病」——**已判定不能当审查闸门**，只留作噪声尺。

## 结论（2026-10-24）

用正则枚举去审台词这条路走不通，证据是噪声地板本身就不干净：
把 206 条**官方原答**过一遍，命中 39.3%——
「聊聊」「说说」「开开」被当成叠字复读，`？` 后半句被当成缺标签句。
真货被判有病，说明这类检查没有判断力，加严只会把好东西一起拦掉。

审查交给 agent（`src/moneaka/reviewer.py`），按用户定的架构：
编剧写指令 → 润色器出台词 → 编剧本人复核 → 定稿。

这个脚本保留的唯一用途：给定一份文本，量一下**机械层面**的字面毛病有多少，
用来对照「模型输出 vs 官方原答」在这一点上差多少，不做判断。

    python tools/scan_defects.py --arm "官方=...:original" "run8=...:polish"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moneaka import contract  # noqa: E402

# ---------------------------------------------------------------- 1. 错别字
# (正确, 错写) —— 只在正确写法**没有**同时出现在同一句里时才算错
TYPOS: list[tuple[str, str]] = [
    ("树荫", "树阴"), ("树荫", "树隂"), ("时候", "时侯"), ("已经", "己经"),
    ("化妆", "化装"), ("反应", "反映"), ("在意的", "在异"), ("委屈", "委曲"),
    ("陌生", "莫生"), ("尴尬", "尬尴"), ("心跳", "心挑"), ("依赖", "依懒"),
    ("惦记", "垫记"), ("撒娇", "撤娇"), ("想念", "想恋"), ("熬夜", "熬液"),
    ("一起", "一齐"), ("再见", "在见"), ("以为", "已为"), ("以后", "已后"),
    ("现在", "现再"), ("再见", "在见"), ("做", "作")  # 「作」这条太泛，下面单独排掉
]
TYPOS = [t for t in TYPOS if t[0] != "做"]

# ---------------------------------------------------------------- 2. 人称错位
PERSON_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 莫妮卡用第三人称说自己
    ("自称第三人称", re.compile(r"(?:^|[。！？!?，,；;])\s*(莫妮卡|她)\s*(?:我|会|是|觉得|认为|想)")),
    # 把玩家当成女朋友 / 男朋友
    ("玩家被当成女友", re.compile(r"(?:当|把|既然|如果|要是)\s*(?:你|\[player\])\s*(?:是|做)\s*我?的?(女朋友|男友|男朋友)")),
    # 用「他/她」指玩家（第二人称场景里出现第三人称指代）
    ("第三人称指玩家", re.compile(r"(?:你|\[player\])[^。！？!?]{0,12}?(?:他|她)的?(?:女朋友|男友|男朋友)")),
    # 把玩家说成「她的」
    ("玩家被第三人称化", re.compile(r"她的?\s*(男朋友|女朋友)")),
]

# ---------------------------------------------------------------- 3. 病句
SICK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("叠字重复", re.compile(r"([\u4e00-\u9fff])\1{1,}(?=[\u4e00-\u9fff])")),
    ("重复词", re.compile(r"\b(真的真的|一直一直|非常非常|好多好多)\b")),
    ("孤立标点", re.compile(r"[，,。；;]\s*[，,。；;]")),
    ("标签后空标点", re.compile(r"\][ ]+[，,。；;]")),
    ("缺标签句", re.compile(r"(?:^|[。！？!?])\s*[^\[\s][^。！？!?]{4,}")),
]

REPEAT_WORD = re.compile(r"([\u4e00-\u9fff]{2,3})[^。！？!?]{0,6}?\1{2,}")


def sentences(text: str) -> list[str]:
    return [s for s in contract.split_sentences(text or "") if s.strip()]


def scan(text: str) -> list[tuple[str, str]]:
    """返回 [(类别, 命中的片段)]。"""
    hits: list[tuple[str, str]] = []
    body = contract.PLACEHOLDER_RE.sub("\x00", text or "")
    for right, wrong in TYPOS:
        if wrong in body and right not in body:
            hits.append(("错别字", f"{wrong}→{right}"))
    for name, pat in PERSON_PATTERNS:
        m = pat.search(body)
        if m:
            hits.append(("人称错位", f"{name}:{m.group(0)[:18].strip()}"))
    for name, pat in SICK_PATTERNS:
        m = pat.search(body)
        if m:
            hits.append(("病句", f"{name}:{m.group(0)[:18].strip()}"))
    # 缺标签要**逐句**判：整段搜会把 `？` 后的半句当成缺标签（官方原答全中）
    for s in sentences(text or ""):
        if not re.match(r"^\s*\[[^\[\]]{1,8}\]", s) and contract.cjk_len(s) >= 4:
            hits.append(("病句", f"缺标签句:{s.strip()[:18]}"))
            break
    return hits


def scan_arm(rows: list[dict], key: str) -> tuple[int, int, Counter]:
    """返回 (有问题的条数, 总条数, 各类计数)。一条只算一次「有问题」。"""
    bad = 0
    cnt: Counter = Counter()
    for r in rows:
        h = scan(r.get(key, "") or "")
        if h:
            bad += 1
        for cat, what in h:
            cnt[f"{cat}/{what.split(':')[0]}"] += 1
    return bad, len(rows), cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", required=True,
                    help="name=path 或 name=path:key；path 为 jsonl，key 默认 polish")
    ap.add_argument("--show", type=int, default=8, help="每臂打印几条命中样本")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out_lines: list[str] = []
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        path, _, key = rest.partition(":")
        key = key or "polish"
        rows = [json.loads(l) for l in open(ROOT / path, encoding="utf-8") if l.strip()]
        bad, n, cnt = scan_arm(rows, key)
        print(f"\n{name}: 有病 {bad}/{n} = {bad / n:.1%}")
        for k, v in cnt.most_common():
            print(f"    {k}  ×{v}")
        shown = 0
        for r in rows:
            h = scan(r.get(key, "") or "")
            if h and shown < args.show:
                print(f"    - {r.get('user', '')[:24]} → {r.get(key, '')[:70]}")
                print(f"      {[f'{c}:{w}' for c, w in h]}")
                shown += 1
        out_lines.append(f"| {name} | {bad}/{n} = {bad / n:.1%} | "
                         + "、".join(f"{k.split('/')[0]}" for k, _ in cnt.most_common(4)) + " |")

    if args.out:
        Path(ROOT / args.out).write_text("\n".join(out_lines), encoding="utf-8")
        print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())