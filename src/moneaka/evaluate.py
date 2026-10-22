"""步骤 3：评测三条出话路线。

| 路线 | 来源 | 意义 |
|---|---|---|
| ``reference`` | 官方数据里的 chosen（真实 DAA4 输出） | 天花板 |
| ``direct``    | 通用模型按官方骨架 prompt 直接扮演 | 竞品方案（纯 prompt 路线） |
| ``polish``    | 通用模型出草稿 → 润色器改写 | 本项目主张 |

四个指标：
- ``contract_pass`` —— 机械校验（句长/逐句标签/标签白名单/中文），零成本、无争议
- ``semantic``      —— 评委打的语义保持度 0-4（以 draft 为真值）
- ``style``         —— 评委打的「像不像她」0-4，评分轴来自官方 chosen/rejected 对照
- ``cos_ref``       —— 与 reference 的 embedding 余弦（粗略的风格邻近度）

语义相似度不放进表格主列：embedding 对风格差异敏感，草稿和莫妮卡稿本来就不该近。
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict, dataclass, field
from typing import Iterable

from . import prompts
from .contract import check
from .llm import LLM

ARMS = ("reference", "direct", "polish")


@dataclass
class Scored:
    turn_id: str
    arm: str
    text: str
    contract_ok: bool
    violations: list[str] = field(default_factory=list)
    semantic: int | None = None
    style: int | None = None
    cos_ref: float | None = None
    cjk_punct_ratio: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# embedding（可选依赖：没配就跳过 cos 列）
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def cos(self, a: str, b: str) -> float:
        import numpy as np
        va, vb = self.model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(va, vb))


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------

def score_arm(llm: LLM, rows: Iterable[dict], arm: str, *,
              embedder: Embedder | None = None, judge: bool = True) -> list[Scored]:
    out: list[Scored] = []
    for r in rows:
        text = r.get(arm) or ""
        res = check(text)
        s = Scored(
            turn_id=r["turn_id"], arm=arm, text=text,
            contract_ok=res.ok, violations=[str(v) for v in res.violations],
            cjk_punct_ratio=res.cjk_punct_ratio,
        )
        if judge and text:
            try:
                s.semantic = llm.score("judge", prompts.JUDGE_SEMANTIC_SYSTEM,
                                       prompts.JUDGE_SEMANTIC_USER.format(
                                           draft=r["draft"], polished=text))
                s.style = llm.score("judge", prompts.JUDGE_STYLE_SYSTEM,
                                    prompts.JUDGE_STYLE_USER.format(
                                        draft=r["draft"], polished=text))
            except Exception as exc:
                print(f"  ! {r['turn_id']} 评委失败: {exc}")
        if embedder and text and r.get("original"):
            from .restyle import strip_tagged
            try:
                s.cos_ref = round(embedder.cos(strip_tagged(r["original"]), strip_tagged(text)), 3)
            except Exception:
                pass
        out.append(s)
    return out


def summarize(scored: list[Scored]) -> dict:
    agg: dict[str, dict] = {}
    for arm in ARMS:
        rows = [s for s in scored if s.arm == arm and s.text]
        if not rows:
            continue
        def _mean(key):
            vals = [getattr(r, key) for r in rows if getattr(r, key) is not None]
            return round(statistics.mean(vals), 3) if vals else None
        n = len(rows)
        agg[arm] = {
            "n": n,
            "contract_pass_rate": round(sum(r.contract_ok for r in rows) / n, 3),
            "semantic_mean": _mean("semantic"),
            "style_mean": _mean("style"),
            "cos_ref_mean": _mean("cos_ref"),
            "cjk_punct": round(statistics.mean([r.cjk_punct_ratio for r in rows]), 3),
        }
    return agg


def violation_census(scored: list[Scored], top: int = 12) -> list[tuple[str, int]]:
    """按违规码统计，定位是校验阈值太紧还是模型真不行。"""
    census: dict[str, int] = {}
    for s in scored:
        if s.arm == "reference":   # 只统计真数据的违规——那是阈值错了
            for v in s.violations:
                code = v.split(":", 1)[0]
                census[code] = census.get(code, 0) + 1
    return sorted(census.items(), key=lambda kv: -kv[1])[:top]


def write_report(out_dir: str, scored: list[Scored], agg: dict,
                 ref_violations: list[tuple[str, int]], pairs: list[dict]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scored.jsonl"), "w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(json.dumps(s.as_dict(), ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"aggregate": agg, "reference_violations": ref_violations},
                  fh, ensure_ascii=False, indent=1)

    by_id = {p["turn_id"]: p for p in pairs}
    lines = ["# 阶段 0 评测\n",
             "| arm | n | contract_pass | semantic | style | cos_ref | cjk_punct |",
             "|---|---|---|---|---|---|---|"]
    for arm, a in agg.items():
        lines.append(f"| {arm} | {a['n']} | {a['contract_pass_rate']} | "
                     f"{a['semantic_mean']} | {a['style_mean']} | {a['cos_ref_mean']} | "
                     f"{a['cjk_punct']} |")
    if ref_violations:
        lines.append("\n> ⚠️ 真数据（reference）也触发的违规码 = 校验阈值定错了，先修阈值再谈模型：")
        lines += [f"> - `{c}` × {n}" for c, n in ref_violations]
    lines.append("\n## 逐条对照（前 20 条）\n")
    shown = 0
    seen: dict[str, dict[str, str]] = {}
    for s in scored:
        seen.setdefault(s.turn_id, {})[s.arm] = s.text
    for tid, arms in list(seen.items()):
        if shown >= 20:
            break
        p = by_id.get(tid, {})
        lines.append(f"### {tid}\n")
        lines.append(f"- USER: {p.get('user', '')}\n- DRAFT: {p.get('draft', '')}")
        for arm in ARMS:
            if arms.get(arm):
                lines.append(f"- **{arm}**: {arms[arm]}")
        lines.append("")
        shown += 1
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
