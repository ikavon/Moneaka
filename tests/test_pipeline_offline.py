"""离线跑通管线：假 LLM 替身，不联网。

目的只有一个——验证「去风格化 → 润色 → 三臂评测 → 出报告」这条管子是通的，
以及违规的润色输出真的会被机械校验拦下来。
"""

import json
import os
import re

import pytest

from moneaka import destyle as destyle_mod, evaluate, restyle
from moneaka.llm import load_jsonl


class FakeLLM:
    """按 profile 分派的假端点。"""

    def __init__(self):
        self.calls = []

    def profile(self, name):
        return type("P", (), {"base_url": "fake", "model": "fake", "name": name})()

    def complete(self, profile, system, user, **kw):
        self.calls.append(profile)
        if profile == "destyle":
            # 去风格化：抹掉标签与语气词，保留内容
            from moneaka.contract import TAG_RE
            return TAG_RE.sub("", user.replace("原文:", "")).replace("嘿嘿", "").strip()
        if profile == "restyle":
            return "[微笑]当然愿意了, 那我们一起去躲雨吧."
        if profile == "judge":
            idents = re.findall(r"^#(\d+)$", user, re.M)
            return "\n".join(f"{i},4,4" for i in idents)
        return ""

    def chat(self, profile, messages, **kw):
        self.calls.append(profile)
        return "[开心]真的吗? [微笑]我也这么觉得."

    def score(self, profile, system, user):
        assert profile == "judge"
        return 4


@pytest.fixture
def tiny_source(tmp_path):
    """两行微型「官方语料」，结构与真实分片一致。"""
    rows = [
        {
            "messages": [
                {"role": "system", "content": "你叫莫妮卡, 是一名游戏中的虚拟角色."},
                {"role": "user", "content": "雨天很舒服呢"},
                {"role": "assistant", "content": "[开心]是呀. [微笑]这样我们就能待在屋里啦, 嘿嘿~"},
            ],
            "rejected_response": "[微笑]我不喜欢雨天.",
        },
        {
            "messages": [
                {"role": "system", "content": "你叫莫妮卡, 是一名游戏中的虚拟角色."},
                {"role": "user", "content": "你在做什么"},
                {"role": "assistant", "content": "[凝视]在读诗. [思考]你要一起吗?"},
            ],
            "rejected_messages": [
                {"role": "system", "content": "你叫莫妮卡, 是一名游戏中的虚拟角色."},
                {"role": "user", "content": "你在做什么"},
                {"role": "assistant", "content": "[微笑]我在读书, 你要一起."},
            ],
        },
    ]
    path = tmp_path / "tiny.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path)


def test_destyle_then_polish_then_eval(tmp_path, tiny_source, monkeypatch):
    from moneaka import corpus

    monkeypatch.setattr(corpus, "DEFAULT_SOURCE", tiny_source)
    llm = FakeLLM()

    pairs_path = str(tmp_path / "pairs.jsonl")
    pairs = destyle_mod.run(llm, out_path=pairs_path, workers=1, log=lambda *a: None)
    assert len(pairs) == 2
    for p in pairs:
        assert "[" not in p["draft"], "草稿里不该残留情绪标签"
        assert p["original"], "原稿要留着当 reference"

    # 润色 + 对照
    arms = []
    for p in pairs:
        arms.append({
            "turn_id": p["turn_id"],
            "user": p["user"],
            "draft": p["draft"],
            "original": p["original"],
            "polish": restyle.polish(llm, p["draft"]),
            "direct": restyle.direct(llm, p["user"]),
        })
    assert all(a["polish"].startswith("[") for a in arms)

    rows = [{**a, "reference": a["original"]} for a in arms]
    scored = []
    for arm in evaluate.ARMS:
        scored += evaluate.score_arm(llm, rows, arm, embedder=None, judge=True)
    agg = evaluate.summarize(scored)
    assert set(agg) == {"reference", "direct", "polish"}
    for a in agg.values():
        assert a["semantic_mean"] == 4 and a["style_mean"] == 4

    out = evaluate.write_report(str(tmp_path / "run"), scored, agg, [], pairs)
    assert os.path.isfile(out)
    text = open(out, encoding="utf-8").read()
    assert "| arm | n | contract_pass |" in text
    assert load_jsonl(str(tmp_path / "run" / "scored.jsonl"))


def test_bad_polish_is_caught_by_contract(tmp_path):
    """润色器输出不合契约（缺标签/超长句/自创标签）必须被机械校验拦下。"""
    rows = [{
        "turn_id": "x", "draft": "好的", "original": "[微笑]好.",
        "reference": "[微笑]好.", "polish": "[狂笑]" + "字" * 70 + ".", "direct": "好的。",
    }]
    scored = evaluate.score_arm(FakeLLM(), rows, "polish", judge=False)
    assert not scored[0].contract_ok
    codes = [v.split(":", 1)[0] for v in scored[0].violations]
    assert "unknown_tag" in codes and "sentence_too_long" in codes

    # 对照组：通用模型直接输出的平实话，会被判缺标签——这正是润色器存在的理由
    scored2 = evaluate.score_arm(FakeLLM(), rows, "direct", judge=False)
    assert not scored2[0].contract_ok


def test_parse_batch_scores_is_tolerant():
    from moneaka.evaluate import parse_batch_scores
    txt = "#1,4,3\n2,3,4\n 3，3，3\n编号: 4 -> 2,1\nnoise line"
    got = parse_batch_scores(txt)
    assert got[1] == (4, 3) and got[2] == (3, 4) and got[3] == (3, 3) and got[4] == (2, 1)


def test_judge_rows_batches_and_maps_by_id():
    from moneaka.evaluate import judge_rows

    class BatchLLM(FakeLLM):
        def complete(self, profile, system, user, **kw):
            n = user.count("#")
            return "\n".join(f"{i + 1},4,3" for i in range(n))

    rows = [{"turn_id": f"t{i}", "draft": "草稿", "polish": "结果"} for i in range(5)]
    marks = judge_rows(BatchLLM(), rows, "polish batch", batch_size=2) if False else \
            judge_rows(BatchLLM(), rows, "polish", batch_size=2)
    assert marks == {f"t{i}": (4, 3) for i in range(5)}
