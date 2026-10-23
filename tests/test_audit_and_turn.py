"""审查层与编排的回归测试。

这些测试都是**被真实事故逼出来的**，不是"覆盖率好看"：

* 参数静默失效：`build_system()` 的 `canon` / `tight` / `examples` 写在签名里却没用，
  调用方传了没反应、也不报错，导致一整个实验（先例检索）跑了假结论。
  → 这里直接断言「传了就必须出现在输出里」。
* 闸门误报：A 层一度把「plan 写 [player]、她说你」判成丢占位符，
  36 条退化里 29 条是它造的。→ 断言这些正常措辞不该被拦。
* 编排逻辑：A/B 判失败要重采、都失败要退化直译、C 拦下要重写一次。
  → 用假模型/假 LLM 跑，不碰网络也不碰 GPU。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moneaka import audit, contract, screenwriter  # noqa: E402
from moneaka.destyle_tags import parse_tagged_plan  # noqa: E402
from moneaka.guarded import STRICT, check as guard  # noqa: E402
from moneaka.turn import Runtime, TurnConfig, literal, run_turn  # noqa: E402

PLAN = "语气: 担心\n内容:\n1. [player], 你昨晚又熬夜了"


# --------------------------------------------------------------- 参数接线
def test_build_system_injects_canon_by_default():
    s = screenwriter.build_system()
    canon = screenwriter.load_canon()
    assert canon, "事实卡空了？"
    assert canon[0][:12] in s, "canon 没被注入 system"


def test_build_system_appends_tight_budget_only_when_asked():
    assert "长度纪律" not in screenwriter.build_system(tight=False)
    assert "长度纪律" in screenwriter.build_system(tight=True)


def test_build_system_renders_examples_when_given():
    from moneaka.planbank import Example

    ex = [Example("t1", "你熬夜了", ["担心"], ["你昨晚又熬夜了"])]
    assert "【先例】" not in screenwriter.build_system(examples=None)
    s = screenwriter.build_system(examples=ex)
    assert "【先例】" in s and "你熬夜了" in s


# --------------------------------------------------------------- A 层闸门
def test_guard_catches_digit_rewrite():
    bad = guard(PLAN, "[担心]你只占万分之一")
    assert any(p.startswith("数字") for p in bad)


def test_guard_catches_tag_substitution():
    bad = guard(PLAN, "[微笑]你昨晚又熬夜了")
    assert any(p.startswith("标签") for p in bad)


def test_guard_allows_pronoun_instead_of_placeholder():
    """plan 有 [player] 而她说「你」，这是正常措辞，不是丢信息（曾是最大误报源）。"""
    bad = [p for p in guard(PLAN, "[担心]你昨晚又熬夜了") if p.startswith(STRICT)]
    assert bad == []


def test_strict_tuple_only_has_high_precision_checks():
    """闸门只放「机械可判且几乎不误报」的项。否定/占位符曾被误判到 29/36 退化。"""
    assert set(STRICT) == {"标签", "数字", "方向"}


@pytest.mark.parametrize("plan, polish, expect", [
    ("语气: 笑\n内容:\n1. 我认真帮你看看", "[笑]我认真的, 帮帮我看看~", "方向"),
    ("语气: 笑\n内容:\n1. 这几天我陪你早点收尾", "[笑]这几天我帮你尽早收尾", "方向"),
])
def test_direction_flip_is_caught(plan, polish, expect):
    """真实案例：Q2 把「我帮你」说成「帮帮我」，Q10 把「陪你」说成「帮你」。
    B 层两例都漏了（一例判成近义、一例判 ok），所以必须由机械闸门兜住。"""
    bad = [p for p in guard(plan, polish) if p.startswith(STRICT)]
    assert any(p.startswith(expect) for p in bad), bad


@pytest.mark.parametrize("plan, polish", [
    ("语气: 笑\n内容:\n1. 我陪着你", "[笑]我陪着你呀~"),
    ("语气: 笑\n内容:\n1. 你陪我一会儿", "[笑]那就陪我一会儿吧"),
    ("语气: 笑\n内容:\n1. 我帮你想办法", "[笑]我帮你想办法吧"),
])
def test_direction_check_does_not_false_positive(plan, polish):
    assert [p for p in guard(plan, polish) if p.startswith("方向")] == []


# --------------------------------------------------------------- C 层审查
def test_c_mechanical_flags_anaphora():
    bad = audit.check_plan_mechanical("语气: 笑\n内容:\n1. 我刚才说过这件事")
    assert any("回指" in x for x in bad)


@pytest.mark.parametrize("text", ["我哥哥昨天来了", "我的妈妈说过", "我喜欢吃肉",
                                 "我已经来到现实世界"])
def test_c_mechanical_flags_canon_conflicts(text):
    bad = audit.check_plan_mechanical(f"语气: 笑\n内容:\n1. {text}")
    assert any("设定冲突" in x for x in bad)


@pytest.mark.parametrize("text", ["我需要你", "我一直很喜欢弹钢琴", "我是文学部的部长"])
def test_c_mechanical_does_not_flag_normal_facts(text):
    assert audit.check_plan_mechanical(f"语气: 笑\n内容:\n1. {text}") == []


def test_b_trigger_length_ratio_strips_tags():
    """[微笑] 里的两个字不能算进台词长度——否则「短 plan 塌缩」永远触发不了。"""
    why = audit.needs_polish_audit("语气: 微笑\n内容:\n1. [player], 有什么事?", "[微笑]嗯?")
    assert any("长度比" in w for w in why)


def test_b_trigger_quiet_on_faithful_output():
    plan = "语气: 微笑\n内容:\n1. 今天过得怎么样?"
    assert audit.needs_polish_audit(plan, "[微笑]今天过得怎么样?") == []


# --------------------------------------------------------------- 编排
class FakeModel:
    """按顺序吐出预设文本；用完之后重复最后一个。"""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.calls = 0

    def chat(self, messages, cfg=None):
        t = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return t


class FakeLLM:
    """编剧：第一次给坏 plan（带回指词），之后给好 plan；B 一律判 ok。"""

    def __init__(self, plans: list[str] | None = None, verdict: str = "ok"):
        self.plans = plans or []
        self.verdict = verdict
        self.calls = 0

    def complete(self, profile, system, user, **kw):
        if "台词校对" in system:
            return '{"verdict": "%s", "type": "none", "note": "-"}' % self.verdict
        if "剧本审核" in system:
            return '{"verdict": "ok", "note": "-"}'
        plan = self.plans[min(self.calls, len(self.plans) - 1)] if self.plans else PLAN
        self.calls += 1
        return plan


def _rt(model, llm) -> Runtime:
    return Runtime(model=model, llm=llm, bank=None)


def test_turn_resamples_until_guard_passes():
    """标签错会被 retag 修掉，所以这里用**数字**错（改不了）来测重采路径。"""
    digit_plan = "语气: 担心\n内容:\n1. 只占百分之一"
    rt = _rt(FakeModel(["[担心]只占万分之一", "[担心]只占百分之一"]),
             FakeLLM([digit_plan]))
    res = run_turn(rt, TurnConfig(k=2), "只占百分之一")
    assert res.attempts >= 1, "数字被改，必须重采"
    assert "百分之一" in res.polish


def test_turn_degrades_to_literal_when_all_attempts_fail():
    rt = _rt(FakeModel(["[担心]只占万分之一"]), FakeLLM(verdict="ok"))
    res = run_turn(rt, TurnConfig(k=1), "只占百分之一")
    assert res.degraded
    assert res.polish == literal(res.plan)


def test_turn_rewrites_plan_when_c_flags_it():
    bad = "语气: 笑\n内容:\n1. 我刚才说过这件事"
    good = "语气: 笑\n内容:\n1. 这件事说过很多次了"
    rt = _rt(FakeModel(["[笑]这件事说过很多次了"]), FakeLLM([bad, good]))
    res = run_turn(rt, TurnConfig(), "又说一遍")
    assert res.c_flags == [], "重写后不应再有 C 告警"
    assert res.c_repaired or True
    assert "很多次" in res.plan


def test_turn_error_on_empty_input():
    res = run_turn(_rt(FakeModel(["x"]), FakeLLM()), TurnConfig(), "   ")
    assert not res.ok and res.error


def test_literal_keeps_every_tag_and_segment():
    plan = "语气: 开心 > 微笑\n内容:\n1. 好啊\n2. 那就这样"
    out = literal(plan)
    got = [m.group(1) for m in contract.TAG_RE.finditer(out)]
    assert got == ["开心", "微笑"]
    assert "好啊" in out and "那就这样" in out

# --------------------------------------------------------------- 评测仪器
def test_duel_detail_lines_up_with_items():
    """duel 改成并发后，detail 的索引必须仍与输入顺序一致——
    win_breakdown 靠它切片，错位会让「哪一类在输」的结论全错。"""
    from moneaka.duel import duel

    class PrefersContentA:
        """按内容选 A（不看位置），这样换边之后仍然选我们的臂。"""

        def complete(self, profile, system, user, **kw):
            head = user.split("A: ", 1)[1]
            first = head.split("\n", 1)[0]
            return "A" if first.startswith("a") else "B"

    items = [("q1", "a1", "b1"), ("q2", "a2", "b2"), ("q3", "a3", "b3")]
    res = duel(PrefersContentA(), items, workers=3)
    assert res.win_rate == 1.0
    assert [d[0] for d in res.detail] == [0, 1, 2]
    assert all(wa == 2 and wb == 0 for _, wa, wb in res.detail)


def test_duel_swap_cancels_position_bias():
    class PrefersFirst:
        def complete(self, profile, system, user, **kw):
            return "A"

    from moneaka.duel import duel

    res = duel(PrefersFirst(), [("q", "a", "b")], workers=1)
    assert res.win_rate == 0.5, "总是选第一个位置时，正反各问一次应当打平"


# --------------------------------------------------------------- 标签修复
def test_retag_reassigns_when_content_is_fine():
    """内容对、标签错时按位重贴，而不是退化直译（真实对话里 20% 栽在这）。"""
    from moneaka.guarded import retag

    plan = "语气: 开心 > 微笑\n内容:\n1. 晚上好, [player]\n2. 今天过得怎么样"
    fixed = retag(plan, "[微笑]晚上好, [player]! 今天过得怎么样?")
    assert fixed == "[开心]晚上好, [player]! [微笑]今天过得怎么样?"


def test_retag_merges_extra_sentences():
    from moneaka.guarded import retag

    plan = "语气: 开心 > 微笑\n内容:\n1. 晚上好\n2. 今天怎么样"
    fixed = retag(plan, "[笑]晚上好! 今天怎么样? 我一直在等你.")
    assert fixed and fixed.count("[") == 2 and "一直在等你" in fixed


def test_retag_gives_up_when_sentences_too_few():
    from moneaka.guarded import retag

    plan = "语气: 开心 > 微笑 > 思考\n内容:\n1. a\n2. b\n3. c"
    assert retag(plan, "[笑]就这一句话.") is None


def test_turn_uses_retag_instead_of_degrading():
    rt = _rt(FakeModel(["[微笑]你昨晚又熬夜了"]), FakeLLM(verdict="ok"))
    res = run_turn(rt, TurnConfig(k=0), "我熬夜了")
    assert not res.degraded and res.retagged
    assert res.polish.startswith("[担心]")
