"""契约校验器的考题就是官方真实数据本身。

reference（真实 DAA4 输出）不过 = 阈值/切句写错了，不是模型错了。
见 docs/DESIGN.md「契约是从数据里读出来的」。
"""

import os
import pytest

from moneaka import contract, corpus


@pytest.fixture(scope="module")
def chosen():
    if not os.path.isfile(corpus.DEFAULT_SOURCE):
        pytest.skip("官方语料分片不在本机")
    return corpus.chosen_turns()


def test_reference_all_pass(chosen):
    """地基测试：真数据必须 100% 通过。曾经只有 34%（按终止符切句导致）。"""
    bad = [(t, contract.check(t.chosen)) for t in chosen]
    failed = [(t, r) for t, r in bad if not r.ok]
    assert not failed, "真实输出被判违规，说明校验器错了：\n" + "\n".join(
        f"  [{t.turn_id}] {t.chosen[:60]} -> {[str(v) for v in r.violations]}"
        for t, r in failed[:5]
    )


def test_tag_units_are_per_sentence(chosen):
    """一个标签 = 一句；语料里中位 2 句、最多不超过 5 句。"""
    counts = [len(contract.split_sentences(t.chosen)) for t in chosen]
    assert max(counts) <= 5
    assert sorted(counts)[len(counts) // 2] == 2


def test_placeholder_is_not_an_emotion_tag():
    res = contract.check("[微笑]当然愿意了, [player].")
    assert res.ok, [str(v) for v in res.violations]
    assert res.n_tags == 1  # [player] 不计入标签


def test_missing_leading_tag_is_caught():
    """标签单元前的裸文本 = 违规。"""
    res = contract.check("好啊. [微笑]那我们一起去看看.")
    codes = [v.code for v in res.violations]
    assert "missing_tag" in codes


def test_multi_sentence_unit_is_advisory_not_violation():
    """一个标签套两句——真语料里占 19.5%，不能判违规，否则契约比 DAA4 自己还严。"""
    res = contract.check("[尴尬]说实话我有点意外. 不过也没关系.")
    assert res.ok, [str(v) for v in res.violations]
    assert res.multi_sentence_units == 1


def test_too_long_sentence_is_caught():
    long = "字" * (contract.SENTENCE_MAX_CHARS + 5)
    res = contract.check(f"[微笑]{long}.")
    assert any(v.code == "sentence_too_long" for v in res.violations)


def test_unknown_tag_is_caught():
    res = contract.check("[狂笑]哈哈哈哈.")
    assert any(v.code == "unknown_tag" for v in res.violations)


def test_synonym_group_is_tolerated():
    """笑/开心、脸红/害羞 这类近义标签官方语料自己就并存，不该判违规。"""
    assert contract.check("[笑]真的吗.").ok
    assert contract.check("[害羞]嗯.").ok


def test_english_only_is_caught():
    res = contract.check("[smile]Sure thing.")
    codes = [v.code for v in res.violations]
    assert "no_cjk" in codes or "unknown_tag" in codes


def test_contract_text_covers_four_official_rules():
    txt = contract.contract_text()
    for frag in ("方括号", str(contract.SENTENCE_MAX_CHARS), "中文", "编造"):
        assert frag in txt


def test_skeleton_prompt_matches_official_wording(chosen):
    """草稿端要用的骨架 prompt 必须逐字来自官方数据，不许我们改写。"""
    skel = corpus.skeleton_prompt()
    assert "每个句子自然且不超过60字" in skel
    assert "并在每个句子前用方括号中的文字表示情绪" in skel
