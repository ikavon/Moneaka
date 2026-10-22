"""受控 schema 的回归测试：分割、渲染/解析往返、中性化。"""

from __future__ import annotations

import pytest

from moneaka import contract
from moneaka.destyle_tags import (
    MAX_SEGMENTS,
    SegmentError,
    neutralize,
    parse_tagged_plan,
    render_plan,
    split_tagged,
)


def test_split_anchors_on_tags_not_terminal_punctuation():
    """一句里有多个终止符（`...`/`!`/`~`）也算同一段——切分必须按标签。"""
    tags, bodies = split_tagged("[不满]笨蛋, 你怎么会这么想? [脸红]当然愿意了啊! [意味深长]我才不要其他人呢.")
    assert tags == ["不满", "脸红", "意味深长"]
    assert bodies[0].startswith("笨蛋")


def test_split_keeps_player_placeholder_as_content():
    """`[player]` 不是标签，切分时不能把它当段边界，也不能丢掉。"""
    tags, bodies = split_tagged("[微笑]不用谢的啦, [player]! [宠爱]我是你的女朋友")
    assert tags == ["微笑", "宠爱"]
    # 正文用《player》哨兵发给模型（避免它把方括号当标签），故这里认哨兵
    assert "player" in bodies[0]


def test_split_rejects_untagged_leading_unit():
    """ds_basis 里 9% 的轮次开头直接开口没标签——这类必须拒，不能修补。"""
    with pytest.raises(SegmentError):
        split_tagged("天哪...很难相信我们相遇已经一年了. [微笑]真的很快啊.")


def test_split_rejects_too_many_segments():
    many = " ".join(f"[微笑]第{i}句" for i in range(MAX_SEGMENTS + 1))
    with pytest.raises(SegmentError):
        split_tagged(many)


def test_render_parse_roundtrip():
    tags = ["担心", "微笑"]
    contents = ["我不会嫌弃你", "[player]就是我的[player]"]
    plan = render_plan(tags, contents)
    assert plan.startswith("语气: 担心 > 微笑")
    got_tags, got_contents = parse_tagged_plan(plan)
    assert got_tags == tags
    assert got_contents == contents


def test_parse_rejects_segment_count_mismatch():
    plan = "语气: 微笑 > 笑\n内容:\n1. 只有一段"
    with pytest.raises(SegmentError):
        parse_tagged_plan(plan)


@pytest.mark.parametrize("raw,want", [
    ("我明白你的意思，[player]", "我明白你的意思, [player]"),
    ("这是真的吗？能讲给我听吗？", "这是真的吗? 能讲给我听吗?"),
    ("这样啊~ 我也一直在等你", "这样啊 我也一直在等你"),
    ("我...我只是有点意外", "我 我只是有点意外"),
])
def test_neutralize_strips_style_but_keeps_placeholder(raw, want):
    assert neutralize(raw) == want


def test_neutralize_fixes_bare_player_name():
    """模型偶发把占位符写成裸 `Player`；不归位就会被当成真名写进训练数据。"""
    assert "[player]" in neutralize("那么, 你希望我这样吗, Player?")


def test_every_tag_in_schema_is_in_whitelist():
    """schema 用的标签必须都来自官方 28 个白名单（`很开心` 是语料变体，允许）。"""
    from moneaka.destyle_tags import split_tagged as split

    src = "[生气]哼. [伤心]我有点难过. [惊喜]真的吗?"
    tags, _ = split(src)
    for t in tags:
        assert t in contract.EMOTION_TAGS, t
