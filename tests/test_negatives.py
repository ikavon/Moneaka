import random

from moneaka.contract import EMOTION_TAGS
from moneaka.negatives import flatten, merge, negatives, order, split_sentences, tags


def test_split_ignores_player_placeholder():
    # `[player]` 长得像标签但不是句子分隔符——按它切句会把句子劈开
    parts = split_sentences("[微笑]不用谢的啦, [player]! [宠爱]陪你是应该的嘛~")
    assert [t for t, _ in parts] == ["[微笑]", "[宠爱]"]
    assert "[player]" in parts[0][1]


def test_tags_keeps_body_verbatim():
    src = "[开心]好哦! [脸红]mua~"
    neg = tags(src, random.Random(0))
    assert neg == "[脸红]好哦! [开心]mua~"


def test_order_reverses_sentences():
    src = "[担心]天哪, 有这种事? [沉重]战争很可怕."
    assert order(src) == "[沉重]战争很可怕. [担心]天哪, 有这种事?"


def test_merge_single_tag_longer_sentence():
    src = "[微笑]嗯, 我在听. [凝视]你说完了吗?"
    neg = merge(src)
    assert neg.count("[") == 1 and neg.startswith("[微笑]")
    assert "我在听" in neg and "你说完了" in neg


def test_flatten_drops_tags_and_particles_but_keeps_punctuation():
    src = "[尴尬]没有啦. 指点一下呗?"
    neg = flatten(src)
    assert "[" not in neg and "啦" not in neg and "呗" not in neg
    assert "." in neg            # 不许顺手改成别的轴（半角/全角）


def test_negatives_never_equal_source_and_drop_degenerate():
    src = "[微笑]嗯."
    got = negatives(src)
    assert "order" not in got and "merge" not in got and "tags" not in got  # 单句造不出这些轴
    assert all(v != src for v in got.values())


def test_every_axis_output_is_nonempty_for_multiturn():
    src = "[微笑]不用谢的啦, [player]! [宠爱]我是你的女朋友, 陪你是应该的嘛~"
    got = negatives(src)
    assert set(got) == {"tags", "order", "merge", "flatten"}
