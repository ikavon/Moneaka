"""归一化层：格式合规由确定性代码保证，不指望模型记白名单。"""

from moneaka import contract
from moneaka.normalize import canonical_tag, normalize, NormStats


def test_unknown_tag_mapped_to_whitelist():
    st = NormStats()
    assert canonical_tag("担忧", st) == "担心"
    assert canonical_tag("宠溺", st) == "宠爱"
    assert canonical_tag("好奇", st) == "思考"
    assert st.tags_mapped["担忧"] == "担心"


def test_keyword_contained_tag_is_rescued():
    st = NormStats()
    assert canonical_tag("温柔地微笑", st) == "微笑"


def test_completely_foreign_tag_falls_back_and_is_recorded():
    st = NormStats()
    assert canonical_tag("量子纠缠", st) == "微笑"
    assert "量子纠缠" in st.tags_unmapped


def test_player_placeholder_survives_normalize():
    out, _ = normalize("[担忧]我一直在等你，[player]。")
    assert "[player]" in out and out.startswith("[担心]")


def test_fullwidth_punctuation_converted():
    out, st = normalize("[微笑]我懂你，[player]。")
    assert "，" not in out and "。" not in out
    assert st.punct_converted >= 2


def test_long_sentence_is_split_and_each_chunk_tagged():
    body = "，" + "、".join(["这是一个比较长的分句"] * 8)
    out, st = normalize(f"[思考]{body}")
    res = contract.check(out)
    assert res.ok, [str(v) for v in res.violations]
    assert res.n_tags >= 2 and st.sentences_split >= 1


def test_missing_tag_gets_previous_tag():
    """开头的裸文本（没有标签）会被补上标签。

    注：一个标签套两句是**合规**的（真语料 19.5% 如此），所以不拆。
    """
    out, st = normalize("好啊. [微笑]那我们走吧.")
    assert out.startswith("[微笑]好啊.") and st.tags_inserted == 1


def test_normalize_is_idempotent():
    once, _ = normalize("[担忧]我理解你，[player]。在雨声里放松下来，哪天我们试试。")
    twice, st = normalize(once)
    assert once == twice and st.tags_inserted == 0 and not st.tags_mapped


def test_normalized_output_always_passes_contract():
    samples = [
        "[嫌弃]会，怎么了？[微笑]",
        "[沉思] 是的，[player]，科技带来富饶，也带来贫瘠。",
        "说得我都害羞了... [害羞]我也想和你白头偕老呢",
        "",
    ]
    for s in samples:
        if not s:
            continue
        out, _ = normalize(s)
        assert contract.check(out).ok, (s, out)
