"""Spec #68: engine-declared non-verbal tags + normalization exemptions.

Contract-level assertions through the public HTTP surface: each engine's
``GET /engines`` entry carries its declared ``nonverbal_tags`` data (empty =
the engine has no body-tag support), and ``POST /normalize`` leaves the
digit-bearing marker syntaxes (MiniMax ``<#x#>`` pauses, IndexTTS-2.5
``<汉字|PINYIN>`` pinyin annotations) byte-identical while other numbers are
still rewritten.
"""

from __future__ import annotations

import pytest

VOXCPM2_TAG_TEXTS = [
    "[laughing]",
    "[sigh]",
    "[breath]",
    "[Uhm]",
    "[Shh]",
    "[Question-ah]",
    "[Question-ei]",
    "[Question-en]",
    "[Question-oh]",
    "[Confirmation-en]",
]

MINIMAX_TAG_TEXTS = [
    "(laughs)",
    "(chuckle)",
    "(sighs)",
    "(groans)",
    "(humming)",
    "(breath)",
    "(pant)",
    "(inhale)",
    "(exhale)",
    "(gasps)",
    "(sniffs)",
    "(snorts)",
    "(coughs)",
    "(clear-throat)",
    "(burps)",
    "(lip-smacking)",
    "(hissing)",
    "(emm)",
    "(sneezes)",
    "<#1#>",
]


def _engine_ids(client) -> set[str]:
    r = client.get("/engines")
    assert r.status_code == 200
    return {e["id"] for e in r.json()["engines"]}


def _caps(client, engine_id: str) -> dict:
    r = client.get("/engines")
    assert r.status_code == 200
    matches = [e for e in r.json()["engines"] if e["id"] == engine_id]
    assert len(matches) == 1, engine_id
    return matches[0]["capabilities"]


# --- engine-declared tag sets ------------------------------------------------


def test_voxcpm2_declares_full_cookbook_tag_set(client):
    ids = _engine_ids(client)
    # Platform-dependent registration (mps on macOS arm64, cuda on win32);
    # at least one VoxCPM2 variant must be present wherever tests run.
    present = [i for i in ("voxcpm2-mps", "voxcpm2-cuda") if i in ids]
    assert present, "no VoxCPM2 engine registered on this platform"
    for engine_id in present:
        tags = _caps(client, engine_id)["nonverbal_tags"]
        assert [t["text"] for t in tags] == VOXCPM2_TAG_TEXTS
        by_text = {t["text"]: t for t in tags}
        # measured stays on the two real-machine verified tags only.
        assert by_text["[laughing]"]["verification"] == "measured"
        assert by_text["[sigh]"]["verification"] == "measured"
        assert [t["verification"] for t in tags if t["text"] not in ("[laughing]", "[sigh]")] == [
            "vendor"
        ] * 8
        # The engine's own quick-insert favorites (ADR-0018, data-driven).
        assert [t["text"] for t in tags if t["common"]] == ["[laughing]", "[sigh]", "[breath]"]
        # Every tag is complete declaration data (spec #68 shape).
        for t in tags:
            assert set(t) == {"text", "category", "label", "verification", "common"}
            assert t["category"] in ("笑叹", "呼吸停顿", "疑问确认")


def test_minimax_declares_verbal_tags_and_pause_inserter(client):
    tags = _caps(client, "minimax-speech-cloud")["nonverbal_tags"]
    assert len(tags) == 20  # 19 verbal sounds + the <#x#> pause inserter
    assert [t["text"] for t in tags][:19] == MINIMAX_TAG_TEXTS[:19]
    assert tags[-1]["text"] == "<#1#>"
    assert tags[-1]["category"] == "停顿"
    # 前三个常用词是引擎自声明的快捷插入项（ADR-0018）。
    assert [t["text"] for t in tags if t["common"]] == ["(laughs)", "(chuckle)", "(sighs)"]
    # 情绪不进标签集：MiniMax 情绪走既有 emotion 参数（两套机制不打架）。
    assert not any("开心" in t["label"] or "情绪" in t["label"] or "悲伤" in t["label"] for t in tags)


def test_engines_without_tag_support_return_empty_set(client):
    """空集是数据：不支持正文标记的引擎返回空列表，前端整体隐藏控件。"""
    ids = _engine_ids(client)
    candidates = ["indextts-25-cuda", "indextts-25-mps", "fireredtts3-mps", "qwen3-tts-mlx",
                  "qwen3-tts-vc-cloud", "qwen3-tts-vd-cloud"]
    empty_engines = [i for i in candidates if i in ids]
    assert empty_engines, "no tag-less engine registered on this platform"
    for engine_id in empty_engines:
        caps = _caps(client, engine_id)
        assert caps["nonverbal_tags"] == [], engine_id


def test_fake_engine_declares_a_tag_set_for_contract_coverage(client):
    tags = _caps(client, "fake")["nonverbal_tags"]
    assert [t["text"] for t in tags] == ["[laughing]", "[sigh]", "[breath]"]
    # fake declares exactly ONE common favorite (data-driven quickTags).
    assert [t["text"] for t in tags if t["common"]] == ["[laughing]"]


def test_every_engine_has_nonverbal_tags_key(client):
    """The key is ALWAYS present — absence must never read as an empty set
    by accident; engines declare the field explicitly (ADR-0018)."""
    r = client.get("/engines")
    for e in r.json()["engines"]:
        assert "nonverbal_tags" in e["capabilities"], e["id"]


# --- normalization exemptions (US8 / US9) ------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # MiniMax pause marker: decimal seconds survive byte-identical.
        ("你好<#2.5#>世界2024年3月5日", "你好<#2.5#>世界二零二四年三月五日"),
        ("等一下<#0.01#>再<#99.99#>说3个", "等一下<#0.01#>再<#99.99#>说三个"),
        # IndexTTS-2.5 pinyin annotation: tone digits survive.
        ("我要<行|xing2>，买了3个", "我要<行|xing2>，买了三个"),
        ("<长得|ZHANG3>真快，2024年了", "<长得|ZHANG3>真快，二零二四年了"),
        # Multiple markers keep their order and content.
        ("A<#1#>B<汉|HAN4>C5", "A<#1#>B<汉|HAN4>C五"),
        # Only-digit content: the marker is the number, nothing else to do —
        # still byte-identical.
        ("停顿<#2.5#>", "停顿<#2.5#>"),
    ],
)
def test_marker_syntax_is_exempt_from_number_rewriting(client, text, expected):
    r = client.post("/normalize", json={"text": text})
    assert r.status_code == 200
    assert r.json()["normalized"] == expected


def test_exempt_markers_stay_byte_identical_but_other_digits_change(client):
    text = "时长<#2.5#>秒，成本12.5元，发音<例|li4>，编号13800138000"
    r = client.post("/normalize", json={"text": text})
    normalized = r.json()["normalized"]
    assert "<#2.5#>" in normalized
    assert "<例|li4>" in normalized
    assert "十二元五角" in normalized
    assert "一三八零零一三八零零零" in normalized


def test_normalize_is_idempotent_with_markers(client):
    text = "你好<#2.5#>，2024年3月5日<行|xing2>买了3个"
    once = client.post("/normalize", json={"text": text}).json()["normalized"]
    twice = client.post("/normalize", json={"text": once}).json()["normalized"]
    assert once == twice
