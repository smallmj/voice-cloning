"""Issue #23 / ADR-0018: declared parameter surface.

Covers the extended ParamSpec (kinds, intervals, not-exposed data, wire
mapping), the canonical layer + per-engine adapters, and the three
pronunciation grammars (IndexTTS / MiniMax / dots.tts).
"""

from __future__ import annotations

import json

import pytest

from voiceclone_sidecar.capabilities import (
    NOT_EXPOSED_REASONS,
    AppliesTo,
    ObjectField,
    ParamSpec,
)
from voiceclone_sidecar.engines.fake import FakeKeyEngine
from voiceclone_sidecar.pronunciation import (
    mark_pinyin,
    parse_annotations,
    rewrite,
    to_dots,
    to_indextts,
    to_minimax,
)

# --- ParamSpec validation ----------------------------------------------------


def test_exposed_spec_requires_applies_to():
    with pytest.raises(ValueError, match="applies_to"):
        ParamSpec(name="x", label="X", kind="text")


def test_unexposed_spec_requires_reason_from_closed_vocabulary():
    with pytest.raises(ValueError, match="not_exposed_reason"):
        ParamSpec(name="x", label="X", kind="text", exposed=False)
    with pytest.raises(ValueError, match="not_exposed_reason"):
        ParamSpec(
            name="x", label="X", kind="text", exposed=False, not_exposed_reason="because"
        )
    for reason in NOT_EXPOSED_REASONS:
        ParamSpec(
            name="x",
            label="X",
            kind="text",
            exposed=False,
            not_exposed_reason=reason,
            applies_to=AppliesTo(engine="e"),
        )


def test_canonical_spec_requires_to_wire_adapter():
    # The whole point of ADR-0018 decision 1: no adapter, no canonical slot.
    with pytest.raises(ValueError, match="to_wire"):
        ParamSpec(
            name="speed", label="语速", kind="number", layer="canonical",
            applies_to=AppliesTo(engine="e"),
        )
    ParamSpec(
        name="speed", label="语速", kind="number", layer="canonical",
        applies_to=AppliesTo(engine="e"), to_wire=lambda v: v,
    )


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        ParamSpec(name="x", label="X", kind="slider", applies_to=AppliesTo(engine="e"))


def test_to_dict_is_json_safe():
    spec = ParamSpec(
        name="vol", label="音量", kind="number",
        min=0, max=10, min_open=True, unit="", integer=False,
        applies_to=AppliesTo(engine="e", model="m", mode="cloning"),
        to_wire=lambda v: v,
        items=(ObjectField(name="voice_id", label="音色", kind="text"),),
        max_items=4,
    )
    assert json.loads(json.dumps(spec.to_dict()))["max_items"] == 4
    assert spec.to_dict()["wire_map"] is True
    assert spec.to_dict()["applies_to"] == {"engine": "e", "model": "m", "mode": "cloning"}


# --- pronunciation grammars ---------------------------------------------------


def test_parse_annotations_canonical_format():
    ann, errors = parse_annotations("行 = xing2\n重: chong2\n\n误=1\n的了x")
    assert ann == {"行": "xing2", "重": "chong2"}
    assert len(errors) == 2  # 误=1 (bad pinyin) and 的了x (unparseable line)


def test_indextts_grammar_uppercase_digit_tone():
    text = "他在银行里走了半天"
    _ann, _ = parse_annotations("行=xing2\n行2=hang2")
    out = to_indextts(text, {"行": "xing2"})
    assert "<行|XING2>" in out
    # Multi-char pinyin stays as-is; tone digit is kept, never a diacritic.
    ann2, _ = parse_annotations("重=zhong4")
    assert "<重|ZHONG4>" in to_indextts("重要", ann2)


def test_minimax_grammar_lowercase_parenthesized():
    ann, _ = parse_annotations("吃=chu3\n里=li3")
    assert to_minimax("吃里扒外", ann) == "(chu3)(li3)扒外"


def test_dots_grammar_tone_marked_pinyin():
    ann, _ = parse_annotations("号=hao4\n行=xing2")
    assert to_dots("号", ann) == "hào"
    assert to_dots("银行", {"行": "hang2"}) == "银háng"


def test_mark_pinyin_standard_placement():
    assert mark_pinyin("hao4") == "hào"
    assert mark_pinyin("xing2") == "xíng"
    assert mark_pinyin("zhong1") == "zhōng"
    assert mark_pinyin("liu4") == "liù"  # trailing iu -> mark the u
    assert mark_pinyin("gui4") == "guì"  # trailing ui -> mark the i
    assert mark_pinyin("de") == "de"  # neutral tone: no mark


def test_rewrite_dispatches_by_grammar_and_skips_bad_lines():
    out = rewrite("银行", "行=xing2\n坏的", "indextts", log=lambda m: None)
    assert out == "银<行|XING2>"
    assert rewrite("文本", "", "indextts") == "文本"
    with pytest.raises(ValueError):
        rewrite("文本", "行=xing2", "klingon")


# --- engine declarations (ADR-0018 contract) ----------------------------------


def _all_specs(registry):
    for engine in registry.list():
        for spec in engine.param_specs():
            yield engine, spec


@pytest.fixture()
def registry():
    from voiceclone_sidecar.registry import default_registry

    return default_registry()


def test_every_engine_spec_satisfies_the_adr_contract(registry):
    """Acceptance: every exposed spec declares what it was verified against;
    every not-exposed spec carries a reason from the closed vocabulary; the
    to_dict projection is JSON-safe (it is what the UI renders)."""
    for engine, spec in _all_specs(registry):
        d = spec.to_dict()
        json.dumps(d)
        if spec.exposed:
            assert spec.applies_to is not None, (engine.engine_id, spec.name)
            assert d["applies_to"]["engine"]
        else:
            assert spec.not_exposed_reason in NOT_EXPOSED_REASONS


def test_indextts_engines_expose_pronunciation_and_speed(registry):
    engines = {
        e.engine_id: e
        for e in registry.list()
        if e.engine_id.startswith("indextts")
    }
    assert engines, "IndexTTS engines must be registered on this platform"
    for engine in engines.values():
        names = {s.name for s in engine.param_specs() if s.exposed}
        # The capability flag always said True; the UI finally has an entry.
        assert engine.capabilities().pronunciation_control is True
        assert "pronunciation" in names
        assert "speed" in names


def test_canonical_speed_adapter_inverts_index_tts_duration_factor(registry):
    for engine in registry.list():
        if not engine.engine_id.startswith("indextts"):
            continue
        spec = next(s for s in engine.param_specs() if s.name == "speed")
        assert spec.to_wire(2.0) == 0.5  # faster rate -> smaller duration factor
        assert spec.to_wire(0.5) == 2.0
        assert spec.to_wire(1.0) == 1.0
        assert spec.to_wire("abc") is None
        assert spec.to_wire(-1) is None


def test_prepare_synthesis_rewrites_text_and_maps_wire(registry):
    from voiceclone_sidecar.engines.indextts25 import prepare_synthesis

    logs: list[str] = []
    text, wire = prepare_synthesis(
        "他在银行里",
        {"speed": 1.25, "language": "en", "pronunciation": "行=hang2\n坏=1"},
        logs.append,
    )
    assert "<行|HANG2>" in text
    assert wire == {"duration_factor": 0.8, "lang": "en"}


def test_prepare_synthesis_defaults_and_rejects_unknown_lang():
    from voiceclone_sidecar.engines.indextts25 import prepare_synthesis

    text, wire = prepare_synthesis("你好", {"language": "fr"}, lambda m: None)
    assert text == "你好"  # no annotations -> text untouched
    assert wire["lang"] == "zh"  # unknown canonical value falls back, never lies
    assert "duration_factor" not in wire


def test_mlx_dead_speed_is_data_not_a_slider(registry):
    engine = next(e for e in registry.list() if e.engine_id == "qwen3-tts-mlx")
    speed = next(s for s in engine.param_specs() if s.name == "speed")
    assert speed.exposed is False
    assert speed.not_exposed_reason == "no-op"
    # And the language slot carries the (fixed, issue #47) wire mapping:
    # lang_code, lowercased. Issue #68: the no-op verdict is REVOKED —
    # re-verification proved the installed mlx-audio consumes lang_code,
    # so the selector is exposed canonical data again.
    language = next(s for s in engine.param_specs() if s.name == "language")
    assert language.exposed is True
    assert language.not_exposed_reason is None
    assert language.to_wire("Chinese") == "chinese"


def test_fake_key_engine_spec_has_applies_to():
    spec = FakeKeyEngine().param_specs()[0]
    assert spec.applies_to is not None
    assert spec.exposed


# --- issue #56: 英文 CMUDict 发音标注路由 -------------------------------------


def test_parse_annotations_accepts_english_pipe_form():
    """工单 #56 形态：`词|ARPAbet`（`|` 也可作行内分隔符）与 `词=值` 共用入口。"""
    ann, errors = parse_annotations("world|W ER1 L D")
    assert ann == {"world": "W ER1 L D"}
    assert errors == []
    ann2, errors2 = parse_annotations("world = world|W ER1 L D")
    assert ann2 == {"world": "world|W ER1 L D"}
    assert errors2 == []


def test_voxcpm_explicit_arpabet_annotation():
    """`world|W ER1 L D` -> `{W ER1 L D}`（全大写 + 重音数字，官方音素语法）。"""
    out = rewrite("hello world", "world|W ER1 L D", "voxcpm")
    assert out == "hello {W ER1 L D}"


def test_voxcpm_cmudict_lookup_uses_bundled_offline_dictionary():
    """标注值为纯英文单词时查仓库打包的精简 CMUDict（离线、无网络）。"""
    from voiceclone_sidecar import pronunciation

    phones = pronunciation.cmudict_lookup("world")
    assert tuple(phones) == ("W", "ER1", "L", "D")
    out = rewrite("hello world", "world=world", "voxcpm")
    assert out == "hello {W ER1 L D}"


def test_voxcpm_bad_english_annotation_degrades_to_original_text():
    """坏标注（词典未收录 / 音素非法）降级原文，不阻塞生成；仅记日志。"""
    logs: list[str] = []
    out = rewrite("hello world", "zzzqqx=zzzqqx", "voxcpm", log=logs.append)
    assert out == "hello world"  # 原文保留
    assert logs and "zzzqqx" in logs[0]
    out2 = rewrite("hello world", "world|not a phoneme", "voxcpm", log=logs.append)
    assert out2 == "hello world"


def test_chinese_and_english_annotations_share_one_entry_and_route_by_value():
    """同一标注块内中文拼音与英文 CMUDict 共存，按标注值内容分流词典。"""
    out = rewrite(
        "你好 world",
        "你=ni3\nworld|W ER1 L D",
        "voxcpm",
    )
    assert out == "{ni3}好 {W ER1 L D}"


def test_english_annotation_replaces_whole_words_only():
    """英文词替换按词边界进行：world 不误伤 worldwide。"""
    out = rewrite("worldwide world", "world|W ER1 L D", "voxcpm")
    assert out == "worldwide {W ER1 L D}"


def test_english_annotations_only_apply_to_voxcpm_grammar():
    """CMUDict 路由是 voxcpm 专属扩展；其他语法保持既有行为（pinyin 值）。"""
    assert rewrite("银行", "行=xing2", "indextts") == "银<行|XING2>"


def test_english_annotation_matches_case_insensitively_whole_words():
    """Review 锁定（issue #54 review 修复 8）：英文标注的整词替换是大小写
    不敏感的——正文 World / WORLD 与标注 key world 同样命中（语义披露在
    docs/audit/local-engine-param-gap-audit.md §2.3），词边界语义不变：
    worldwide 不受影响。"""
    out = rewrite("World worldwide WORLD", "world|W ER1 L D", "voxcpm")
    assert out == "{W ER1 L D} worldwide {W ER1 L D}"
