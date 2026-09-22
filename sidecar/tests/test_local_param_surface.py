"""Local-engine request-surface locks (issue #52).

Parity lock with the issue-38 cloud paradigm
(``test_param_specs_cover_the_documented_request_surface`` in
test_minimax_cloud.py): every field of each local engine's documented
request surface (docs/audit/local-engine-param-gap-audit.md §2) is either
exposed (with applies_to / to_wire) or ``exposed=False`` with a
closed-vocabulary not_exposed_reason. No gap a reviewer has to remember.

Also locks, in one place:
- the two silent-failure bug fixes (audit §1): qwen3 lang_code wire mapping
  (issue #47) and voxcpm2 seed via worker-side manual_seed (issue #49);
- the capability-matrix end state: every new / flipped claim for these
  engines is backed by a real-machine evidence entry (audit §4, V1–V7).
"""

from __future__ import annotations

import json
from pathlib import Path

from voiceclone_sidecar.engines.dots_tts_cuda import DotsTtsCudaEngine
from voiceclone_sidecar.engines.fireredtts3_mps import FireRedTts3MpsEngine
from voiceclone_sidecar.engines.qwen3_tts import Qwen3TtsMlxEngine
from voiceclone_sidecar.engines.voxcpm2_mps import VoxCPM2MpsEngine
from voiceclone_sidecar.registry import GenerationRequest

MATRIX = json.loads(
    (Path(__file__).resolve().parents[1] / "voiceclone_sidecar" / "data" / "capability_matrix.json")
    .read_text(encoding="utf-8")
)
MATRIX_BY_ENGINE = {e["engine_id"]: e for e in MATRIX["engines"]}


def _specs(engine) -> dict:
    specs = {}
    for s in engine.param_specs():
        assert s.name not in specs, f"duplicate spec {s.name}"
        specs[s.name] = s
        if s.exposed:
            # ADR-0018: an exposed declaration must say where it applies and
            # how it reaches the wire.
            assert s.applies_to is not None, s.name
    return specs


def _lock_surface(engine, expected_exposed: tuple, expected_hidden: dict):
    specs = _specs(engine)
    assert set(specs) == set(expected_exposed) | set(expected_hidden), (
        f"{engine.engine_id}: declared surface drifted — "
        f"extra={set(specs) - set(expected_exposed) - set(expected_hidden)}, "
        f"missing={set(expected_exposed) | set(expected_hidden) - set(specs)}"
    )
    for name in expected_exposed:
        assert specs[name].exposed, f"{engine.engine_id}.{name} must stay exposed"
    for name, reason in expected_hidden.items():
        assert not specs[name].exposed, f"{engine.engine_id}.{name} must stay hidden"
        assert specs[name].not_exposed_reason == reason, (
            f"{engine.engine_id}.{name}: {specs[name].not_exposed_reason!r} != {reason!r}"
        )


# --- qwen3-tts-mlx (audit §2.1) ----------------------------------------------


def test_qwen3_request_surface_matches_audit():
    engine = Qwen3TtsMlxEngine(root=Path("/tmp/ascii-root"))
    # language stays DECLARED but exposed=False/no-op: V3 real-machine proof
    # that mlx-audio never consumes lang_code (issue #47 final state). The
    # four sampling params are exposed in the engine folded area (issue #48,
    # always sent — mlx-audio's silent defaults differ from the official
    # generation_config). speed stays no-op data. max_tokens/instruct/voice/
    # seed/streaming_* are NOT declared rows: their disposition is recorded
    # as capability-matrix evidence entries, which the matrix test below locks.
    _lock_surface(
        engine,
        ("temperature", "top_p", "top_k", "repetition_penalty"),
        {"language": "no-op", "speed": "no-op"},
    )


def test_qwen3_silent_failure_lang_code_regression():
    """Audit §1.1 / issue #47: the old wire sent ``language=`` (silently
    dropped into upstream **kwargs) with capitalized values that never
    matched the weight codec_language_id keys. Locked forever: the wire key
    is lowercase ``lang_code``, Auto = key absent, and the ``language=`` key
    must never reappear in the payload."""
    engine = Qwen3TtsMlxEngine(root=Path("/tmp/ascii-root"))
    payload = engine.build_worker_payload(
        GenerationRequest(generation_id="g1", text="你好", params={"language": "Japanese"}),
        "/tmp/weights", Path("/tmp/out.wav"),
    )
    assert payload["lang_code"] == "japanese"
    assert "language" not in payload  # the old silent no-op key
    auto = engine.build_worker_payload(
        GenerationRequest(generation_id="g2", text="你好", params={"language": "Auto"}),
        "/tmp/weights", Path("/tmp/out.wav"),
    )
    assert "lang_code" not in auto and "language" not in auto


def test_qwen3_sampling_defaults_match_official_generation_config():
    engine = Qwen3TtsMlxEngine(root=Path("/tmp/ascii-root"))
    defaults = {s.name: s.to_wire(None) for s in engine.param_specs()
                if s.name in ("temperature", "top_p", "top_k", "repetition_penalty")}
    assert defaults == {
        "temperature": 0.9, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.05,
    }


# --- voxcpm2 (audit §2.3) -----------------------------------------------------


def test_voxcpm2_request_surface_matches_audit():
    engine = VoxCPM2MpsEngine(output_dir=Path("/tmp/a"), root=Path("/tmp/a"), env={})
    # §2.3 end state: seed exposed via worker-side manual_seed (issue #49);
    # min_len/max_len + retry_badcase* exposed after 3090 verification
    # (issue #50); denoise demoted to no-op data (V7: load_denoiser=False
    # makes it byte-identical); normalize breaks-pipeline (ADR-0008);
    # denoise_output wrong-mode; ref_audio/ref_text server-injected.
    _lock_surface(
        engine,
        ("cfg_value", "inference_timesteps", "min_len", "max_len",
         "retry_badcase", "retry_badcase_max_times",
         "retry_badcase_ratio_threshold", "seed",
         # issue #56: engine-layer multi-line control instruction (Voice
         # Design / style control); concatenated AFTER normalization.
         "control_instruction"),
        {"denoise": "no-op", "normalize": "breaks-pipeline",
         "denoise_output": "wrong-mode", "ref_audio": "server-injected",
         "ref_text": "server-injected"},
    )


def test_voxcpm2_silent_failure_seed_regression():
    """Audit §1.2 / issue #49: upstream VoxCPM 2.0.3 _generate() has NO seed
    parameter and no **kwargs — sending seed= raised TypeError on the real
    machine. The reproducibility contract (worker-side torch.manual_seed,
    never generate(seed=...)) is locked in detail by
    test_voxcpm2.test_seed_is_never_passed_to_generate_and_pins_manual_seed;
    this lock keeps the spec honest too: seed is exposed ENGINE-side while
    its wire route is deliberately the worker seam."""
    engine = VoxCPM2MpsEngine(output_dir=Path("/tmp/a"), root=Path("/tmp/a"), env={})
    seed = next(s for s in engine.param_specs() if s.name == "seed")
    assert seed.exposed
    assert "manual_seed" in seed.help


# --- fireredtts3-mps (audit §2.2) ---------------------------------------------


def test_fireredtts3_request_surface_matches_audit():
    engine = FireRedTts3MpsEngine(output_dir=Path("/tmp/a"), root=Path("/tmp/a"), env={})
    # §2.2 end state: the three quality params exposed (issue #51, V5/V6);
    # stop_threshold measured as byte-identical across levels → not a user
    # param and (per the #51 decision) not even a declared row; instruct is
    # wrong-mode (Base weights only); ref_audio/ref_text server-injected.
    _lock_surface(
        engine,
        ("seed", "inference_cfg", "n_timesteps", "cross_fade_ms"),
        {"instruct_mode": "wrong-mode", "ref_audio": "server-injected",
         "ref_text": "server-injected"},
    )


# --- dots-tts-cuda (audit §2.4) ------------------------------------------------


def test_dots_request_surface_matches_audit():
    engine = DotsTtsCudaEngine(output_dir=Path("/tmp/a"), root=Path("/tmp/a"), env={})
    # §2.4: most complete of the four — declared params + anti-spoof no-op
    # data (temperature/speed don't exist upstream) + server-injected refs.
    _lock_surface(
        engine,
        ("pronunciation", "speaker_scale", "num_steps", "guidance_scale", "seed"),
        {"normalize_text": "breaks-pipeline", "temperature": "no-op",
         "speed": "no-op", "ref_audio": "server-injected",
         "ref_text": "server-injected"},
    )


# --- capability_matrix end state (audit §4, V1–V7) -----------------------------


def _evidence_entry(engine_id: str, field: str):
    entry = MATRIX_BY_ENGINE[engine_id]
    matches = [ev for ev in entry.get("evidence", []) if ev.get("field") == field]
    assert matches, f"{engine_id}: no evidence entry for field {field!r}"
    return matches[0]


def test_matrix_qwen3_lang_code_claim_is_real_machine_backed():
    """V3: lang_code no-op finding — measured, 2026-09-22, and it must state
    the honest end state (correct key, still no effect)."""
    ev = _evidence_entry("qwen3-tts-mlx", "语种参数（lang_code）")
    assert ev["verification"] == "measured"
    assert ev["date"] == "2026-09-22"
    assert "lang_code" in ev["value"] and "no-op" in ev["value"]


def test_matrix_qwen3_sampling_claim_matches_spec_defaults():
    """Issue #48 end state: matrix says the four sampling params are exposed
    AND always sent with the official generation_config defaults — the same
    numbers the specs carry."""
    ev = _evidence_entry("qwen3-tts-mlx", "采样参数（temperature/top_p/top_k/repetition_penalty）")
    assert ev["verification"] in ("measured", "verified")
    for token in ("0.9", "1.0", "50", "1.05", "恒发送"):
        assert token in ev["value"], token
    engine = Qwen3TtsMlxEngine(root=Path("/tmp/ascii-root"))
    assert {s.name: s.to_wire(None) for s in engine.param_specs()
            if s.name.startswith(("temperature", "top_", "repetition"))} == {
        "temperature": 0.9, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.05,
    }


def test_matrix_qwen3_max_tokens_is_disclosed_not_declared():
    """V4 end state: max_tokens is disclosed as matrix evidence (long-text
    degradation finding) but deliberately has NO spec row — locked in
    test_qwen3_request_surface_matches_audit; here we lock the matrix side."""
    ev = _evidence_entry("qwen3-tts-mlx", "max_tokens（上游默认 1200/段）")
    assert ev["verification"] == "measured"
    assert ev["date"] == "2026-09-22"


def test_matrix_voxcpm2_generation_params_claim_is_real_machine_backed():
    """V1/V2/V7: seed disposition, length/retry params, denoise no-op — one
    evidence entry carries the end state for both mps and cuda variants."""
    for engine_id in ("voxcpm2-mps", "voxcpm2-cuda"):
        ev = _evidence_entry(engine_id, "生成参数")
        assert ev["verification"] == "verified", engine_id
        value = ev["value"]
        for token in ("seed", "cfg_value", "inference_timesteps", "retry_badcase"):
            assert token in value, (engine_id, token)


def test_matrix_voxcpm2_languages_claim_is_declarative_official_list():
    """Issue #56: the 30-language claim is the OFFICIAL list, marked
    declarative (官方口径) — real-machine spot checks live in the
    verification ticket, per the honest-marking rule."""
    official = {
        "zh", "en", "ar", "my", "da", "nl", "fi", "fr", "de", "el", "he",
        "hi", "id", "it", "ja", "km", "ko", "lo", "ms", "no", "pl", "pt",
        "ru", "es", "sw", "sv", "tl", "th", "tr", "vi",
    }
    for engine_id in ("voxcpm2-mps", "voxcpm2-cuda"):
        engine = (VoxCPM2MpsEngine(output_dir=Path("/tmp/a"), root=Path("/tmp/a"), env={})
                  if engine_id == "voxcpm2-mps" else None)
        if engine is not None:
            langs = set(engine.capabilities().languages)
            assert langs == official, engine_id
        # mps carries the dedicated 语种与方言 entry; cuda shares the
        # surface via its 能力面 entry (same model, same declaration).
        ev = _evidence_entry(
            engine_id,
            "语种与方言" if engine_id == "voxcpm2-mps" else "能力面",
        )
        assert ev["verification"] in ("vendor", "verified")
        assert "声明性支持（官方口径）" in ev["value"], engine_id
        assert "真机抽查见验证工单" in ev["value"], engine_id


def test_matrix_voxcpm2_control_instruction_is_declared_with_timing():
    """Issue #56: control_instruction is an engine-layer declared param,
    concatenated AFTER normalization (never normalized itself); live
    listening checks are deferred to the verification ticket."""
    for engine_id in ("voxcpm2-mps", "voxcpm2-cuda"):
        ev = _evidence_entry(engine_id, "控制指令（control_instruction）")
        assert ev["verification"] in ("verified", "vendor")
        for token in ("control_instruction", "归一化之后", "(指令)正文"):
            assert token in ev["value"], (engine_id, token)
        assert "验证工单" in ev["value"], engine_id


def test_matrix_fireredtts3_quality_params_claim_is_real_machine_backed():
    """V5/V6 end state (issue #51): the three quality params measured on real
    MPS hardware, 2026-09-22, including the stop_threshold byte-identical
    finding that keeps it out of the surface."""
    ev = _evidence_entry("fireredtts3-mps", "生成参数")
    assert ev["verification"] == "measured"
    assert ev["date"] == "2026-09-22"
    for token in ("inference_cfg", "n_timesteps", "cross_fade_ms", "stop_threshold"):
        assert token in ev["value"], token


def test_matrix_dots_antispoof_claim_locks_nonexistent_params():
    ev = _evidence_entry("dots-tts-cuda", "不存在的参数（防伪声明）")
    assert ev["verification"] == "verified"
    assert "temperature" in ev["value"] and "不存在" in ev["value"]


def test_local_engine_audit_doc_matches_declared_surfaces():
    """AC: the audit checklist and the code stay in lockstep — the §2 end
    state recorded in the doc names the same engines the locks above cover,
    so a future engine or param change fails here until the doc and the
    locks move together."""
    doc = (Path(__file__).resolve().parents[2] / "docs" / "audit"
           / "local-engine-param-gap-audit.md").read_text(encoding="utf-8")
    for engine_id, section in (
        ("qwen3-tts-mlx", "### 2.1 qwen3-tts-mlx"),
        ("fireredtts3-mps", "### 2.2 fireredtts3-mps"),
        ("voxcpm2", "### 2.3 voxcpm2-mps / -cuda"),
        ("dots-tts-cuda", "### 2.4 dots-tts-cuda"),
    ):
        assert section in doc, engine_id
    # The test-lock work item (§3.5) must stay in the doc until this file
    # replaces the human-memory gap list.
    assert "test_param_specs_cover_the_documented_request_surface" in doc
