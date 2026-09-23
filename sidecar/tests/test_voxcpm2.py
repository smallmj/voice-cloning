"""VoxCPM2 engine tests (issue #25): install wiring, three cloning modes,
MPS/CUDA gates. Everything is stubbed at the runtime seam — no network, no
model load. Gate tests run the REAL worker script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from voiceclone_sidecar.engines import voxcpm2_base, voxcpm2_mps
from voiceclone_sidecar.engines.voxcpm2_mps import VoxCPM2MpsEngine
from voiceclone_sidecar.pronunciation import rewrite
from voiceclone_sidecar.registry import GenerationRequest
from voiceclone_sidecar.runtime import uvman


@pytest.fixture()
def engine(tmp_path: Path) -> VoxCPM2MpsEngine:
    return VoxCPM2MpsEngine(output_dir=tmp_path / "audio", root=tmp_path / "runtime", env={})


def test_install_step_order(engine):
    steps = [s.id for s in engine.install_steps()]
    assert steps == ["python", "venv", "torch", "engine", "weights"]


def test_torch_step_pins_pypi_versions_on_mps(engine, tmp_path, monkeypatch):
    installed = {}
    monkeypatch.setattr(
        uvman, "pip_install",
        lambda uv, venv, packages, env=None, log=None: installed.setdefault("packages", packages),
    )
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")
    engine.install_steps()[2].run(lambda m: None, lambda *a: None)
    assert installed["packages"] == ["torch==2.8.0", "torchaudio==2.8.0"]


def test_cuda_variant_fetches_exact_wheels(tmp_path, monkeypatch):
    from voiceclone_sidecar.engines.voxcpm2_cuda import VoxCPM2CudaEngine

    engine = VoxCPM2CudaEngine(output_dir=tmp_path / "audio", root=tmp_path / "runtime", env={})
    wheels = []
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")
    monkeypatch.setattr(uvman, "pip_install", lambda *a, **k: wheels.append(a[2]))
    # Stub the wheel download (real wheels are GB-sized; the URL shape is
    # what this test asserts).
    from voiceclone_sidecar.runtime import downloader

    def fake_download(spec, dest_dir, sources, progress=None, log=None):
        dest = dest_dir / spec.dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"wheel")
        return dest

    monkeypatch.setattr(downloader, "download_file", fake_download)
    engine.install_steps()[2].run(lambda m: None, lambda *a: None)
    # Exact CUDA wheel URLs, installed from local files — no index resolution.
    # local files carry a literal "+" (the URL name uses %2B)
    assert any("torch-2.8.0+cu128-cp311-cp311-win_amd64.whl" in p for p in wheels[0])


def test_engine_step_pins_voxcpm_version(engine, tmp_path, monkeypatch):
    installed = {}
    monkeypatch.setattr(
        uvman, "pip_install",
        lambda uv, venv, packages, env=None, log=None: installed.setdefault("packages", packages),
    )
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")
    engine.install_steps()[3].run(lambda m: None, lambda *a: None)
    assert installed["packages"] == ["voxcpm==2.0.3"]


def test_pronunciation_grammar_voxcpm():
    assert rewrite("你好世界", "你=ni3", "voxcpm") == "{ni3}好世界"


def test_synthesize_requires_install(engine):
    with pytest.raises(RuntimeError, match="not installed"):
        engine.synthesize(GenerationRequest(generation_id="g1", text="你好", params={}), lambda m: None)


def test_synthesize_requires_reference_audio(engine):
    engine.is_installed = lambda: True
    with pytest.raises(RuntimeError, match="参考音频"):
        engine.synthesize(GenerationRequest(generation_id="g2", text="你好", params={}), lambda m: None)


def test_registry_registers_mps_engine_on_apple_silicon(tmp_path):
    from voiceclone_sidecar.registry import default_registry

    registry = default_registry(output_dir=tmp_path / "audio")
    assert registry.get("voxcpm2-mps") is not None
    assert registry.get("voxcpm2-mps").display_name == "VoxCPM2（本地 · MPS）"


def test_param_specs_exposed_params_carry_applies_to(engine):
    for spec in engine.param_specs():
        if spec.exposed:
            assert spec.applies_to is not None
            assert spec.applies_to.engine == "voxcpm2-mps"


def test_normalize_param_is_declared_not_exposed(engine):
    """Engine-side TN duplicates this app's ADR-0008 layer — declared data."""
    spec = next(s for s in engine.param_specs() if s.name == "normalize")
    assert spec.exposed is False
    assert spec.not_exposed_reason == "breaks-pipeline"


def test_denoise_param_not_exposed_without_denoiser(engine):
    spec = next(s for s in engine.param_specs() if s.name == "denoise")
    assert spec.exposed is False


def test_worker_gate_refuses_without_torch(tmp_path):
    """The REAL worker on this interpreter (no torch in the engine venv
    sense) must exit with the gate code and an actionable message."""
    worker = Path(voxcpm2_mps.__file__).with_name("voxcpm2_worker.py")
    proc = subprocess.run(
        [sys.executable, "-u", str(worker), "mps"],
        input="", capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 3
    assert '"fatal": true' in proc.stdout
    assert "torch is not importable" in proc.stdout


def test_worker_refuses_missing_gate_argument():
    worker = Path(voxcpm2_mps.__file__).with_name("voxcpm2_worker.py")
    proc = subprocess.run(
        [sys.executable, "-u", str(worker)],
        input="", capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 3
    assert "gate not specified" in proc.stdout


def test_cuda_and_mps_share_the_base():
    from voiceclone_sidecar.engines.voxcpm2_cuda import VoxCPM2CudaEngine

    assert issubclass(VoxCPM2MpsEngine, voxcpm2_base.VoxCPM2EngineBase)
    assert issubclass(VoxCPM2CudaEngine, voxcpm2_base.VoxCPM2EngineBase)
    # Identical model, identical capability surface.
    a = VoxCPM2MpsEngine(output_dir=Path("/tmp/a"), root=Path("/tmp/a"), env={})
    b = VoxCPM2CudaEngine(output_dir=Path("/tmp/a"), root=Path("/tmp/a"), env={})
    assert a.capabilities().to_dict() == b.capabilities().to_dict()


# --- issues #49 / #50: seed fix + validation params (worker payload) --------


def _run_worker_synthesis(monkeypatch, request):
    """Drive the REAL worker _synthesis with a stubbed runtime seam: torch
    (manual_seed capture), soundfile (write capture) and the post-check."""
    import types

    from voiceclone_sidecar.engines import voxcpm2_worker as worker

    captured: dict = {}
    torch_stub = types.SimpleNamespace(
        manual_seed=lambda s: captured.setdefault("manual_seed", s)
    )
    sf_stub = types.SimpleNamespace(
        write=lambda path, wav, sr, subtype=None: captured.update(
            written=(str(path), wav, int(sr))
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "soundfile", sf_stub)
    monkeypatch.setattr(
        worker.common, "verify_wav_output", lambda out, log, label: {"ok": True}
    )

    generate_calls: list[dict] = []

    class FakeTTSModel:
        sample_rate = 48000

    class FakeModel:
        tts_model = FakeTTSModel()

        def generate(self, **kwargs):
            generate_calls.append(kwargs)
            return [0.0] * 480  # no numpy in the sidecar test venv

    request = {
        "output": "/tmp/out.wav",
        "ref_audio": "/tmp/ref.wav",
        "model_dir": "/tmp/weights",
        **request,
    }
    result = worker._synthesis(
        request, lambda model_dir: FakeModel(), lambda: {}, lambda msg: None
    )
    assert result["audio_path"] == "/tmp/out.wav"
    return captured, generate_calls


def test_seed_is_never_passed_to_generate_and_pins_manual_seed(monkeypatch):
    """Issue #49 regression lock: pinned VoxCPM 2.0.3 _generate() has NO seed
    parameter and no **kwargs — the old worker sent seed= and would raise
    TypeError on the real machine. Reproducibility now goes through
    worker-side torch.manual_seed (verified byte-identical on real synthesis,
    2026-09-22)."""
    captured, calls = _run_worker_synthesis(monkeypatch, {"text": "你好", "seed": 42})
    assert len(calls) == 1
    assert "seed" not in calls[0]  # the TypeError trigger is gone
    assert captured["manual_seed"] == 42
    # 3090 real-machine finding (2026-09-22): same seed is byte-reproducible
    # with manual_seed + the model-load warmup — the deterministic-algorithm
    # flags are NOT needed (runs 2+ were identical without them).


def test_seed_absent_leaves_rng_untouched(monkeypatch):
    captured, calls = _run_worker_synthesis(monkeypatch, {"text": "你好"})
    assert len(calls) == 1
    assert "manual_seed" not in captured  # random by default


def test_load_performs_seed_reproducibility_warmup(monkeypatch):
    """Issue #49: the FIRST generation after model load deviates from
    same-seed reruns on CUDA (3090 实测) — the worker must run one throwaway
    warmup generation at load (synthetic sine reference, 4 timesteps) so the
    first user request is already byte-reproducible. Upstream masks the same
    effect with its own warmup when optimize=True; we pin optimize=False."""
    source = Path(voxcpm2_mps.__file__).with_name("voxcpm2_worker.py").read_text()
    assert "seed-reproducibility warmup" in source
    assert "inference_timesteps=4" in source
    assert "torch.manual_seed(0)" in source


def test_validated_length_and_retry_params_are_forwarded(monkeypatch):
    """Issue #50: min_len/max_len and the retry_badcase triple are real
    upstream _generate() parameters (2.0.3 signature, real-machine verified)
    and must reach generate() with their declared types."""
    _, calls = _run_worker_synthesis(
        monkeypatch,
        {
            "text": "你好",
            "min_len": 2,
            "max_len": 4096,
            "retry_badcase": True,
            "retry_badcase_max_times": 3,
            "retry_badcase_ratio_threshold": 6.0,
        },
    )
    assert calls[0]["min_len"] == 2
    assert calls[0]["max_len"] == 4096
    assert calls[0]["retry_badcase"] is True
    assert calls[0]["retry_badcase_max_times"] == 3
    assert calls[0]["retry_badcase_ratio_threshold"] == 6.0


def test_validated_params_absent_when_not_chosen(monkeypatch):
    _, calls = _run_worker_synthesis(monkeypatch, {"text": "你好"})
    for key in (
        "min_len",
        "max_len",
        "retry_badcase",
        "retry_badcase_max_times",
        "retry_badcase_ratio_threshold",
    ):
        assert key not in calls[0]  # upstream defaults apply unchanged


# --- issue #50: declaration surface -----------------------------------------


@pytest.mark.parametrize(
    "name,kind,default",
    [
        ("min_len", "number", 2),
        ("max_len", "number", 4096),
        ("retry_badcase", "bool", True),
        ("retry_badcase_max_times", "number", 3),
        ("retry_badcase_ratio_threshold", "number", 6.0),
    ],
)
def test_validation_params_exposed_with_upstream_defaults(engine, name, kind, default):
    """Declared defaults MUST equal the upstream core-layer defaults
    (min_len=2, max_len=4096, retry_badcase=True/3/6.0) so the UI shows what
    the engine actually does."""
    spec = next(s for s in engine.param_specs() if s.name == name)
    assert spec.exposed is True
    assert spec.kind == kind
    assert spec.default == default
    assert spec.applies_to is not None
    assert spec.applies_to.engine == "voxcpm2-mps"
    assert spec.layer == "engine"  # engine-specific collapsed area


def test_denoise_is_no_op_data_after_real_machine_verification(engine):
    """Issue #50 V7: with load_denoiser=False (offline install), upstream
    silently skips denoise (self.denoiser is None) — output byte-identical
    with denoise=True on the real machine (2026-09-22)."""
    spec = next(s for s in engine.param_specs() if s.name == "denoise")
    assert spec.exposed is False
    assert spec.not_exposed_reason == "no-op"


def test_prepare_synthesis_forwards_seed_for_worker_side_handling(engine):
    _text, wire = voxcpm2_base.prepare_synthesis("你好", {"seed": 7}, lambda m: None)
    assert wire["seed"] == 7  # consumed by the worker via torch.manual_seed
    assert "denoise" not in wire  # stays a declared no-op, never forwarded


# --- issue #56: control_instruction 参数 + 归一化后拼接时序 -------------------


def test_control_instruction_param_declaration(engine):
    """引擎层参数：string 多行（textarea）、默认空、默认折叠在引擎层；
    help 含工单要求的文案。"""
    spec = next(s for s in engine.param_specs() if s.name == "control_instruction")
    assert spec.exposed is True
    assert spec.kind == "textarea"
    assert spec.default in ("", None)
    assert spec.layer == "engine"
    assert spec.applies_to is not None
    for token in ("建议英文描述", "warm female voice", "音色设计", "克隆风格控制"):
        assert token in spec.help, token


def test_control_instruction_is_appended_after_normalization(engine):
    """拼接时序锁定（spec #54 / ADR-0008）：正文先归一化、后拼控制指令。
    prepare_synthesis 收到的就是归一化后的正文；含数字的英文指令必须原样
    到达引擎（永不经归一化层），且格式为官方语法 `(指令)正文`。"""
    normalized_body = "今天是二零二四年三月五日"  # pipeline 归一化产物
    text, _wire = voxcpm2_base.prepare_synthesis(
        normalized_body,
        {"control_instruction": "a 30-year-old warm voice"},
        lambda m: None,
    )
    assert text == "(a 30-year-old warm voice)今天是二零二四年三月五日"


def test_control_instruction_empty_is_never_sent(engine):
    """空值不下发：既无括号前缀，也不出现在 wire 参数里。"""
    for params in ({}, {"control_instruction": ""}, {"control_instruction": "   "}):
        text, wire = voxcpm2_base.prepare_synthesis("你好", params, lambda m: None)
        assert text == "你好"
        assert "control_instruction" not in wire


def test_control_instruction_concatenates_with_pronunciation_rewrite(engine):
    """与发音标注改写同时序：指令在最前，标注改写作用在正文。"""
    text, _wire = voxcpm2_base.prepare_synthesis(
        "你好世界",
        {"control_instruction": "warm female voice", "pronunciation": "你=ni3"},
        lambda m: None,
    )
    assert text == "(warm female voice){ni3}好世界"


def test_control_instruction_help_is_user_facing(engine):
    """Review 修复：help 是用户可见文案——不带 spec/ADR/issue 等内部引用，
    也没有重复句（同一短句只出现一次）。"""
    spec = next(s for s in engine.param_specs() if s.name == "control_instruction")
    for token in ("spec #", "ADR-", "issue #"):
        assert token not in spec.help, token
    assert spec.help.count("用于音色设计与克隆风格控制") == 1


def test_full_synthesis_text_returns_the_complete_text_for_preview(engine):
    """US19（issue #54 review 修复）：预览钩子返回将要合成的完整文本——
    与 prepare_synthesis 走同一个 adapt_synthesis_text，归一化后的正文 +
    发音标注改写 + 控制指令前缀，预览与真实合成不可能漂移。"""
    full = engine.full_synthesis_text(
        "你好世界",
        {"control_instruction": "warm female voice", "pronunciation": "你=ni3"},
    )
    assert full == "(warm female voice){ni3}好世界"
    # 与 prepare_synthesis 的文本路径逐字节一致（同一函数的同一实现）。
    adapted, _wire = voxcpm2_base.prepare_synthesis(
        "你好世界",
        {"control_instruction": "warm female voice", "pronunciation": "你=ni3"},
        lambda m: None,
    )
    assert full == adapted


def test_full_synthesis_text_without_instruction_is_identity(engine):
    """无控制指令（或缺省参数）时预览文本不变——默认引擎上下文不撒谎。"""
    for params in (None, {}, {"control_instruction": "  "}):
        assert engine.full_synthesis_text("你好", params) == "你好"


def test_voxcpm2_capabilities_declare_official_30_languages(engine):
    """语种声明扩为官方 30 语种清单（HF 模型卡 language 字段口径，声明性
    支持——真机抽查随验证工单，不加语言选择 UI）。清单在生产侧只定义一份
    （voxcpm2_base.VOXCPM2_LANGUAGES）；capabilities 与测试都从它派生，
    互锁的是「capabilities 忠实于唯一常量」这一层。"""
    official = voxcpm2_base.VOXCPM2_LANGUAGES
    assert len(official) == 30
    assert len(set(official)) == 30  # the single source carries no duplicates
    caps = engine.capabilities()
    assert len(caps.languages) == 30
    assert set(caps.languages) == set(official)
    assert caps.languages == official  # order also pinned to the one source


# --- issue #57: 原生 Voice Design（无参考生成） ------------------------------


class _RecordingSupervisor:
    """Fake worker seam: records payloads, writes a real playable WAV."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.payloads: list[dict] = []

    def request(self, payload: dict, on_log=None, timeout_s=None) -> dict:
        self.payloads.append(payload)
        out = self.tmp_path / "audio" / Path(payload["output"]).name
        out.parent.mkdir(parents=True, exist_ok=True)
        from voiceclone_sidecar.engines.fake import write_tone_wav

        write_tone_wav(out, 0.6)
        return {"audio_path": str(out), "sample_rate": 48000}


def _design_engine(tmp_path: Path) -> tuple[VoxCPM2MpsEngine, _RecordingSupervisor]:
    eng = VoxCPM2MpsEngine(output_dir=tmp_path / "audio", root=tmp_path / "runtime", env={})
    sup = _RecordingSupervisor(tmp_path)
    eng.is_installed = lambda: True
    eng._get_supervisor = lambda: sup
    return eng, sup


def test_capabilities_declare_voice_design(engine):
    """能力位：基类声明（MPS/CUDA 共享），/engines、/voices/design 门禁据此放行。"""
    assert engine.capabilities().voice_design is True


def test_design_voice_sends_parenthesized_prefix_without_reference(tmp_path):
    """design_voice = 控制指令括号前缀 + 试听文本，无参考音频；描述与试听
    文本不经归一化（design 调用不走管线，逐字到达引擎）。"""
    eng, sup = _design_engine(tmp_path)
    result = eng.design_voice(
        "Warm female voice with 2.5Hz pacing, 数字 123 保持原样",
        "晚上好，欢迎收听。",
        lambda m: None,
    )
    payload = sup.payloads[0]
    assert payload["design"] is True
    assert "ref_audio" not in payload
    assert "prompt_text" not in payload
    # 描述（含数字与英文）+ 试听文本原样拼接，永不经归一化层。
    assert payload["text"] == "(Warm female voice with 2.5Hz pacing, 数字 123 保持原样)晚上好，欢迎收听。"
    # 返回契约：本地生成 voice_id + 预览样本路径 + transcript=试听文本。
    assert result["voice_id"].startswith("voxcpm2-design-")
    assert Path(result["sample_audio_path"]).name.startswith("design-")
    assert result["transcript"] == "晚上好，欢迎收听。"


def test_design_voice_rejects_empty_description_or_preview(tmp_path):
    eng, sup = _design_engine(tmp_path)
    with pytest.raises(RuntimeError, match="声音描述"):
        eng.design_voice("  ", "晚上好", lambda m: None)
    with pytest.raises(RuntimeError, match="试听文本"):
        eng.design_voice("低沉男声", "  ", lambda m: None)
    assert sup.payloads == []  # nothing reached the worker


def test_design_voice_requires_installed_engine(tmp_path):
    eng = VoxCPM2MpsEngine(output_dir=tmp_path / "audio", root=tmp_path / "runtime", env={})
    with pytest.raises(RuntimeError, match="not installed"):
        eng.design_voice("低沉男声", "晚上好", lambda m: None)


def test_synthesize_payload_never_sets_design_flag(tmp_path):
    """普通合成的强制 ref_audio 不变，且永不携带 design 标记——design 是
    design_voice 的专用入口。"""
    eng, sup = _design_engine(tmp_path)
    eng.synthesize(
        GenerationRequest(
            generation_id="g57", text="你好", params={"ref_audio": "/tmp/ref.wav"}
        ),
        lambda m: None,
    )
    payload = sup.payloads[0]
    assert "design" not in payload
    assert payload["ref_audio"] == str(Path("/tmp/ref.wav").resolve())


def test_worker_design_path_generates_without_reference(monkeypatch):
    """worker 无参考生成路径：design=True 时 generate() 不带 reference_wav_path。"""
    import types

    from voiceclone_sidecar.engines import voxcpm2_worker as worker

    sf_stub = types.SimpleNamespace(
        write=lambda path, wav, sr, subtype=None: None
    )
    monkeypatch.setitem(sys.modules, "soundfile", sf_stub)
    monkeypatch.setattr(
        worker.common, "verify_wav_output", lambda out, log, label: {"ok": True}
    )
    calls: list[dict] = []

    class FakeModel:
        tts_model = types.SimpleNamespace(sample_rate=48000)

        def generate(self, **kwargs):
            calls.append(kwargs)
            return [0.0] * 480

    result = worker._synthesis(
        {
            "output": "/tmp/out.wav",
            "design": True,  # no ref_audio — the design entry
            "model_dir": "/tmp/weights",
            "text": "(warm female voice)晚上好。",
        },
        lambda model_dir: FakeModel(),
        lambda: {},
        lambda msg: None,
    )
    assert result["audio_path"] == "/tmp/out.wav"
    assert calls[0]["text"] == "(warm female voice)晚上好。"
    assert "reference_wav_path" not in calls[0]


def test_worker_without_design_still_requires_reference(monkeypatch):
    """非 design 场景的强制 ref_audio 不变（无 ref_audio 且无 design → 报错）。"""
    import types

    from voiceclone_sidecar.engines import voxcpm2_worker as worker

    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(write=lambda *a, **k: None))

    with pytest.raises(RuntimeError, match="参考音频"):
        worker._synthesis(
            {"output": "/tmp/out.wav", "model_dir": "/tmp/weights", "text": "你好"},
            lambda model_dir: None,
            lambda: {},
            lambda msg: None,
        )


# --- issue #68: control_instruction 强制 reference-only 降级 ------------------


def test_control_instruction_with_ref_text_skips_prompt_pair(tmp_path):
    """有控制指令 + 有参考转写 → 不传 prompt pair（reference-only），指令前缀
    照拼在文本里；降级原因在日志中披露（issue #68 wrong-mode 降级）。"""
    eng, sup = _design_engine(tmp_path)
    logs: list[str] = []
    eng.synthesize(
        GenerationRequest(
            generation_id="g68",
            text="你好，世界。",
            params={
                "ref_audio": "/tmp/ref.wav",
                "ref_text": "你好，我是参考转写。",
                "control_instruction": "warm female voice",
            },
        ),
        logs.append,
    )
    payload = sup.payloads[0]
    assert "prompt_text" not in payload
    # 指令前缀照拼（reference-only 模式实测生效）。
    assert payload["text"] == "(warm female voice)你好，世界。"
    assert any("Hi-Fi prompt pairing disabled" in m for m in logs)
    assert any("reference-only" in m for m in logs)


def test_no_control_instruction_keeps_hifi_prompt_pair(tmp_path):
    """无控制指令 → Hi-Fi prompt pair 照旧（行为回归锁定）。"""
    eng, sup = _design_engine(tmp_path)
    eng.synthesize(
        GenerationRequest(
            generation_id="g68b",
            text="你好，世界。",
            params={"ref_audio": "/tmp/ref.wav", "ref_text": "你好，我是参考转写。"},
        ),
        lambda m: None,
    )
    payload = sup.payloads[0]
    assert payload["prompt_text"] == "你好，我是参考转写。"


def test_blank_control_instruction_counts_as_absent(tmp_path):
    """空白指令与空值同义（与 control_prefix 的规则一致）：不降级。"""
    eng, sup = _design_engine(tmp_path)
    eng.synthesize(
        GenerationRequest(
            generation_id="g68c",
            text="你好",
            params={"ref_audio": "/tmp/ref.wav", "ref_text": "转写", "control_instruction": "  "},
        ),
        lambda m: None,
    )
    assert sup.payloads[0]["prompt_text"] == "转写"


def test_control_instruction_without_ref_text_is_unchanged(tmp_path):
    """本来就没有转写的纯音色模式不受影响（也不报降级日志）。"""
    eng, sup = _design_engine(tmp_path)
    logs: list[str] = []
    eng.synthesize(
        GenerationRequest(
            generation_id="g68d",
            text="你好",
            params={"ref_audio": "/tmp/ref.wav", "control_instruction": "calm voice"},
        ),
        logs.append,
    )
    assert "prompt_text" not in sup.payloads[0]
    assert sup.payloads[0]["text"] == "(calm voice)你好"
    assert not any("Hi-Fi prompt pairing disabled" in m for m in logs)


def test_control_instruction_help_discloses_reference_only_demotion(engine):
    """参数 help 披露方言用法与降级行为（issue #68 文案）。"""
    spec = next(p for p in engine.param_specs() if p.name == "control_instruction")
    assert "方言" in spec.help
    assert "reference-only" in spec.help or "参考音频-only" in spec.help
