"""Shared VoxCPM2 engine base (issue #25).

VoxCPM2 (OpenBMB, Apache-2.0 code + weights, 2.29B params, 48 kHz output)
runs on both macOS (MPS, officially supported via ``device="mps"``) and
Windows (CUDA). Like the IndexTTS-2.5 base (issue #32), this module holds
everything the two platform variants share — install steps, weight chain,
worker supervision, and the synthesis flow — and each variant declares only
its platform deltas: the torch source and the worker gate.

Facts this adapter encodes (all verified against upstream source at tag
2.0.3, see the capability-matrix entry):

- ``VoxCPM.from_pretrained`` accepts a LOCAL DIRECTORY as ``hf_model_id``
  (``os.path.isdir`` short-circuit) — weights are downloaded by our own
  chain and loaded fully offline;
- three cloning modes: (1) ``reference_wav_path`` alone (timbre-only, no
  transcript — VoxCPM2-only), (2) ``prompt_wav_path`` + ``prompt_text``
  (Hi-Fi continuation; the pair is REQUIRED together — upstream raises
  ValueError on one without the other), (3) both (max similarity);
- ``generate()`` returns a bare 1-D float32 numpy waveform; sample rate
  comes from ``model.tts_model.sample_rate``;
- on MPS the loader forces float32 (upstream ``pick_runtime_dtype`` —
  bfloat16/float16 are documented as glitched on MPS);
- text normalization is ``wetext`` (pynini-free), lazily imported only on
  ``normalize=True`` — we keep it OFF: normalization is this repo's own
  layer before the engine (ADR-0008), so the engine's TN is declared as a
  non-exposed parameter with reason ``breaks-pipeline``;
- phoneme input syntax ``{ni3}`` requires ``normalize=False`` — which the
  pronunciation adapter (pronunciation.to_voxcpm) guarantees.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from .. import pronunciation, sources
from ..capabilities import AppliesTo, Capabilities, ParamSpec
from ..engine_config import EngineConfig, resolve_seam
from ..registry import GenerationRequest, GenerationResult, InstallableEngine
from ..runtime import installer, paths, uvman
from .worker_supervisor import WorkerDegradedError, WorkerSupervisor

REPO = "openbmb/VoxCPM2"
MS_REPO = "OpenBMB/VoxCPM2"  # ModelScope twin verified 2026-09-20
PYTHON_SPEC = "3.11"
TORCH_VERSION = "2.8.0"  # ADR-0002 rule 1: pinned below the TorchAudio 2.9 TorchCodec cutover
VOXCPM_VERSION = "2.0.3"

# Local-directory load layout (verified: from_local reads config.json,
# model.safetensors, audiovae.pth and the tokenizer via the same directory).
WEIGHTS_FILES = [
    "config.json",
    "model.safetensors",
    "audiovae.pth",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenization_voxcpm2.py",  # custom tokenizer code, loaded by LlamaTokenizerFast
]

IDLE_TIMEOUT_S = 300.0
MAX_REQUESTS_PER_WORKER = 20

# Vendor guidance: split long text into sentences — long single requests
# cause speed-up/buzzing. 200 chars/request is our conservative vendor-based
# cap (verification level: vendor — recorded in the capability matrix).
MAX_CHARS_PER_REQUEST = 200


def param_specs_for(engine_id: str) -> list[ParamSpec]:
    """Declared parameters, verified against upstream generate() at 2.0.3.

    No canonical ``language`` parameter: VoxCPM2 has NO language argument —
    language follows the text itself (30 languages, vendor-verified)."""
    return [
        ParamSpec(
            name="cfg_value",
            label="引导强度 (CFG)",
            kind="number",
            default=2.0,
            min=1.0,
            max=3.0,
            step=0.1,
            layer="engine",
            help="控制文本引导强度；长文本下偏低（约 1.5–1.6）更稳定。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        ParamSpec(
            name="inference_timesteps",
            label="推理步数",
            kind="number",
            integer=True,
            default=10,
            min=4,
            max=30,
            step=1,
            layer="engine",
            help="扩散步数：越高越稳、越慢；官方建议 4–30。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        ParamSpec(
            name="denoise",
            label="参考音频降噪",
            kind="bool",
            exposed=False,
            not_exposed_reason="no-op",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="上游参数真实存在，但本引擎离线安装形态 load_denoiser=False（不装 ModelScope 的 ZipEnhancer），denoise=True 被静默忽略——真机实测开与不开输出逐字节相同（issue #50，2026-09-22），故按 no-op 数据声明。",
        ),
        ParamSpec(
            name="min_len",
            label="最短音频长度",
            kind="number",
            integer=True,
            default=2,
            min=1,
            max=60,
            step=1,
            layer="engine",
            help="避免生成过短音频的下限（上游默认 2，token 口径）；一般保持默认即可。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        ParamSpec(
            name="max_len",
            label="单段生成长度上限",
            kind="number",
            integer=True,
            default=4096,
            min=100,
            max=4096,
            step=1,
            layer="engine",
            help="单次生成的 token 上限（上游默认 4096）。坏例重试开启时实际上限为 min(文本长度×阈值+10, 本值)。与本应用 200 字/请求的分段上限相互独立。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        ParamSpec(
            name="retry_badcase",
            label="坏例自动重试",
            kind="bool",
            default=True,
            layer="engine",
            help="生成结果与文本长度比例失衡时自动重试（上游 core 层默认 True；注意上游 model 层默认 False，实际生效以 core 传参为准，已真机验证）。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        ParamSpec(
            name="retry_badcase_max_times",
            label="坏例重试次数上限",
            kind="number",
            integer=True,
            default=3,
            min=1,
            max=10,
            step=1,
            layer="engine",
            help="坏例重试的最大尝试次数（上游默认 3）。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        ParamSpec(
            name="retry_badcase_ratio_threshold",
            label="坏例判定阈值",
            kind="number",
            default=6.0,
            min=2.0,
            max=12.0,
            step=0.5,
            layer="engine",
            help="音频/文本长度比例超过该阈值视为坏例并触发重试（上游默认 6.0）。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        ParamSpec(
            name="control_instruction",
            label="控制指令",
            kind="textarea",
            default="",
            layer="engine",
            help="控制指令（VoxCPM 官方语法，英文圆括号前缀）：写在待合成文本开头的自然语言声音描述，用于音色设计与克隆风格控制；建议英文描述（如 warm female voice）；用于音色设计与克隆风格控制。在文本归一化之后拼接进正文开头（永不经归一化层，spec #54 / ADR-0008 时序），克隆与设计音色通用。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        ParamSpec(
            name="seed",
            label="随机种子",
            kind="number",
            integer=True,
            min=0,
            max=2**31 - 1,
            layer="engine",
            help="留空则不固定种子；相同种子 + 相同参数可复现结果（worker 侧 torch.manual_seed + 加载预热实现——上游 generate() 无 seed 参数，见 issue #49；MPS/CUDA 真机均已逐字节验证）。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        # 「不支持是数据」（ADR-0018 decision 3）。每一条都对应一个真实存在
        # 的上游参数，逐条给出不暴露的原因。
        ParamSpec(
            name="normalize",
            label="引擎内文本归一化",
            kind="bool",
            exposed=False,
            not_exposed_reason="breaks-pipeline",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="归一化由本仓库自建归一化层在进引擎前完成（ADR-0008）；引擎内 TN 与其职责重复，且音素输入语法 {ni3} 要求此项为 False。",
        ),
        ParamSpec(
            name="denoise_output",
            label="输出降噪",
            kind="text",
            exposed=False,
            not_exposed_reason="wrong-mode",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="上游 denoise 只作用于参考音频，不作用于输出——不存在「输出降噪」。",
        ),
        ParamSpec(
            name="ref_audio",
            label="参考音频",
            kind="text",
            exposed=False,
            not_exposed_reason="server-injected",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="由音色解析自动注入，不作为用户参数。",
        ),
        ParamSpec(
            name="ref_text",
            label="参考文本",
            kind="text",
            exposed=False,
            not_exposed_reason="server-injected",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="可选（三种复刻模式）：有转写走 Hi-Fi 模式，无转写走纯音色模式；由参考音频转写自动填充。",
        ),
    ]


def prepare_synthesis(text: str, params: dict, log) -> tuple[str, dict]:
    """Canonical -> wire adaptation: pronunciation rewriting + params.

    The user-visible values never reach the engine unconverted (ADR-0018).

    Fixed ordering (spec #54 / CONTEXT.md「控制指令」「发音标注」): the text
    arriving here is ALREADY normalized (the pipeline applies the ADR-0008
    layer centrally before any engine adapter runs). Pronunciation rewriting
    and the control-instruction prefix are appended HERE, after
    normalization — a control instruction containing digits or English
    descriptions therefore never passes through the normalization layer and
    reaches the engine verbatim.
    """
    params = params or {}
    wire: dict = {}
    ann = params.get("pronunciation")
    if ann and str(ann).strip():
        # {ni3} phoneme syntax is only valid with normalize=False — the
        # adapter OWNS that invariant (upstream usage guide).
        text = pronunciation.rewrite(text, str(ann), "voxcpm", log)
    # Control instruction: official `(instruction)text` parenthesized prefix,
    # concatenated AFTER normalization (see docstring). Empty value is never
    # forwarded — neither into the text nor onto the wire.
    control = str(params.get("control_instruction") or "").strip()
    if control:
        text = f"({control}){text}"
    for key in (
        "cfg_value",
        "inference_timesteps",
        "min_len",
        "max_len",
        "retry_badcase",
        "retry_badcase_max_times",
        "retry_badcase_ratio_threshold",
    ):
        if params.get(key) not in (None, ""):
            wire[key] = params[key]
    # seed is handled worker-side (torch.manual_seed) — upstream generate()
    # has no seed parameter (issue #49); the worker forwards it separately.
    if params.get("seed") not in (None, ""):
        wire["seed"] = params["seed"]
    # denoise is declared-but-not-exposed (no denoiser installed); it is
    # never forwarded — the engine receives only verified parameters.
    return text, wire


class VoxCPM2EngineBase(InstallableEngine):

    # This engine keeps a trained model resident in memory between
    # generations (see the unload()/worker supervisor); run_generation
    # evicts other resident-worker engines before this one loads so
    # multi-local sessions never hold several models at once.
    resident_worker = True
    """Everything the MPS and CUDA VoxCPM2 variants share verbatim.

    Subclasses declare the platform deltas (torch source + worker gate via
    ``worker_gate``) and implement exactly one install hook
    (``_step_torch``) — nothing else.
    """

    engine_id: str = ""
    display_name: str = ""
    worker_filename = "voxcpm2_worker.py"
    worker_gate: str = ""  # "mps" | "cuda" — argv[1] of the worker script
    gate_label: str = ""

    def __init__(self, output_dir: Path | None = None, root: Path | None = None,
                 env: dict | None = None,
                 idle_timeout_s: float = IDLE_TIMEOUT_S,
                 max_requests: int = MAX_REQUESTS_PER_WORKER,
                 config: EngineConfig | None = None) -> None:
        output_dir, env = resolve_seam(config, self.engine_id, output_dir, env)
        self.output_dir = Path(output_dir) if output_dir else None
        self.env = env
        self.root = root if root is not None else paths.runtime_root(self.env)
        self.idle_timeout_s = idle_timeout_s
        self.max_requests = max_requests
        self._supervisor: WorkerSupervisor | None = None
        self._supervisor_lock = threading.Lock()

    # -- capabilities ------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            # Official 30-language list (issue #56; the `language` field of
            # the HF model card openbmb/VoxCPM2 — declarative support per
            # the vendor's own claim; real-machine spot checks are tracked
            # by the verification ticket. No language selector: the model
            # follows the text itself, there is NO language argument).
            languages=(
                "zh", "en", "ar", "my", "da", "nl", "fi", "fr", "de", "el",
                "he", "hi", "id", "it", "ja", "km", "ko", "lo", "ms", "no",
                "pl", "pt", "ru", "es", "sw", "sv", "tl", "th", "tr", "vi",
            ),
            voice_cloning=True,
            # Native Voice Design (issue #57): description + preview text,
            # NO reference audio — official `(description)text` syntax.
            # Declared here so both platform variants share it.
            voice_design=True,
            pronunciation_control=True,  # {ni3} phoneme syntax (verified)
            emotion=False,
            commercial_license=True,  # Apache-2.0 code + weights
            cross_device_use=False,
            upload_used_for_training=False,
            api_closed_loop=False,
            # Reference transcript is OPTIONAL: mode ① (reference-only,
            # timbre) needs none; modes ②③ use it when available.
            requires_reference_text=False,
            max_chars_per_request=MAX_CHARS_PER_REQUEST,
        )

    def param_specs(self) -> list[ParamSpec]:
        return param_specs_for(self.engine_id)

    # -- install surface ----------------------------------------------------

    def _ctx(self) -> installer.InstallContext:
        return installer.InstallContext(engine_id=self.engine_id, root=self.root, env=self.env)

    def is_installed(self) -> bool:
        return bool(installer.load_state(self._ctx()).get("installed"))

    def install_state(self) -> dict:
        state = installer.load_state(self._ctx())
        return {"installed": bool(state.get("installed")), "steps": state.get("steps", {})}

    def install_steps(self) -> list[installer.InstallStep]:
        from ..runtime import downloader

        weights_dir = paths.engine_weights_dir(self.root, self.engine_id)
        venv = uvman.engine_venv_dir(self.root, self.engine_id)

        def find_uv(log):
            return uvman.find_uv(self.root, env=self.env, log=log)

        def step_python(log, progress):
            uvman.ensure_python(find_uv(log), PYTHON_SPEC, env=self.env, log=log)

        def step_venv(log, progress):
            uvman.create_venv(find_uv(log), self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)

        def step_torch(log, progress):
            uv = find_uv(log)
            venv_ = uvman.create_venv(uv, self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)
            self._step_torch(uv, venv_, log, progress)

        def step_engine(log, progress):
            uv = find_uv(log)
            venv_ = uvman.create_venv(uv, self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)
            # Pure-python wheel; pulls transformers/einops/librosa/etc. Its
            # optional heavies are NOT in the inference path (verified):
            # gradio (demo app only), funasr (absent from src/), datasets
            # (training only) — but pip installs what the wheel declares.
            uvman.pip_install(uv, venv_, [f"voxcpm=={VOXCPM_VERSION}"], env=self.env, log=log)

        def step_weights(log, progress):
            chain = sources.weight_sources(REPO, ms_repo=MS_REPO, preferred=sources.weight_pref(self.env))
            for rel in WEIGHTS_FILES:
                log(f"weights: downloading {rel}")
                downloader.download_file(
                    downloader.DownloadSpec(path=rel, dest_name=rel),
                    weights_dir, chain,
                    progress=lambda name, done, total, _n=rel: progress(f"weights:{_n}", done, total),
                    log=log,
                )
            log(f"weights: {len(WEIGHTS_FILES)} files ready in {weights_dir}")

        return [
            installer.InstallStep("python", f"安装 uv 托管的 Python {PYTHON_SPEC}", step_python),
            installer.InstallStep("venv", "创建引擎独立 venv", step_venv, artifact=venv),
            installer.InstallStep("torch", self.torch_step_label, step_torch, artifact=venv),
            installer.InstallStep("engine", f"安装 voxcpm {VOXCPM_VERSION} 引擎代码", step_engine, artifact=venv),
            installer.InstallStep("weights", self.weights_step_label, step_weights),
        ]

    def _step_torch(self, uv, venv: Path, log, progress) -> None:
        raise NotImplementedError

    def install(self, log, progress=None) -> dict:
        return installer.run_install(self._ctx(), self.install_steps(), log, progress)

    # -- worker management ---------------------------------------------------

    def _get_supervisor(self) -> WorkerSupervisor:
        with self._supervisor_lock:
            if self._supervisor is None:
                venv = paths.engine_venv_dir(self.root, self.engine_id)
                python = uvman.venv_python(venv)
                if not python.exists():
                    raise RuntimeError(
                        f"engine venv is broken (no python at {python}); reinstall the engine"
                    )
                worker = Path(__file__).with_name(self.worker_filename)
                self._supervisor = WorkerSupervisor(
                    command=[str(python), "-u", str(worker), self.worker_gate],
                    cwd=str(self.root),
                    idle_timeout_s=self.idle_timeout_s,
                    max_requests=self.max_requests,
                )
            return self._supervisor

    def unload(self) -> None:
        with self._supervisor_lock:
            if self._supervisor is not None:
                self._supervisor.unload()

    def worker_stats(self) -> dict:
        with self._supervisor_lock:
            return self._supervisor.stats() if self._supervisor else {"alive": False}

    # -- generation ---------------------------------------------------------

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        if not self.is_installed():
            raise RuntimeError(
                f"engine is not installed yet — call POST /engines/{self.engine_id}/install first"
            )
        params = request.params or {}
        if not params.get("ref_audio"):
            raise RuntimeError(
                "VoxCPM2 是零样本复刻引擎，必须提供参考音频：请选择一个音色（或先创建音色）再生成"
            )
        out_dir = (self.output_dir or Path.cwd() / "data" / "audio").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / f"{request.generation_id or uuid.uuid4().hex}.wav").resolve()

        text, wire = prepare_synthesis(request.text, params, log)
        ref_text = (params.get("ref_text") or "").strip()
        payload: dict = {
            "action": "synthesize",
            "model_dir": str(paths.engine_weights_dir(self.root, self.engine_id)),
            "text": text,
            "ref_audio": str(Path(params["ref_audio"]).resolve()),
            "output": str(out_path),
            **wire,
        }
        if ref_text:
            # Mode ③: Hi-Fi continuation (prompt pair) + reference tokens —
            # the vendor's max-similarity combination. Upstream enforces the
            # prompt_wav/prompt_text pairing, so the transcript is passed
            # only when present, and then BOTH are set.
            payload["prompt_text"] = ref_text
            log("voxcpm2: 复刻模式 = Hi-Fi（参考音频 + 参考文本，最高相似度）")
        else:
            # Mode ①: reference tokens only — timbre cloning, no transcript.
            log("voxcpm2: 复刻模式 = 纯音色（无参考文本；提供转写可获得更高相似度）")

        supervisor = self._get_supervisor()
        try:
            result = supervisor.request(payload, on_log=log)
        except WorkerDegradedError as exc:
            raise RuntimeError(f"{self.gate_label} gate refused to start the engine: {exc}") from exc
        if result.get("clipping"):
            log("voxcpm2: output hit full scale — consider lowering gain on the reference audio")
        return GenerationResult(
            audio_path=result["audio_path"],
            sample_rate=result["sample_rate"],
            model_version=f"{REPO} (voxcpm {VOXCPM_VERSION})",
            cost=0.0,
        )

    # -- voice design (issue #57; ADR-0010) ---------------------------------

    def design_voice(self, description: str, preview_text: str, log) -> dict:
        """Native Voice Design: create a voice from a text description.

        Official syntax reuses the control-instruction prefix — the design
        description rides as an English-parenthesis prefix before the
        preview text, exactly the shape ``prepare_synthesis`` builds from
        ``control_instruction``. Unlike normal synthesis the design call
        NEVER goes through the pipeline: no normalization, no pronunciation
        rewriting — the description reaches the engine verbatim (spec #54
        user story 10 / CONTEXT.md「控制指令」).

        The generated preview sample becomes the designed voice's reference
        (ADR-0010「预览样本即参考音频」): the router feeds it through the
        existing ``attach_reference`` and every later synthesis of that
        voice runs the normal reference-cloning path above — the pipeline
        stays untouched. Returns ``{"voice_id", "sample_audio_path",
        "transcript"}`` with a locally minted voice id.
        """
        if not self.is_installed():
            raise RuntimeError(
                f"engine is not installed yet — call POST /engines/{self.engine_id}/install first"
            )
        description = (description or "").strip()
        preview_text = (preview_text or "").strip()
        if not description:
            raise RuntimeError("声音描述不能为空")
        if not preview_text:
            raise RuntimeError("试听文本不能为空")
        out_dir = (self.output_dir or Path.cwd() / "data" / "audio").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / f"design-{uuid.uuid4().hex}.wav").resolve()
        payload: dict = {
            "action": "synthesize",
            # The ONLY no-reference entry: ``design`` routes the worker to
            # the reference-less generate() call; plain synthesize never
            # sets it and keeps enforcing ref_audio.
            "design": True,
            "model_dir": str(paths.engine_weights_dir(self.root, self.engine_id)),
            "text": f"({description}){preview_text}",
            "output": str(out_path),
        }
        log("voxcpm2: 音色设计 = 无参考生成（控制指令前缀 + 试听文本）")
        supervisor = self._get_supervisor()
        try:
            result = supervisor.request(payload, on_log=log)
        except WorkerDegradedError as exc:
            raise RuntimeError(f"{self.gate_label} gate refused to start the engine: {exc}") from exc
        return {
            "voice_id": f"voxcpm2-design-{uuid.uuid4().hex[:12]}",
            "sample_audio_path": result["audio_path"],
            "transcript": preview_text,
        }
