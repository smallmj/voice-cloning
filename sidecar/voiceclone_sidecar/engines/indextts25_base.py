"""Shared IndexTTS-2.5 engine base (issue #32).

The CUDA (Windows) and MPS (Apple Silicon) variants used to be line-for-line
copies of each other: same capability set, same two-step dependency install,
same weight source chain, same worker supervisor wiring, same synthesize
flow. This module holds the shared implementation; each variant declares only
its platform deltas as class attributes plus one ``_step_torch`` hook:

- torch source: explicit CUDA wheel URLs vs pinned PyPI versions;
- device gate + worker script: which worker file supervises generation and
  how its degraded error is phrased;
- local weight reuse: the MPS install can seed weights from an existing
  checkpoint tree (``local_weights_env``); CUDA always downloads.

Everything else — step ids and their order, the engine-package mirror
fallback, the weight source list with loud mirror failure, and the
worker-management / generation surface — lives here once.
"""

from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

from .. import pronunciation, sources
from .. import params as params_mod
from ..capabilities import AppliesTo, Capabilities, ParamSpec
from ..engine_config import EngineConfig, resolve_seam
from ..registry import GenerationRequest, GenerationResult, InstallableEngine
from ..runtime import downloader, installer, paths, uvman
from .worker_supervisor import WorkerDegradedError, WorkerSupervisor

REPO = "IndexTeam/IndexTTS-2.5"
PYTHON_SPEC = "3.11"  # index-tts requires >=3.10,<3.12
TORCH_VERSION = "2.8.0"  # pinned: <2.9 avoids TorchAudio's TorchCodec WAV path (ADR-0002 rule 1)

# The engine CODE package (a GitHub zip, not one of the three download-source
# axes) is engine-specific but shared by both platforms, declared with its own
# mirror pair and the VOICECLONE_INDEXTTS_URL(_FALLBACK) env overrides.
ENGINE_PACKAGE_URL = (
    "https://ghfast.top/https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
)
ENGINE_PACKAGE_URL_FALLBACK = "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"

# Downloader source templates interpolate {path}; these carry {repo}/{ms_repo}
# too, so each repo passes its own identifiers alongside the file path.
MAIN_WEIGHTS_FILES = [
    "config.yaml",
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "feat1.pt",
    "feat2.pt",
    "wav2vec2bert_stats.pt",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "qwen0.6bemo4-merge/model.safetensors",
    "qwen0.6bemo4-merge/config.json",
    "qwen0.6bemo4-merge/tokenizer.json",
    "qwen0.6bemo4-merge/tokenizer_config.json",
    "qwen0.6bemo4-merge/vocab.json",
    "qwen0.6bemo4-merge/merges.txt",
    "qwen0.6bemo4-merge/added_tokens.json",
    "qwen0.6bemo4-merge/special_tokens_map.json",
    "qwen0.6bemo4-merge/chat_template.jinja",
    "qwen0.6bemo4-merge/generation_config.json",
]

# repo_id on HF, {path} inside that repo, dest name under weights/hf_cache,
# ModelScope model id (None = no ModelScope source).
AUX_WEIGHTS = [
    ("facebook/w2v-bert-2.0", "config.json", "hf_cache/w2v-bert-2.0/config.json", "AI-ModelScope/w2v-bert-2.0"),
    ("facebook/w2v-bert-2.0", "preprocessor_config.json", "hf_cache/w2v-bert-2.0/preprocessor_config.json", "AI-ModelScope/w2v-bert-2.0"),
    ("facebook/w2v-bert-2.0", "model.safetensors", "hf_cache/w2v-bert-2.0/model.safetensors", "AI-ModelScope/w2v-bert-2.0"),
    ("amphion/MaskGCT", "semantic_codec/model.safetensors", "hf_cache/semantic_codec_model.safetensors", "amphion/MaskGCT"),
    ("funasr/campplus", "campplus_cn_common.bin", "hf_cache/campplus_cn_common.bin", "iic/speech_campplus_sv_zh-cn_16k-common"),
    # ModelScope twin verified 2026-09: nv-community/bigvgan_v2_22khz_80band_256x
    # serves both files (resolve -> 200). Closes the last bigvgan coverage
    # hole ADR-0016 decision 4 mandated; CUDA and MPS share this list.
    ("nvidia/bigvgan_v2_22khz_80band_256x", "config.json", "hf_cache/bigvgan/config.json", "nv-community/bigvgan_v2_22khz_80band_256x"),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "bigvgan_generator.pt", "hf_cache/bigvgan/bigvgan_generator.pt", "nv-community/bigvgan_v2_22khz_80band_256x"),
]

IDLE_TIMEOUT_S = 300.0  # release GPU/unified memory after five idle minutes
MAX_REQUESTS_PER_WORKER = 20  # recycle the worker to bound slow memory leaks

LANG_CHOICES = ("zh", "en", "ja", "es", "ar")


def param_specs_for(engine_id: str) -> list[ParamSpec]:
    """Shared parameter declaration for both IndexTTS adapters (issue #23).

    They run the same model — only the torch device differs — so the specs,
    including the pronunciation entry the capability flag always promised
    but the UI never had, are declared once and shared.
    """
    return [
        # Canonical speed: user sees a rate (1.0 = normal); the adapter maps
        # it onto the engine's INVERSE duration_factor so "更快" can never
        # mean slower (ADR-0018 decision 1).
        params_mod.canonical_speed(
            engine_id, REPO,
            help="1.0 为原速；>1 更快，<1 更慢。IndexTTS 的原生参数是时长倍率（方向相反），已自动换算。",
        ),
        params_mod.canonical_language(
            engine_id, REPO, LANG_CHOICES, wire_name="lang", default="zh",
        ),
        # The pronunciation entry: capability declared True, model supports
        # it, UI had no entry — the exact gap ADR-0018 closes.
        params_mod.canonical_pronunciation(
            engine_id, REPO,
            grammar_help=(
                "每行一条「汉字=拼音」（小写声调数字，如 行=xing2），将改写为 "
                "IndexTTS 的 <行|XING2> 语法后送入引擎。"
            ),
        ),
        # "不支持是数据"（ADR-0018 decision 3）：上游另有 8 维情感向量 /
        # 情感文本等参数，但需要构造期 QwenEmotion 开关，未在锁定的引擎
        # 版本上验证 —— 暴露即撒谎，先以数据声明不暴露。
        ParamSpec(
            name="emo_vector",
            label="8 维情感向量",
            kind="text",
            exposed=False,
            not_exposed_reason="unverified",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="上游支持（[高兴,愤怒,悲伤,害怕,厌恶,忧郁,惊讶,平静]），但需构造期情感模型开关，本版本未验证，暂不暴露。",
        ),
        # The sidecar injects reference audio itself (voice resolution);
        # declaring it keeps the wire contract honest instead of invisible.
        ParamSpec(
            name="ref_audio",
            label="参考音频",
            kind="text",
            exposed=False,
            not_exposed_reason="server-injected",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="由音色解析自动注入，不作为用户参数。",
        ),
    ]


def prepare_synthesis(text: str, params: dict, log) -> tuple[str, dict]:
    """Canonical -> wire adaptation shared by both IndexTTS adapters.

    Returns the (possibly pronunciation-rewritten) text and the extra worker
    payload keys. This is the per-engine adapter the canonical layer
    requires: user-visible values never reach the engine unconverted.
    """
    params = params or {}
    wire: dict = {}
    speed = params.get("speed")
    if speed not in (None, ""):
        df = params_mod.speed_to_duration_factor(speed)
        if df is not None:
            wire["duration_factor"] = df
    lang = params.get("language") or params.get("lang")
    if lang in LANG_CHOICES:
        wire["lang"] = lang
    else:
        # Not silent: an unknown language value is dropped to the engine's
        # default WITH a log line — never silently substituted (ADR-0018
        # bans rewriting what the engine receives without saying so).
        if lang not in (None, ""):
            log(f"参数：未知语种 {lang!r}，已忽略（使用引擎默认语种 zh）")
        wire["lang"] = "zh"
    ann = params.get("pronunciation")
    if ann and str(ann).strip():
        text = pronunciation.rewrite(text, str(ann), "indextts", log)
    return text, wire


def copy_from_local(local_root: Path, relative: str, dest: Path, log) -> bool:
    """Copy one weight file from a local checkpoint tree. True if done.

    Existing complete destination files are left alone (skip-if-complete,
    same contract as the downloader); a zero-byte local file is treated as
    absent rather than copied.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True
    src = local_root / relative
    if not src.is_file() or src.stat().st_size == 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"weights: reusing local {relative}")
    shutil.copy2(src, dest)
    return True


class IndexTts25EngineBase(InstallableEngine):
    """Everything the CUDA and MPS IndexTTS variants share verbatim.

    Subclasses declare the platform deltas as class attributes and implement
    exactly one install hook (``_step_torch``) — nothing else.
    """

    engine_id: str = ""
    display_name: str = ""
    worker_filename: str = ""  # basename of the worker script in this package
    gate_label: str = ""  # "CUDA" / "MPS", used in the degraded-start error
    torch_step_label: str = ""
    weights_step_label: str = ""
    aux_step_label: str = ""
    # Env var naming an existing checkpoint tree to reuse weights from
    # (None = always download, e.g. the CUDA variant).
    local_weights_env: str | None = None

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
        # Both variants run the SAME model; the capability set is identical.
        return Capabilities(
            languages=("zh", "en", "ja", "es", "ar"),
            voice_cloning=True,
            voice_design=False,
            pronunciation_control=True,  # pinyin / CMU phonemes
            emotion=True,
            commercial_license=False,  # bilibili model license
            cross_device_use=False,
            upload_used_for_training=False,
            api_closed_loop=False,
            requires_reference_text=True,  # the sidecar auto-fills ref_text from the transcript
            max_chars_per_request=500,  # device memory ceiling; keep requests small (issue #13)
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

    def _local_weights_root(self) -> Path | None:
        if not self.local_weights_env:
            return None
        override = self.env.get(self.local_weights_env, "")
        return Path(override).expanduser() if override else None

    def _fetch_weight(self, relative: str, dest_name: str, sources_list: list[str],
                      weights_dir: Path, log, progress, label: str) -> None:
        local = self._local_weights_root()
        if local is not None:
            # Main weights sit at their repo-relative paths in a clone,
            # but aux models live under the pre-seeded hf_cache/ layout —
            # try the destination-relative path first, then repo-relative.
            for local_relative in dict.fromkeys((dest_name, relative)):
                if copy_from_local(local, local_relative, weights_dir / dest_name, log):
                    return
        log(f"{label}: downloading {relative}")
        downloader.download_file(
            downloader.DownloadSpec(path=relative, dest_name=dest_name),
            weights_dir, sources_list,
            progress=lambda name, done, total, _n=relative: progress(f"{label}:{_n}", done, total),
            log=log,
        )

    def _step_torch(self, uv, venv: Path, log, progress) -> None:
        """Platform delta: how torch/torchaudio land in the engine venv."""
        raise NotImplementedError

    def install_steps(self) -> list[installer.InstallStep]:
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
            pkg = self.env.get("VOICECLONE_INDEXTTS_URL", ENGINE_PACKAGE_URL)
            try:
                uvman.pip_install(uv, venv_, [f"indextts @ {pkg}"], env=self.env, log=log)
            except uvman.RuntimeCommandError:
                fallback = self.env.get("VOICECLONE_INDEXTTS_URL_FALLBACK", ENGINE_PACKAGE_URL_FALLBACK)
                log(f"engine package install from {pkg} failed; retrying from {fallback}")
                uvman.pip_install(uv, venv_, [f"indextts @ {fallback}"], env=self.env, log=log)

        def step_weights(log, progress):
            # ADR-0016: preferred weight source from the injected env.
            weight_chain = sources.weight_sources(
                REPO, ms_repo=REPO,
                preferred=sources.weight_pref(self.env),
            )
            for f in MAIN_WEIGHTS_FILES:
                self._fetch_weight(f, f, weight_chain, weights_dir, log, progress, "weights")
            log(f"weights: {len(MAIN_WEIGHTS_FILES)} files ready in {weights_dir}")

        def step_aux(log, progress):
            preferred = sources.weight_pref(self.env)
            for repo, path, dest, ms_repo in AUX_WEIGHTS:
                aux_sources = sources.weight_sources(repo, ms_repo=ms_repo, preferred=preferred)
                self._fetch_weight(path, dest, aux_sources, weights_dir, log, progress, "aux")
            log("aux: w2v-bert-2.0 / semantic codec / campplus / bigvgan ready")

        return [
            installer.InstallStep("python", f"安装 uv 托管的 Python {PYTHON_SPEC}", step_python),
            installer.InstallStep("venv", "创建引擎独立 venv", step_venv, artifact=venv),
            installer.InstallStep("torch", self.torch_step_label, step_torch, artifact=venv),
            installer.InstallStep("engine", "安装 index-tts 引擎代码", step_engine, artifact=venv),
            # No artifact wipe on the download steps: the downloader
            # guarantees per-file completeness (skip-if-complete + .part
            # resume), and wiping the whole weights dir on a retry would
            # re-download many GB that were already fine (observed) and
            # discard locally reused weights.
            installer.InstallStep("weights", self.weights_step_label, step_weights),
            installer.InstallStep("aux", self.aux_step_label, step_aux),
        ]

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
                    command=[str(python), "-u", str(worker)],
                    cwd=str(self.root),
                    idle_timeout_s=self.idle_timeout_s,
                    max_requests=self.max_requests,
                )
            return self._supervisor

    def unload(self) -> None:
        """Explicit memory release: drop the loaded model and exit the worker."""
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
            # Zero-shot cloning with no reference: upstream fails deep inside
            # soundfile with "Invalid file: None" — refuse here with an
            # actionable message instead.
            raise RuntimeError(
                "IndexTTS-2.5 是零样本复刻引擎，必须提供参考音频：请选择一个音色（或先创建音色）再生成"
            )
        out_dir = (self.output_dir or Path.cwd() / "data" / "audio").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / f"{request.generation_id or uuid.uuid4().hex}.wav").resolve()

        text, wire = prepare_synthesis(request.text, params, log)
        payload = {
            "action": "synthesize",
            "model_dir": str(paths.engine_weights_dir(self.root, self.engine_id)),
            "text": text,
            "output": str(out_path),
            **wire,
        }
        if params.get("ref_audio"):
            payload["ref_audio"] = params["ref_audio"]

        supervisor = self._get_supervisor()
        try:
            result = supervisor.request(payload, on_log=log)
        except WorkerDegradedError as exc:
            raise RuntimeError(f"{self.gate_label} gate refused to start the engine: {exc}") from exc
        if result.get("clipping"):
            log("indextts-2.5: output hit full scale — consider lowering gain on the reference audio")
        return GenerationResult(
            audio_path=result["audio_path"],
            sample_rate=result["sample_rate"],
            model_version=REPO,
            cost=0.0,  # local synthesis has no per-run cost
        )
