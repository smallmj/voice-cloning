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

from .. import params as params_mod
from .. import pronunciation, sources
from ..capabilities import AppliesTo, Capabilities, ObjectField, ParamSpec
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

# -- 情感控制四模式（issue #37：对齐官方 webui）----------------------------
# The SELECT CHOICES are the official webui wordings (what the user picks);
# the wire keys are the stable mode identifiers the synthesize adapter gates
# on. The ignored_when predicates compare against the raw choice string —
# that is the single value both the UI and the engine adapter see.
EMO_MODE_SAME = "与参考音频相同"
EMO_MODE_REF_AUDIO = "情感参考音频"
EMO_MODE_VECTOR = "情感向量"
EMO_MODE_TEXT = "情感描述文本"
EMO_MODE_CHOICES = (EMO_MODE_SAME, EMO_MODE_REF_AUDIO, EMO_MODE_VECTOR, EMO_MODE_TEXT)

EMO_MODE_WIRE_KEYS = {
    EMO_MODE_SAME: "same-as-reference",
    EMO_MODE_REF_AUDIO: "reference-audio",
    EMO_MODE_VECTOR: "vector",
    EMO_MODE_TEXT: "text",
}

# 8 维情感向量：固定顺序（官方一致），每维 0–1（ADR-0018 decision 2 的
# 对象数组形态 —— 顺序敏感，故用固定 items 而不是自由输入）。
EMO_VECTOR_ITEMS = tuple(
    ObjectField(name=name, label=label, kind="number", min=0.0, max=1.0)
    for name, label in (
        ("happy", "高兴"),
        ("angry", "愤怒"),
        ("sad", "悲伤"),
        ("afraid", "恐惧"),
        ("disgusted", "厌恶"),
        ("melancholic", "低落"),
        ("surprised", "惊喜"),
        ("calm", "平静"),
    )
)


def emo_mode_to_wire(value):
    """User-facing mode wording -> stable wire key. Unknown values fall back
    to the default mode (same-as-reference), never an invented one."""
    if value in EMO_MODE_WIRE_KEYS:
        return EMO_MODE_WIRE_KEYS[value]
    if value in set(EMO_MODE_WIRE_KEYS.values()):  # already a wire key (rerun)
        return value
    return "same-as-reference"


def emo_vector_to_wire(value):
    """Comma-joined 8-dim string -> list[float] of length 8 (0–1 clamped).

    Empty/invalid input returns None so the engine's own default applies —
    never a padded zeros vector that would silently change the sound.
    """
    if value is None or value == "":
        return None
    parts = value if isinstance(value, (list, tuple)) else str(value).split(",")
    if len(parts) != len(EMO_VECTOR_ITEMS):
        return None
    out = []
    for part in parts:
        try:
            out.append(max(0.0, min(1.0, float(part))))
        except (TypeError, ValueError):
            return None
    return out


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
        # -- 情感控制（issue #37：对齐官方 webui 四模式）-------------------
        # The four official emotion modes. The user sees the official webui
        # wording; to_wire maps it onto a stable mode key that the synthesize
        # adapter uses to gate the emotion wire parameters (ignored_when is
        # enforced BOTH in the UI display and again here, engine-side).
        ParamSpec(
            name="emo_mode",
            label="情感控制",
            kind="select",
            default=EMO_MODE_SAME,
            choices=EMO_MODE_CHOICES,
            layer="canonical",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            to_wire=emo_mode_to_wire,
            help="与参考音频相同 / 使用情感参考音频 / 情感向量 / 情感描述文本（官方 webui 四模式）。",
        ),
        # 情感权重 0–1，官方 webui 默认 0.65。仅在「情感参考音频」与「情感向量」
        # （及情感文本推导出的向量）下生效；与参考音频相同时上游强制 1.0。
        ParamSpec(
            name="emo_weight",
            label="情感权重",
            kind="number",
            default=0.65,
            min=0.0,
            max=1.0,
            step=0.05,
            layer="canonical",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            to_wire=lambda v: None if v in (None, "") else float(v),
            ignored_when=(
                "emo_mode!=" + EMO_MODE_REF_AUDIO,
                "emo_mode!=" + EMO_MODE_VECTOR,
                "emo_mode!=" + EMO_MODE_TEXT,
            ),
            help="0–1，官方默认 0.65：情感参考/向量对结果的加权强度。",
        ),
        # 情感参考音频：新增的生成期上传输入（issue #37）。前端经
        # POST /uploads/audio 上传后把返回的文件引用填进该参数；管线把它
        # 解析为音频目录内受限的绝对路径。引擎收到的最终键是 emo_audio_prompt。
        ParamSpec(
            name="emo_audio",
            label="情感参考音频",
            kind="audio",
            layer="canonical",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            to_wire=lambda v: None if v in (None, "") else str(v),
            ignored_when=("emo_mode!=" + EMO_MODE_REF_AUDIO,),
            help="仅「情感参考音频」模式下使用：上传一段代表目标情绪的音频。",
        ),
        # 情感向量：8 维固定顺序滑条（ADR-0018 decision 2 的对象数组形态）。
        # 上游 normalize_emo_vec 后按 emo_alpha 缩放（issue #37 听感验证通过，
        # 故由 unverified 翻转为 exposed）。
        ParamSpec(
            name="emo_vector",
            label="8 维情感向量",
            kind="array",
            items=EMO_VECTOR_ITEMS,
            max_items=8,
            layer="canonical",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            ignored_when=("emo_mode!=" + EMO_MODE_VECTOR,),
            to_wire=emo_vector_to_wire,
            help="0–1 滑条，顺序与官方一致：[高兴,愤怒,悲伤,恐惧,厌恶,低落,惊喜,平静]。",
        ),
        # 情感描述文本 + 随机采样（实验）：需要 QwenEmotion（use_qwen_emo=True，
        # 两个 worker 构造期均已开启）。
        ParamSpec(
            name="emo_text",
            label="情感描述文本",
            kind="textarea",
            max_length=200,
            layer="canonical",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            to_wire=lambda v: None if v in (None, "") else str(v),
            ignored_when=("emo_mode!=" + EMO_MODE_TEXT,),
            help="实验功能：用自然语言描述情绪，由 QwenEmotion 推导情感向量。",
        ),
        ParamSpec(
            name="emo_random",
            label="情感随机采样（实验）",
            kind="bool",
            default=False,
            layer="canonical",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            to_wire=lambda v: str(v).strip().lower() == "true" if v not in (None, "") else None,
            ignored_when=("emo_mode!=" + EMO_MODE_TEXT,),
            help="实验功能：在文本推导的情感向量上加入随机采样。",
        ),
        # -- 引擎专属层：采样参数（官方 webui 默认值，issue #37）-----------
        ParamSpec(
            name="temperature",
            label="温度",
            kind="number",
            default=0.8,
            min=0.1,
            max=2.0,
            step=0.05,
            layer="engine",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="官方默认 0.8。",
        ),
        ParamSpec(
            name="top_p",
            label="Top P",
            kind="number",
            default=0.8,
            min=0.0,
            max=1.0,
            step=0.05,
            layer="engine",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="官方默认 0.8。",
        ),
        ParamSpec(
            name="top_k",
            label="Top K",
            kind="number",
            integer=True,
            default=30,
            min=0,
            max=100,
            step=1,
            layer="engine",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="官方默认 30。",
        ),
        ParamSpec(
            name="num_beams",
            label="束搜索宽度",
            kind="number",
            integer=True,
            default=3,
            min=1,
            max=10,
            step=1,
            layer="engine",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="官方默认 3。",
        ),
        ParamSpec(
            name="repetition_penalty",
            label="重复惩罚",
            kind="number",
            default=10.0,
            min=1.0,
            max=20.0,
            step=0.1,
            layer="engine",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="官方默认 10.0（注意与其他引擎常用 1.05 量级不同）。",
        ),
        ParamSpec(
            name="length_penalty",
            label="长度惩罚",
            kind="number",
            default=0.0,
            min=-2.0,
            max=2.0,
            step=0.1,
            layer="engine",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="官方默认 0.0。",
        ),
        ParamSpec(
            name="max_mel_tokens",
            label="最大 Mel Token 数",
            kind="number",
            integer=True,
            default=1500,
            min=100,
            max=6000,
            step=50,
            layer="engine",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="官方默认 1500；超出会截断并告警。",
        ),
        ParamSpec(
            name="do_sample",
            label="随机采样",
            kind="bool",
            default=True,
            layer="engine",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="关闭后走贪心解码（temperature/top_p/top_k 不生效）。",
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


def _num(params: dict, name: str, lo: float, hi: float, integer: bool = False):
    """Read a numeric param, clamped into [lo, hi]. None when unset/invalid."""
    raw = params.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    value = max(lo, min(hi, value))
    return round(value) if integer else value


def _bool(params: dict, name: str) -> bool | None:
    raw = params.get(name)
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() == "true"


def prepare_synthesis(text: str, params: dict, log) -> tuple[str, dict]:
    """Canonical -> wire adaptation shared by both IndexTTS adapters.

    Returns the (possibly pronunciation-rewritten) text and the extra worker
    payload keys. This is the per-engine adapter the canonical layer
    requires: user-visible values never reach the engine unconverted.

    情感模式联动（issue #37）：ignored_when 在界面上隐藏不适用的参数，但
    门控的最终裁决在这里 —— 即便前端漏发或重跑带入了陈旧参数，错误模式下
    的情感参数也到不了引擎。
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

    # -- 情感控制（官方 webui 四模式）------------------------------------
    mode = emo_mode_to_wire(params.get("emo_mode") or EMO_MODE_SAME)
    if params.get("emo_mode") not in (None, "") and \
            params.get("emo_mode") not in EMO_MODE_WIRE_KEYS and \
            params.get("emo_mode") not in EMO_MODE_CHOICES:
        log(f"参数：未知情感模式 {params.get('emo_mode')!r}，按「与参考音频相同」处理")
    emo_weight = _num(params, "emo_weight", 0.0, 1.0)
    if emo_weight is None:
        # The official webui default rides along whenever a mode uses alpha
        # mixing (the UI only sends explicitly touched values).
        emo_weight = 0.65

    if mode == "reference-audio":
        emo_audio = params.get("emo_audio")
        if not emo_audio:
            raise ValueError(
                "情感模式为「情感参考音频」但未提供情感参考音频；请上传或切换情感模式"
            )
        wire["emo_audio_prompt"] = str(emo_audio)
        if emo_weight is not None:
            wire["emo_alpha"] = emo_weight
    elif mode == "vector":
        vector = emo_vector_to_wire(params.get("emo_vector"))
        if vector is None:
            raise ValueError("情感模式为「情感向量」但情感向量无效；请设置 8 个 0–1 的滑条")
        wire["emo_vector"] = vector
        if emo_weight is not None:
            wire["emo_alpha"] = emo_weight
    elif mode == "text":
        wire["use_emo_text"] = True
        emo_text = params.get("emo_text")
        if emo_text and str(emo_text).strip():
            wire["emo_text"] = str(emo_text).strip()
        emo_random = _bool(params, "emo_random")
        if emo_random is not None:
            wire["use_random"] = emo_random
        if emo_weight is not None:
            wire["emo_alpha"] = emo_weight
    # mode == "same-as-reference": no emotion overrides at all — upstream
    # forces emo_alpha=1.0 and uses the speaker reference as emotion source.

    # -- 引擎专属层：采样参数（仅在用户显式给出时转发）-------------------
    num_wire = {
        "temperature": _num(params, "temperature", 0.1, 2.0),
        "top_p": _num(params, "top_p", 0.0, 1.0),
        "top_k": _num(params, "top_k", 0, 100, integer=True),
        "num_beams": _num(params, "num_beams", 1, 10, integer=True),
        "repetition_penalty": _num(params, "repetition_penalty", 1.0, 20.0),
        "length_penalty": _num(params, "length_penalty", -2.0, 2.0),
        "max_mel_tokens": _num(params, "max_mel_tokens", 100, 6000, integer=True),
    }
    for key, value in num_wire.items():
        if value is not None:
            wire[key] = value
    do_sample = _bool(params, "do_sample")
    if do_sample is not None:
        wire["do_sample"] = do_sample

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

    # This engine keeps a trained model resident in memory between
    # generations (see the unload()/worker supervisor); run_generation
    # evicts other resident-worker engines before this one loads so
    # multi-local sessions never hold several models at once.
    resident_worker = True
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
