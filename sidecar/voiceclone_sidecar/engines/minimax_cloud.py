"""MiniMax speech-2.8 cloud engine — issue #27.

BYOK only (ADR-0003): the user's own MiniMax API key is read from the
sidecar's key store at call time and never persisted elsewhere. One engine
serves all four capabilities because MiniMax's design/cloned voices are
synthesized by the SAME speech-2.8 model (unlike DashScope where the vd
model differs): cloning, sync synthesis, system voices and voice design.

Vendor contract (verified 2026-09-18 against platform.minimax.io /
platform.minimaxi.com OpenAPI, evidence in
``docs/research/2026-09-minimax-api-and-local-tts.md``):

1. Cloning — two calls: ``POST /v1/files/upload`` (multipart,
   ``purpose=voice_clone``, mp3/m4a/wav, 10s–5min, ≤20MB) then
   ``POST /v1/voice_clone`` with an app-chosen ``voice_id`` (must start
   with a letter, [A-Za-z0-9-_], 8–256 chars, no trailing -/_).
   GroupId is NOT needed anywhere (zero hits across all six OpenAPIs).
2. Activation trap — a cloned voice is INACTIVE until a real T2A synthesis;
   the official FAQ: "Previewing during voice_clone does not activate the
   voice_id". Unactivated voices auto-expire after 7 days. So
   :meth:`bind_reference` silently synthesizes one short sentence right
   after cloning and records ``activated_at`` (the first real synthesis
   also triggers the $1.50/voice cloning charge — cloning itself is free).
   If activation fails, the binding extra carries ``status: "unactivated"``
   which the pipeline persists and the UI surfaces.
3. Health check — ``/v1/get_voice`` is POST and only lists voices that
   already synthesized at least once, so ``check_voice`` is implemented as
   a minimal probe synthesis (issue #12), never as a list query.
4. Synthesis — short text (≤ :data:`SYNC_CHAR_LIMIT`) goes to
   ``POST /v1/t2a_v2`` with ``audio_setting.format=wav`` and
   ``output_format=url`` (the default mp3 + hex would break the pipeline's
   WAV peak self-check; the url is valid 24h — plenty for an immediate
   download). Longer text goes to ``POST /v1/t2a_async_v2`` (single request
   up to 1,000,000 chars — this is how the mainland 20 RPM limit is
   bypassed) and its payload keys DIFFER from sync: ``audio_sample_rate``
   instead of ``sample_rate``, ``english_normalization`` instead of
   ``text_normalization``, and NO ``output_format`` / ``stream`` /
   ``force_cbr``. The two payloads are constructed separately on purpose.
   Task status is compared case-insensitively (docs mix "Processing" and
   "processing"); on success the file is retrieved and downloaded (the
   download URL is valid 9h).
5. Voice design — ``POST /v1/voice_design`` with ``prompt`` +
   ``preview_text`` (≤500 chars); the response's ``trial_audio`` hex is
   decoded and stored as the designed voice's reference sample.
6. Region — the international (.io) and mainland (.cn) deployments are two
   separate accounts with separate quotas (60 vs 10/20 RPM). The region is
   therefore an ENGINE-LEVEL configurable base URL via the
   :data:`BASE_URL_ENV` environment knob (per-engine settings ``env`` block
   works through the ADR-0015 seam). A key must NOT be assumed to work on
   both regions.

Errors: ``base_resp.status_code`` != 0 raises :class:`CloudEngineError`.
1002/1039 (RPM/TPM) carry the shared rate-limit retry markers so the
existing backoff engages; 2038 is rendered as the real-name/paid-tier
requirement it documents ("No cloning permission, please check account
verification status").
"""

from __future__ import annotations

import time
import uuid
import wave
from pathlib import Path

import httpx

from ..capabilities import AppliesTo, Capabilities, ParamSpec
from ..registry import Engine, GenerationRequest, GenerationResult
from .cloud_base import CloudEngineBase, CloudEngineError, voice_missing_blob

DEFAULT_BASE_URL = "https://api.minimax.io"
BASE_URL_ENV = "VOICECLONE_MINIMAX_BASE_URL"

TARGET_MODEL = "speech-2.8-hd"
MODEL_CHOICES = ("speech-2.8-hd", "speech-2.8-turbo")

# USD per 1M characters, pay-as-you-go (pricing-paygo, 2026-09-18).
PRICES_PER_M_CHARS = {"speech-2.8-hd": 100.0, "speech-2.8-turbo": 60.0}

# Text at or below this length uses the synchronous endpoint; anything
# longer goes to the async long-text endpoint. The vendor's sync cap is
# 10,000 chars, but the async route is also the RPM bypass — switch early.
SYNC_CHAR_LIMIT = 2000

# Poll cadence for the async task query endpoint (documented limit: 10
# queries/second — we are far below) and the overall task timeout.
ASYNC_POLL_SECONDS = 2.0
ASYNC_TIMEOUT_SECONDS = 900.0

UPLOAD_PATH = "/v1/files/upload"
CLONE_PATH = "/v1/voice_clone"
SYNC_PATH = "/v1/t2a_v2"
ASYNC_CREATE_PATH = "/v1/t2a_async_v2"
ASYNC_QUERY_PATH = "/v1/query/t2a_async_query_v2"
FILE_RETRIEVE_PATH = "/v1/files/retrieve"
DESIGN_PATH = "/v1/voice_design"

# Reference containers the upload endpoint accepts (mp3/m4a/wav).
REF_MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}
MAX_REF_BYTES = 20 * 1024 * 1024
MAX_REF_SECONDS = 5 * 60  # documented 10s–5min window; upper bound checked

MAX_DESIGN_PROMPT_CHARS = 2000  # vendor does not publish a hard cap; keep sane
MAX_PREVIEW_TEXT_CHARS = 500  # documented voice_design limit

# A short sentence the engine synthesizes right after cloning to activate
# the voice (previewing during voice_clone does NOT activate it — official
# FAQ). First real synthesis bills the $1.50/voice cloning charge either
# way, so this adds no extra cost.
ACTIVATION_TEXT = "你好，这是一句激活音色的测试。"

# Emotions documented for speech-2.8 (whisper/fluent are 2.6-only).
EMOTION_CHOICES = ("happy", "sad", "angry", "fearful", "disgusted",
                   "surprised", "neutral")

LANGUAGE_BOOST_CHOICES = ("auto", "Chinese", "English", "Japanese", "Korean",
                          "French", "German", "Spanish", "Russian",
                          "Portuguese", "Italian")

RATE_LIMIT_CODES = (1002, 1039)

__all__ = [
    "BASE_URL_ENV",
    "DEFAULT_BASE_URL",
    "CloudEngineError",
    "MODEL_CHOICES",
    "PRICES_PER_M_CHARS",
    "SYNC_CHAR_LIMIT",
    "TARGET_MODEL",
    "MiniMaxCloudEngine",
]


class MiniMaxCloudEngine(CloudEngineBase, Engine):
    engine_id = "minimax-speech-cloud"
    display_name = "MiniMax speech-2.8 复刻/设计（MiniMax · 云端）"

    vendor_label = "MiniMax"

    billing_note = (
        "按量付费（USD）：speech-2.8-hd $100 / 百万字符，speech-2.8-turbo $60 / 百万字符；"
        "复刻 $1.5/音色（复刻本身不计费，首次真实合成时收取——本应用复刻后立即合成一句激活语，"
        "因此该费用在创建音色时即产生）；音色设计 $3/音色（同样首次使用时计费），"
        "设计预览字符按 $60/百万字符计。复刻音色 7 天未调用会被删除（激活后永久）；"
        "音色数量受订阅套餐 Voice slots 限制。每次生成的记录中显示估算成本（USD）。"
    )
    data_usage_note = (
        "MiniMax 服务条款明示：其可能将信息用于「改进算法或增强服务」，且该用途不构成保密义务的"
        "违反；是否提供关闭选项未在公开文档中说明。参考音频会上传至 MiniMax 用于创建复刻音色"
        "（可在平台删除音色）。深度合成标识义务落在应用侧：本应用已在生成记录与导出物上标注 "
        "AI 生成内容。复刻需账号完成实名认证/付费档（错误码 2038）；音色设计拒绝模仿真实人物。"
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Region as engine-level config (issue #27): .io (international, 60
        # RPM) vs .cn (mainland, free 10 / paid 20 RPM) are separate accounts
        # with separate quotas. Override through BASE_URL_ENV in the
        # per-engine settings env block (ADR-0015 seam) or the environment.
        self.base_url = (self.env.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")

    # -- capabilities ---------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            languages=("zh", "en", "ja", "ko", "fr", "de", "it", "es", "pt", "ru"),
            voice_cloning=True,
            voice_design=True,
            pronunciation_control=False,  # pronunciation_dict exists but is not exposed yet
            emotion=True,
            commercial_license=True,  # integrated-application use is permitted by ToS
            cross_device_use=True,  # the voice lives in the cloud account
            # ToS reserves the right to use uploaded information to improve
            # algorithms / enhance services — surfaced verbatim in
            # data_usage_note (issue #21 makes this disclosure visible).
            upload_used_for_training=True,
            api_closed_loop=True,
            # Long text is NOT segmented: the async endpoint takes up to 1M
            # chars per request and bypasses the mainland 20 RPM limit that
            # per-request segmentation would run into (issue #27).
            max_chars_per_request=None,
        )

    def param_specs(self) -> list[ParamSpec]:
        applies = AppliesTo(engine=self.engine_id, model=TARGET_MODEL, mode="cloning")
        return [
            ParamSpec(
                name="speed",
                label="语速",
                kind="number",
                default=1.0,
                min=0.5, max=2.0, step=0.05,
                unit="x",
                layer="canonical",
                wire_path="voice_setting.speed",
                applies_to=applies,
                to_wire=lambda v: None if v in (None, "") else float(v),
                help="MiniMax voice_setting.speed，范围 [0.5, 2]",
            ),
            ParamSpec(
                name="model",
                label="模型",
                kind="select",
                default=TARGET_MODEL,
                choices=MODEL_CHOICES,
                layer="engine",
                wire_path="model",
                applies_to=applies,
                help="hd 音质更高，turbo 更快更便宜",
            ),
            ParamSpec(
                name="emotion",
                label="情感",
                kind="select",
                default="auto",
                choices=("auto",) + EMOTION_CHOICES,
                layer="engine",
                wire_path="voice_setting.emotion",
                applies_to=applies,
                to_wire=lambda v: None if v in (None, "", "auto") else v,
                help="auto 时不向引擎发送该参数（speech-2.8 不支持 whisper/fluent）",
            ),
            ParamSpec(
                name="language_boost",
                label="语种增强",
                kind="select",
                default="auto",
                choices=LANGUAGE_BOOST_CHOICES,
                layer="engine",
                wire_path="language_boost",
                applies_to=applies,
                to_wire=lambda v: None if v in (None, "", "auto") else v,
                help="提升指定语种的发音表现；auto 时不发送",
            ),
            ParamSpec(
                name="english_normalization",
                label="英文规范化（长文本）",
                kind="bool",
                default=False,
                layer="engine",
                wire_path="english_normalization",
                applies_to=applies,
                to_wire=lambda v: bool(v) or None,
                help="仅长文本异步接口生效（异步键名与同步不同）；对文本中的英文做规范化",
            ),
            ParamSpec(
                name="system_voice",
                label="系统音色 ID（可选）",
                kind="text",
                default="",
                layer="engine",
                wire_path="voice_setting.voice_id",
                applies_to=applies,
                to_wire=lambda v: (v or "").strip() or None,
                help=(
                    "填写 MiniMax 系统音色 ID（如 male-qn-qingse）后本次生成直接使用该系统"
                    "音色，覆盖当前音色的云端绑定"
                ),
            ),
        ]

    # -- vendor errors ---------------------------------------------------------

    def _base_error(self, resp: httpx.Response) -> tuple[int, str]:
        """Extract (status_code, status_msg) from a MiniMax envelope."""
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 - keep the status line at minimum
            raise CloudEngineError(f"HTTP {resp.status_code}") from None
        base = body.get("base_resp") or {}
        code = base.get("status_code", body.get("status_code", resp.status_code))
        msg = base.get("status_msg") or body.get("message") or body.get("msg") or str(body)
        return int(code or 0), str(msg)

    def _error(self, resp: httpx.Response, action: str) -> CloudEngineError:
        """Render a MiniMax failure as the user-facing cloud error.

        Rate-limit codes carry the shared retry markers ("限流"/"rate
        limit") so the pipeline's exponential backoff engages; 2038 is the
        documented "no cloning permission" account-verification gate.
        """
        code, msg = self._base_error(resp)
        if code in RATE_LIMIT_CODES:
            which = "RPM" if code == 1002 else "TPM"
            return CloudEngineError(
                f"{self.vendor_label}限流（错误码 {code}，{which} 超限 / rate limit）：{msg}"
            )
        if code == 1004 or resp.status_code in (401, 403):
            return CloudEngineError(
                f"API Key 无效或无权限（{self.vendor_label} 错误码 {code}）：{msg}"
            )
        if code == 2038:
            return CloudEngineError(
                f"{self.vendor_label} 错误码 2038：当前账号没有复刻权限——需要完成实名认证并"
                f"开通付费档后再试：{msg}"
            )
        return CloudEngineError(f"{self.vendor_label} {action}失败（错误码 {code}）：{msg}")

    @staticmethod
    def _voice_missing(code: int, msg: str) -> bool:
        """The shared "cloud voice is gone" rule (cloud_base) applied to
        MiniMax's base_resp envelope, whose code/message the engine extracts
        itself — the rule itself lives in exactly one place."""
        return voice_missing_blob(f"{code} {msg}".lower())

    # -- response plumbing -------------------------------------------------------

    def _ok(self, resp: httpx.Response, action: str) -> dict:
        """The one shape every MiniMax call is validated with: HTTP 200 AND
        ``base_resp.status_code == 0``, else the rendered cloud error."""
        if resp.status_code != 200:
            raise self._error(resp, action)
        code, _msg = self._base_error(resp)
        if code != 0:
            raise self._error(resp, action)
        return resp.json()

    def _download(self, url: str) -> bytes:
        dl = self._http().get(url)
        if dl.status_code != 200:
            raise CloudEngineError(f"下载合成音频失败：HTTP {dl.status_code}")
        return dl.content

    def _audio_bytes(self, data: dict) -> bytes:
        """Fetch the synthesis audio: ``data.audio`` is a URL under
        ``output_format=url`` (24h validity) — but decode hex defensively if
        the vendor ever ignores the flag."""
        audio = (data or {}).get("audio")
        if not audio:
            raise CloudEngineError(f"{self.vendor_label} 合成失败：响应中缺少音频：{str(data)[:200]}")
        if isinstance(audio, str) and audio.startswith("http"):
            return self._download(audio)
        return bytes.fromhex(audio)

    def _ensure_wav(self, data: bytes) -> bytes:
        """The pipeline's peak self-check requires real WAV frames. MiniMax
        defaults to mp3 — we always request wav, and a non-RIFF payload is a
        loud error, never a silently mis-parsed artifact."""
        if data[:4] == b"RIFF":
            return data
        raise CloudEngineError(
            f"{self.vendor_label} 返回的音频不是 WAV（应为 audio_setting.format=wav）；"
            "为避免破坏峰值自检已中止，请重试或报告该问题"
        )

    def _write_wav(self, data: bytes, generation_id: str | None) -> tuple[Path, int]:
        data = self._ensure_wav(data)
        out_dir = self.output_dir or Path.cwd() / "data" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{generation_id or uuid.uuid4().hex}.wav"
        path.write_bytes(data)
        sample_rate = self._probe_sample_rate(path, 24000)
        return path, sample_rate

    # -- binding: upload + clone + activate ---------------------------------------

    def bind_reference(self, ref_path, ref_text: str | None, log) -> dict:
        ref = Path(ref_path)
        mime = REF_MIME.get(ref.suffix.lower())
        if mime is None:
            raise CloudEngineError(
                f"{self.vendor_label}复刻仅接受 WAV / MP3 / M4A 参考音频，"
                f"当前为 {ref.suffix[1:].upper() or '(无后缀)'}"
            )
        data = ref.read_bytes()
        if len(data) > MAX_REF_BYTES:
            raise CloudEngineError("参考音频超过 MiniMax 20MB 上限")

        voice_id = "vc" + uuid.uuid4().hex[:12]  # starts with a letter; no trailing -/_

        log(f"cloud: 正在上传参考音频到 {self.vendor_label}（{len(data) / 1024:.0f} KB）…")
        up = self._http().post(
            f"{self.base_url}{UPLOAD_PATH}",
            headers=self._headers(),
            files={"purpose": (None, "voice_clone"),
                   "file": (ref.name, data, mime)},
        )
        file_id = (self._ok(up, "上传参考音频").get("file") or {}).get("file_id")
        if file_id is None:
            raise CloudEngineError(
                f"{self.vendor_label} 上传失败：响应中缺少 file_id：{up.text[:200]}"
            )

        log(f"cloud: 正在创建复刻音色（{voice_id}）…")
        clone = self._http().post(
            f"{self.base_url}{CLONE_PATH}",
            headers=self._headers(),
            json={"file_id": int(file_id), "voice_id": voice_id},
        )
        self._ok(clone, "创建复刻音色")

        # Activation (issue #27): a cloned voice is inactive until one real
        # T2A synthesis; the voice_clone preview does NOT count. Failure
        # must not lose the freshly cloned voice — surface it as an
        # "unactivated" binding instead of raising.
        extra: dict = {"voice_id": voice_id, "activated_at": None}
        log("cloud: 正在合成激活语句（复刻后不激活的音色 7 天会被删除）…")
        try:
            self._probe_synthesis(voice_id, ACTIVATION_TEXT)
            extra["activated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            log(f"cloud: 音色已激活（{voice_id}）")
        except CloudEngineError as exc:
            extra["status"] = "unactivated"
            extra["activation_error"] = str(exc)
            log(f"cloud: 激活失败，绑定标记为 unactivated：{exc}")
        return extra

    # -- voice health (issue #12 / #27) --------------------------------------------

    def _probe_synthesis(self, voice_id: str, text: str) -> None:
        """One minimal real synthesis; raises CloudEngineError on failure."""
        resp = self._http().post(
            f"{self.base_url}{SYNC_PATH}",
            headers=self._headers(),
            json=self._sync_payload(TARGET_MODEL, text, voice_id),
        )
        if resp.status_code != 200:
            raise self._error(resp, "合成")
        code, msg = self._base_error(resp)
        if code != 0:
            if self._voice_missing(code, msg):
                raise _VoiceGone(f"{self.vendor_label} 云端音色 {voice_id} 已失效：{msg}")
            raise self._error(resp, "合成")

    def check_voice(self, voice_id: str, log) -> bool:
        """Probe whether the bound cloud voice still exists.

        get_voice is POST and only lists voices that already synthesized at
        least once — the ONLY reliable check is a probe synthesis (issue
        #27). 200 + status_code 0 = alive; an error clearly naming the
        voice as gone = dead; anything else (bad key, quota, outage) is NOT
        a verdict and raises so the caller never rebuilds on a hunch.
        """
        log(f"cloud: 健康检查云端音色 {voice_id} …")
        try:
            self._probe_synthesis(voice_id, "好")
        except _VoiceGone as exc:
            log(f"cloud: {exc}")
            return False
        return True

    # -- synthesis -----------------------------------------------------------------

    @staticmethod
    def _sync_payload(model: str, text: str, voice_id: str) -> dict:
        """SYNC payload. Keys here are t2a_v2-specific."""
        return {
            "model": model,
            "text": text,
            "voice_setting": {"voice_id": voice_id},
            "audio_setting": {"sample_rate": 24000, "format": "wav"},
            "output_format": "url",
        }

    @staticmethod
    def _async_payload(model: str, text: str, voice_id: str) -> dict:
        """ASYNC long-text payload (t2a_async_v2). The keys DIFFER from the
        sync request — audio_sample_rate / english_normalization, and no
        output_format / stream / force_cbr. Constructed separately on
        purpose; reusing the sync builder silently sends wrong keys."""
        return {
            "model": model,
            "text": text,
            "voice_setting": {"voice_id": voice_id},
            "audio_setting": {"audio_sample_rate": 24000, "format": "wav"},
        }

    def _t2a_request(self, model: str, text: str, voice_id: str, params: dict) -> dict:
        """Build the wire payload for a synthesis, applying the user params
        (both layers, ADR-0018). The structural keys come from the
        sync/async payload builders; params decorate on top."""
        if len(text) > SYNC_CHAR_LIMIT:
            body = self._async_payload(model, text, voice_id)
            # Async-only key (the sync request uses text_normalization in
            # voice_setting instead). Off by default: it rewrites English
            # normalization behavior, so it only fires when asked for.
            if params.get("english_normalization"):
                body["english_normalization"] = True
        else:
            body = self._sync_payload(model, text, voice_id)
        voice_setting = body["voice_setting"]
        speed = params.get("speed")
        if speed not in (None, ""):
            voice_setting["speed"] = float(speed)
        if params.get("emotion"):
            voice_setting["emotion"] = params["emotion"]
        if params.get("language_boost"):
            body["language_boost"] = params["language_boost"]
        return body

    def _synthesize_sync(self, model: str, text: str, voice_id: str,
                         params: dict, log) -> tuple[bytes, dict]:
        payload = self._t2a_request(model, text, voice_id, params)
        resp = self._http().post(
            f"{self.base_url}{SYNC_PATH}", headers=self._headers(), json=payload,
        )
        body = self._ok(resp, "合成")
        log("cloud: 下载合成音频…")
        return self._audio_bytes(body.get("data") or {}), body.get("extra_info") or {}

    def _synthesize_async(self, model: str, text: str, voice_id: str,
                          params: dict, log) -> bytes:
        payload = self._t2a_request(model, text, voice_id, params)
        create = self._http().post(
            f"{self.base_url}{ASYNC_CREATE_PATH}", headers=self._headers(), json=payload,
        )
        task_id = self._ok(create, "创建长文本合成任务").get("task_id")
        if not task_id:
            raise CloudEngineError(
                f"{self.vendor_label} 异步合成失败：响应中缺少 task_id：{create.text[:200]}"
            )
        log(f"cloud: 长文本任务已创建（{task_id}），轮询状态…")

        deadline = time.monotonic() + ASYNC_TIMEOUT_SECONDS
        while True:
            if time.monotonic() > deadline:
                raise CloudEngineError(
                    f"{self.vendor_label} 异步合成超时（>{ASYNC_TIMEOUT_SECONDS:.0f}s）：{task_id}"
                )
            time.sleep(ASYNC_POLL_SECONDS)
            query = self._http().get(
                f"{self.base_url}{ASYNC_QUERY_PATH}",
                headers=self._headers(),
                params={"task_id": task_id},
            )
            task = self._ok(query, "查询长文本合成任务")
            # Case-insensitive on purpose: the docs' example writes
            # "Processing" while the enum says lowercase.
            status = str(task.get("status") or "").lower()
            if status == "success":
                file_id = task.get("file_id")
                if not file_id:
                    raise CloudEngineError(
                        f"{self.vendor_label} 异步合成完成但缺少 file_id：{str(task)[:200]}"
                    )
                break
            if status in ("failed", "expired"):
                raise CloudEngineError(
                    f"{self.vendor_label} 异步合成任务{('失败' if status == 'failed' else '已过期')}"
                    f"（status={status}）：{task_id}"
                )
            log(f"cloud: 任务 {task_id} 状态 {status or 'unknown'}…")

        retrieve = self._http().get(
            f"{self.base_url}{FILE_RETRIEVE_PATH}",
            headers=self._headers(),
            params={"file_id": str(file_id)},
        )
        body = self._ok(retrieve, "获取合成结果文件")
        file_info = body.get("file") or body
        url = file_info.get("download_url")
        if not url:
            raise CloudEngineError(
                f"{self.vendor_label} 获取合成结果失败：缺少 download_url：{retrieve.text[:200]}"
            )
        return self._download(url)

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        started = time.monotonic()
        params = dict(request.params)
        model = params.pop("model", None) or TARGET_MODEL
        if model not in MODEL_CHOICES:
            raise CloudEngineError(f"未知模型 {model}（可选：{', '.join(MODEL_CHOICES)}）")
        # System voices (issue #27): an explicit system voice id overrides
        # the bound cloned voice for this generation.
        voice_id = (params.pop("system_voice", "") or "").strip() or params.get("voice_id")
        if not voice_id:
            raise CloudEngineError(
                "该音色还没有云端绑定；请先对所选音色执行一次绑定（生成时会自动注册云端音色），"
                "或在引擎参数中填写系统音色 ID"
            )

        log(f"cloud: 调用 {model} 合成 {len(request.text)} 字符…")
        if len(request.text) > SYNC_CHAR_LIMIT:
            log("cloud: 长文本走异步接口（绕开大陆 20 RPM 限制）…")
            data = self._synthesize_async(model, request.text, voice_id, params, log)
            extra_info = {}
        else:
            data, extra_info = self._synthesize_sync(model, request.text, voice_id, params, log)

        path, sample_rate = self._write_wav(data, request.generation_id)

        usage = extra_info.get("usage_characters")
        billed = int(usage) if usage else len(request.text)
        cost = round(billed / 1_000_000 * PRICES_PER_M_CHARS[model], 6)

        log(
            f"cloud: 完成（{path.name}，{sample_rate} Hz，"
            f"估算成本 ${cost:.4f}，耗时 {time.monotonic() - started:.2f}s）"
        )
        return GenerationResult(
            audio_path=str(path),
            sample_rate=sample_rate,
            model_version=model,
            cost=cost,
        )

    # -- voice design (issue #27; same engine — same speech-2.8 model) -------------

    def design_voice(self, description: str, preview_text: str, log) -> dict:
        description = (description or "").strip()
        preview_text = (preview_text or "").strip()
        if not description:
            raise CloudEngineError("声音描述不能为空")
        if len(description) > MAX_DESIGN_PROMPT_CHARS:
            raise CloudEngineError(
                f"声音描述过长（{len(description)} 字符，上限 {MAX_DESIGN_PROMPT_CHARS}）"
            )
        if not preview_text:
            raise CloudEngineError("试听文本不能为空")
        if len(preview_text) > MAX_PREVIEW_TEXT_CHARS:
            raise CloudEngineError(
                f"试听文本过长（{len(preview_text)} 字符，上限 {MAX_PREVIEW_TEXT_CHARS}）"
            )

        log("cloud-design: 正在按文字描述创建云端音色…")
        resp = self._http().post(
            f"{self.base_url}{DESIGN_PATH}",
            headers=self._headers(),
            json={"prompt": description, "preview_text": preview_text},
        )
        body = self._ok(resp, "创建设计音色")
        voice_id = body.get("voice_id")
        if not voice_id:
            raise CloudEngineError(
                f"创建设计音色失败：响应中缺少 voice_id：{resp.text[:200]}"
            )
        trial_hex = body.get("trial_audio")
        if not trial_hex:
            raise CloudEngineError(
                f"创建设计音色失败：响应中缺少 trial_audio：{resp.text[:200]}"
            )

        out_dir = self.output_dir or Path.cwd() / "data" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_path = out_dir / f"design-{uuid.uuid4().hex}.wav"
        sample_path.write_bytes(bytes.fromhex(trial_hex))
        try:
            with wave.open(str(sample_path), "rb") as w:
                sample_rate = w.getframerate()
        except wave.Error:
            raise CloudEngineError(
                f"{self.vendor_label} 设计音色的试听音频不是 WAV；已中止以保护峰值自检"
            ) from None

        log(f"cloud-design: 设计音色已创建（{voice_id}），预览样本已保存")
        return {
            "voice_id": voice_id,
            "sample_audio_path": str(sample_path),
            "sample_rate": sample_rate,
            "transcript": preview_text,
        }


class _VoiceGone(CloudEngineError):
    """Internal marker: the probe verdict is "this cloud voice is gone"."""
