"""Qwen3-TTS-VC on Alibaba Cloud Model Studio (阿里百炼) — the cloud engine.

BYOK only (ADR-0003): the user's own DashScope API key is read from the
sidecar's key store at call time and is never persisted anywhere else.

Flow mirrors the vendor's documented contract (verified 2026-09 against
help.aliyun.com):

1. Enrollment — ``POST /api/v1/services/audio/tts/customization`` with
   ``model: qwen-voice-enrollment``. Unlike CosyVoice enrollment (public-URL
   only), Qwen-TTS accepts the reference as a base64 data URL, which is what
   makes it usable from a local desktop app.
2. Synthesis — ``POST /api/v1/services/aigc/multimodal-generation/generation``
   with the cloned voice id; non-streaming returns a JSON payload whose
   ``output.audio.url`` points at a WAV valid for 24 hours. The engine
   downloads it and writes it into the app's audio dir, so the artifact is
   owned locally like every other engine's output.

Billing (Qwen 3-TTS-VC, 华北 2 北京): 0.8 元/万输入字符，输出不计费；
阿里口径「1 个汉字 = 2 个字符」.
"""

from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path


from ..capabilities import Capabilities, ParamSpec
from ..registry import Engine, GenerationRequest, GenerationResult
from .cloud_base import (
    CloudEngineBase,
    CloudEngineError,
    billed_chars,
    billing_cost,
    voice_missing_error,
)

ENROLL_PATH = "/services/audio/tts/customization"
SYNTH_PATH = "/services/aigc/multimodal-generation/generation"

# Vendor-specific: the DashScope base URL stays with the engines that call it
# (ADR-0015 decision 3 — only base URL, model names, payloads and error
# classification are vendor-specific; shared behavior lives in cloud_base).
BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

__all__ = [
    "BASE_URL",
    "CloudEngineError",
    "Qwen3TtsVcCloudEngine",
    "billing_cost",
    "billed_chars",
    "voice_missing_error",
]

TARGET_MODEL = "qwen3-tts-vc-2026-01-22"
ENROLL_MODEL = "qwen-voice-enrollment"

# Aliyun billing: 1 CJK char = 2 chars; 0.8 CNY per 10k input chars.
PRICE_PER_10K_CHARS = 0.8

# Reference containers the enrollment endpoint accepts as data URLs.
DATA_URL_MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}
MAX_REF_BYTES = 10 * 1024 * 1024

# language_type values documented for Qwen-TTS synthesis.
LANGUAGE_CHOICES = (
    "Chinese", "English", "German", "Italian", "Portuguese",
    "Spanish", "Japanese", "Korean", "French", "Russian",
)


# (VOICE_MISSING_MARKERS / _voice_missing_error / billed_chars / billing_cost
# moved to the vendor-neutral cloud base in ADR-0015; re-exported above.)


class DashScopeEngine(CloudEngineBase):
    """Cloud base pinned to Alibaba Cloud Model Studio (阿里百炼).

    Vendor identity lives with the vendor's engines (ADR-0015 decision 3):
    the neutral base carries only label-driven behavior.
    """

    vendor_label = "阿里百炼"


class Qwen3TtsVcCloudEngine(DashScopeEngine, Engine):
    engine_id = "qwen3-tts-vc-cloud"
    display_name = "Qwen3-TTS 复刻（阿里百炼 · 云端）"

    billing_note = (
        "qwen3-tts-vc-2026-01-22：0.8 元 / 万输入字符（1 个汉字计 2 个字符），输出不计费；"
        "北京地域每个模型有 1 万字符免费额度（开通后 90 天内有效）。按实际字符计费，"
        "本应用在每次生成的记录中显示估算成本。"
    )
    data_usage_note = (
        "按阿里云隐私声明，百炼的模型输入输出数据不会用于模型训练；但参考音频会上传至"
        "阿里云用于创建克隆音色，并留存于你的百炼账号下（可在控制台删除音色）。"
        "服务条款明确禁止转售本服务（BYOK 下你是阿里云的直接客户）。"
    )

    # -- capabilities --------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            languages=("zh", "en", "ja", "ko", "fr", "de", "it", "es", "pt", "ru"),
            voice_cloning=True,
            voice_design=False,  # qwen3-tts-vd exists but is not implemented yet
            pronunciation_control=False,
            emotion=False,
            commercial_license=True,  # paid commercial API use is permitted
            cross_device_use=True,  # the cloned voice lives in the cloud account
            upload_used_for_training=False,
            api_closed_loop=True,
            # Conservative per-request cap: the vendor counts 1 Chinese
            # character as 2 and allows far more per call, but prosody
            # quality degrades on very long single requests — segment
            # earlier rather than pushing the API ceiling (issue #13).
            max_chars_per_request=2000,
        )

    def param_specs(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="language_type",
                label="发音语种",
                kind="select",
                default="auto",
                choices=("auto",) + LANGUAGE_CHOICES,
                help="建议与文本语种一致以获得自然发音；auto 时不向引擎发送该参数",
            ),
        ]

    # -- binding: enrollment ---------------------------------------------------

    def bind_reference(self, ref_path, ref_text: str | None, log) -> dict:
        ref = Path(ref_path)
        mime = DATA_URL_MIME.get(ref.suffix.lower())
        if mime is None:
            raise CloudEngineError(
                f"阿里百炼复刻仅接受 WAV / MP3 / M4A 参考音频，当前为 {ref.suffix[1:].upper()}"
            )
        data = ref.read_bytes()
        if len(data) > MAX_REF_BYTES:
            raise CloudEngineError("参考音频超过阿里百炼 10MB 上限")
        log(f"cloud: 正在上传参考音频到阿里百炼（{len(data) / 1024:.0f} KB）…")

        payload = {
            "model": ENROLL_MODEL,
            "input": {
                "action": "create",
                "target_model": TARGET_MODEL,
                "preferred_name": "vc" + uuid.uuid4().hex[:8],
                "audio": {"data": f"data:{mime};base64,{base64.b64encode(data).decode()}"},
            },
        }
        if ref_text:
            payload["input"]["text"] = ref_text

        resp = self._http().post(
            f"{BASE_URL}{ENROLL_PATH}",
            headers=self._headers(),
            json=payload,
        )
        if resp.status_code != 200:
            raise CloudEngineError(f"创建云端音色失败：{self._vendor_error(resp)}")
        voice = (resp.json().get("output") or {}).get("voice")
        if not voice:
            raise CloudEngineError(f"创建云端音色失败：响应中缺少 voice 字段：{resp.text[:200]}")
        log(f"cloud: 云端音色已创建（{voice}）")
        return {"voice_id": voice, "target_model": TARGET_MODEL}

    # -- voice health (issue #12) ------------------------------------------------

    def check_voice(self, voice_id: str, log) -> bool:
        """Probe whether the bound cloud voice still exists on 百炼.

        Vendors silently recycle enrolled voices; the cheapest documented
        way to ask "is this voice alive" is a minimal one-character synthesis
        against the target model. 200 = alive; a vendor error that clearly
        names the voice as missing/gone = dead; anything else (bad key,
        quota, outage) is NOT a verdict about the voice and raises instead —
        the caller must not rebuild a binding on an unrelated failure.
        Cost trade-off (deliberate): the probe bills the minimum 2 chars
        (≈ ¥0.00016) per generation — negligible next to the correctness
        guarantee that a recycled voice never fails a real run.
        """
        log(f"cloud: 健康检查云端音色 {voice_id} …")
        resp = self._http().post(
            f"{BASE_URL}{SYNTH_PATH}",
            headers=self._headers(),
            json={"model": TARGET_MODEL, "input": {"text": "好", "voice": voice_id}},
        )
        if resp.status_code == 200:
            return True
        if voice_missing_error(resp):
            log(f"cloud: 云端音色 {voice_id} 已被厂商删除或失效")
            return False
        raise CloudEngineError(f"云端音色健康检查失败：{self._vendor_error(resp)}")

    # -- synthesis ---------------------------------------------------------------

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        started = time.monotonic()
        params = request.params
        voice_id = params.get("voice_id")
        if not voice_id:
            raise CloudEngineError(
                "该音色还没有云端绑定；请先对所选音色执行一次绑定（生成时会自动注册云端音色）"
            )

        body: dict = {"text": request.text, "voice": voice_id}
        language = params.get("language_type")
        if language and language in LANGUAGE_CHOICES:
            body["language_type"] = language

        log(f"cloud: 调用 {TARGET_MODEL} 合成 {billed_chars(request.text)} 计费字符…")
        resp = self._http().post(
            f"{BASE_URL}{SYNTH_PATH}", headers=self._headers(),
            json={"model": TARGET_MODEL, "input": body},
        )
        if resp.status_code != 200:
            raise CloudEngineError(f"云端合成失败：{self._vendor_error(resp)}")
        audio_url = ((resp.json().get("output") or {}).get("audio") or {}).get("url")
        if not audio_url:
            raise CloudEngineError(f"云端合成失败：响应中缺少音频 URL：{resp.text[:200]}")

        log("cloud: 下载合成音频…")
        dl = self._http().get(audio_url)
        if dl.status_code != 200:
            raise CloudEngineError(f"下载合成音频失败：HTTP {dl.status_code}")

        out_dir = self.output_dir or Path.cwd() / "data" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{request.generation_id or uuid.uuid4().hex}.wav"
        out_path.write_bytes(dl.content)

        sample_rate = self._probe_sample_rate(out_path, 24000)

        cost = billing_cost(request.text, PRICE_PER_10K_CHARS)
        log(
            f"cloud: 完成（{out_path.name}，{sample_rate} Hz，"
            f"估算成本 ¥{cost:.4f}，耗时 {time.monotonic() - started:.2f}s）"
        )
        return GenerationResult(
            audio_path=str(out_path),
            sample_rate=sample_rate,
            model_version=TARGET_MODEL,
            cost=cost,
        )
