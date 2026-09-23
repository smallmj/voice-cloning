"""Qwen3-TTS-Instruct-Flash on Alibaba Cloud Model Studio (阿里百炼) — issue #68.

The instruction-control sibling of the qwen3-tts-vc cloud engine: instead of
a cloned voice it exposes the vendor's natural-language ``instructions``
parameter (语气/语速/方言 as free text, verified 2026-09 against
help.aliyun.com/zh/model-studio/qwen3-tts-instruct-flash and the DashScope
非实时语音合成 docs). Same vendor, same BYOK key, same multimodal-generation
endpoint and response envelope as the VC engine, so the whole cloud plumbing
(key access, injectable client, vendor errors, audio download, billing) is
reused from the vendor-neutral base and DashScopeEngine (ADR-0015).

Vendor facts baked in here (厂商口径, see data/capability_matrix.json):
- model ``qwen3-tts-instruct-flash`` (snapshot equivalent
  ``qwen3-tts-instruct-flash-2026-01-26``); Instruct 调节 verified for the
  vendor's 25 system timbres (中英文);
- ``instructions``: natural-language synthesis-direction text, sent inside
  ``input`` when non-empty; absent means "do not send";
- ``language_type``: same 10-value enum as the rest of the Qwen-TTS family;
- billing 0.8 元/万输入字符 (1 CJK char = 2 chars), output not billed.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from ..capabilities import AppliesTo, Capabilities, ParamSpec
from ..registry import Engine, GenerationRequest, GenerationResult
from .cloud_base import (
    CloudEngineError,
    billed_chars,
    billing_cost,
    voice_missing_error,
)
from .qwen_tts_cloud import (
    BASE_URL,
    DashScopeEngine,
    LANGUAGE_CHOICES,
    PRICE_PER_10K_CHARS,
    SYNTH_PATH,
)

TARGET_MODEL = "qwen3-tts-instruct-flash"

__all__ = ["BASE_URL", "CloudEngineError", "Qwen3TtsInstructCloudEngine", "TARGET_MODEL"]

# The vendor documents a natural-language instruction, not a hard char cap;
# keep a generous guard so a runaway paste fails here instead of at the API.
MAX_INSTRUCTION_CHARS = 2000

ENROLL_PATH = "/services/audio/tts/customization"


class Qwen3TtsInstructCloudEngine(DashScopeEngine, Engine):
    engine_id = "qwen3-tts-instruct-cloud"
    display_name = "Qwen3-TTS 指令控制（阿里百炼 · 云端）"

    billing_note = (
        "qwen3-tts-instruct-flash：0.8 元 / 万输入字符（1 个汉字计 2 个字符），输出不计费。"
        "本应用在每次生成的记录中显示估算成本。"
    )
    data_usage_note = (
        "按阿里云隐私声明，百炼的模型输入输出数据不会用于模型训练；"
        "instructions 指令与文本会上传至阿里云用于本次合成。"
        "服务条款明确禁止转售本服务（BYOK 下你是阿里云的直接客户）。"
    )

    # -- capabilities --------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            languages=("zh", "en", "ja", "ko", "fr", "de", "it", "es", "pt", "ru"),
            voice_cloning=False,  # instruct 控制，不做音色复刻 enrollment
            voice_design=False,
            pronunciation_control=False,
            # Issue #68 术语口径：instruct 的情感控制走 instructions 参数
            # （自然语言指令），不是 API 级情感枚举参数。
            emotion=False,
            commercial_license=True,
            cross_device_use=True,
            upload_used_for_training=False,
            api_closed_loop=True,
            # Same conservative per-request cap as the sibling Qwen cloud
            # engines (issue #13): segment earlier rather than pushing the
            # API ceiling.
            max_chars_per_request=2000,
        )

    def param_specs(self) -> list[ParamSpec]:
        def to_wire(value):
            if value in (None, "", "auto"):
                return None
            return value if value in LANGUAGE_CHOICES else None

        return [
            ParamSpec(
                name="instructions",
                label="合成指令",
                kind="textarea",
                default="",
                layer="engine",
                wire_path="instructions",
                max_length=MAX_INSTRUCTION_CHARS,
                applies_to=AppliesTo(
                    engine=self.engine_id, model=TARGET_MODEL, mode="generation"
                ),
                help=(
                    "自然语言指令控制语气/语速/方言，如「用兴奋的语气、稍快的语速说」；"
                    "留空则不向引擎发送该参数"
                ),
            ),
            # Canonical language (ADR-0018), same shape as the VC engine:
            # the wire key is the vendor's `language_type`; "auto" sends
            # nothing.
            ParamSpec(
                name="language",
                label="发音语种",
                kind="select",
                default="auto",
                choices=("auto", *LANGUAGE_CHOICES),
                layer="canonical",
                wire_path="language_type",
                applies_to=AppliesTo(
                    engine=self.engine_id, model=TARGET_MODEL, mode="generation"
                ),
                to_wire=to_wire,
                help="建议与文本语种一致以获得自然发音；auto 时不向引擎发送该参数",
            ),
            # ADR-0018 decision 4: the voice id is server-injected (it comes
            # from the app's cloud binding / selected voice, never from a
            # user-editable field) — declared as data so the closed parameter
            # list stays honest.
            ParamSpec(
                name="voice",
                label="音色",
                kind="text",
                exposed=False,
                not_exposed_reason="server-injected",
                layer="engine",
                applies_to=AppliesTo(
                    engine=self.engine_id, model=TARGET_MODEL, mode="generation"
                ),
                help="音色 id 由应用的云端绑定注入（server-injected），不作为用户参数暴露",
            ),
        ]

    # -- binding: enrollment (same 口径 as the VC engine) ----------------------

    def bind_reference(self, ref_path, ref_text: str | None, log) -> dict:
        """Enroll the reference against the instruct target model.

        Mirrors the VC engine's enrollment 1:1 (same endpoint, same data-URL
        payload) with ``target_model`` pinned to this engine's model, so the
        binding/health-check 口径 stays identical across the Qwen cloud
        engines. If the vendor rejects enrollment for the instruct model the
        vendor error surfaces verbatim (no silent fallback).
        """
        from .qwen_tts_cloud import DATA_URL_MIME, MAX_REF_BYTES

        ref = Path(ref_path)
        mime = DATA_URL_MIME.get(ref.suffix.lower())
        if mime is None:
            raise CloudEngineError(
                f"阿里百炼仅接受 WAV / MP3 / M4A 参考音频，当前为 {ref.suffix[1:].upper()}"
            )
        data = ref.read_bytes()
        if len(data) > MAX_REF_BYTES:
            raise CloudEngineError("参考音频超过阿里百炼 10MB 上限")
        log(f"cloud: 正在上传参考音频到阿里百炼（{len(data) / 1024:.0f} KB）…")

        import base64

        payload = {
            "model": "qwen-voice-enrollment",
            "input": {
                "action": "create",
                "target_model": TARGET_MODEL,
                "preferred_name": "ins" + uuid.uuid4().hex[:8],
                "audio": {"data": f"data:{mime};base64,{base64.b64encode(data).decode()}"},
            },
        }
        if ref_text:
            payload["input"]["text"] = ref_text

        resp = self._http().post(
            f"{BASE_URL}{ENROLL_PATH}", headers=self._headers(), json=payload
        )
        if resp.status_code != 200:
            raise CloudEngineError(f"创建云端音色失败：{self._vendor_error(resp)}")
        voice = (resp.json().get("output") or {}).get("voice")
        if not voice:
            raise CloudEngineError(f"创建云端音色失败：响应中缺少 voice 字段：{resp.text[:200]}")
        log(f"cloud: 云端音色已创建（{voice}）")
        return {"voice_id": voice, "target_model": TARGET_MODEL}

    # -- voice health (issue #12, same probe as the VC engine) -----------------

    def check_voice(self, voice_id: str, log) -> bool:
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

    # -- synthesis -------------------------------------------------------------

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        started = time.monotonic()
        params = request.params
        voice_id = params.get("voice_id")
        if not voice_id:
            raise CloudEngineError(
                "该音色还没有云端绑定；请先对所选音色执行一次绑定（生成时会自动注册云端音色）"
            )

        instructions = (params.get("instructions") or "").strip()
        if len(instructions) > MAX_INSTRUCTION_CHARS:
            raise CloudEngineError(
                f"合成指令过长（{len(instructions)} 字符，上限 {MAX_INSTRUCTION_CHARS}）"
            )

        body: dict = {"text": request.text, "voice": voice_id}
        if instructions:
            body["instructions"] = instructions
        # Canonical `language` (ADR-0018) with the legacy wire key as
        # fallback for reruns of old records; "auto"/empty sends nothing.
        language = params.get("language", params.get("language_type"))
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
