"""Qwen 声音设计 on Alibaba Cloud Model Studio (阿里百炼) — issue #11.

Creates a voice from a text description alone (no reference audio), BYOK
only (ADR-0003). Vendor contract (help.aliyun.com 声音设计 API, verified
2026-09):

1. Create — ``POST /services/audio/tts/customization`` with
   ``model: qwen-voice-design``, ``input.action: create``,
   ``input.target_model`` + ``input.voice_prompt`` (the description) +
   ``input.preview_text``. The response carries ``output.voice`` plus a
   base64-encoded ``output.preview_audio`` — the preview sample the app
   stores as the designed voice's reference.
2. Synthesis — ``POST /services/aigc/multimodal-generation/generation``
   with ``model: qwen3-tts-vd-2026-01-26`` and the designed ``voice``;
   non-streaming returns ``output.audio.url`` (a WAV valid for 24 hours)
   exactly like the cloning engine.

The ``target_model`` in design MUST match the synthesis ``model``, or the
vendor rejects the call — both are pinned to TARGET_MODEL here.

Note: ``preview_text`` is handed to the vendor raw. It is a design-preview
utterance the vendor speaks to produce the sample, not app-generated
content — the ADR-0008 normalization layer applies at generation time,
where the sidecar controls the pipeline end to end.
"""

from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path

import httpx

from ..capabilities import Capabilities
from ..registry import Engine, GenerationRequest, GenerationResult
from .dashscope_base import BASE_URL, CloudEngineError, DashScopeEngine
from .qwen_tts_cloud import _voice_missing_error

ENROLL_PATH = "/services/audio/tts/customization"
SYNTH_PATH = "/services/aigc/multimodal-generation/generation"

# Design must target the model that will later synthesize with the voice.
DESIGN_MODEL = "qwen-voice-design"
TARGET_MODEL = "qwen3-tts-vd-2026-01-26"

MAX_PROMPT_CHARS = 2048  # vendor limit for Qwen-TTS voice descriptions

__all__ = ["BASE_URL", "CloudEngineError", "Qwen3TtsVdCloudEngine"]


class Qwen3TtsVdCloudEngine(DashScopeEngine, Engine):
    engine_id = "qwen3-tts-vd-cloud"
    display_name = "Qwen3-TTS 音色设计（阿里百炼 · 云端）"

    billing_note = (
        "qwen-voice-design：按次计费（以百炼控制台价格为准）；"
        "后续用设计音色合成按 qwen3-tts-vd 模型的字符单价计费。"
        "本应用在每次生成的记录中显示估算成本。"
    )
    data_usage_note = (
        "按阿里云隐私声明，百炼的模型输入输出数据不会用于模型训练；"
        "声音描述会上传至阿里云用于创建音色（可在百炼控制台删除）。"
        "服务条款明确禁止转售本服务（BYOK 下你是阿里云的直接客户）。"
    )

    # -- capabilities --------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            languages=("zh", "en", "ja", "ko", "fr", "de", "it", "es", "pt", "ru"),
            voice_cloning=False,
            voice_design=True,
            pronunciation_control=False,
            emotion=False,
            commercial_license=True,
            cross_device_use=True,  # the designed voice lives in the cloud account
            upload_used_for_training=False,
            api_closed_loop=True,
        )

    # -- voice design ----------------------------------------------------------

    def design_voice(self, description: str, preview_text: str, log) -> dict:
        description = (description or "").strip()
        preview_text = (preview_text or "").strip()
        if not description:
            raise CloudEngineError("声音描述不能为空")
        if len(description) > MAX_PROMPT_CHARS:
            raise CloudEngineError(
                f"声音描述过长（{len(description)} 字符，上限 {MAX_PROMPT_CHARS}）"
            )

        log("cloud-design: 正在按文字描述创建云端音色…")
        payload = {
            "model": DESIGN_MODEL,
            "input": {
                "action": "create",
                "target_model": TARGET_MODEL,
                "preferred_name": "vd" + uuid.uuid4().hex[:8],
                "voice_prompt": description,
                "preview_text": preview_text,
            },
        }
        resp = self._http().post(
            f"{BASE_URL}{ENROLL_PATH}", headers=self._headers(), json=payload
        )
        if resp.status_code != 200:
            raise CloudEngineError(f"创建设计音色失败：{self._vendor_error(resp)}")
        output = resp.json().get("output") or {}
        voice = output.get("voice") or output.get("voice_id")
        if not voice:
            raise CloudEngineError(
                f"创建设计音色失败：响应中缺少 voice 字段：{resp.text[:200]}"
            )

        preview = output.get("preview_audio") or {}
        data_b64 = preview.get("data")
        if not data_b64:
            raise CloudEngineError(
                f"创建设计音色失败：响应中缺少预览音频：{resp.text[:200]}"
            )
        sample_rate = int(preview.get("sample_rate") or 24000)

        out_dir = self.output_dir or Path.cwd() / "data" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_path = out_dir / f"design-{uuid.uuid4().hex}.wav"
        sample_path.write_bytes(base64.b64decode(data_b64))
        sample_rate = self._probe_sample_rate(sample_path, sample_rate)

        log(f"cloud-design: 设计音色已创建（{voice}），预览样本已保存")
        return {
            "voice_id": voice,
            "sample_audio_path": str(sample_path),
            "sample_rate": sample_rate,
            "transcript": preview_text or None,
        }

    # -- voice health (issue #12) ------------------------------------------------

    def check_voice(self, voice_id: str, log) -> bool:
        """Probe whether the designed cloud voice still exists on 百炼.

        Same minimal-synthesis probe as the cloning engine (issue #12). A
        designed voice can never be rebuilt from a reference — the rebuild
        path fails with the explicit design-again reason, and the sidecar
        marks the binding unavailable instead of failing mid-run.
        """
        log(f"cloud-design: 健康检查云端音色 {voice_id} …")
        resp = self._http().post(
            f"{BASE_URL}{SYNTH_PATH}",
            headers=self._headers(),
            json={"model": TARGET_MODEL, "input": {"text": "好", "voice": voice_id}},
        )
        if resp.status_code == 200:
            return True
        if _voice_missing_error(resp):
            log(f"cloud-design: 云端音色 {voice_id} 已被厂商删除或失效")
            return False
        raise CloudEngineError(f"云端音色健康检查失败：{self._vendor_error(resp)}")

    # -- binding ---------------------------------------------------------------

    def bind_reference(self, ref_path, ref_text: str | None, log) -> dict:
        """A designed voice cannot be re-bound from a reference sample.

        The vendor's designed voice exists only as a design call. If the
        binding was lost, the user must run the design again — surfaced as
        a user-facing cloud error, never as a raw 500.
        """
        raise CloudEngineError(
            "设计音色无法通过参考音频重建绑定；请删除后用文字描述重新设计该音色"
        )

    # -- synthesis -------------------------------------------------------------

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        started = time.monotonic()
        params = request.params
        voice_id = params.get("voice_id")
        if not voice_id:
            raise CloudEngineError(
                "该音色还没有云端绑定；请先对所选设计音色执行一次设计（创建时会自动绑定）"
            )

        log(f"cloud-design: 调用 {TARGET_MODEL} 合成…")
        resp = self._http().post(
            f"{BASE_URL}{SYNTH_PATH}",
            headers=self._headers(),
            json={"model": TARGET_MODEL, "input": {"text": request.text, "voice": voice_id}},
        )
        if resp.status_code != 200:
            raise CloudEngineError(f"云端合成失败：{self._vendor_error(resp)}")
        audio_url = ((resp.json().get("output") or {}).get("audio") or {}).get("url")
        if not audio_url:
            raise CloudEngineError(f"云端合成失败：响应中缺少音频 URL：{resp.text[:200]}")

        log("cloud-design: 下载合成音频…")
        dl = self._http().get(audio_url)
        if dl.status_code != 200:
            raise CloudEngineError(f"下载合成音频失败：HTTP {dl.status_code}")

        out_dir = self.output_dir or Path.cwd() / "data" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{request.generation_id or uuid.uuid4().hex}.wav"
        out_path.write_bytes(dl.content)
        sample_rate = self._probe_sample_rate(out_path, 24000)

        log(
            f"cloud-design: 完成（{out_path.name}，{sample_rate} Hz，"
            f"耗时 {time.monotonic() - started:.2f}s）"
        )
        return GenerationResult(
            audio_path=str(out_path),
            sample_rate=sample_rate,
            model_version=TARGET_MODEL,
            cost=None,  # the vd model's per-char price is not published; no estimate
        )
