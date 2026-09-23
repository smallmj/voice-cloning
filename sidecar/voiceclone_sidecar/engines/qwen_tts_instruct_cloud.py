"""Qwen3-TTS-Instruct-Flash on Alibaba Cloud Model Studio (阿里百炼) — issue #68.

The instruction-control sibling of the qwen3-tts-vc cloud engine: instead of
a cloned voice it exposes the vendor's natural-language ``instructions``
parameter (语气/语速/方言 as free text, verified 2026-09 against
help.aliyun.com/zh/model-studio/qwen3-tts-instruct-flash and the DashScope
非实时语音合成 docs). Same vendor, same BYOK key, same multimodal-generation
endpoint and response envelope as the VC engine.

ADR-0015 decision 4 (review fix): the whole cloud plumbing — enrollment,
voice health check, synthesis/download/billing flow — lives ONCE on
``DashScopeEngine``; this subclass declares only its differences as data
(target model, enrollment prefix, price) plus the ``instructions`` knob and
the language adapter wiring in ``_synthesis_body``.

Vendor facts baked in here (厂商口径, see data/capability_matrix.json):
- model ``qwen3-tts-instruct-flash`` (snapshot equivalent
  ``qwen3-tts-instruct-flash-2026-01-26``); Instruct 调节 verified for the
  vendor's 25 system timbres (中英文);
- ``instructions``: natural-language synthesis-direction text, sent inside
  ``input`` when non-empty; absent means "do not send";
- ``language_type``: same 10-value enum as the rest of the Qwen-TTS family;
- billing 0.8 元/万输入字符 (1 CJK char = 2 chars), output not billed.

NOTE: the matrix records that cloning-voice availability for the instruct
model is NOT verified (the engine still routes a reference through the
shared enrollment flow when a voice record is bound; the vendor error
surfaces verbatim if the vendor rejects it for this model).
"""

from __future__ import annotations

from ..capabilities import AppliesTo, Capabilities, ParamSpec
from ..registry import Engine, GenerationRequest
from .cloud_base import CloudEngineError
from .qwen_tts_cloud import (
    LANGUAGE_CHOICES,
    PRICE_PER_10K_CHARS,
    DashScopeEngine,
)

TARGET_MODEL = "qwen3-tts-instruct-flash"

__all__ = ["Qwen3TtsInstructCloudEngine", "TARGET_MODEL"]

# The vendor documents a natural-language instruction, not a hard char cap;
# keep a generous guard so a runaway paste fails here instead of at the API.
MAX_INSTRUCTION_CHARS = 2000


class Qwen3TtsInstructCloudEngine(DashScopeEngine, Engine):
    engine_id = "qwen3-tts-instruct-cloud"
    display_name = "Qwen3-TTS 指令控制（阿里百炼 · 云端）"

    # Differences from the shared DashScope flow (ADR-0015 decision 4):
    target_model = TARGET_MODEL
    enroll_prefix = "ins"  # preferred_name prefix for the enrollment call
    price_per_10k_chars = PRICE_PER_10K_CHARS

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

    # -- synthesis (shared flow; only the wire knobs differ) ---------------------

    def _synthesis_body(self, request: GenerationRequest) -> dict:
        params = request.params
        instructions = (params.get("instructions") or "").strip()
        if len(instructions) > MAX_INSTRUCTION_CHARS:
            raise CloudEngineError(
                f"合成指令过长（{len(instructions)} 字符，上限 {MAX_INSTRUCTION_CHARS}）"
            )

        body: dict = {"text": request.text, "voice": params["voice_id"]}
        if instructions:
            body["instructions"] = instructions
        # Canonical `language` (ADR-0018) mapped through THIS engine's own
        # declared to_wire adapter (legacy `language_type` key stays as a
        # rerun fallback for old records); "auto"/empty sends nothing.
        language_wire = self._language_wire(params)
        if language_wire:
            body["language_type"] = language_wire
        return body
