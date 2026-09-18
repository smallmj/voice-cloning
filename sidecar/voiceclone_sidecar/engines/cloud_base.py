"""Vendor-neutral cloud engine base (ADR-0015).

Formerly ``dashscope_base.py`` — a name that was vendor-specific while the
content was mostly generic. Everything every BYOK cloud engine shares lives
here exactly once: BYOK key access, the injectable HTTP client (contract
tests pass an ``httpx.MockTransport``), vendor-error rendering, WAV sample
rate probing, the "this cloud voice is gone" classifier (``voice_missing_error``,
issue #12) and the Aliyun char-billing helpers.

Vendor identity is a LABEL (:attr:`CloudEngineBase.vendor_label`), not
hardcoded copy: every user-facing message that used to name 阿里百炼 is
generated from the label, so a MiniMax engine on the same code path shows
MiniMax text. What stays vendor-specific lives in the concrete engines: base
URL, model names, payload construction and error classification.

``CloudEngineError`` defined here is THE user-facing cloud failure type —
every cloud engine raises it, and the sidecar's error mapping catches it.
"""

from __future__ import annotations

import wave
from pathlib import Path

import httpx

from ..engine_config import EngineConfig, resolve_seam


class CloudEngineError(RuntimeError):
    """A user-facing cloud failure: config problem or a vendor-side error."""


class CloudEngineBase:
    """Base for BYOK cloud engines (ADR-0003: key from the OS key store)."""

    engine_id: str
    requires_key = True
    # Vendor display label used to build every user-facing message; e.g.
    # "阿里百炼" for DashScope, "MiniMax" for MiniMax.
    vendor_label: str = "云端引擎"

    def __init__(self, output_dir: Path | None = None, key_store=None,
                 client: httpx.Client | None = None,
                 config: "EngineConfig | None" = None) -> None:
        self.key_store = key_store
        # Same seam as local engines; cloud engines currently consume only
        # output_dir from it, but the env dict is carried for uniformity.
        output_dir, env = resolve_seam(config, self.engine_id, output_dir, None)
        self.output_dir = output_dir
        self.env = env
        # Injectable client: contract tests pass an httpx.MockTransport.
        self._client = client

    # -- vendor copy ----------------------------------------------------------

    def key_missing_hint(self) -> str:
        """User-facing message for the pipeline's key-missing (409) path.

        Lives on the engine, not in the pipeline: the pipeline must not know
        which vendor an engine talks to (ADR-0015 decision 4).
        """
        return (
            f"引擎 {self.engine_id} 需要{self.vendor_label} API Key；"
            "请在「设置」页配置后再生成"
        )

    # -- key access ----------------------------------------------------------

    def _key(self) -> str:
        if self.key_store is None:
            raise CloudEngineError("应用内部错误：密钥存储未初始化")
        key = self.key_store.get(self.engine_id)
        if not key:
            raise CloudEngineError(
                f"尚未配置{self.vendor_label} API Key；请在「设置」页填入你的 API Key 后重试"
            )
        return key

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=120.0)
        return self._client

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key()}"}

    @staticmethod
    def _vendor_error(resp: httpx.Response) -> str:
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 - keep the status line at minimum
            return f"HTTP {resp.status_code}"
        msg = body.get("message") or body.get("msg") or body
        code = body.get("code")
        detail = f"{code}：{msg}" if code else str(msg)
        if resp.status_code in (401, 403):
            return f"API Key 无效或无权限（HTTP {resp.status_code}）：{detail}"
        return f"HTTP {resp.status_code}：{detail}"

    @staticmethod
    def _probe_sample_rate(path: Path, default: int) -> int:
        """Read a WAV's sample rate, falling back to `default` when the file
        does not carry a RIFF header the stdlib can parse."""
        try:
            with wave.open(str(path), "rb") as w:
                return w.getframerate()
        except wave.Error:
            return default


# -- shared cloud helpers -----------------------------------------------------


# Vendor error shapes that mean "this enrolled voice no longer exists"
# (verified against DashScope error bodies: code/message mention the voice
# and that it is missing, deleted or expired). Matched on the lowercased
# code+message blob; anything else must NOT be read as a dead voice.
VOICE_MISSING_MARKERS = (
    "不存在",
    "not exist",
    "not found",
    "已删除",
    "已被删除",
    "已过期",
    "expired",
    "deleted",
)


def voice_missing_error(resp: httpx.Response) -> bool:
    """Classify a vendor response as "the cloud voice is gone" (issue #12).

    Lives in the vendor-neutral base, not in one adapter: both the cloning
    and the voice-design engines need the same verdict, and the health-check
    caller must never rebuild a binding on an unrelated failure.
    """
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 - no parseable body: no verdict
        return False
    code = str(body.get("code") or "")
    message = str(body.get("message") or body.get("msg") or "")
    blob = f"{code} {message}".lower()
    if "voice" not in blob and "音色" not in blob:
        return False
    return any(marker in blob for marker in VOICE_MISSING_MARKERS)


def billed_chars(text: str) -> int:
    """Aliyun's char-counting rule: one CJK/full-width char = 2 chars."""
    return sum(2 if ord(c) > 127 else 1 for c in text)


def billing_cost(text: str, price_per_10k_chars: float) -> float:
    """Estimated cost under ``price_per_10k_chars`` 元/万计费字符."""
    return round(billed_chars(text) / 10000 * price_per_10k_chars, 4)
