"""Shared DashScope (阿里百炼) client base for the Qwen cloud engines.

One vendor, one adapter layer (ADR-0006): BYOK key access, the HTTP client,
auth headers and vendor-error rendering live here exactly once. The
`CloudEngineError` defined here is THE user-facing cloud failure type —
every cloud engine raises it, and the sidecar's error mapping catches it.
"""

from __future__ import annotations

import wave
from pathlib import Path

import httpx

BASE_URL = "https://dashscope.aliyuncs.com/api/v1"


class CloudEngineError(RuntimeError):
    """A user-facing cloud failure: config problem or a vendor-side error."""


class DashScopeEngine:
    """Base for engines calling dashscope.aliyuncs.com with a BYOK key."""

    engine_id: str
    requires_key = True

    def __init__(self, output_dir: Path | None = None, key_store=None,
                 client: httpx.Client | None = None) -> None:
        self.output_dir = output_dir
        self.key_store = key_store
        # Injectable client: contract tests pass an httpx.MockTransport.
        self._client = client

    # -- key access ----------------------------------------------------------

    def _key(self) -> str:
        if self.key_store is None:
            raise CloudEngineError("应用内部错误：密钥存储未初始化")
        key = self.key_store.get(self.engine_id)
        if not key:
            raise CloudEngineError(
                "尚未配置阿里百炼 API Key；请在「设置」页填入你的 API Key 后重试"
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
