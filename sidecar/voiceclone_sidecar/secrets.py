"""API key storage: BYOK keys live in the OS key store (ADR-0003).

Keys are never written to the app's data files or logs. The default backend
delegates to the ``keyring`` package, which resolves to the macOS Keychain,
Windows Credential Manager, or the freedesktop Secret Service depending on
platform. Tests swap in an in-memory backend.

The sidecar keeps keys in memory only for the duration of a run; they are
read back from the key store on demand. Nothing here ever returns a key
value through the HTTP contract — only "configured or not".
"""

from __future__ import annotations

import abc

SERVICE_NAME = "voiceclone-api-keys"
MAX_KEY_CHARS = 4096


class KeyStoreError(RuntimeError):
    """Raised when the OS key store cannot be reached or refuses a write."""


class KeyBackend(abc.ABC):
    """A secret store keyed by (service, account)."""

    @abc.abstractmethod
    def get(self, service: str, account: str) -> str | None: ...

    @abc.abstractmethod
    def set(self, service: str, account: str, value: str) -> None: ...

    @abc.abstractmethod
    def delete(self, service: str, account: str) -> None: ...


class MemoryBackend(KeyBackend):
    """In-memory backend for tests and as an explicit fallback target."""

    def __init__(self) -> None:
        self._secrets: dict[tuple[str, str], str] = {}

    def get(self, service: str, account: str) -> str | None:
        return self._secrets.get((service, account))

    def set(self, service: str, account: str, value: str) -> None:
        self._secrets[(service, account)] = value

    def delete(self, service: str, account: str) -> None:
        self._secrets.pop((service, account), None)


class KeyringBackend(KeyBackend):
    """Backend over the ``keyring`` package (OS key store).

    Import is lazy so that environments without ``keyring`` (or without a
    usable backend) fail with an actionable message instead of at import
    time — the rest of the sidecar keeps working without cloud engines.
    """

    def _lib(self):
        try:
            import keyring
        except ImportError as exc:  # pragma: no cover - depends on env
            raise KeyStoreError(
                "未找到 keyring 组件，无法访问系统钥匙串；请重新安装应用"
            ) from exc
        try:
            # Fail fast on "no usable backend" (headless Linux, etc.). Some
            # backends (e.g. the chainer wrapper) carry no priority — treat
            # that as usable rather than failing the whole key store.
            priority = getattr(keyring.get_keyring(), "priority", None)
            if priority is not None and priority < 1:
                raise KeyStoreError("系统没有可用的钥匙串后端")
        except KeyStoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - backend probe varies
            raise KeyStoreError(f"系统钥匙串不可用：{exc}") from exc
        return keyring

    def get(self, service: str, account: str) -> str | None:
        try:
            return self._lib().get_password(service, account)
        except KeyStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise KeyStoreError(f"读取系统钥匙串失败：{exc}") from exc

    def set(self, service: str, account: str, value: str) -> None:
        try:
            self._lib().set_password(service, account, value)
        except KeyStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise KeyStoreError(f"写入系统钥匙串失败：{exc}") from exc

    def delete(self, service: str, account: str) -> None:
        try:
            # PasswordDeleteError is raised for a missing entry on some
            # backends; deleting a non-existent key is a no-op here.
            self._lib().delete_password(service, account)
        except Exception as exc:  # noqa: BLE001 - entry-missing vs real failure
            if "not found" not in str(exc).lower() and "nopassword" not in type(exc).__name__.lower():
                raise KeyStoreError(f"删除系统钥匙串条目失败：{exc}") from exc


def _default_backend() -> KeyBackend:
    """The platform key store via keyring, or an explicit test seam.

    ``VOICECLONE_KEY_BACKEND=memory`` swaps in an in-memory backend — used by
    the sidecar contract tests so they never touch the developer's real
    keychain.
    """
    import os

    if os.environ.get("VOICECLONE_KEY_BACKEND") == "memory":
        # Issue #19: the memory backend silently drops every BYOK key on
        # exit. That is fine for tests, but a real process must never lose
        # its keys without being told loudly.
        import sys

        in_tests = "PYTEST_CURRENT_TEST" in os.environ
        allowed = os.environ.get("VOICECLONE_ALLOW_MEMORY_KEY_BACKEND") == "1"
        if not (in_tests or allowed):
            print(
                "WARNING: VOICECLONE_KEY_BACKEND=memory is active outside the "
                "test suite — BYOK API keys will NOT be persisted and are lost "
                "when this process exits. Unset the variable to use the OS key "
                "store, or set VOICECLONE_ALLOW_MEMORY_KEY_BACKEND=1 to silence "
                "this warning.",
                file=sys.stderr,
                flush=True,
            )
        return MemoryBackend()
    return KeyringBackend()


class KeyStore:
    """Engine-scoped access to BYOK API keys."""

    def __init__(self, backend: KeyBackend | None = None) -> None:
        self._backend: KeyBackend = backend if backend is not None else _default_backend()

    def set(self, engine_id: str, key: str) -> None:
        key = (key or "").strip()
        if not key:
            raise KeyStoreError("API Key 不能为空")
        if len(key) > MAX_KEY_CHARS:
            raise KeyStoreError(f"API Key 过长（最多 {MAX_KEY_CHARS} 字符）")
        self._backend.set(SERVICE_NAME, engine_id, key)

    def get(self, engine_id: str) -> str | None:
        return self._backend.get(SERVICE_NAME, engine_id)

    def delete(self, engine_id: str) -> None:
        self._backend.delete(SERVICE_NAME, engine_id)

    def status(self, engine_ids: list[str]) -> list[dict]:
        """Configured-flags for the given engines — never key values."""
        return [{"engine_id": e, "configured": self.get(e) is not None} for e in engine_ids]
