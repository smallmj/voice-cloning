"""Unit tests for uvman: Windows-safe triple resolution and pinned uv version.

Issue #29: ``uv_triple()`` used to evaluate ``os.uname()`` before the win32
branch, raising AttributeError on a clean Windows machine (os.uname is
Unix-only). The download URL also hit ``releases/latest``, which is both
unreproducible and GitHub-dependent (plan §3: GitHub unavailable in CN).
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

from voiceclone_sidecar.runtime import uvman


def _without_os_uname(monkeypatch):
    """Simulate a platform without os.uname (i.e. Windows)."""
    saved = getattr(__import__("os"), "uname", None)
    if saved is not None:
        monkeypatch.delattr(__import__("os"), "uname")


def test_uv_triple_windows_does_not_touch_os_uname(monkeypatch):
    # os.uname() must never be reached on win32 (it does not exist there).
    _without_os_uname(monkeypatch)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(uvman, "platform_machine", lambda: "AMD64")
    assert uvman.uv_triple() == "x86_64-pc-windows-msvc"


def test_uv_triple_windows_arm64(monkeypatch):
    _without_os_uname(monkeypatch)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(uvman, "platform_machine", lambda: "ARM64")
    assert uvman.uv_triple() == "aarch64-pc-windows-msvc"


def test_uv_triple_macos_arm64(monkeypatch):
    monkeypatch.setattr(uvman, "platform_machine", lambda: "arm64")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert uvman.uv_triple() == "aarch64-apple-darwin"


def test_uv_triple_linux_x86_64(monkeypatch):
    monkeypatch.setattr(uvman, "platform_machine", lambda: "x86_64")
    monkeypatch.setattr(sys, "platform", "linux")
    assert uvman.uv_triple() == "x86_64-unknown-linux-gnu"


def test_uv_version_is_pinned():
    # A concrete version constant must exist — no floating releases/latest.
    version = uvman.UV_VERSION
    assert version
    parts = version.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts), version


def test_download_url_uses_pinned_version(monkeypatch, tmp_path):
    urls: list[str] = []

    def fake_download_file(spec, dest, sources, log=None):
        urls.extend(sources)
        import io
        import tarfile

        dest_dir = Path(dest)
        dest_dir.mkdir(parents=True, exist_ok=True)
        archive = dest_dir / spec.dest_name
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            data = b"#!/bin/sh\n"
            info = tarfile.TarInfo(name="uv-aarch64-apple-darwin/uv")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        archive.write_bytes(buf.getvalue())
        return archive

    monkeypatch.setattr(uvman, "download_file", fake_download_file)
    monkeypatch.setattr(uvman.shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(uvman, "platform_machine", lambda: "arm64")

    uv = uvman.find_uv(tmp_path, env={}, log=lambda *_: None)
    assert uv == tmp_path / "bin" / "uv"
    assert urls == [
        f"https://github.com/astral-sh/uv/releases/download/{uvman.UV_VERSION}/uv-aarch64-apple-darwin.tar.gz"
    ]


def test_bundled_uv_wins_over_download(tmp_path):
    bundled = tmp_path / "bin" / "uv"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("")
    bundled.chmod(0o755)
    assert uvman.find_uv(tmp_path, env={}) == bundled


def test_env_override_uv_path_wins(monkeypatch, tmp_path):
    override = tmp_path / "my-uv"
    override.write_text("")
    monkeypatch.setattr(uvman.shutil, "which", lambda name: None)
    uv = uvman.find_uv(tmp_path, env={"VOICECLONE_UV_PATH": str(override)})
    assert uv == override


def test_path_uv_used_before_download(monkeypatch, tmp_path):
    monkeypatch.setattr(uvman.shutil, "which", lambda name: str(tmp_path / "on-path-uv") if name == "uv" else None)
    uv = uvman.find_uv(tmp_path, env={})
    assert uv == tmp_path / "on-path-uv"
