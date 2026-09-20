"""Unit tests for the bundled runtime: paths, downloader, install orchestrator."""

from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path

import pytest

from voiceclone_sidecar.runtime import downloader, installer, paths


# --- paths -----------------------------------------------------------------


def test_default_runtime_root_is_ascii():
    root = paths.runtime_root()
    assert str(root).isascii()


def test_non_ascii_root_is_refused():
    with pytest.raises(paths.PathNotAsciiError):
        paths.runtime_root(env={"VOICECLONE_RUNTIME_ROOT": "/Users/小张/模型"})


def test_engine_dirs_are_isolated():
    root = Path("/tmp/fake-root")
    a = paths.engine_venv_dir(root, "engine-a")
    b = paths.engine_venv_dir(root, "engine-b")
    assert a != b
    assert a.is_relative_to(paths.engine_dir(root, "engine-a"))


# --- downloader ------------------------------------------------------------


class RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b"x" * 100_000

    def do_GET(self):
        rng = self.headers.get("Range")
        if self.path.startswith("/missing/"):
            self.send_error(404)
            return
        if rng:
            start = int(rng.removeprefix("bytes=").split("-")[0])
            chunk = self.payload[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(self.payload)-1}/{len(self.payload)}")
            self.send_header("Content-Length", str(len(chunk)))
        else:
            start = 0
            chunk = self.payload
            self.send_response(200)
            self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args):
        pass


@pytest.fixture()
def http_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_download_reports_byte_progress(http_server, tmp_path):
    events = []
    out = downloader.download_file(
        downloader.DownloadSpec(path="f.bin", dest_name="f.bin"),
        tmp_path,
        [f"{http_server}/{{path}}"],
        progress=lambda n, done, total: events.append((done, total)),
    )
    assert out.read_bytes() == RangeHandler.payload
    assert events[-1] == (100_000, 100_000)  # final byte-accurate report
    assert any(done > 0 for done, _ in events)


def test_download_resumes_from_partial(http_server, tmp_path):
    part = tmp_path / "f.bin.part"
    part.write_bytes(RangeHandler.payload[:30_000])
    seen = []

    class Recording(RangeHandler):
        def do_GET(self):
            seen.append(self.headers.get("Range"))
            super().do_GET()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Recording)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        out = downloader.download_file(
            downloader.DownloadSpec(path="f.bin", dest_name="f.bin"),
            tmp_path,
            [f"{url}/{{path}}"],
        )
    finally:
        server.shutdown()
    assert "bytes=30000-" in seen
    assert out.read_bytes() == RangeHandler.payload


def test_download_falls_back_to_second_source(http_server, tmp_path):
    out = downloader.download_file(
        downloader.DownloadSpec(path="f.bin", dest_name="f.bin"),
        tmp_path,
        [f"{http_server}/missing/{{path}}", f"{http_server}/{{path}}"],
    )
    assert out.read_bytes() == RangeHandler.payload


# --- installer ---------------------------------------------------------------


def test_installer_skips_completed_and_retries_failed(tmp_path):
    ctx = installer.InstallContext(engine_id="e", root=tmp_path)
    calls = []
    marker = tmp_path / "engines" / "e" / "artifact"

    def ok(log, progress):
        calls.append("ok")
        marker.mkdir(parents=True, exist_ok=True)
        marker.joinpath("data.bin").write_bytes(b"data")

    def boom(log, progress):
        marker.mkdir(parents=True, exist_ok=True)
        if calls.count("boom") == 0:
            calls.append("boom")
            marker.joinpath("data.bin").write_bytes(b"half")
            raise RuntimeError("interrupted mid-download")
        calls.append("boom")
        marker.joinpath("data.bin").write_bytes(b"data")

    steps = [
        installer.InstallStep("s1", "always skipped", ok),
        installer.InstallStep("s2", "fails first", boom, artifact=marker),
    ]

    # First run: s2 fails, artifacts are left for inspection.
    with pytest.raises(RuntimeError):
        installer.run_install(ctx, steps, lambda m: None)
    state = installer.load_state(ctx)
    assert state["steps"]["s1"]["status"] == "completed"
    assert state["steps"]["s2"]["status"] == "failed"
    assert marker.joinpath("data.bin").read_bytes() == b"half"

    # Retry: failed artifacts are cleaned and the step reruns; s1 is skipped.
    result = installer.run_install(ctx, steps, lambda m: None)
    assert result["installed"] is True
    assert calls == ["ok", "boom", "boom"]  # s1 skipped; s2 reruns and now passes
    assert marker.joinpath("data.bin").read_bytes() == b"data"


def test_installer_state_survives_reload(tmp_path):
    ctx = installer.InstallContext(engine_id="e", root=tmp_path)
    installer.run_install(ctx, [installer.InstallStep("s", "d", lambda l, p: None)], lambda m: None)
    reloaded = installer.load_state(
        installer.InstallContext(engine_id="e", root=tmp_path)
    )
    assert reloaded["installed"] is True
    assert json.loads((tmp_path / "engines" / "e" / "install_state.json").read_text())["installed"]


def test_download_file_skips_already_complete_file(tmp_path, monkeypatch):
    """A pre-seeded file at its full size is not re-downloaded."""
    from voiceclone_sidecar.runtime import downloader

    dest_dir = tmp_path / "w"
    dest = dest_dir / "f.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"x" * 100)

    calls = []
    monkeypatch.setattr(downloader, "remote_size", lambda url: 100)
    monkeypatch.setattr(downloader, "_download_from", lambda *a, **k: calls.append(a))

    result = downloader.download_file(
        downloader.DownloadSpec(path="f.bin", dest_name="f.bin"),
        dest_dir,
        ["https://example.com/{path}"],
    )
    assert result == dest
    assert calls == []  # no transfer happened


def test_retry_after_artifact_wipe_reruns_shared_artifact_steps(tmp_path, monkeypatch):
    """issue #25: a failed engine step wipes the shared venv; the earlier
    torch step (same artifact) must rerun on retry, not stay 'completed'."""
    import json

    from voiceclone_sidecar.runtime.installer import (
        InstallContext,
        InstallStep,
        load_state,
        run_install,
    )

    ctx = InstallContext(engine_id="e", root=tmp_path)
    venv = tmp_path / "venv"
    calls = []

    def make_step(step_id, fail=False):
        def run(log, progress):
            calls.append(step_id)
            venv.mkdir(parents=True, exist_ok=True)
            if fail:
                raise RuntimeError("boom")
        return InstallStep(step_id, step_id, run, artifact=venv)

    with pytest.raises(RuntimeError):
        run_install(ctx, [make_step("torch"), make_step("engine", fail=True)], lambda m: None)

    state = load_state(ctx)
    assert state["steps"]["torch"]["status"] == "completed"
    assert state["steps"]["engine"]["status"] == "failed"

    run_install(ctx, [make_step("torch"), make_step("engine")], lambda m: None)
    # torch reruns because its artifact (the venv) was wiped by the engine retry
    assert calls == ["torch", "engine", "torch", "engine"]
    assert load_state(ctx)["installed"] is True
