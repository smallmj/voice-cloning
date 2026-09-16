"""Resumable, progress-reporting, multi-source file downloader.

Used for engine weights (Hugging Face / ModelScope / hf-mirror) and for
bootstrapping the uv binary itself. Requirements from the spec:

- resume from what is already on disk via HTTP Range (a full 200 response
  restarts from zero — some mirrors ignore Range);
- report progress in bytes so the UI can render a real progress bar;
- fall back through a list of sources instead of dying on the first one.

Transfer engine is ``curl`` (bundled with macOS and Windows 10+), NOT
``urllib``: on many China networks Python's TLS ClientHello gets reset
mid-handshake (SSL UNEXPECTED_EOF) while curl's is let through. curl also
picks up system proxies and handles redirects/retries for free.

Progress is reported by polling the .part file size while curl runs — simple,
reliable, and decoupled from curl's progress meter format.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

CURL_TIMEOUT = 60 * 60  # big weights files on slow links


@dataclass(frozen=True)
class DownloadSpec:
    """One file. ``path`` is interpolated into each source template."""

    path: str
    dest_name: str


def _curl(url: str, part: Path, resume: bool) -> tuple[int, int]:
    """Run one curl attempt. Returns (http_code, exit_code).

    ``resume`` uses ``-C -`` so curl computes the offset from the .part file
    itself and fails cleanly when the server cannot serve ranges.
    """
    cmd = [
        "curl",
        "-L",
        "--fail",
        "-sS",
        "--connect-timeout", "20",
        "--max-time", str(CURL_TIMEOUT),
        "--retry", "5",
        "--retry-all-errors",
        "--retry-delay", "2",
        "-D", "-",  # response headers on stdout
        "-o", str(part),
        "-w", "\nDSCODE:%{http_code}",  # final status marker after headers
        url,
    ]
    if resume:
        cmd.insert(1, "-C")
        cmd.insert(2, "-")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CURL_TIMEOUT + 30)
    code = 0
    length: int | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("DSCODE:"):
            try:
                code = int(line.split(":", 1)[1].strip())
            except ValueError:
                code = 0
        elif line.lower().startswith("content-length:"):
            try:
                length = int(line.split(":", 1)[1].strip())
            except ValueError:
                length = None
    return code, length if length is not None else -(proc.returncode)


def download_file(
    spec: DownloadSpec,
    dest_dir: Path,
    sources: list[str],
    progress=None,
    log=None,
) -> Path:
    """Download ``spec`` into ``dest_dir`` trying each source template in order.

    ``sources`` entries are URL templates containing ``{path}``. ``progress``
    receives ``(dest_name, downloaded_bytes, total_bytes_or_None)``.
    Returns the final path; raises the last error if every source fails.
    """
    dest = dest_dir / spec.dest_name
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for source in sources:
        url = source.format(path=spec.path)
        try:
            _download_from(url, part, spec.dest_name, progress, log)
            part.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - try the next source
            last_error = exc
            if log:
                log(f"download of {spec.dest_name} from {url} failed: {exc}")
    raise RuntimeError(f"all sources failed for {spec.dest_name}") from last_error


def _download_from(url: str, part: Path, name: str, progress, log) -> None:
    stop = threading.Event()

    def poll_progress(total_hint: int | None) -> None:
        while not stop.is_set():
            done = part.stat().st_size if part.exists() else 0
            if progress:
                progress(name, done, total_hint)
            time.sleep(0.5)

    # Attempt 1: resume if a partial file exists; attempt 2: fresh download.
    for resume in ([part.exists() and part.stat().st_size > 0, False] if part.exists() else [False]):
        if resume:
            if log:
                log(f"resuming {name} at byte {part.stat().st_size}")
        elif part.exists():
            part.unlink()
        poller = threading.Thread(target=poll_progress, args=(None,), daemon=True)
        poller.start()
        try:
            http_code, extra = _curl(url, part, resume=resume)
        finally:
            stop.set()
            poller.join()
        if http_code == 200 and resume:
            # Server ignored Range and restarted the body — don't trust a file
            # where resumed bytes precede fresh bytes.
            if log:
                log(f"{name}: server ignored Range; restarting from scratch")
            continue
        if http_code in (200, 206) and part.exists():
            if progress:
                progress(name, part.stat().st_size, part.stat().st_size or None)
            return
        raise RuntimeError(f"curl failed (http={http_code}, exit={extra})")

    raise RuntimeError(f"unable to download {name}")


def remote_size(spec: DownloadSpec, sources: list[str]) -> int | None:
    """Best-effort total size of a file via ``curl -I``, for pre-flight checks."""
    for source in sources:
        url = source.format(path=spec.path)
        proc = subprocess.run(
            ["curl", "-sIL", "--connect-timeout", "20", url],
            capture_output=True, text=True, timeout=90,
        )
        if proc.returncode != 0:
            continue
        sizes = [
            int(line.split(":", 1)[1].strip())
            for line in proc.stdout.splitlines()
            if line.lower().startswith("content-length:")
            and line.split(":", 1)[1].strip().isdigit()
        ]
        if sizes:
            return sizes[-1]  # last response in the redirect chain
    return None
