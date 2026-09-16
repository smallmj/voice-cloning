"""Resumable, progress-reporting, multi-source file downloader.

Used for engine weights (Hugging Face / hf-mirror) and for bootstrapping the
uv binary itself. Requirements from the spec:

- resume from what is already on disk via HTTP Range (a full 200 response
  restarts from zero — some mirrors ignore Range);
- report progress in bytes — including the TOTAL, fetched via HEAD, so the UI
  can render a real progress bar;
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


def _run_curl(args: list[str]) -> tuple[int, str]:
    """Run one curl attempt; return (final http code, stderr tail)."""
    proc = subprocess.run(
        ["curl", "--connect-timeout", "20", *args],
        capture_output=True, text=True, timeout=CURL_TIMEOUT + 30,
    )
    code = 0
    for line in proc.stdout.splitlines():
        if line.startswith("DSCODE:"):
            try:
                code = int(line.split(":", 1)[1].strip())
            except ValueError:
                code = 0
    return code, proc.stderr.strip()[-300:]


def remote_size(url: str) -> int | None:
    """Best-effort total size of a URL via ``curl -I`` (follows redirects)."""
    proc = subprocess.run(
        ["curl", "-sIL", "--connect-timeout", "20", url],
        capture_output=True, text=True, timeout=90,
    )
    if proc.returncode != 0:
        return None
    # Walk the redirect chain: trust Content-Length only from the FINAL
    # response, and only when it actually succeeded.
    total: int | None = None
    last_status_ok = False
    for line in proc.stdout.splitlines():
        if line.startswith("HTTP/"):
            last_status_ok = line.split(" ")[1].startswith("2")
        elif line.lower().startswith("content-length:") and last_status_ok:
            value = line.split(":", 1)[1].strip()
            if value.isdigit():
                total = int(value)  # last 2xx response wins
    return total


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
    total = remote_size(url)
    stop = threading.Event()

    def poll_progress() -> None:
        while not stop.is_set():
            done = part.stat().st_size if part.exists() else 0
            if progress:
                progress(name, done, total)
            time.sleep(0.5)

    # Attempt 1: resume if a partial file exists; attempt 2: fresh download.
    attempts = [part.exists() and part.stat().st_size > 0, False] if part.exists() else [False]
    for resume in attempts:
        if resume and log:
            log(f"resuming {name} at byte {part.stat().st_size}")
        args = [
            "-L", "--fail", "-sS",
            "--max-time", str(CURL_TIMEOUT),
            "--retry", "5", "--retry-all-errors", "--retry-delay", "2",
            "-o", str(part),
            "-w", "\nDSCODE:%{http_code}",  # final status after the body
            url,
        ]
        if resume:
            args[:0] = ["-C", "-"]
        elif part.exists():
            part.unlink()
        poller = threading.Thread(target=poll_progress, daemon=True)
        poller.start()
        try:
            code, stderr = _run_curl(args)
        finally:
            stop.set()
            poller.join()
        if code == 200 and resume:
            # Server ignored Range and restarted the body — don't trust a file
            # where resumed bytes precede fresh bytes.
            if log:
                log(f"{name}: server ignored Range; restarting from scratch")
            continue
        if code in (200, 206) and part.exists():
            if progress:
                progress(name, part.stat().st_size, total or part.stat().st_size)
            return
        raise RuntimeError(f"curl failed (http={code}): {stderr}")

    raise RuntimeError(f"unable to download {name}")
