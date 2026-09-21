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


def serving_label(source_template: str) -> str:
    """Human-readable name of the source a template belongs to (ADR-0016).

    Best-effort on purpose: the downloader must never fail because labeling
    failed, and callers outside the sidecar package may pass custom
    templates (label falls back to the host).
    """
    try:
        from .. import sources as _sources

        return _sources.source_label(source_template)
    except Exception:  # noqa: BLE001 - labeling is cosmetic
        return source_template.split("/")[2] if source_template.startswith("http") else source_template


@dataclass(frozen=True)
class DownloadSpec:
    """One file. ``path`` is interpolated into each source template."""

    path: str
    dest_name: str


def _run_curl(args: list[str]) -> tuple[int, str, str | None]:
    """Run one curl attempt; return (final http code, stderr tail, effective URL).

    ADR-0016: the effective URL is what makes the "actual serving source" log
    honest — hf-mirror 308-redirects境外 traffic back to huggingface.co, and
    only ``%{url_effective}`` reveals that the bytes came from the real site,
    not the mirror the user picked.
    """
    proc = subprocess.run(
        ["curl", "--connect-timeout", "20", *args],
        capture_output=True, text=True, timeout=CURL_TIMEOUT + 30,
    )
    code = 0
    effective: str | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("DSCODE:"):
            fields = line.split(":", 1)[1].strip().split()
            if fields:
                try:
                    code = int(fields[0])
                except ValueError:
                    code = 0
            for f in fields[1:]:
                if f.startswith("DSURL:"):
                    effective = f[len("DSURL:"):]
    return code, proc.stderr.strip()[-300:], effective


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

    ADR-0016: the log always states which source ACTUALLY served the file —
    hf-mirror's failure mode is silent (a 308 back to huggingface.co shows up
    as slowness, not an error), so "preferred source" without "served by"
    leaves the user guessing.
    """
    dest = dest_dir / spec.dest_name
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for source in sources:
        url = source.format(path=spec.path)
        # Idempotent installs: a file already on disk at its full size is not
        # re-downloaded (pre-seeded media, retried installs).
        total = remote_size(url)
        if total is not None and dest.exists() and dest.stat().st_size == total:
            if log:
                log(f"{spec.dest_name}: already complete ({total} bytes), skipping")
            return dest
        # A complete `.part` from an interrupted finalize must never be fed
        # back to curl as a resume: the server answers 416 (Range Not
        # Satisfiable) for an offset at EOF and the transfer fails forever
        # (seen live 2026-09-21: hf-mirror→xet, fully-downloaded 1.19 GB file).
        if total is not None and part.exists() and part.stat().st_size == total:
            part.replace(dest)
            if log:
                log(f"{spec.dest_name}: partial download was already complete ({total} bytes), finalizing")
            return dest
        try:
            served_url = _download_from(url, part, spec.dest_name, progress, log, total=total)
            part.replace(dest)
            if log:
                log(
                    f"{spec.dest_name}: 本次由 {serving_label(served_url or url)} 提供"
                    f"（{total if total is not None else '?'} bytes）"
                )
            return dest
        except Exception as exc:  # noqa: BLE001 - try the next source
            last_error = exc
            if log:
                log(f"download of {spec.dest_name} from {url} failed: {exc}")
    raise RuntimeError(f"all sources failed for {spec.dest_name}") from last_error


def _download_from(url: str, part: Path, name: str, progress, log, total: int | None = None) -> str | None:
    if total is None:
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
            # Abort a stalled transfer (hung TCP, zero bytes) instead of
            # blocking until the full CURL_TIMEOUT: treat <10 KB/s over 30s
            # as dead and let the retry/fallback machinery take over.
            "--speed-time", "30", "--speed-limit", "10240",
            "-o", str(part),
            "-w", "\nDSCODE:%{http_code} DSURL:%{url_effective}",
            url,
        ]
        if resume:
            args[:0] = ["-C", "-"]
        elif part.exists():
            part.unlink()
        poller = threading.Thread(target=poll_progress, daemon=True)
        poller.start()
        try:
            code, stderr, effective_url = _run_curl(args)
        finally:
            stop.set()
            poller.join()
        if code == 200 and resume:
            # Server ignored Range and restarted the body — don't trust a file
            # where resumed bytes precede fresh bytes.
            if log:
                log(f"{name}: server ignored Range; restarting from scratch")
            continue
        if code == 416 and resume:
            # Range Not Satisfiable on resume = the offset is already at (or
            # past) EOF. If the part holds the full file, the download was
            # DONE and only the finalize was interrupted — treat it as
            # complete (live 2026-09-21). Otherwise the part is corrupt/too
            # long: fall through to the fresh-download attempt instead of
            # aborting the whole source.
            if total is not None and part.exists() and part.stat().st_size == total:
                if log:
                    log(f"{name}: resume hit 416 but the file is already complete ({total} bytes)")
                return effective_url or url
            if log:
                log(f"{name}: resume hit 416 (bad partial size {part.stat().st_size if part.exists() else 0}); restarting from scratch")
            continue
        if code in (200, 206) and part.exists():
            if progress:
                progress(name, part.stat().st_size, total or part.stat().st_size)
            # The redirect chain may have silently left the requested source:
            # label the bytes by where they ACTUALLY came from, falling back
            # to the attempted URL when curl did not report one.
            return effective_url or url
        raise RuntimeError(f"curl failed (http={code}): {stderr}")

    raise RuntimeError(f"unable to download {name}")
