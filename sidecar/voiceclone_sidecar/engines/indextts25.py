"""IndexTTS-2.5 on Windows + CUDA (PyTorch) — the local engine for NVIDIA rigs.

A thin platform declaration over the shared IndexTTS-2.5 base
(indextts25_base.py, issue #32). The CUDA-specific deltas are exactly:

- torch/torchaudio are declared as direct deps of the torch step and fetched
  as EXACT wheel URLs (the explicit-index requirement, taken to its strongest
  form: no index resolution, no extra-index leaks) with a mirror fallback
  list, then installed from the local wheel files — the CUDA index can never
  leak into the rest of the install. Weights always download (no local
  checkpoint reuse on this platform).
- the worker gates on CUDA (indextts25_worker.py, which delegates to the
  shared worker runtime).

Everything else — the two-step dependency split, the weight source list
(Hugging Face → hf-mirror → ModelScope) with loud mirror failure, the
pre-seeded ``hf_cache`` aux layout, HF_HUB_OFFLINE at generation time, and
the worker supervisor (idle recycle / crash restart) — is shared with the
MPS engine in the base.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from .. import sources
from ..runtime import downloader, paths, uvman
from .indextts25_base import (
    AUX_WEIGHTS,
    ENGINE_PACKAGE_URL,
    ENGINE_PACKAGE_URL_FALLBACK,
    LANG_CHOICES,
    MAIN_WEIGHTS_FILES,
    REPO,
    TORCH_VERSION,
    IndexTts25EngineBase,
    param_specs_for,
    prepare_synthesis,
)

CUDA_TAG = "cu128"
PY_TAG = "cp311"
PLATFORM_TAG = "win_amd64"  # this engine only registers on Windows
TORCH_WHEELS = [
    f"torch-{TORCH_VERSION}%2B{CUDA_TAG}-{PY_TAG}-{PY_TAG}-{PLATFORM_TAG}.whl",
    f"torchaudio-{TORCH_VERSION}%2B{CUDA_TAG}-{PY_TAG}-{PY_TAG}-{PLATFORM_TAG}.whl",
]
# CUDA wheel sources live in the central sources module (ADR-0015): exact
# wheel URLs, no index resolution. The chain (preferred CUDA source first,
# the rest as silent fallback) is resolved at INSTALL time from the injected
# env — axis 3 is separately switchable, never merged with the others.
# Measured 2026-09-17 on the Windows rig: download.pytorch.org ~800 KB/s
# (after an initial TLS-blackhole window), aliyun ~240 KB/s per connection,
# so the official source stays first and aliyun is the fallback.

__all__ = [
    "AUX_WEIGHTS",
    "ENGINE_PACKAGE_URL",
    "ENGINE_PACKAGE_URL_FALLBACK",
    "LANG_CHOICES",
    "MAIN_WEIGHTS_FILES",
    "REPO",
    "TORCH_VERSION",
    "IndexTts25CudaEngine",
    "param_specs_for",
    "prepare_synthesis",
]


class IndexTts25CudaEngine(IndexTts25EngineBase):
    """Local engine on Windows + CUDA. Carries the InstallableEngine surface."""

    engine_id = "indextts-25-cuda"
    display_name = "IndexTTS-2.5（本地 · CUDA）"
    worker_filename = "indextts25_worker.py"
    gate_label = "CUDA"
    local_weights_env = None  # CUDA always downloads; no checkpoint reuse

    torch_step_label = f"安装 torch/torchaudio {TORCH_VERSION}（CUDA {CUDA_TAG}，显式 wheel 源）"
    weights_step_label = f"下载引擎权重 {REPO}"
    aux_step_label = "下载辅助模型（w2v-bert / MaskGCT / CAMPPlus / BigVGAN）"

    def _step_torch(self, uv, venv: Path, log, progress) -> None:
        wheels_dir = paths.engine_dir(self.root, self.engine_id) / "wheels"
        torch_chain = sources.torch_wheel_sources(
            CUDA_TAG, preferred=self.env.get(sources.PREFERRED_CUDA_ENV)
        )
        local_wheels = []
        for wheel in TORCH_WHEELS:
            log(f"torch: fetching {wheel}")
            local_wheels.append(
                # URL name carries %2B; the local file must carry a literal
                # "+" or uv rejects the wheel filename.
                downloader.download_file(
                    downloader.DownloadSpec(path=wheel, dest_name=urllib.parse.unquote(wheel)),
                    wheels_dir, torch_chain,
                    progress=lambda name, done, total, _w=wheel: progress(f"torch:{_w}", done, total),
                    log=log,
                )
            )
        # Install from the local wheel files — no index is consulted at all.
        uvman.pip_install(uv, venv, [str(w) for w in local_wheels], env=self.env, log=log)
