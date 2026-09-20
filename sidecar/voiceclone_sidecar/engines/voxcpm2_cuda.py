"""VoxCPM2 on Windows + CUDA — the local engine for NVIDIA rigs (issue #25).

A thin platform declaration over the shared VoxCPM2 base:

- torch/torchaudio are fetched as EXACT CUDA wheel URLs (ADR-0002 rule 2:
  no index resolution at all) with the central mirror chain, then installed
  from the local wheel files — identical to the IndexTTS-2.5 CUDA variant;
- the worker gates on ``torch.cuda.is_available()``.

No flash-attn, no pynini, no compiled wheels in the dependency set
(verified against the upstream pyproject) — a plain CUDA torch install
works. Community reports on Windows (#184, #258) are recorded in the
capability matrix; the RTX 3090 measurement is pending.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from .. import sources
from ..runtime import downloader, paths, uvman
from .voxcpm2_base import TORCH_VERSION, VoxCPM2EngineBase

CUDA_TAG = "cu128"
PY_TAG = "cp311"
PLATFORM_TAG = "win_amd64"  # this engine only registers on Windows
TORCH_WHEELS = [
    f"torch-{TORCH_VERSION}%2B{CUDA_TAG}-{PY_TAG}-{PY_TAG}-{PLATFORM_TAG}.whl",
    f"torchaudio-{TORCH_VERSION}%2B{CUDA_TAG}-{PY_TAG}-{PY_TAG}-{PLATFORM_TAG}.whl",
]


class VoxCPM2CudaEngine(VoxCPM2EngineBase):
    """Local engine on Windows + CUDA. Carries the InstallableEngine surface."""

    engine_id = "voxcpm2-cuda"
    display_name = "VoxCPM2（本地 · CUDA）"
    worker_gate = "cuda"
    gate_label = "CUDA"

    torch_step_label = f"安装 torch/torchaudio {TORCH_VERSION}（CUDA {CUDA_TAG}，显式 wheel 源）"
    weights_step_label = "下载引擎权重 openbmb/VoxCPM2（约 5 GB）"

    def _step_torch(self, uv, venv: Path, log, progress) -> None:
        wheels_dir = paths.engine_dir(self.root, self.engine_id) / "wheels"
        torch_chain = sources.torch_wheel_sources(
            CUDA_TAG, preferred=self.env.get(sources.PREFERRED_CUDA_ENV)
        )
        local_wheels = []
        for wheel in TORCH_WHEELS:
            log(f"torch: fetching {wheel}")
            local_wheels.append(
                downloader.download_file(
                    downloader.DownloadSpec(path=wheel, dest_name=urllib.parse.unquote(wheel)),
                    wheels_dir, torch_chain,
                    progress=lambda name, done, total, _w=wheel: progress(f"torch:{_w}", done, total),
                    log=log,
                )
            )
        uvman.pip_install(uv, venv, [str(w) for w in local_wheels], env=self.env, log=log)
