"""VoxCPM2 on Apple Silicon (MPS) — the local engine for Macs (issue #25).

A thin platform declaration over the shared VoxCPM2 base (voxcpm2_base.py):

- torch comes from the standard PyPI index pinned at 2.8.0 (macOS arm64
  wheels bundle MPS); upstream officially supports MPS (device auto → MPS),
  and its loader forces float32 on MPS (bfloat16/float16 are documented as
  glitched in the diffusion AR loop) — see the capability-matrix entry for
  the two known MPS issues (#232 closed, #301 unverified here);
- the worker gates on ``torch.backends.mps`` and pins ``device="mps"``.

Everything else — install steps, weight chain (HF → hf-mirror → ModelScope),
worker supervision, and the three-mode synthesis flow — lives in the base.
"""

from __future__ import annotations

from ..runtime import uvman
from .voxcpm2_base import TORCH_VERSION, VoxCPM2EngineBase


class VoxCPM2MpsEngine(VoxCPM2EngineBase):
    """Local engine for Apple Silicon. Carries the InstallableEngine surface."""

    engine_id = "voxcpm2-mps"
    display_name = "VoxCPM2（本地 · MPS）"
    worker_gate = "mps"
    gate_label = "MPS"

    torch_step_label = f"安装 torch/torchaudio {TORCH_VERSION}（PyPI，macOS arm64 自带 MPS）"
    weights_step_label = "下载引擎权重 openbmb/VoxCPM2（约 5 GB）"

    def _step_torch(self, uv, venv, log, progress) -> None:
        uvman.pip_install(
            uv, venv,
            [f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}"],
            env=self.env, log=log,
        )
