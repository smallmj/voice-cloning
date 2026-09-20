"""IndexTTS-2.5 on Apple Silicon (MPS, PyTorch) — the local engine for Macs.

A thin platform declaration over the shared IndexTTS-2.5 base
(indextts25_base.py, issue #32). The MPS-specific deltas are exactly:

- torch comes from the standard PyPI index — the macOS arm64 wheels bundle
  MPS support, so there is no explicit CUDA wheel URL step. Versions are
  still pinned exactly (ADR-0002 rule 1) and stay at 2.8.0, below the
  TorchAudio 2.9 TorchCodec cutover that silently clips WAV output.
- the worker gates on ``torch.backends.mps`` and pins ``device="mps"`` with
  fp32 (the only precision verified end to end on Apple Silicon); see
  indextts25_mps_worker.py, which delegates to the shared worker runtime.
- the weights step reuses weights that already exist on the machine before
  touching the network: point ``VOICECLONE_INDEXTTS25_LOCAL_WEIGHTS`` at an
  existing IndexTTS-2.5 checkpoint directory (e.g. a prior full clone) and
  every matching file is copied in instead of downloaded. The downloader's
  skip-if-complete logic covers whatever remains.

Everything else — the two-step dependency split, the weight source list
(Hugging Face → hf-mirror → ModelScope) with loud mirror failure, the
pre-seeded ``hf_cache`` aux layout, HF_HUB_OFFLINE at generation time, and
the worker supervisor (idle recycle / crash restart) — is shared with the
CUDA engine in the base.
"""

from __future__ import annotations

from ..runtime import uvman
from .indextts25_base import TORCH_VERSION, IndexTts25EngineBase
from .indextts25_mps_worker import MPS_GATE_EXIT_CODE  # noqa: F401 - re-exported for tests

LOCAL_WEIGHTS_ENV = "VOICECLONE_INDEXTTS25_LOCAL_WEIGHTS"


class IndexTts25MpsEngine(IndexTts25EngineBase):
    """Local engine for Apple Silicon. Carries the InstallableEngine surface."""

    engine_id = "indextts-25-mps"
    display_name = "IndexTTS-2.5（本地 · MPS）"
    worker_filename = "indextts25_mps_worker.py"
    gate_label = "MPS"
    local_weights_env = LOCAL_WEIGHTS_ENV

    torch_step_label = f"安装 torch/torchaudio {TORCH_VERSION}（PyPI，macOS arm64 自带 MPS）"
    weights_step_label = "准备引擎权重 IndexTeam/IndexTTS-2.5（本地复用优先）"
    aux_step_label = "准备辅助模型（w2v-bert / MaskGCT / CAMPPlus / BigVGAN，本地复用优先）"

    def _step_torch(self, uv, venv, log, progress) -> None:
        # macOS arm64 wheels from the configured index bundle MPS — no
        # explicit CUDA wheel URLs, but versions stay pinned exactly.
        uvman.pip_install(
            uv, venv,
            [f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}"],
            env=self.env, log=log,
        )
