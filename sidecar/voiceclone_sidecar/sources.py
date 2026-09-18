"""Central resolution of the three download axes (ADR-0015 decision 2).

Every engine used to carry its own module-level source constants and its own
``_runtime_env()`` copy; four files each re-implemented the fallback chain
slightly differently. This module is now the ONLY place that turns "which
file do I need" into "which URLs serve it":

- **weight sources** — Hugging Face → hf-mirror → ModelScope templates;
- **package index** — the PyPI index (or mirror) used by ``uv pip install``;
- **CUDA wheel sources** — the explicit PyTorch CUDA wheel URLs.

ADR-0002's two hard rules are enforced by shape here and cannot be bypassed
by an engine: CUDA wheels are fetched as EXACT wheel URLs (no index
resolution at all), and the package install path never emits
``--extra-index-url`` (uvman only passes ``UV_DEFAULT_INDEX``). Mixing the
axes into one "mirror" switch would re-create the torch accident recorded in
plan §3: transitive dependencies resolving through a CUDA index downgraded
CUDA torch to a CPU build (29 s of audio took 700 s).
"""

from __future__ import annotations

from .engine_config import BUILTIN_DEFAULTS

# -- axis 1: weight sources ---------------------------------------------------
# Templates interpolate {repo} (Hugging Face repo id), {ms_repo} (ModelScope
# model id) and {path} (file path inside the repo). Mirror failure is LOUD:
# the downloader surfaces per-source errors instead of silently degrading.
HF_TEMPLATE = "https://huggingface.co/{repo}/resolve/main/{path}"
HF_MIRROR_TEMPLATE = "https://hf-mirror.com/{repo}/resolve/main/{path}"
MODELSCOPE_TEMPLATE = "https://modelscope.cn/models/{ms_repo}/resolve/master/{path}"

# Endpoint used by huggingface_hub-based downloads (snapshot_download), whose
# constants snapshot at import time — the caller injects it into the CHILD's
# environment. Falls back to the same mirror as the weight-source chain.
HF_ENDPOINT_DEFAULT = "https://huggingface.co"
HF_ENDPOINT_MIRROR = "https://hf-mirror.com"


def weight_sources(repo: str, ms_repo: str | None = None) -> list[str]:
    """The weight-source fallback chain for one repo.

    ``{path}`` stays as a literal placeholder for the downloader to
    interpolate per file. ModelScope is appended only when the engine
    declares a ModelScope counterpart (``ms_repo``); it is NOT a universal
    mirror — repos without a ModelScope twin would 404 there.
    """
    sources = [HF_TEMPLATE.replace("{repo}", repo), HF_MIRROR_TEMPLATE.replace("{repo}", repo)]
    if ms_repo:
        sources.append(MODELSCOPE_TEMPLATE.replace("{ms_repo}", ms_repo))
    return sources


# -- axis 2: package index ----------------------------------------------------


def pypi_index(env: dict | None = None) -> str:
    """The PyPI index for non-torch engine packages.

    ``UV_DEFAULT_INDEX`` wins (user override), the Aliyun mirror is the
    built-in default. uv installs with ``UV_DEFAULT_INDEX`` only — never
    ``--extra-index-url`` — so transitive deps resolve through the SAME index
    and a CUDA wheel can never leak here or vice versa. Installers read the
    env dict directly; this helper exists for the resolved value (logs, UI).
    """
    return (env or {}).get("UV_DEFAULT_INDEX") or BUILTIN_DEFAULTS["UV_DEFAULT_INDEX"]


def python_install_mirror(env: dict | None = None) -> str:
    """uv-managed Python build mirror (python-build-standalone). uv reads
    ``UV_PYTHON_INSTALL_MIRROR`` from the injected env directly; this helper
    exists for callers that need the resolved value (logs, UI)."""
    return (env or {}).get("UV_PYTHON_INSTALL_MIRROR") or BUILTIN_DEFAULTS[
        "UV_PYTHON_INSTALL_MIRROR"
    ]


# -- axis 3: CUDA wheel sources -----------------------------------------------


def torch_wheel_sources(cuda_tag: str) -> list[str]:
    """Source templates for exact CUDA wheel URLs, ``{path}`` = wheel name.

    download.pytorch.org is canonical; the Aliyun pytorch-wheels mirror is
    the fallback that actually answers on many China networks (measured
    2026-09-17: download.pytorch.org TLS-blackholed while Aliyun responded).
    These are EXACT wheel URLs — no index is ever consulted, which is the
    strongest form of ADR-0002's explicit-index rule.
    """
    return [
        f"https://download.pytorch.org/whl/{cuda_tag}/{{path}}",
        f"https://mirrors.aliyun.com/pytorch-wheels/{cuda_tag}/{{path}}",
    ]
