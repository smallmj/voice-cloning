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

# -- ADR-0016: preferred source + silent fallback chain ------------------------

# Env knobs the download-source settings feed into the ADR-0015 seam
# (effective_env maps the settings `sources` block onto these). Engines read
# them from their injected env at INSTALL time — never as module constants,
# which would freeze the choice at import and survive no settings change.
PREFERRED_WEIGHT_ENV = "VOICECLONE_WEIGHT_SOURCE"
PREFERRED_CUDA_ENV = "VOICECLONE_CUDA_SOURCE"


def weight_pref(env: dict | None) -> str | None:
    """The preferred weight source an engine reads from its INJECTED env
    (ADR-0015) at install time — never a module constant, which would freeze
    the choice at import and survive no settings change."""
    return (env or {}).get(PREFERRED_WEIGHT_ENV)

# The three weight sources. The user picks ONE as 首选; the other two stay as
# a silent fallback chain (ADR-0016 decision 2) — the picker must stay simple
# without losing the mandated three-source degradation. ModelScope only ever
# serves repos that declare an ms_repo counterpart (see weight_sources).
WEIGHT_SOURCE_CHOICES: dict[str, dict[str, str]] = {
    "hf": {"label": "HF 官方", "template": HF_TEMPLATE},
    "hf-mirror": {"label": "hf-mirror", "template": HF_MIRROR_TEMPLATE},
    "modelscope": {"label": "ModelScope", "template": MODELSCOPE_TEMPLATE},
}
DEFAULT_WEIGHT_SOURCE = "hf"

# Axis 2 choices: the PyPI index for non-torch engine packages. Aliyun is the
# China-friendly default; PyPI official is the escape hatch when a mirror
# serves stale wheels.
PYPI_CHOICES: dict[str, dict[str, str]] = {
    "aliyun": {"label": "阿里云 PyPI 镜像", "index": "https://mirrors.aliyun.com/pypi/simple/"},
    "official": {"label": "PyPI 官方", "index": "https://pypi.org/simple/"},
}
DEFAULT_PYPI_SOURCE = "aliyun"

# Axis 3 choices: CUDA wheel hosts (exact wheel URLs, never an index —
# ADR-0002). download.pytorch.org is canonical; Aliyun answered on many China
# networks when the canonical host was TLS-blackholed (measured 2026-09-17).
CUDA_CHOICES: dict[str, dict[str, str]] = {
    "official": {
        "label": "download.pytorch.org（官方）",
        "template": "https://download.pytorch.org/whl/{cuda_tag}/{path}",
    },
    "aliyun": {
        "label": "阿里云 pytorch-wheels 镜像",
        "template": "https://mirrors.aliyun.com/pytorch-wheels/{cuda_tag}/{path}",
    },
}
DEFAULT_CUDA_SOURCE = "official"


def order_ids(choices: dict, preferred: str | None) -> list[str]:
    """Source ids with the preferred one FIRST, the rest in default order.

    An unknown/None preferred value is ignored (defaults apply) — the sources
    layer never invents a source the catalog does not know. The fallback
    order is FIXED, never auto-re-ranked by speed: ADR-0016 §后果 — automatic
    reordering would make the serving source diverge from what the user
    configured, defeating the "actual source" display.
    """
    ids = list(choices)
    if preferred in choices:
        ids.remove(preferred)
        ids.insert(0, preferred)
    return ids


def weight_sources(
    repo: str, ms_repo: str | None = None, preferred: str | None = None
) -> list[str]:
    """The weight-source fallback chain for one repo, preferred source first.

    ``{path}`` stays as a literal placeholder for the downloader to
    interpolate per file. ModelScope is appended only when the engine
    declares a ModelScope counterpart (``ms_repo``); it is NOT a universal
    mirror — repos without a ModelScope twin would 404 there (verified
    2026-09: bigvgan and mlx-community/whisper-small have none). A preferred
    ``modelscope`` on such a repo silently degrades to the HF-first order —
    the source simply is not applicable.
    """
    ids = order_ids(WEIGHT_SOURCE_CHOICES, preferred)
    if "modelscope" in ids and not ms_repo:
        ids.remove("modelscope")
    out = []
    for sid in ids:
        template = WEIGHT_SOURCE_CHOICES[sid]["template"]
        url = template.replace("{repo}", repo)
        if ms_repo:
            url = url.replace("{ms_repo}", ms_repo)
        out.append(url)
    return out


def source_label(url: str) -> str:
    """Human-readable name of the source a URL belongs to.

    Matched against the known templates (preferred over host parsing so the
    log line says "ModelScope", not "modelscope.cn"). Unknown hosts fall back
    to their hostname — a custom template still produces an honest line.
    """
    base = url.split("{", 1)[0] if "{" in url else url
    for catalog in (WEIGHT_SOURCE_CHOICES, CUDA_CHOICES):
        for spec in catalog.values():
            template = spec["template"]
            prefix = template.split("{", 1)[0]
            if base.startswith(prefix):
                return spec["label"]
    from urllib.parse import urlparse

    return urlparse(url).hostname or url


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


def torch_wheel_sources(cuda_tag: str, preferred: str | None = None) -> list[str]:
    """Source templates for exact CUDA wheel URLs, ``{path}`` = wheel name.

    download.pytorch.org is canonical; the Aliyun pytorch-wheels mirror is
    the fallback that actually answers on many China networks (measured
    2026-09-17: download.pytorch.org TLS-blackholed while Aliyun responded).
    These are EXACT wheel URLs — no index is ever consulted, which is the
    strongest form of ADR-0002's explicit-index rule. ``preferred`` (an id
    from :data:`CUDA_CHOICES`) moves its host to the front of the chain.
    """
    return [
        CUDA_CHOICES[sid]["template"].replace("{cuda_tag}", cuda_tag)
        for sid in order_ids(CUDA_CHOICES, preferred)
    ]
