"""Neutral constants shared between the IndexTTS-2.5 MPS worker script and
the sidecar-side engine module.

This module MUST stay free of side effects: the sidecar imports it for
``MPS_GATE_EXIT_CODE``, and the previous arrangement (importing the constant
out of the worker script) dragged ``indextts25_common_worker`` — which force-
sets ``os.environ["HF_HUB_OFFLINE"] = "1"`` for the worker process — into the
SIDECAR process. That poisoned the sidecar's whole environment, so every
install-time download child (e.g. the local-transcription weights step)
inherited offline mode, refused all network traffic, and failed instantly
(diagnosed 2026-09: the engine-page install card looked dead).
"""

from __future__ import annotations

# Exit code when the machine cannot serve the engine's device gate (no MPS).
MPS_GATE_EXIT_CODE = 3
