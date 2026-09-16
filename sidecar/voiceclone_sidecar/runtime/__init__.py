"""Bundled-runtime support: ASCII-safe paths, resumable downloads, uv-managed
Python, per-engine isolated venvs and a per-step install orchestrator.

ADR-0002 rules enforced here (not just documented):
- the runtime root must be pure ASCII (non-ASCII paths break CUDA/ninja builds
  and silently degrade);
- every engine gets its own venv — engines never share dependencies;
- a failed install must not wedge the app: its artifacts are cleaned and the
  retry starts from a defined state.
"""
