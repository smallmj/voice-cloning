"""Single source of the sidecar version string.

Imported by ``main`` (health/contract) and ``aigc`` (ISFT metadata) so the
two never drift apart — no circular import, no comment-kept copies.
"""

SIDECAR_VERSION = "1.2.0"
