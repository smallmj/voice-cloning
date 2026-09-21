/** Shared rendering helpers for install download progress (issue #35).

The sidecar persists ``progress: {file, done_bytes, total_bytes}`` in the
install state; both engine installs and (via T2) transcription-tool installs
render it with these helpers, so the structure and formatting stay identical.
*/

import type { InstallProgress } from "./api";

export function progressPercent(p: InstallProgress): number | null {
  if (p.total_bytes === null || p.total_bytes <= 0) return null;
  return Math.min(100, Math.floor((p.done_bytes / p.total_bytes) * 100));
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
