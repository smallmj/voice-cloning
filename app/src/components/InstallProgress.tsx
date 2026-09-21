/** Shared install-progress rendering (issues #35 + #42).

Both engine installs and the transcription-tool card render the byte-level
``progress: {file, done_bytes, total_bytes}`` structure the sidecar persists,
plus the per-step status list, with exactly these helpers so the two surfaces
never drift apart.
*/

import type { EngineInstallStatus } from "../api";
import { STEP_LABELS } from "../labels";
import { formatBytes, progressPercent } from "../install-progress";

export function InstallProgressLine({
  progress,
}: {
  progress: NonNullable<EngineInstallStatus["progress"]>;
}) {
  const pct = progressPercent(progress);
  return (
    <div
      className="install-progress"
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct ?? undefined}
    >
      <div className="install-progress-bar">
        <div className="install-progress-fill" style={{ width: `${pct ?? 0}%` }} />
      </div>
      <div className="install-progress-text">
        {pct !== null ? `${pct}%` : "下载中…"} · {progress.file}
        {`（${formatBytes(progress.done_bytes)}${
          progress.total_bytes !== null ? ` / ${formatBytes(progress.total_bytes)}` : ""
        }）`}
      </div>
    </div>
  );
}

export function InstallStepsList({ steps }: { steps: EngineInstallStatus["steps"] }) {
  return (
    <div className="install-steps">
      {Object.entries(steps).map(([id, s]) => (
        <div key={id} className={`install-step step-${s.status}`}>
          {STEP_LABELS[id] ?? id}：{s.status}
          {s.status === "failed" && s.error ? `（${s.error}）` : ""}
        </div>
      ))}
    </div>
  );
}
