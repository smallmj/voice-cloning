import type { EngineInfo, EngineInstallStatus } from "../api";
import { CAP_LABELS, STEP_LABELS } from "../labels";
import { CapBadge } from "./bits";
import { formatBytes, progressPercent } from "../install-progress";

function InstallProgressLine({
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

export function EngineCard({
  engine,
  selected,
  installStatus,
  installing,
  onInstall,
  onSelect,
}: {
  engine: EngineInfo;
  selected: boolean;
  installStatus: EngineInstallStatus | null;
  installing: boolean;
  onInstall: () => void;
  onSelect: () => void;
}) {
  const c = engine.capabilities;
  const installed = installStatus ? installStatus.installed : (engine.installed ?? true);
  return (
    <div
      className={`engine-card ${selected ? "selected" : ""}`}
      onClick={onSelect}
      role="button"
      aria-pressed={selected}
      tabIndex={0}
      onKeyDown={(ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          onSelect();
        }
      }}
    >
      <strong>{engine.display_name}</strong>{" "}
      {engine.requires_key && (
        <span className={`badge ${engine.key_configured ? "badge-on" : "badge-off"}`}>
          {engine.key_configured ? "API Key 已配置" : "未配置 API Key"}
        </span>
      )}
      {installed ? (
        <span className="badge badge-on">已就绪</span>
      ) : (
        <button
          className="install-btn"
          onClick={(ev) => {
            ev.stopPropagation();
            onInstall();
          }}
          disabled={installing}
        >
          {installing ? "安装中…" : "安装引擎"}
        </button>
      )}
      {installing && installStatus?.progress && (
        <InstallProgressLine progress={installStatus.progress} />
      )}
      {!installed && installStatus && Object.keys(installStatus.steps).length > 0 && (
        <div className="install-steps">
          {Object.entries(installStatus.steps).map(([id, s]) => (
            <div key={id} className={`install-step step-${s.status}`}>
              {STEP_LABELS[id] ?? id}：{s.status}
              {s.status === "failed" && s.error ? `（${s.error}）` : ""}
            </div>
          ))}
        </div>
      )}
      <div className="badges">
        <CapBadge
          label={`${CAP_LABELS.languages}:${c.languages.join("/") || "—"}`}
          on={c.languages.length > 0}
        />
        <CapBadge label={CAP_LABELS.voice_cloning} on={c.voice_cloning} />
        <CapBadge label={CAP_LABELS.voice_design} on={c.voice_design} />
        <CapBadge label={CAP_LABELS.emotion} on={c.emotion} />
        <CapBadge label={CAP_LABELS.commercial_license} on={c.commercial_license} />
        <CapBadge label={CAP_LABELS.upload_used_for_training} on={c.upload_used_for_training} />
      </div>
    </div>
  );
}
