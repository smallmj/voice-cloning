import { useState } from "react";
import type { EngineInfo, EngineInstallStatus } from "../api";
import { CAP_LABELS } from "../labels";
import { CapBadge } from "./bits";
import { InstallProgressLine, InstallStepsList } from "./InstallProgress";
import { engineDetailParagraphs } from "../ui";
import type { MatrixEngine } from "../capability-matrix";

export function EngineCard({
  engine,
  selected,
  installStatus,
  installing,
  matrixEntry,
  designOnly = false,
  onInstall,
  onSelect,
}: {
  engine: EngineInfo;
  selected: boolean;
  installStatus: EngineInstallStatus | null;
  installing: boolean;
  /** Curated capability-matrix entry powering the 「详情」 copy (issue #40). */
  matrixEntry: MatrixEngine | null;
  /** Issue #68: design-only engines are not selectable for generation. */
  designOnly?: boolean;
  onInstall: () => void;
  onSelect: () => void;
}) {
  const c = engine.capabilities;
  const installed = installStatus ? installStatus.installed : (engine.installed ?? true);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const detailParagraphs = detailsOpen ? engineDetailParagraphs(engine, matrixEntry) : [];
  // Issue #68: a design-only engine cannot be selected for generation —
  // clicks land on the card body but selection never fires.
  const handleSelect = () => {
    if (!designOnly) onSelect();
  };
  return (
    <div
      className={`engine-card ${selected ? "selected" : ""}`}
      onClick={handleSelect}
      role={designOnly ? undefined : "button"}
      aria-pressed={designOnly ? undefined : selected}
      tabIndex={0}
      onKeyDown={(ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          // Don't hijack keys meant for the card's inner controls.
          if ((ev.target as HTMLElement).closest("button, input, select, textarea")) return;
          ev.preventDefault();
          handleSelect();
        }
      }}
    >
      <strong>{engine.display_name}</strong>{" "}
      {designOnly && <span className="badge badge-on">音色设计专用</span>}
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
        <InstallStepsList steps={installStatus.steps} />
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
      {/* Issue #53: key inputs moved OUT of the engine card — one row per
          vendor in VendorKeysSection above the card list (the two Qwen3-TTS
          cloud engines share one 阿里百炼 key). The card keeps only the
          configured/missing badge. */}
      {/* Issue #40: per-engine model description lives behind 「详情」. */}
      <button
        className="details-toggle"
        aria-expanded={detailsOpen}
        onClick={(ev) => {
          ev.stopPropagation();
          setDetailsOpen((v) => !v);
        }}
      >
        {detailsOpen ? "收起详情" : "详情"}
      </button>
      {detailsOpen && (
        <div className="engine-details">
          {matrixEntry?.regression_summary && (
            <div className="engine-note">
              <span className="engine-note-label">回归</span>
              {`中文回归集 ${matrixEntry.regression_summary.ok} 成功 / ${matrixEntry.regression_summary.failed} 失败${
                matrixEntry.regression_summary.rtf_mean != null
                  ? ` · RTF 均值 ${matrixEntry.regression_summary.rtf_mean}`
                  : ""
              }`}
            </div>
          )}
          {detailParagraphs.map((p, i) => (
            <p key={i} className="engine-detail-paragraph">
              {p}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
