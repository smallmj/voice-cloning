import type { EngineInfo, EngineInstallStatus } from "../api";
import { CapabilityMatrixSection } from "../components/CapabilityMatrixSection";
import { DownloadSourcesSection } from "../components/DownloadSourcesSection";
import { EngineCard } from "../components/EngineCard";
import type { SectionId } from "../ui";

export function EnginesSection({
  baseUrl,
  token,
  engines,
  selectedEngine,
  installStatus,
  installing,
  onInstall,
  onSelect,
  onNavigate,
}: {
  baseUrl: string;
  token: string;
  engines: EngineInfo[];
  selectedEngine: string | null;
  installStatus: Record<string, EngineInstallStatus>;
  installing: Record<string, boolean>;
  onInstall: (id: string) => void;
  onSelect: (id: string) => void;
  onNavigate: (section: SectionId) => void;
}) {
  return (
    <section>
      <h2>引擎</h2>
      <div className="hint">
        点击卡片选择用于生成的引擎；在此安装本地引擎、查看能力矩阵与回归结果。
      </div>
      <div className="engine-list">
        {engines.map((e) => (
          <EngineCard
            key={e.id}
            engine={e}
            selected={e.id === selectedEngine}
            installStatus={installStatus[e.id] ?? null}
            installing={!!installing[e.id]}
            onInstall={() => onInstall(e.id)}
            onSelect={() => onSelect(e.id)}
          />
        ))}
      </div>
      <DownloadSourcesSection baseUrl={baseUrl} token={token} />
      <CapabilityMatrixSection
        baseUrl={baseUrl}
        token={token}
        engines={engines}
        onNavigate={onNavigate}
      />
    </section>
  );
}
