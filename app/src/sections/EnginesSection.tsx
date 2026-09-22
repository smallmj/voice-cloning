import { useEffect, useState } from "react";
import type { EngineInfo, EngineInstallStatus } from "../api";
import { apiJson } from "../client";
import {
  matrixEntryFor,
  useCapabilityMatrix,
} from "../capability-matrix";
import { CapabilityMatrixSection } from "../components/CapabilityMatrixSection";
import { EngineCard } from "../components/EngineCard";
import { TranscribeToolCard } from "../components/TranscribeToolCard";
import { VendorKeysSection } from "../components/VendorKeysSection";
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
  refreshEngines,
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
  refreshEngines: () => Promise<void>;
}) {
  const matrix = useCapabilityMatrix(baseUrl, token);
  // Vendor-level BYOK keys (issue #53): one input per vendor, applied to
  // every engine of that vendor via the same per-engine keychain endpoints.
  const [keyBusy, setKeyBusy] = useState(false);
  const [keyError, setKeyError] = useState<string | null>(null);

  async function saveKey(engineIds: string[], key: string) {
    if (!key) {
      setKeyError("API Key 不能为空");
      return;
    }
    setKeyBusy(true);
    setKeyError(null);
    try {
      for (const engineId of engineIds) {
        await apiJson(baseUrl, token, "/settings/keys", {
          method: "PUT",
          body: JSON.stringify({ engine_id: engineId, key }),
        });
      }
      await refreshEngines();
    } catch (e) {
      setKeyError(e instanceof Error ? e.message : String(e));
    } finally {
      setKeyBusy(false);
    }
  }

  async function deleteKey(engineIds: string[]) {
    setKeyBusy(true);
    setKeyError(null);
    try {
      for (const engineId of engineIds) {
        await apiJson(baseUrl, token, `/settings/keys/${engineId}`, { method: "DELETE" });
      }
      await refreshEngines();
    } catch (e) {
      setKeyError(e instanceof Error ? e.message : String(e));
    } finally {
      setKeyBusy(false);
    }
  }

  return (
    <section>
      <h2>引擎</h2>
      <div className="hint">
        点击卡片选择用于生成的引擎；在上方「厂商 API Key」区填写云端密钥、安装本地引擎、查看能力矩阵。
      </div>
      <VendorKeysSection
        engines={engines}
        onSaveKey={saveKey}
        onDeleteKey={deleteKey}
        keyBusy={keyBusy}
        keyError={keyError}
      />
      <div className="engine-list">
        {engines.map((e) => (
          <EngineCard
            key={e.id}
            engine={e}
            selected={e.id === selectedEngine}
            installStatus={installStatus[e.id] ?? null}
            installing={!!installing[e.id]}
            matrixEntry={matrixEntryFor(matrix, e.id)}
            onInstall={() => onInstall(e.id)}
            onSelect={() => onSelect(e.id)}
          />
        ))}
      </div>
      {/* Issue #42: the transcription tool's ONLY install entry in the app. */}
      <TranscribeToolCard baseUrl={baseUrl} token={token} />
      <RegressionAdvancedArea baseUrl={baseUrl} token={token} />
      {/* Issue #44: 默认下载源 moved to the settings page — the 引擎 page
          keeps only per-engine cards (keys, installs) and the matrix. */}
      <CapabilityMatrixSection baseUrl={baseUrl} token={token} engines={engines} onNavigate={onNavigate} />
    </section>
  );
}

// ---------------------------------------------------------------------------
// Issue #40: 「运行中文回归集」 moved here from the capability-matrix block,
// demoted into an advanced area with an explanation of what it runs and
// where results land.
// ---------------------------------------------------------------------------

function RegressionAdvancedArea({
  baseUrl,
  token,
}: {
  baseUrl: string;
  token: string;
}) {
  const [regression, setRegression] = useState<{ id: string; status: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The suite runs on a background sidecar thread: after starting it, poll
  // until it completes, then point the user at the matrix for the results.
  useEffect(() => {
    if (!regression || regression.status !== "running") return;
    const t = setTimeout(async () => {
      try {
        setRegression(
          await apiJson<{ id: string; status: string }>(
            baseUrl,
            token,
            `/regression/${regression.id}`,
          ),
        );
      } catch {
        /* keep polling */
      }
    }, 1500);
    return () => clearTimeout(t);
  }, [regression, baseUrl, token]);

  async function runRegression() {
    setBusy(true);
    setError(null);
    try {
      setRegression(
        await apiJson<{ id: string; status: string }>(baseUrl, token, "/regression/run", {
          method: "POST",
          body: JSON.stringify({}),
        }),
      );
    } catch (err) {
      setError(`启动回归失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="engines-advanced">
      <div className="settings-engine-head">
        <strong>高级</strong>
      </div>
      <div className="hint">
        「运行中文回归集」会逐引擎生成一组中文难例文本（数字 / 日期 / 金额 /
        中英混读 / 多音字），验证各引擎的中文基础读法。它在后台运行，耗时取决于引擎数量；
        完成后结果自动合入下方能力矩阵（成功/失败计数与 RTF 均值显示在各引擎行），
        不产生可保存的音频，也不影响你的音色与历史记录。
      </div>
      <div className="key-row">
        <button disabled={busy} onClick={() => void runRegression()}>
          {busy ? "启动中…" : "运行中文回归集"}
        </button>
        {regression && (
          <span className="hint">
            {regression.status === "running"
              ? "回归运行中…（引擎逐条生成，完成后自动合入能力矩阵）"
              : "最近一次回归已完成，结果已合入能力矩阵。"}
          </span>
        )}
      </div>
      {error && <div className="error">{error}</div>}
    </div>
  );
}
