import { useEffect, useState } from "react";
import type { EngineInfo } from "../api";
import {
  apiJson,
  authHeaders,
  downloadPostWithAuth,
  downloadWithAuth,
} from "../client";
import { CAP_LABELS } from "../labels";
import type { ThemePref, UiPrefs } from "../hooks";

const THEME_OPTIONS: { value: ThemePref; label: string }[] = [
  { value: "system", label: "跟随系统" },
  { value: "dark", label: "深色" },
  { value: "light", label: "浅色" },
];

export function SettingsSection({
  baseUrl,
  token,
  engines,
  prefs,
  updatePrefs,
  refreshEngines,
}: {
  baseUrl: string;
  token: string;
  engines: EngineInfo[];
  prefs: UiPrefs;
  updatePrefs: (partial: Partial<UiPrefs>) => void;
  refreshEngines: () => Promise<void>;
}) {
  // BYOK keys (issue #9): inputs + per-engine busy flags; values live in the
  // OS key store after save — the renderer never persists them.
  const [keyInputs, setKeyInputs] = useState<Record<string, string>>({});
  const [keyBusy, setKeyBusy] = useState<Record<string, boolean>>({});
  const [keyError, setKeyError] = useState<string | null>(null);
  // Transcription provider (issue #8) + local install lifecycle.
  const [transProviders, setTransProviders] = useState<{
    provider: string;
    local: { supported: boolean; installed: boolean; label: string | null };
    engines: { id: string; display_name: string }[];
  } | null>(null);
  const [localTransInstalling, setLocalTransInstalling] = useState(false);
  const [transError, setTransError] = useState<string | null>(null);
  // Issue #14: whole-library backup/restore.
  const [backupBusy, setBackupBusy] = useState(false);
  const [restoreBusy, setRestoreBusy] = useState(false);
  const [backupMessage, setBackupMessage] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);

  useEffect(() => {
    apiJson<typeof transProviders>(baseUrl, token, "/transcription/providers")
      .then((p) => setTransProviders(p))
      .catch(() => {
        /* sidecar restarting — keep the last snapshot */
      });
  }, [baseUrl, token]);

  useEffect(() => {
    if (!localTransInstalling) return;
    const t = setInterval(async () => {
      let st: { installed: boolean; installing: boolean };
      try {
        st = await apiJson<{ installed: boolean; installing: boolean }>(
          baseUrl,
          token,
          "/transcription/local/status",
        );
      } catch {
        return; // sidecar restarting — keep polling
      }
      if (!st.installing) {
        setLocalTransInstalling(false);
        apiJson<typeof transProviders>(baseUrl, token, "/transcription/providers")
          .then((p) => setTransProviders(p))
          .catch(() => undefined);
      }
    }, 1000);
    return () => clearInterval(t);
  }, [baseUrl, token, localTransInstalling]);

  async function setTransProvider(provider: string) {
    setTransError(null);
    try {
      await apiJson(baseUrl, token, "/transcription/provider", {
        method: "PUT",
        body: JSON.stringify({ provider }),
      });
      setTransProviders(
        await apiJson(baseUrl, token, "/transcription/providers"),
      );
    } catch (e) {
      setTransError(`切换转写提供方失败：${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function installLocalTranscriber() {
    if (localTransInstalling) return;
    setLocalTransInstalling(true);
    setTransError(null);
    try {
      await apiJson(baseUrl, token, "/transcription/local/install", { method: "POST" });
    } catch (e) {
      setLocalTransInstalling(false);
      setTransError(`发起本地转写安装失败：${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function saveKey(engineId: string) {
    const key = (keyInputs[engineId] ?? "").trim();
    if (!key) {
      setKeyError("API Key 不能为空");
      return;
    }
    setKeyBusy((p) => ({ ...p, [engineId]: true }));
    setKeyError(null);
    try {
      await apiJson(baseUrl, token, "/settings/keys", {
        method: "PUT",
        body: JSON.stringify({ engine_id: engineId, key }),
      });
      setKeyInputs((p) => ({ ...p, [engineId]: "" }));
      await refreshEngines();
    } catch (e) {
      setKeyError(e instanceof Error ? e.message : String(e));
    } finally {
      setKeyBusy((p) => ({ ...p, [engineId]: false }));
    }
  }

  async function deleteKey(engineId: string) {
    setKeyBusy((p) => ({ ...p, [engineId]: true }));
    setKeyError(null);
    try {
      await apiJson(baseUrl, token, `/settings/keys/${engineId}`, { method: "DELETE" });
      await refreshEngines();
    } catch (e) {
      setKeyError(e instanceof Error ? e.message : String(e));
    } finally {
      setKeyBusy((p) => ({ ...p, [engineId]: false }));
    }
  }

  async function backupLibrary() {
    if (backupBusy) return;
    setBackupBusy(true);
    setBackupMessage(null);
    try {
      const stamp = new Date().toISOString().slice(0, 10);
      await downloadPostWithAuth(`${baseUrl}/backup`, token, `voice-library-backup-${stamp}.zip`);
      setBackupMessage("备份已下载。请妥善保存备份文件。");
    } catch (err) {
      setBackupMessage(`备份失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBackupBusy(false);
    }
  }

  // ADR-0017: the index is SQLite now; this export restores the "readable
  // files" inspectability of the old JSON indexes.
  async function exportIndex() {
    if (exportBusy) return;
    setExportBusy(true);
    setBackupMessage(null);
    try {
      const stamp = new Date().toISOString().slice(0, 10);
      await downloadWithAuth(
        `${baseUrl}/diagnostics/export`, token, `voice-library-index-${stamp}.json`,
      );
      setBackupMessage("索引已导出（JSON，仅供诊断查看）。");
    } catch (err) {
      setBackupMessage(`导出失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setExportBusy(false);
    }
  }

  async function restoreLibrary(file: File) {
    if (restoreBusy) return;
    if (
      !window.confirm(
        "恢复备份会覆盖当前全部数据（音色、历史、设置）。\n确定继续吗？",
      )
    )
      return;
    setRestoreBusy(true);
    setBackupMessage(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${baseUrl}/restore`, {
        method: "POST",
        headers: authHeaders(token),
        body,
      });
      if (!res.ok) {
        setBackupMessage(`恢复失败（HTTP ${res.status}）`);
        return;
      }
      const j = (await res.json()) as { voices: number; generations: number };
      setBackupMessage(
        `恢复完成：${j.voices} 个音色、${j.generations} 条历史记录。API Key 不在备份内，仍在系统钥匙串中。`,
      );
    } catch (err) {
      setBackupMessage(`恢复失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRestoreBusy(false);
    }
  }

  return (
    <section>
      <h2>设置</h2>
      <div className="hint">
        云端引擎一律 BYOK（Bring Your Own Key）：API Key 只保存在本机系统钥匙串中，
        不落明文文件，也不经过任何第三方服务器。
      </div>

      <div className="settings-engine">
        <div className="settings-engine-head">
          <strong>外观</strong>
        </div>
        <div className="voice-picker">
          <label>
            主题：
            <select
              value={prefs.theme}
              onChange={(e) => updatePrefs({ theme: e.target.value as ThemePref })}
            >
              {THEME_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="settings-engine">
        <div className="settings-engine-head">
          <strong>备份与恢复</strong>
        </div>
        <div className="hint">
          整库备份把全部数据（音色、参考音频、历史记录、设置）打包成一个 .zip，
          可拷贝到其他机器后恢复。API Key 不包含在备份中。不会自动云同步。
        </div>
        <div className="key-row">
          <button disabled={backupBusy} onClick={() => void backupLibrary()}>
            {backupBusy ? "备份中…" : "备份整库"}
          </button>
          <button disabled={exportBusy} onClick={() => void exportIndex()}>
            {exportBusy ? "导出中…" : "导出索引（诊断）"}
          </button>
          <label>
            恢复备份（.zip）：
            <input
              type="file"
              accept=".zip"
              disabled={restoreBusy}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void restoreLibrary(f);
                e.target.value = "";
              }}
            />
          </label>
        </div>
        {backupMessage && <div className="hint">{backupMessage}</div>}
      </div>

      <div className="settings-engine">
        <div className="settings-engine-head">
          <strong>转写设置</strong>
        </div>
        <div className="voice-picker">
          <label>
            参考音频转写：
            <select
              value={transProviders?.provider ?? "local"}
              onChange={(e) => void setTransProvider(e.target.value)}
            >
              <option value="local">
                本地转写（{transProviders?.local.label ?? "mlx-whisper / faster-whisper"}）
              </option>
              {(transProviders?.engines ?? []).map((e) => (
                <option key={e.id} value={e.id}>
                  云端转写：{e.display_name}
                </option>
              ))}
            </select>
          </label>
          {transProviders?.provider === "local" && transProviders.local.supported && (
            <>
              <button onClick={() => void installLocalTranscriber()} disabled={localTransInstalling}>
                {localTransInstalling
                  ? "安装中…"
                  : transProviders.local.installed
                    ? "重新安装"
                    : "安装本地转写"}
              </button>
              <span className="hint">
                {localTransInstalling
                  ? "正在下载并安装本地转写模型，请稍候…"
                  : transProviders.local.installed
                    ? "本地转写已就绪（离线可用）。"
                    : "本地转写尚未安装；安装后转写与参考文本自动补全均离线完成。"}
              </span>
            </>
          )}
        </div>
        {transError && <div className="error">{transError}</div>}
      </div>

      <div className="settings-list">
        {engines.map((e) => (
          <div className="settings-engine" key={e.id}>
            <div className="settings-engine-head">
              <strong>{e.display_name}</strong>
              {e.requires_key ? (
                <span className={`badge ${e.key_configured ? "badge-on" : "badge-off"}`}>
                  {e.key_configured ? "API Key 已配置（存于系统钥匙串）" : "未配置 API Key"}
                </span>
              ) : (
                <span className="badge badge-on">本地引擎，无需密钥</span>
              )}
            </div>
            {e.requires_key && (
              <div className="key-row">
                <input
                  type="password"
                  autoComplete="off"
                  placeholder="粘贴 API Key（DashScope，sk-…）"
                  value={keyInputs[e.id] ?? ""}
                  onChange={(ev) =>
                    setKeyInputs((p) => ({ ...p, [e.id]: ev.target.value }))
                  }
                />
                <button
                  disabled={keyBusy[e.id] || !(keyInputs[e.id] ?? "").trim()}
                  onClick={() => void saveKey(e.id)}
                >
                  {keyBusy[e.id] ? "保存中…" : "保存"}
                </button>
                {e.key_configured && (
                  <button disabled={keyBusy[e.id]} onClick={() => void deleteKey(e.id)}>
                    删除
                  </button>
                )}
              </div>
            )}
            <div className="engine-note">
              <span className="engine-note-label">能力</span>
              {`${CAP_LABELS.languages}：${e.capabilities.languages.join("/") || "—"} · `}
              {(
                ["voice_cloning", "voice_design", "pronunciation_control", "emotion"] as const
              )
                .map((k) => `${CAP_LABELS[k]} ${e.capabilities[k] ? "✓" : "✗（不支持）"}`)
                .join(" · ")}
            </div>
            {e.billing_note && (
              <div className="engine-note">
                <span className="engine-note-label">计费口径</span>
                {e.billing_note}
              </div>
            )}
            {e.data_usage_note && (
              <div className="engine-note">
                <span className="engine-note-label">数据与训练</span>
                {e.data_usage_note}
              </div>
            )}
          </div>
        ))}
      </div>
      {keyError && <div className="error">{keyError}</div>}
    </section>
  );
}
