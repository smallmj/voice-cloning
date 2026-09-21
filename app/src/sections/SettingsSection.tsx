import type { TranscriptionEngineInfo } from "../api";
import { useEffect, useState } from "react";
import {
  apiJson,
  authHeaders,
  downloadPostWithAuth,
  downloadWithAuth,
} from "../client";
import type { ThemePref, UiPrefs } from "../hooks";
import { JumpLink } from "../components/bits";
import type { SectionId } from "../ui";

const THEME_OPTIONS: { value: ThemePref; label: string }[] = [
  { value: "system", label: "跟随系统" },
  { value: "dark", label: "深色" },
  { value: "light", label: "浅色" },
];

export function SettingsSection({
  baseUrl,
  token,
  prefs,
  updatePrefs,
  onNavigate,
}: {
  baseUrl: string;
  token: string;
  prefs: UiPrefs;
  updatePrefs: (partial: Partial<UiPrefs>) => void;
  onNavigate: (section: SectionId) => void;
}) {
  // Transcription provider (issue #8; issue #42: the install entry moved to
  // the 引擎 page's transcription-tool card — this page keeps only the
  // provider choice and a guidance hint).
  const [transProviders, setTransProviders] = useState<{
    provider: string;
    local: { supported: boolean; installed: boolean; label: string | null };
    engines: TranscriptionEngineInfo[];
  } | null>(null);
  const [transError, setTransError] = useState<string | null>(null);
  // Issue #14: whole-library backup/restore.
  const [backupBusy, setBackupBusy] = useState(false);
  const [restoreBusy, setRestoreBusy] = useState(false);
  const [backupMessage, setBackupMessage] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  // Issue #33: warn BEFORE transcribing when the selected cloud provider's
  // key is absent — the provider reuses an engine's key, it never has its own.
  const selectedTransEngineMissingKey = (transProviders?.engines ?? []).some(
    (e) => e.id === transProviders?.provider && e.requires_key && !e.key_configured,
  );

  useEffect(() => {
    apiJson<typeof transProviders>(baseUrl, token, "/transcription/providers")
      .then((p) => setTransProviders(p))
      .catch(() => {
        /* sidecar restarting — keep the last snapshot */
      });
  }, [baseUrl, token]);

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
        软件级设置：外观、备份恢复、转写提供方。引擎与转写工具的密钥、安装入口都在「引擎」页的卡片里
        （云端引擎一律 BYOK：API Key 只保存在本机系统钥匙串中）。
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
                  {e.requires_key && !e.key_configured ? "（未配置 API Key）" : ""}
                </option>
              ))}
            </select>
          </label>
          {selectedTransEngineMissingKey && (
            <span className="hint">
              该云端转写使用已接入引擎的 API Key（不新增密钥）；请先到「引擎」页该引擎卡片内配置该厂商的 API Key。
            </span>
          )}
          {/* Issue #42: no install button here — the 引擎 page's transcription
              tool card is the only install entry. Only guidance remains. */}
          {transProviders?.provider === "local" &&
            transProviders.local.supported &&
            !transProviders.local.installed && (
              <span className="hint">
                本地转写尚未安装；
                <JumpLink
                  target="engines"
                  label="到「引擎」页的转写工具卡片安装"
                  onNavigate={onNavigate}
                />
                ，安装后转写与参考文本自动补全均离线完成。
              </span>
            )}
        </div>
        {transError && <div className="error">{transError}</div>}
      </div>

    </section>
  );
}
