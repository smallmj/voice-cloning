// Issue #66: 「关于 / 更新」 card in the software-level settings section
// (issue #44 structure). Renderer half of the online-update feature (spec
// #62, ADR-0020): current/latest version, manual 检查更新, the 更新渠道
// three-way selector + mirror prefix, release notes, download progress
// (InstallProgress visual pattern), and 跳过此版本. The startup popup itself
// lives in #65/main — this card only registers onNewVersionAvailable so a
// startup detection refreshes the card state.
//
// All preload access goes through ../update-client (guarded, single module).
// `initialState` is a test seam: renderToStaticMarkup-based tests and
// jsdom interaction tests start the card in a given state without waiting
// on the preload bridge.

import { useEffect, useReducer, useState } from "react";
import { formatBytes } from "../install-progress";
import {
  CHANNEL_OPTIONS,
  RELEASES_PAGE_URL,
  type UpdateEvent,
  type UpdateSettings,
  type UpdateUiAction,
  type UpdateUiState,
  type UpdaterResult,
  cancelUpdateDownload,
  checkForUpdate,
  effectiveChannelLabel,
  getUpdateStatus,
  initialUpdateState,
  onNewVersionAvailable,
  onUpdateEvent,
  releaseNotesParagraphs,
  saveUpdateSettings,
  settingsForChannelChange,
  settingsForMirrorChange,
  skipVersion,
  startUpdateDownload,
  updateReducer,
} from "../update-client";

const WARNING_MANIFEST_MISSING = "已跳过 sha512 校验（清单缺失）";

export function AboutUpdateCard({ initialState }: { initialState?: UpdateUiState }) {
  const [state, dispatch] = useReducer(
    (s: UpdateUiState, a: UpdateUiAction) => updateReducer(s, a),
    initialState ?? initialUpdateState,
  );
  const [currentVersion, setCurrentVersion] = useState<string | null>(null);
  const [settings, setSettings] = useState<UpdateSettings | null>(null);
  const [bridgeMissing, setBridgeMissing] = useState(false);
  const [skipBusy, setSkipBusy] = useState(false);

  // Load the initial status + persisted settings, and subscribe to both
  // event streams. The startup popup (#65) triggers onNewVersionAvailable —
  // here we only refresh the card.
  useEffect(() => {
    let cancelled = false;
    getUpdateStatus()
      .then((status) => {
        if (cancelled) return;
        if (!status) {
          setBridgeMissing(true);
          return;
        }
        setCurrentVersion(status.currentVersion);
        setSettings(status.settings);
        if (status.lastCheck) {
          dispatch({ type: "check-result", result: status.lastCheck });
        }
      })
      .catch(() => setBridgeMissing(true));
    const offEvents = onUpdateEvent((e: UpdateEvent) => {
      if (cancelled) return;
      if (e.type === "progress") {
        dispatch({
          type: "progress",
          percent: e.percent,
          received: e.received,
          total: e.total,
        });
      } else if (e.type === "done") {
        dispatch({ type: "done", warning: e.warning });
      } else if (e.type === "failed") {
        dispatch({ type: "failed", message: e.message });
      }
    });
    const offNewVersion = onNewVersionAvailable((r) => {
      if (cancelled) return;
      // Startup detection found a version; surface it in the card as if the
      // user had manually checked (manual 检查更新 ignores skippedTag, but a
      // push event for a skipped tag simply re-renders the known result).
      dispatch({
        type: "check-result",
        // r IS the real available result (status discriminant) — pass through.
        result: r,
      });
    });
    return () => {
      cancelled = true;
      offEvents();
      offNewVersion();
    };
  }, []);

  async function handleCheck() {
    dispatch({ type: "check-start" });
    try {
      const result = await checkForUpdate(true);
      if (result) {
        dispatch({ type: "check-result", result });
      } else {
        setBridgeMissing(true);
        dispatch({ type: "failed", message: "更新接口不可用（开发模式未加载 preload）" });
      }
    } catch (e) {
      dispatch({
        type: "failed",
        message: `检查更新失败：${e instanceof Error ? e.message : String(e)}`,
      });
    }
  }

  function persist(next: UpdateSettings) {
    setSettings(next);
    void saveUpdateSettings(next).catch(() => {
      /* persistence failure is non-fatal; keep the local value */
    });
  }

  async function handleDownload() {
    dispatch({ type: "download-start" });
    try {
      await startUpdateDownload();
    } catch (e) {
      dispatch({
        type: "failed",
        message: `启动下载失败：${e instanceof Error ? e.message : String(e)}`,
      });
    }
  }

  async function handleSkip() {
    if (!result || result.status !== "available" || skipBusy) return;
    setSkipBusy(true);
    try {
      await skipVersion(result.latestTag);
      dispatch({
        type: "check-result",
        result: {
          status: "skipped",
          latestTag: result.latestTag,
          effectiveChannel: result.effectiveChannel,
        },
      });
    } finally {
      setSkipBusy(false);
    }
  }

  const result: UpdaterResult | null = state.result;
  const notes =
    result?.status === "available" && result.releaseNotes
      ? releaseNotesParagraphs(result.releaseNotes)
      : [];

  return (
    <div className="settings-engine" id="about-update">
      <div className="settings-engine-head">
        <strong>关于 / 更新</strong>
      </div>
      <div className="hint">
        应用在线更新（半自动）：检测新版本、选择更新渠道、应用内下载安装包。模型权重的下载源设置在下方「默认下载源」，与此处无关。
      </div>

      {/* 当前版本 / 最新版本 row */}
      <div className="voice-picker">
        <span>
          当前版本：
          <code>{currentVersion ?? "…"}</code>
        </span>
        {state.phase === "up-to-date" && state.result?.status === "up-to-date" && (
          <span>
            最新版本：<code>{state.result.latestTag}</code> — <strong>已是最新</strong>
          </span>
        )}
        {state.phase === "available" && state.result?.status === "available" && (
          <span>
            最新版本：<code>{state.result.latestTag}</code>（有可用更新）
          </span>
        )}
        {state.phase === "skipped" && state.result?.status === "skipped" && (
          <span>
            最新版本：<code>{state.result.latestTag}</code>（已跳过此版本）
          </span>
        )}
      </div>

      {/* 更新渠道 three-way selector + mirror prefix */}
      {settings && (
        <div className="voice-picker">
          <label>
            更新渠道：
            <select
              value={settings.channelMode}
              onChange={(e) =>
                persist(
                  settingsForChannelChange(
                    settings,
                    e.target.value as UpdateSettings["channelMode"],
                  ),
                )
              }
            >
              {CHANNEL_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            镜像前缀：
            <input
              type="text"
              value={settings.mirrorPrefix}
              onChange={(e) => setSettings({ ...settings, mirrorPrefix: e.target.value })}
              onBlur={(e) => persist(settingsForMirrorChange(settings, e.target.value))}
            />
          </label>
          <span className="hint">公共服务，可自行更换</span>
        </div>
      )}

      <div className="key-row">
        <button disabled={state.phase === "checking"} onClick={() => void handleCheck()}>
          {state.phase === "checking" ? "检查中…" : "检查更新"}
        </button>
        {state.phase === "downloading" && (
          <button onClick={() => void cancelUpdateDownload()}>取消下载</button>
        )}
      </div>

      {bridgeMissing && (
        <div className="hint">更新接口未加载（开发模式或 preload 缺失），设置与检测暂不可用。</div>
      )}

      {/* available view: notes + effective channel + warning + actions */}
      {state.phase === "available" && result?.status === "available" && (
        <div className="update-available">
          <div>实际生效渠道：{effectiveChannelLabel(result.effectiveChannel)}</div>
          {result.warning === "manifest-missing" && (
            <div className="hint">{WARNING_MANIFEST_MISSING}</div>
          )}
          {notes.length > 0 && (
            <div className="update-notes">
              {notes.map((p, i) => (
                <p key={i}>{p}</p>
              ))}
            </div>
          )}
          <div className="key-row">
            <button onClick={() => void handleDownload()}>下载更新</button>
            <button disabled={skipBusy} onClick={() => void handleSkip()}>
              跳过此版本
            </button>
          </div>
        </div>
      )}

      {/* download progress: InstallProgress visual pattern */}
      {state.phase === "downloading" && (
        <div
          className="install-progress"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={state.progress?.percent}
        >
          <div className="install-progress-bar">
            <div
              className="install-progress-fill"
              style={{ width: `${state.progress?.percent ?? 0}%` }}
            />
          </div>
          <div className="install-progress-text">
            {state.progress
              ? `${state.progress.percent}%（${formatBytes(state.progress.received)}${
                  state.progress.total > 0 ? ` / ${formatBytes(state.progress.total)}` : ""
                }）`
              : "下载中…"}
          </div>
        </div>
      )}

      {state.phase === "done" && (
        <div className="update-done">
          更新包下载完成，安装器即将启动。请按系统提示完成安装。
          {state.doneWarning === "manifest-missing" && (
            <div className="hint">{WARNING_MANIFEST_MISSING}</div>
          )}
        </div>
      )}

      {state.phase === "failed" && (
        <div className="update-failed">
          <div className="error">更新失败：{state.failMessage}</div>
          <a href={RELEASES_PAGE_URL} target="_blank" rel="noreferrer">
            打开发布页
          </a>
        </div>
      )}
    </div>
  );
}
