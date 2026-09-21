import { useCallback, useEffect, useRef, useState } from "react";
import type { EngineInstallStatus, TranscriptionProviders } from "../api";
import { apiJson } from "../client";
import { InstallProgressLine, InstallStepsList } from "./InstallProgress";

// ---------------------------------------------------------------------------
// Issue #42: the local transcription tool becomes a proper card on the 引擎
// page — install button, byte-level progress (same structure as engine
// installs), installed state, and an uninstall button that only exists once
// installed. This is the ONLY install entry for the tool in the whole app;
// the settings page keeps just the provider picker.
// ---------------------------------------------------------------------------

export function TranscribeToolCard({
  baseUrl,
  token,
}: {
  baseUrl: string;
  token: string;
}) {
  const [status, setStatus] = useState<EngineInstallStatus | null>(null);
  const [label, setLabel] = useState("本地转写");
  const [supported, setSupported] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);

  // The card is self-contained: it learns its display name and whether the
  // platform supports the tool at all from the providers endpoint.
  useEffect(() => {
    apiJson<TranscriptionProviders>(baseUrl, token, "/transcription/providers")
      .then((p) => {
        if (!alive.current) return;
        if (p.local.label) setLabel(p.local.label);
        setSupported(p.local.supported);
      })
      .catch(() => undefined);
  }, [baseUrl, token]);

  const load = useCallback(async () => {
    try {
      const s = await apiJson<EngineInstallStatus>(
        baseUrl,
        token,
        "/transcription/local/status",
      );
      if (alive.current) setStatus(s);
    } catch {
      /* sidecar restarting — keep the last snapshot */
    }
  }, [baseUrl, token]);

  useEffect(() => {
    void load();
    // Poll only while an install is actually running (same cadence as
    // engine installs); when it settles the status stops being `installing`
    // and the poller tears itself down — the final poll carried the result.
    if (!status?.installing) return;
    const t = setInterval(() => void load(), 1000);
    return () => clearInterval(t);
  }, [load, status?.installing]);

  useEffect(() => {
    return () => {
      alive.current = false;
    };
  }, []);

  const installed = !!status?.installed;
  const installing = !!status?.installing;

  const install = async () => {
    if (installing) return;
    setBusy(true);
    setError(null);
    try {
      await apiJson(baseUrl, token, "/transcription/local/install", { method: "POST" });
      await load(); // the poller takes over from here
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const uninstall = async () => {
    if (!installed || installing) return;
    if (!window.confirm(`卸载「${label}」将删除其模型与运行环境，确定吗？`)) return;
    setBusy(true);
    setError(null);
    try {
      await apiJson(baseUrl, token, "/engines/transcribe-local/model", { method: "DELETE" });
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="engine-card" role="group" aria-label="转写工具">
      <strong>{label}</strong>{" "}
      <span className="hint">参考音频转写工具（离线）</span>{" "}
      {!supported ? (
        <span className="hint">当前平台不支持本地转写；可在设置中选择云端转写。</span>
      ) : (
        <>
          {installed ? (
            <span className="badge badge-on">已就绪</span>
          ) : (
            <button
              className="install-btn"
              disabled={busy || installing}
              onClick={() => void install()}
            >
              {installing ? "安装中…" : "安装"}
            </button>
          )}
          {/* Uninstall exists only once installed — no dead gray button
              next to 「未安装」. */}
          {installed && (
            <button disabled={busy || installing} onClick={() => void uninstall()}>
              卸载
            </button>
          )}
          {installing && status?.progress && <InstallProgressLine progress={status.progress} />}
          {status && Object.keys(status.steps).length > 0 && !installed && (
            <InstallStepsList steps={status.steps} />
          )}
        </>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
}
