import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import type {
  Capabilities,
  EngineInfo,
  EngineInstallStatus,
  GenerationRecord,
  LogEvent,
  SidecarInfo,
} from "./api";

const CAP_LABELS: Record<keyof Capabilities, string> = {
  languages: "语种",
  voice_cloning: "复刻",
  voice_design: "音色设计",
  pronunciation_control: "发音控制",
  emotion: "情感",
  commercial_license: "商业许可",
  cross_device_use: "跨设备",
  upload_used_for_training: "上传用于训练",
  api_closed_loop: "API 闭环",
};

function CapBadge({ label, on }: { label: string; on: boolean }) {
  return (
    <span className={`badge ${on ? "badge-on" : "badge-off"}`}>
      {label} {on ? "✓" : "✗"}
    </span>
  );
}

const STEP_LABELS: Record<string, string> = {
  python: "Python 运行时",
  venv: "引擎独立环境",
  packages: "依赖安装",
  weights: "引擎权重下载",
};

function EngineCard({
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
    >
      <strong>{engine.display_name}</strong>{" "}
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
        <CapBadge label={`${CAP_LABELS.languages}:${c.languages.join("/") || "—"}`} on={c.languages.length > 0} />
        <CapBadge label={CAP_LABELS.voice_cloning} on={c.voice_cloning} />
        <CapBadge label={CAP_LABELS.voice_design} on={c.voice_design} />
        <CapBadge label={CAP_LABELS.emotion} on={c.emotion} />
        <CapBadge label={CAP_LABELS.commercial_license} on={c.commercial_license} />
        <CapBadge label={CAP_LABELS.upload_used_for_training} on={c.upload_used_for_training} />
      </div>
    </div>
  );
}

function Waveform({ url, headers }: { url: string; headers: Record<string, string> }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#5c6bc0",
      progressColor: "#82b1ff",
      height: 80,
      url,
      fetchParams: { headers },
    });
    wsRef.current = ws;
    return () => {
      ws.destroy();
      wsRef.current = null;
    };
  }, [url, headers]);

  return (
    <div className="waveform">
      <div ref={containerRef} />
      <button
        onClick={() => {
          if (wsRef.current) wsRef.current.playPause();
        }}
      >
        ▶ / ⏸
      </button>
    </div>
  );
}

export default function App() {
  const [info, setInfo] = useState<SidecarInfo | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [installStatus, setInstallStatus] = useState<Record<string, EngineInstallStatus>>({});
  const [installing, setInstalling] = useState<Record<string, boolean>>({});
  const [selectedEngine, setSelectedEngine] = useState<string | null>(null);
  const [text, setText] = useState("你好，世界。这是一次端到端生成测试。Hello, world!");
  const [record, setRecord] = useState<GenerationRecord | null>(null);
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [logsOpen, setLogsOpen] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const installingRef = useRef<Record<string, boolean>>({});

  async function installEngine(id: string) {
    if (!info || installing[id]) return;
    setInstalling((p) => ({ ...p, [id]: true }));
    installingRef.current[id] = true;
    setLogsOpen(true);
    try {
      await fetch(`${info.baseUrl}/engines/${id}/install`, {
        method: "POST",
        headers: { Authorization: `Bearer ${info.token}` },
      });
      // Completion is observed by the shared status poller: fetchStatus
      // clears the installing flag once the engine reports a settled state.
    } catch {
      setInstalling((p) => ({ ...p, [id]: false }));
      installingRef.current[id] = false;
    }
  }

  useEffect(() => {
    window.voiceclone.onSidecarError((message) => setConnectionError(message));
  }, []);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let interval: ReturnType<typeof setInterval> | null = null;
    (async () => {
      const sidecarInfo = await window.voiceclone.getSidecarInfo();
      if (!sidecarInfo) return;
      setInfo(sidecarInfo);

      const headers = { Authorization: `Bearer ${sidecarInfo.token}` };
      const res = await fetch(`${sidecarInfo.baseUrl}/engines`, { headers });
      if (!res.ok) {
        setConnectionError(`加载引擎列表失败（HTTP ${res.status}）`);
        return;
      }
      const data = await res.json();
      setEngines(data.engines);
      if (data.engines.length > 0) setSelectedEngine(data.engines[0].id);

      // Fetch per-engine install status; poll only engines that are mid-install.
      const fetchStatus = async (id: string) => {
        const r = await fetch(`${sidecarInfo.baseUrl}/engines/${id}/status`, { headers });
        if (!r.ok) return;
        const status = (await r.json()) as EngineInstallStatus;
        setInstallStatus((prev) => ({ ...prev, [id]: status }));
        if (!status.installing && installingRef.current[id]) {
          installingRef.current[id] = false;
          setInstalling((p) => ({ ...p, [id]: false }));
        }
      };
      for (const e of data.engines) await fetchStatus(e.id);
      // Poll only engines that are mid-install.
      interval = setInterval(() => {
        for (const e of data.engines) {
          if (installingRef.current[e.id]) void fetchStatus(e.id);
        }
      }, 1000);

      ws = new WebSocket(
        `${sidecarInfo.baseUrl.replace("http", "ws")}/ws/logs?token=${sidecarInfo.token}`,
      );
      ws.onmessage = (ev) => {
        const event = JSON.parse(ev.data) as LogEvent;
        setLogs((prev) => [...prev.slice(-499), event]);
      };
      wsRef.current = ws;
    })();
    // StrictMode double-mounts effects in dev — close the socket this
    // mount created, otherwise the first connection leaks.
    return () => {
      if (interval) clearInterval(interval);
      ws?.close();
      if (ws && wsRef.current === ws) wsRef.current = null;
    };
  }, []);

  async function generate() {
    if (!info || !selectedEngine || generating) return;
    setGenerating(true);
    setRecord(null);
    setGenerateError(null);
    try {
      const res = await fetch(`${info.baseUrl}/generations`, {
        method: "POST",
        headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ engine_id: selectedEngine, text }),
      });
      if (!res.ok) {
        setGenerateError(`生成请求失败（HTTP ${res.status}）`);
        return;
      }
      const data: GenerationRecord = await res.json();
      setRecord(data);
      setLogsOpen(true);
    } catch (err) {
      setGenerateError(`生成请求失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setGenerating(false);
    }
  }

  const audioHeaders: Record<string, string> = info ? { Authorization: `Bearer ${info.token}` } : {};

  return (
    <div className="app">
      <header>
        <h1>声音复刻工作台</h1>
        {connectionError ? (
          <span className="status down">{connectionError}</span>
        ) : (
          <span className={`status ${info ? "ok" : "down"}`}>
            {info ? "sidecar 已连接" : "sidecar 连接中…"}
          </span>
        )}
      </header>

      <section>
        <h2>引擎</h2>
        <div className="engine-list">
          {engines.map((e) => (
            <EngineCard
              key={e.id}
              engine={e}
              selected={e.id === selectedEngine}
              installStatus={installStatus[e.id] ?? null}
              installing={!!installing[e.id]}
              onInstall={() => void installEngine(e.id)}
              onSelect={() => setSelectedEngine(e.id)}
            />
          ))}
        </div>
      </section>

      <section>
        <h2>生成</h2>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} />
        <button
          className="primary"
          onClick={generate}
          disabled={
            generating ||
            !selectedEngine ||
            (installStatus[selectedEngine]
              ? !installStatus[selectedEngine].installed
              : false)
          }
        >
          {generating ? "生成中…" : "生成"}
        </button>
        {selectedEngine &&
          installStatus[selectedEngine] &&
          !installStatus[selectedEngine].installed && (
            <div className="hint">该引擎尚未安装，请先在上方点击「安装引擎」。</div>
          )}

        {record && record.status === "succeeded" && record.audio_url && info && (
          <div className="result">
            <Waveform url={`${info.baseUrl}${record.audio_url}`} headers={audioHeaders} />
          </div>
        )}
        {record && record.status === "failed" && (
          <div className="error">生成失败：{record.error}</div>
        )}
        {generateError && <div className="error">{generateError}</div>}
      </section>

      <section className={`log-drawer ${logsOpen ? "open" : ""}`}>
        <button className="log-toggle" onClick={() => setLogsOpen(!logsOpen)}>
          {logsOpen ? "▾" : "▸"} 实时日志（{logs.length}）
        </button>
        {logsOpen && (
          <pre className="logs">
            {logs.map((l, i) => (
              <div key={i}>[{l.generation_id.slice(0, 6)}] {l.message}</div>
            ))}
          </pre>
        )}
      </section>
    </div>
  );
}
