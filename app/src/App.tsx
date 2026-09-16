import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import type { Capabilities, EngineInfo, GenerationRecord, LogEvent, SidecarInfo } from "./api";

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

function EngineCard({
  engine,
  selected,
  onSelect,
}: {
  engine: EngineInfo;
  selected: boolean;
  onSelect: () => void;
}) {
  const c = engine.capabilities;
  return (
    <div
      className={`engine-card ${selected ? "selected" : ""}`}
      onClick={onSelect}
      role="button"
      aria-pressed={selected}
    >
      <strong>{engine.display_name}</strong>
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
  }, [url]);

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
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const [selectedEngine, setSelectedEngine] = useState<string | null>(null);
  const [text, setText] = useState("你好，世界。这是一次端到端生成测试。Hello, world!");
  const [record, setRecord] = useState<GenerationRecord | null>(null);
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [logsOpen, setLogsOpen] = useState(true);
  const [generating, setGenerating] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    (async () => {
      const sidecarInfo = await window.voiceclone.getSidecarInfo();
      if (!sidecarInfo) return;
      setInfo(sidecarInfo);

      const headers = { Authorization: `Bearer ${sidecarInfo.token}` };
      const res = await fetch(`${sidecarInfo.baseUrl}/engines`, { headers });
      const data = await res.json();
      setEngines(data.engines);
      if (data.engines.length > 0) setSelectedEngine(data.engines[0].id);

      const ws = new WebSocket(`${sidecarInfo.baseUrl.replace("http", "ws")}/ws/logs?token=${sidecarInfo.token}`);
      ws.onmessage = (ev) => {
        const event = JSON.parse(ev.data) as LogEvent;
        setLogs((prev) => [...prev.slice(-499), event]);
      };
      wsRef.current = ws;
    })();
    return () => wsRef.current?.close();
  }, []);

  async function generate() {
    if (!info || !selectedEngine || generating) return;
    setGenerating(true);
    setRecord(null);
    try {
      const res = await fetch(`${info.baseUrl}/generations`, {
        method: "POST",
        headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ engine_id: selectedEngine, text }),
      });
      const data = await res.json();
      setRecord(data);
      setLogsOpen(true);
    } finally {
      setGenerating(false);
    }
  }

  const audioHeaders: Record<string, string> = info ? { Authorization: `Bearer ${info.token}` } : {};

  return (
    <div className="app">
      <header>
        <h1>声音复刻工作台</h1>
        <span className={`status ${info ? "ok" : "down"}`}>{info ? "sidecar 已连接" : "sidecar 连接中…"}</span>
      </header>

      <section>
        <h2>引擎</h2>
        <div className="engine-list">
          {engines.map((e) => (
            <EngineCard
              key={e.id}
              engine={e}
              selected={e.id === selectedEngine}
              onSelect={() => setSelectedEngine(e.id)}
            />
          ))}
        </div>
      </section>

      <section>
        <h2>生成</h2>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} />
        <button className="primary" onClick={generate} disabled={generating || !selectedEngine}>
          {generating ? "生成中…" : "生成"}
        </button>

        {record && record.status === "succeeded" && record.audio_url && info && (
          <div className="result">
            <Waveform url={`${info.baseUrl}${record.audio_url}`} headers={audioHeaders} />
          </div>
        )}
        {record && record.status === "failed" && (
          <div className="error">生成失败：{record.error}</div>
        )}
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
