import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import type {
  Capabilities,
  ReferenceAnalysis,
  TranscriptionProviders,
  EngineInfo,
  EngineInstallStatus,
  GenerationListResult,
  GenerationRecord,
  LogEvent,
  NormalizeResult,
  SidecarInfo,
  Voice,
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
  requires_reference_text: "需参考文本",
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

const DIAG_LABELS: Record<string, string> = {
  snr: "信噪比",
  speaker: "说话人",
  clipping: "削波",
  silence: "静音段",
};

function VoiceCard({
  voice,
  baseUrl,
  token,
  selected,
  onSelect,
  onDelete,
  onTranscribed,
}: {
  voice: Voice;
  baseUrl: string;
  token: string;
  selected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onTranscribed: (voice: Voice) => void;
}) {
  const ref = voice.reference;
  const [analysis, setAnalysis] = useState<ReferenceAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [cardError, setCardError] = useState<string | null>(null);

  async function diagnose() {
    setAnalyzing(true);
    setCardError(null);
    try {
      const res = await fetch(`${baseUrl}/voices/${voice.id}/diagnose`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const detail = (await res.json().catch(() => null)) as { detail?: string } | null;
        setCardError(detail?.detail ?? `诊断失败（HTTP ${res.status}）`);
        return;
      }
      setAnalysis((await res.json()) as ReferenceAnalysis);
    } finally {
      setAnalyzing(false);
    }
  }

  async function transcribe() {
    setTranscribing(true);
    setCardError(null);
    try {
      const res = await fetch(`${baseUrl}/voices/${voice.id}/transcribe`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) {
        const detail = (await res.json().catch(() => null)) as { detail?: string } | null;
        setCardError(detail?.detail ?? `转写失败（HTTP ${res.status}）`);
        return;
      }
      onTranscribed((await res.json()) as Voice);
    } finally {
      setTranscribing(false);
    }
  }

  return (
    <div
      className={`voice-card ${selected ? "selected" : ""}`}
      onClick={onSelect}
      role="button"
      aria-pressed={selected}
    >
      <div className="voice-head">
        {voice.avatar ? (
          <img
            className="voice-avatar"
            src={`${baseUrl}/voices/${voice.id}/avatar?token=${token}`}
            alt={voice.name}
          />
        ) : (
          <div className="voice-avatar placeholder">{voice.name.slice(0, 1)}</div>
        )}
        <div className="voice-meta">
          <strong>{voice.name}</strong>
          <div className="voice-ref-info">
            {ref.format.toUpperCase()} · {ref.duration_seconds.toFixed(1)}s ·{" "}
            {(ref.size_bytes / 1024 / 1024).toFixed(1)}MB
          </div>
        </div>
        <button
          className="voice-delete"
          title="删除音色（连带删除全部绑定与本地文件）"
          onClick={(ev) => {
            ev.stopPropagation();
            onDelete();
          }}
        >
          删除
        </button>
      </div>
      {voice.description && <div className="voice-desc">{voice.description}</div>}
      {Object.keys(voice.bindings).length > 0 && (
        <div className="voice-bindings">
          绑定：
          {Object.entries(voice.bindings)
            .map(([engineId, b]) => `${engineId}（${b.status}）`)
            .join("、")}
        </div>
      )}
      {ref.transcript && (
        <div className="voice-transcript">
          转写文本：<code>{ref.transcript}</code>
        </div>
      )}
      <div className="voice-tools" onClick={(ev) => ev.stopPropagation()}>
        <button onClick={() => void diagnose()} disabled={analyzing}>
          {analyzing ? "诊断中…" : "诊断"}
        </button>
        <button onClick={() => void transcribe()} disabled={transcribing}>
          {transcribing ? "转写中…" : ref.transcript ? "重新转写" : "转写"}
        </button>
      </div>
      {cardError && <div className="error">{cardError}</div>}
      {analysis && (
        <div className="voice-diagnostics" onClick={(ev) => ev.stopPropagation()}>
          {analysis.diagnostics.map((d) => (
            <div key={d.id} className={`diagnostic diag-${d.status}`}>
              <span className="diag-label">{DIAG_LABELS[d.id] ?? d.id}</span>
              <span className="diag-message">{d.message}</span>
              <span className="diag-advice">建议：{d.advice}</span>
            </div>
          ))}
        </div>
      )}
      <audio
        controls
        preload="none"
        src={`${baseUrl}/voices/${voice.id}/reference?token=${token}`}
        onClick={(ev) => ev.stopPropagation()}
      />
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

const HISTORY_PAGE_SIZE = 10;

function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  return seconds < 1 ? `${Math.round(seconds * 1000)}ms` : `${seconds.toFixed(2)}s`;
}

function fmtCost(cost: number | null | undefined): string {
  if (cost == null) return "—";
  return cost === 0 ? "本地（免费）" : `¥${cost.toFixed(4)}`;
}

function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return iso.replace("T", " ").replace("Z", " UTC");
}

function HistorySection({
  baseUrl,
  token,
  engines,
  onRerun,
  refreshKey,
}: {
  baseUrl: string;
  token: string;
  engines: EngineInfo[];
  onRerun: (record: GenerationRecord) => void;
  refreshKey: number;
}) {
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [status, setStatus] = useState("");
  const [engineFilter, setEngineFilter] = useState("");
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<GenerationListResult | null>(null);
  const [detailId, setDetailId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Debounce the search box so typing doesn't hammer the sidecar.
  useEffect(() => {
    const t = setTimeout(() => setDebounced(query), 300);
    return () => clearTimeout(t);
  }, [query]);

  useEffect(() => {
    setOffset(0);
  }, [debounced, status, engineFilter]);

  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({
      limit: String(HISTORY_PAGE_SIZE),
      offset: String(offset),
    });
    if (debounced.trim()) params.set("q", debounced.trim());
    if (status) params.set("status", status);
    if (engineFilter) params.set("engine_id", engineFilter);
    (async () => {
      try {
        const res = await fetch(`${baseUrl}/generations?${params}`, {
          headers: { Authorization: `Bearer ${token}` },
          signal: controller.signal,
        });
        if (!res.ok) {
          setError(`加载历史失败（HTTP ${res.status}）`);
          return;
        }
        setError(null);
        setData((await res.json()) as GenerationListResult);
      } catch {
        /* aborted or sidecar restarting — keep the last page */
      }
    })();
    return () => controller.abort();
  }, [baseUrl, token, debounced, status, engineFilter, offset, refreshKey]);

  async function remove(id: string) {
    if (!window.confirm("删除该生成记录？其音频文件也会被一并删除。")) return;
    const res = await fetch(`${baseUrl}/generations/${id}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      setError(`删除记录失败（HTTP ${res.status}）`);
      return;
    }
    setDetailId((prev) => (prev === id ? null : prev));
    setData((prev) =>
      prev
        ? { records: prev.records.filter((r) => r.id !== id), total: prev.total - 1 }
        : prev,
    );
  }

  const total = data?.total ?? 0;
  const page = offset / HISTORY_PAGE_SIZE + 1;
  const pageCount = Math.max(1, Math.ceil(total / HISTORY_PAGE_SIZE));

  return (
    <section>
      <h2>历史</h2>
      <div className="history-controls">
        <input
          className="history-search"
          placeholder="搜索文本、音色名或记录 ID…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select value={engineFilter} onChange={(e) => setEngineFilter(e.target.value)}>
          <option value="">全部引擎</option>
          {engines.map((e) => (
            <option key={e.id} value={e.id}>
              {e.display_name}
            </option>
          ))}
        </select>
        <select value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="">全部状态</option>
          <option value="succeeded">成功</option>
          <option value="failed">失败</option>
        </select>
      </div>
      {error && <div className="error">{error}</div>}
      {!data ? (
        <div className="hint">加载中…</div>
      ) : total === 0 ? (
        <div className="hint">还没有生成记录。生成一次后，这里会留下完整档案。</div>
      ) : (
        <div className="history-list">
          {data.records.map((rec) => (
            <div key={rec.id} className={`history-item history-${rec.status}`}>
              <div className="history-row">
                <span className="history-time">{fmtTime(rec.created_at)}</span>
                <span className={`badge ${rec.status === "succeeded" ? "badge-on" : "badge-off"}`}>
                  {rec.status === "succeeded" ? "成功" : rec.status === "failed" ? "失败" : "进行中"}
                </span>
                <span className="history-engine">{rec.engine_id}</span>
                {rec.voice_name && <span className="history-voice">{rec.voice_name}</span>}
                <span className="history-dur">{fmtDuration(rec.duration_seconds)}</span>
                <span className="history-cost">{fmtCost(rec.cost)}</span>
                <span className="history-actions">
                  <button onClick={() => setDetailId((prev) => (prev === rec.id ? null : rec.id))}>
                    {detailId === rec.id ? "收起" : "详情"}
                  </button>
                  <button onClick={() => onRerun(rec)}>重跑</button>
                  <button onClick={() => void remove(rec.id)}>删除</button>
                </span>
              </div>
              <div className="history-text">{rec.text}</div>
              {detailId === rec.id && (
                <div className="history-detail">
                  <div className="history-detail-grid">
                    <div>引擎：{rec.engine_id}{rec.model_version ? `（模型 ${rec.model_version}）` : ""}</div>
                    <div>音色：{rec.voice_name ?? "（未使用）"}</div>
                    <div>耗时：{fmtDuration(rec.duration_seconds)}</div>
                    <div>成本：{fmtCost(rec.cost)}</div>
                    <div>采样率：{rec.sample_rate ?? "—"}</div>
                    <div>完成：{fmtTime(rec.finished_at)}</div>
                  </div>
                  {rec.normalized_text && (
                    <div>
                      归一化文本：<code>{rec.normalized_text}</code>
                    </div>
                  )}
                  <div className="history-params">
                    参数：
                    <pre>{JSON.stringify(rec.params ?? {}, null, 2)}</pre>
                  </div>
                  {rec.status === "failed" && rec.error && (
                    <div className="error">失败原因：{rec.error}</div>
                  )}
                  <div className="history-logs">
                    日志（{rec.logs.length}）：
                    <pre>
                      {rec.logs.map((l, i) => (
                        <div key={i}>[{l.ts}] {l.message}</div>
                      ))}
                    </pre>
                  </div>
                  {rec.status === "succeeded" && rec.audio_url && (
                    <audio controls preload="none" src={`${baseUrl}${rec.audio_url}?token=${token}`} />
                  )}
                </div>
              )}
            </div>
          ))}
          <div className="history-pager">
            <button disabled={offset === 0} onClick={() => setOffset(offset - HISTORY_PAGE_SIZE)}>
              上一页
            </button>
            <span>
              第 {page} / {pageCount} 页 · 共 {total} 条
            </span>
            <button
              disabled={offset + HISTORY_PAGE_SIZE >= total}
              onClick={() => setOffset(offset + HISTORY_PAGE_SIZE)}
            >
              下一页
            </button>
          </div>
        </div>
      )}
    </section>
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
  const [normalized, setNormalized] = useState<NormalizeResult | null>(null);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null);
  const [newVoiceName, setNewVoiceName] = useState("");
  const [newVoiceDesc, setNewVoiceDesc] = useState("");
  const [newVoiceFile, setNewVoiceFile] = useState<File | null>(null);
  const [newVoiceAvatar, setNewVoiceAvatar] = useState<File | null>(null);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [creatingVoice, setCreatingVoice] = useState(false);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [rerunHint, setRerunHint] = useState<string | null>(null);
  const [transProviders, setTransProviders] = useState<TranscriptionProviders | null>(null);
  const [localTransInstalling, setLocalTransInstalling] = useState(false);
  // Parameters carried back from a rerun: everything the record stored except
  // server-side paths the sidecar re-injects itself (ref_audio). They ride
  // along on the next generate call unless the user clears them.
  const [rerunParams, setRerunParams] = useState<Record<string, unknown> | null>(null);
  const generateRef = useRef<HTMLDivElement>(null);
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

  // 归一化预览：文本变化后防抖请求 sidecar 的 /normalize（与生成同一层）。
  useEffect(() => {
    if (!info) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${info.baseUrl}/normalize`, {
          method: "POST",
          headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
          signal: controller.signal,
        });
        if (res.ok) setNormalized((await res.json()) as NormalizeResult);
      } catch {
        /* aborted or sidecar restarting — keep the last preview */
      }
    }, 300);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [info, text]);

  async function refreshVoices() {
    if (!info) return;
    const res = await fetch(`${info.baseUrl}/voices`, {
      headers: { Authorization: `Bearer ${info.token}` },
    });
    if (res.ok) setVoices(((await res.json()) as { voices: Voice[] }).voices);
  }

  useEffect(() => {
    void refreshVoices();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info]);

  async function refreshTransProviders() {
    if (!info) return;
    const res = await fetch(`${info.baseUrl}/transcription/providers`, {
      headers: { Authorization: `Bearer ${info.token}` },
    });
    if (res.ok) setTransProviders((await res.json()) as TranscriptionProviders);
  }

  useEffect(() => {
    void refreshTransProviders();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info]);

  useEffect(() => {
    if (!info || !localTransInstalling) return;
    const t = setInterval(async () => {
      const res = await fetch(`${info.baseUrl}/transcription/local/status`, {
        headers: { Authorization: `Bearer ${info.token}` },
      });
      if (!res.ok) return;
      const st = (await res.json()) as { installed: boolean; installing: boolean };
      if (!st.installing) {
        setLocalTransInstalling(false);
        void refreshTransProviders();
      }
    }, 1000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info, localTransInstalling]);

  async function setTransProvider(provider: string) {
    if (!info) return;
    const res = await fetch(`${info.baseUrl}/transcription/provider`, {
      method: "PUT",
      headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ provider }),
    });
    if (res.ok) void refreshTransProviders();
  }

  async function installLocalTranscriber() {
    if (!info || localTransInstalling) return;
    setLocalTransInstalling(true);
    setLogsOpen(true);
    try {
      await fetch(`${info.baseUrl}/transcription/local/install`, {
        method: "POST",
        headers: { Authorization: `Bearer ${info.token}` },
      });
    } catch {
      setLocalTransInstalling(false);
    }
  }

  async function createVoice() {
    if (!info || !newVoiceFile || creatingVoice) return;
    setCreatingVoice(true);
    setVoiceError(null);
    try {
      const form = new FormData();
      form.append("name", newVoiceName);
      form.append("description", newVoiceDesc);
      form.append("file", newVoiceFile);
      if (newVoiceAvatar) form.append("avatar", newVoiceAvatar);
      const res = await fetch(`${info.baseUrl}/voices`, {
        method: "POST",
        headers: { Authorization: `Bearer ${info.token}` },
        body: form,
      });
      if (!res.ok) {
        const detail = (await res.json().catch(() => null)) as { detail?: string } | null;
        setVoiceError(detail?.detail ?? `创建音色失败（HTTP ${res.status}）`);
        return;
      }
      const voice = (await res.json()) as Voice;
      setNewVoiceName("");
      setNewVoiceDesc("");
      setNewVoiceFile(null);
      setNewVoiceAvatar(null);
      await refreshVoices();
      setSelectedVoice(voice.id);
    } catch (err) {
      setVoiceError(`创建音色失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCreatingVoice(false);
    }
  }

  async function deleteVoice(id: string) {
    if (!info) return;
    if (!window.confirm("删除该音色？其全部引擎绑定与本地参考音频文件都会被删除。")) return;
    const res = await fetch(`${info.baseUrl}/voices/${id}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${info.token}` },
    });
    if (res.ok) {
      setVoices((prev) => prev.filter((v) => v.id !== id));
      setSelectedVoice((prev) => (prev === id ? null : prev));
    } else {
      setVoiceError(`删除音色失败（HTTP ${res.status}）`);
    }
  }

  async function generate() {
    if (!info || !selectedEngine || generating) return;
    setGenerating(true);
    setRecord(null);
    setGenerateError(null);
    try {
      const res = await fetch(`${info.baseUrl}/generations`, {
        method: "POST",
        headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          engine_id: selectedEngine,
          text,
          ...(selectedVoice ? { voice_id: selectedVoice } : {}),
          ...(rerunParams ? { params: rerunParams } : {}),
        }),
      });
      if (!res.ok) {
        setGenerateError(`生成请求失败（HTTP ${res.status}）`);
        return;
      }
      const data: GenerationRecord = await res.json();
      setRecord(data);
      setRerunParams(null);
      setLogsOpen(true);
      setHistoryRefreshKey((k) => k + 1);
    } catch (err) {
      setGenerateError(`生成请求失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setGenerating(false);
    }
  }

  function rerun(record: GenerationRecord) {
    // Rerun = load the record's inputs back into the generate panel: engine,
    // text, voice, and every stored parameter (minus server-side paths the
    // sidecar re-injects from the voice). The user adjusts and regenerates —
    // never an automatic re-execution.
    setRerunHint(null);
    const params = { ...(record.params ?? {}) };
    delete params.ref_audio;
    setRerunParams(Object.keys(params).length > 0 ? params : null);
    setSelectedEngine(record.engine_id);
    setText(record.text);
    if (record.voice_id && voices.some((v) => v.id === record.voice_id)) {
      setSelectedVoice(record.voice_id);
    } else {
      setSelectedVoice(null);
      if (record.voice_id) {
        setRerunHint("原音色已删除，已预填文本与引擎；请重新选择音色后再生成。");
      }
    }
    generateRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
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
        <h2>音色库</h2>
        <div className="voice-create">
          <div className="voice-create-row">
            <input
              placeholder="音色名称（必填）"
              value={newVoiceName}
              onChange={(e) => setNewVoiceName(e.target.value)}
            />
            <input
              placeholder="文字说明（可选）"
              value={newVoiceDesc}
              onChange={(e) => setNewVoiceDesc(e.target.value)}
            />
          </div>
          <div className="voice-create-row">
            <label>
              参考音频（WAV/MP3/FLAC/M4A/OGG，3–120s，≤20MB）：
              <input
                type="file"
                accept=".wav,.mp3,.flac,.m4a,.ogg"
                onChange={(e) => setNewVoiceFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <label>
              头像（可选）：
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.webp,.gif"
                onChange={(e) => setNewVoiceAvatar(e.target.files?.[0] ?? null)}
              />
            </label>
            <button
              onClick={createVoice}
              disabled={!newVoiceName.trim() || !newVoiceFile || creatingVoice}
            >
              {creatingVoice ? "创建中…" : "创建音色"}
            </button>
          </div>
          {voiceError && <div className="error">{voiceError}</div>}
        </div>
        {voices.length === 0 ? (
          <div className="hint">还没有音色。上传一段参考音频创建第一个音色。</div>
        ) : (
          <div className="voice-list">
            {voices.map((v) => (
              <VoiceCard
                key={v.id}
                voice={v}
                baseUrl={info?.baseUrl ?? ""}
                token={info?.token ?? ""}
                selected={v.id === selectedVoice}
                onSelect={() => setSelectedVoice(v.id === selectedVoice ? null : v.id)}
                onDelete={() => void deleteVoice(v.id)}
                onTranscribed={(updated) =>
                  setVoices((prev) => prev.map((x) => (x.id === updated.id ? updated : x)))
                }
              />
            ))}
          </div>
        )}
      </section>

      <section>
        <h2>转写设置</h2>
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
      </section>

      <section ref={generateRef}>
        <h2>生成</h2>
        <div className="voice-picker">
          <label>
            使用音色：
            <select
              value={selectedVoice ?? ""}
              onChange={(e) => setSelectedVoice(e.target.value || null)}
            >
              <option value="">（不用音色，直接合成）</option>
              {voices.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name}
                </option>
              ))}
            </select>
          </label>
          {selectedVoice &&
            selectedEngine &&
            engines.find((e) => e.id === selectedEngine)?.capabilities.voice_cloning ===
              false && (
              <span className="hint">该引擎不支持声音复刻，将忽略参考音频。</span>
            )}
            {selectedVoice &&
              selectedEngine &&
              engines.find((e) => e.id === selectedEngine)?.capabilities
                .requires_reference_text && (
                <span className="hint">
                  该引擎需要参考文本：将自动转写所选音色的参考音频，无需手动输入。
                </span>
              )}
        </div>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} />

        {normalized && normalized.changed && (
          <div className="normalize-preview">
            <div className="normalize-title">归一化预览（生成前引擎将收到以下文本）</div>
            <div className="normalize-text">{normalized.normalized}</div>
          </div>
        )}
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
        {rerunHint && <div className="hint">{rerunHint}</div>}
        {rerunParams && (
          <div className="rerun-params">
            重跑参数已载入：
            <code>{Object.keys(rerunParams).join("、")}</code>
            <button onClick={() => setRerunParams(null)}>清除</button>
          </div>
        )}
      </section>

      {info && (
        <HistorySection
          baseUrl={info.baseUrl}
          token={info.token}
          engines={engines}
          onRerun={rerun}
          refreshKey={historyRefreshKey}
        />
      )}

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
