import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import type {
  Capabilities,
  CompareSession,
  ConsentInfo,
  PreferenceProfile,
  ReferenceAnalysis,
  TranscriptionProviders,
  EngineInfo,
  KeyStatus,
  EngineInstallStatus,
  GenerationListResult,
  GenerationRecord,
  GenerationJob,
  JobStatus,
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
  // Never rendered as a badge — the settings matrix lists its keys explicitly.
  max_chars_per_request: "单次字符上限",
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

// Issue #14: download a sidecar file (voice package / library backup) with
// bearer auth, then hand it to the browser's normal save flow.
async function downloadWithAuth(url: string, token: string, filename: string): Promise<void> {
  const res = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
  if (!res.ok) throw new Error(await errorDetail(res, `下载失败（HTTP ${res.status}）`));
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function errorDetail(res: Response, fallback: string): Promise<string> {
  try {
    const j = (await res.json()) as { detail?: string };
    return j.detail ?? fallback;
  } catch {
    return fallback;
  }
}

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
  const designed = voice.origin === "designed";
  const [analysis, setAnalysis] = useState<ReferenceAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [cardError, setCardError] = useState<string | null>(null);

  async function exportPackage() {
    setExporting(true);
    setCardError(null);
    try {
      const safeName = (voice.name || voice.id).replace(/[\\/:*?"<>|\s]+/g, "_");
      await downloadWithAuth(
        `${baseUrl}/voices/${voice.id}/export`,
        token,
        `${safeName}.voice.zip`,
      );
    } catch (e) {
      setCardError(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting(false);
    }
  }

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
          <strong>{voice.name}</strong>{" "}
          <span className={`badge ${designed ? "badge-design" : "badge-clone"}`}>
            {designed ? "文字设计" : "参考音频复刻"}
          </span>
          <div className="voice-ref-info">
            {ref
              ? `${ref.format.toUpperCase()} · ${ref.duration_seconds.toFixed(1)}s · ${(
                  ref.size_bytes / 1024 / 1024
                ).toFixed(1)}MB`
              : "尚无参考样本"}
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
            .map(
              ([engineId, b]) =>
                `${engineId}（${b.status}${b.error ? `：${b.error}` : ""}）`,
            )
            .join("、")}
        </div>
      )}
      {ref?.transcript && (
        <div className="voice-transcript">
          转写文本：<code>{ref.transcript}</code>
        </div>
      )}
      <div className="voice-tools" onClick={(ev) => ev.stopPropagation()}>
        {ref ? (
          <>
            <button onClick={() => void diagnose()} disabled={analyzing}>
              {analyzing ? "诊断中…" : "诊断"}
            </button>
            <button onClick={() => void transcribe()} disabled={transcribing}>
              {transcribing ? "转写中…" : ref.transcript ? "重新转写" : "转写"}
            </button>
            <button onClick={() => void exportPackage()} disabled={exporting} title="导出为音色包（.zip），可分享给他人">
              {exporting ? "导出中…" : "导出"}
            </button>
          </>
        ) : (
          <span className="hint">设计失败遗留的音色，请删除后重新设计。</span>
        )}
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
      {ref && (
        <audio
          controls
          preload="none"
          src={`${baseUrl}/voices/${voice.id}/reference?token=${token}`}
          onClick={(ev) => ev.stopPropagation()}
        />
      )}
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

// --- blind comparison + preference profile (issue #10, ADR-0009) ---

const COMPARE_TEXT_TYPES = [
  { value: "general", label: "通用" },
  { value: "narration", label: "旁白/朗读" },
  { value: "dialogue", label: "对话" },
  { value: "news", label: "新闻" },
  { value: "poetry", label: "诗歌/文学" },
] as const;

const TEXT_TYPE_LABELS: Record<string, string> = Object.fromEntries(
  COMPARE_TEXT_TYPES.map((t) => [t.value, t.label]),
);

function CompareSection({
  baseUrl,
  token,
  voices,
  engines,
}: {
  baseUrl: string;
  token: string;
  voices: Voice[];
  engines: EngineInfo[];
}) {
  const [voiceId, setVoiceId] = useState("");
  const [text, setText] = useState(
    "同一音色、同一文本，交给多个引擎并排生成。The comparison is blind until you reveal it.",
  );
  const [picked, setPicked] = useState<Record<string, boolean>>({});
  const [textType, setTextType] = useState("general");
  const [session, setSession] = useState<CompareSession | null>(null);
  const [scores, setScores] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [profile, setProfile] = useState<PreferenceProfile | null>(null);

  const pickedIds = engines.filter((e) => picked[e.id]).map((e) => e.id);

  async function runCompare() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${baseUrl}/compare`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          voice_id: voiceId,
          text,
          engine_ids: pickedIds,
          text_type: textType,
        }),
      });
      if (!res.ok) {
        const detail = (await res.json().catch(() => null)) as { detail?: string } | null;
        setError(detail?.detail ?? `创建对比失败（HTTP ${res.status}）`);
        return;
      }
      const data: CompareSession = await res.json();
      setSession(data);
      setScores(
        Object.fromEntries(data.entries.map((e) => [e.label, e.score ?? 0])),
      );
    } catch (err) {
      setError(`创建对比失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  async function reloadSession(reveal: boolean) {
    if (!session) return;
    const res = await fetch(
      `${baseUrl}/compare/${session.id}?reveal=${reveal ? "true" : "false"}`,
      { headers: { Authorization: `Bearer ${token}` } },
    );
    if (res.ok) setSession(await res.json());
  }

  async function submitScores() {
    if (!session) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${baseUrl}/compare/${session.id}/scores`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ scores }),
      });
      if (!res.ok) {
        const detail = (await res.json().catch(() => null)) as { detail?: string } | null;
        setError(detail?.detail ?? `提交评分失败（HTTP ${res.status}）`);
        return;
      }
      const data: CompareSession = await res.json();
      setSession(data);
      await refreshProfile();
    } finally {
      setBusy(false);
    }
  }

  async function refreshProfile() {
    const res = await fetch(`${baseUrl}/preferences`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.ok) setProfile((await res.json()) as PreferenceProfile);
  }

  useEffect(() => {
    void refreshProfile();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const allScored = !!session && session.scored;
  const canReveal = !!session && !allScored;

  return (
    <div className="compare-mode">
      <h3>对比盲听（生成内模式）</h3>
      <div className="hint">
        同一音色、同一文本交给多个引擎并排生成；所有结果先做响度归一化（LUFS −16），
        再以 A/B/C 盲标呈现。评分沉淀为本机偏好画像，仅作参考，不影响默认引擎或任何自动路由。
      </div>
      <div className="voice-picker">
        <label>
          对比音色：
          <select value={voiceId} onChange={(e) => setVoiceId(e.target.value)}>
            <option value="">（请选择音色）</option>
            {voices.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          文本类型：
          <select value={textType} onChange={(e) => setTextType(e.target.value)}>
            {COMPARE_TEXT_TYPES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="param-row">
        {engines.map((e) => (
          <label key={e.id} className="compare-engine-toggle">
            <input
              type="checkbox"
              checked={!!picked[e.id]}
              onChange={(ev) =>
                setPicked((p) => ({ ...p, [e.id]: ev.target.checked }))
              }
            />
            {e.display_name}
          </label>
        ))}
      </div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} />
      <button
        className="primary"
        onClick={runCompare}
        disabled={busy || !voiceId || pickedIds.length < 2 || !text.trim()}
      >
        {busy ? "生成中…" : `并排生成（已选 ${pickedIds.length} 个引擎）`}
      </button>
      {error && <div className="error">{error}</div>}

      {session && (
        <div className="compare-session">
          <div className="compare-head">
            <strong>
              盲听（{session.language === "zh" ? "中文" : "英文"} ·{" "}
              {TEXT_TYPE_LABELS[session.text_type] ?? session.text_type} · 目标 {session.target_lufs} LUFS）
            </strong>
            {canReveal && (
              <button onClick={() => void reloadSession(true)}>揭示引擎身份</button>
            )}
            {!allScored && (
              <button
                onClick={() => void submitScores()}
                disabled={busy || !session.entries.every((e) => (scores[e.label] ?? 0) > 0)}
                title="为每个对比项打分后可提交"
              >
                提交评分
              </button>
            )}
            {allScored && <span className="badge badge-on">已评分 · 身份已揭示</span>}
          </div>
          <div className="compare-grid">
            {session.entries.map((e) => (
              <div className="compare-card" key={e.label}>
                <div className="compare-card-head">
                  <strong>{e.label}</strong>
                  <span className="hint">
                    {e.engine_id
                      ? engines.find((x) => x.id === e.engine_id)?.display_name ?? e.engine_id
                      : "引擎已隐藏"}
                  </span>
                </div>
                <audio
                  controls
                  preload="none"
                  src={`${baseUrl}${e.normalized_audio_url}?token=${encodeURIComponent(token)}`}
                />
                <div className="hint">
                  原始 {e.original_lufs ?? "—"} LUFS → 归一化 {e.achieved_lufs ?? "—"} LUFS
                  （增益 {e.gain_db != null && e.gain_db > 0 ? "+" : ""}
                  {e.gain_db ?? "—"} dB{e.peak_limited ? "，峰值受限" : ""}）
                </div>
                <div className="score-row">
                  {[1, 2, 3, 4, 5].map((n) => (
                    <button
                      key={n}
                      className={`score-star ${(scores[e.label] ?? 0) >= n ? "on" : ""}`}
                      onClick={() => setScores((p) => ({ ...p, [e.label]: n }))}
                      title={`${n} 分`}
                    >
                      ★
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
          {(session.failed_count ?? 0) > 0 && (
            <div className="hint">
              {session.failed_count} 个引擎未能生成（未安装或缺少 API Key），已跳过。
            </div>
          )}
          {session.failed.length > 0 && (
            <div className="error">
              {session.failed.map((f) => `${f.engine_id}：${f.error}`).join("；")}
            </div>
          )}
        </div>
      )}

      <div className="pref-block">
        <div className="compare-head">
          <strong>偏好画像</strong>
          <button onClick={() => void refreshProfile()}>刷新</button>
        </div>
        <div className="hint">
          只读统计：按语种与文本类型汇总你的盲听评分。它不改变默认引擎，也不参与任何自动路由。
        </div>
        {!profile || profile.cells.length === 0 ? (
          <div className="hint">暂无评分数据。完成一次对比盲听并评分后，画像会出现在这里。</div>
        ) : (
          profile.cells.map((cell) => (
            <div className="pref-cell" key={`${cell.language}-${cell.text_type}`}>
              <div className="pref-cell-title">
                {cell.language === "zh" ? "中文" : "英文"} ·{" "}
                {TEXT_TYPE_LABELS[cell.text_type] ?? cell.text_type}
              </div>
              <table className="pref-table">
                <thead>
                  <tr>
                    <th>引擎</th>
                    <th>平均分</th>
                    <th>评分次数</th>
                    <th>5 分次数</th>
                  </tr>
                </thead>
                <tbody>
                  {cell.engines.map((row) => (
                    <tr key={row.engine_id}>
                      <td>
                        {engines.find((x) => x.id === row.engine_id)?.display_name ?? row.engine_id}
                      </td>
                      <td>{row.average_score}</td>
                      <td>{row.score_count}</td>
                      <td>{row.wins}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))
        )}
      </div>
    </div>
  );
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

// --- long-text jobs: queue / segmentation / cancel (issue #13) ---

const JOB_STATUS_LABELS: Record<JobStatus, string> = {
  queued: "排队中",
  running: "进行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

function JobsSection({
  baseUrl,
  token,
  onSettled,
}: {
  baseUrl: string;
  token: string;
  onSettled: () => void;
}) {
  const [jobs, setJobs] = useState<GenerationJob[] | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [cancelError, setCancelError] = useState<string | null>(null);
  // Latest callback without re-subscribing the poller on every render.
  const settledRef = useRef(onSettled);
  settledRef.current = onSettled;
  const prevStatuses = useRef<Record<string, JobStatus>>({});

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const res = await fetch(`${baseUrl}/jobs?limit=20`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok || !alive) return;
        const data = (await res.json()) as { jobs: GenerationJob[] };
        setJobs(data.jobs);
        for (const j of data.jobs) {
          const prev = prevStatuses.current[j.id];
          if (prev && prev !== j.status && prev !== "cancelled" && j.status !== "running" && j.status !== "queued") {
            settledRef.current();
          }
          prevStatuses.current[j.id] = j.status;
        }
      } catch {
        /* sidecar restarting — keep the last snapshot */
      }
    };
    void poll();
    const timer = setInterval(poll, 2000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [baseUrl, token]);

  async function cancel(id: string) {
    const res = await fetch(`${baseUrl}/jobs/${id}/cancel`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      const detail = (await res.json().catch(() => null)) as { detail?: string } | null;
      setCancelError(detail?.detail ?? `取消失败（HTTP ${res.status}）`);
    } else {
      setCancelError(null);
    }
  }

  function progress(j: GenerationJob): string {
    const done = j.segments.filter((s) => s.status === "succeeded").length;
    return `${done}/${j.segment_count} 段`;
  }

  return (
    <section>
      <h2>任务队列</h2>
      <p className="section-hint">
        超长稿件会按引擎单次字符上限自动分段排队生成，完成后拼接为一条完整音频。
      </p>
      {cancelError && <div className="error">{cancelError}</div>}
      {jobs && jobs.length === 0 && <div className="history-empty">暂无分段任务。</div>}
      <div className="jobs-list">
        {(jobs ?? []).map((j) => (
          <div key={j.id} className={`job-item job-${j.status}`}>
            <div className="history-row">
              <span className="history-time">{fmtTime(j.created_at)}</span>
              <span className={`job-status job-status-${j.status}`}>
                {JOB_STATUS_LABELS[j.status]}
              </span>
              <span className="history-engine">{j.engine_id}</span>
              {j.voice_name && <span className="history-voice">{j.voice_name}</span>}
              <span className="history-dur">{progress(j)}</span>
              <span className="history-actions">
                {(j.status === "queued" || j.status === "running") && (
                  <button onClick={() => void cancel(j.id)}>取消</button>
                )}
                {(j.audio_url || j.segments.length > 0) && (
                  <button onClick={() => setExpanded(expanded === j.id ? null : j.id)}>
                    {expanded === j.id ? "收起分段" : "分段明细"}
                  </button>
                )}
              </span>
            </div>
            <div className="history-text">{j.text.length > 120 ? `${j.text.slice(0, 120)}…` : j.text}</div>
            {j.status === "failed" && j.error && <div className="error">{j.error}</div>}
            {j.audio_url && (
              <div className="job-audio">
                <Waveform url={`${baseUrl}${j.audio_url}`} headers={{ Authorization: `Bearer ${token}` }} />
              </div>
            )}
            {expanded === j.id && (
              <div className="job-segments">
                {j.segments.map((s) => (
                  <div key={s.index} className={`job-segment job-segment-${s.status}`}>
                    <span className="job-segment-index">#{s.index + 1}</span>
                    <span className={`job-segment-status`}>{s.status}</span>
                    <span className="job-segment-text">
                      {s.text.length > 60 ? `${s.text.slice(0, 60)}…` : s.text}
                    </span>
                    {s.error && <span className="error">{s.error}</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
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
  // Issue #13: hint shown when long text was routed to the segmented job queue.
  const [jobHint, setJobHint] = useState<string | null>(null);
  const [normalized, setNormalized] = useState<NormalizeResult | null>(null);
  const [voices, setVoices] = useState<Voice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null);
  const [newVoiceName, setNewVoiceName] = useState("");
  const [newVoiceDesc, setNewVoiceDesc] = useState("");
  const [newVoiceFile, setNewVoiceFile] = useState<File | null>(null);
  const [newVoiceAvatar, setNewVoiceAvatar] = useState<File | null>(null);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [creatingVoice, setCreatingVoice] = useState(false);
  // Issue #14: voice-package import + whole-library backup/restore.
  const [importingVoice, setImportingVoice] = useState(false);
  const [backupBusy, setBackupBusy] = useState(false);
  const [restoreBusy, setRestoreBusy] = useState(false);
  const [backupMessage, setBackupMessage] = useState<string | null>(null);
  // Voice design (issue #11): create a voice from a text description.
  const [designEngineId, setDesignEngineId] = useState("");
  const [designName, setDesignName] = useState("");
  const [designPrompt, setDesignPrompt] = useState("");
  const [designPreviewText, setDesignPreviewText] = useState("你好，很高兴认识你。");
  const [designing, setDesigning] = useState(false);
  const [designError, setDesignError] = useState<string | null>(null);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [rerunHint, setRerunHint] = useState<string | null>(null);
  const [transProviders, setTransProviders] = useState<TranscriptionProviders | null>(null);
  const [localTransInstalling, setLocalTransInstalling] = useState(false);
  const [transError, setTransError] = useState<string | null>(null);
  // BYOK keys (issue #9): inputs + per-engine busy flags; values live in the
  // OS key store after save — the renderer never persists them.
  const [keyInputs, setKeyInputs] = useState<Record<string, string>>({});
  const [keyBusy, setKeyBusy] = useState<Record<string, boolean>>({});
  const [keyError, setKeyError] = useState<string | null>(null);
  // Per-engine generation parameters, driven strictly by the engine's
  // declared param specs (unsupported parameters are never rendered).
  const [engineParams, setEngineParams] = useState<Record<string, string>>({});
  // Parameters carried back from a rerun: everything the record stored except
  // server-side paths the sidecar re-injects itself (ref_audio). They ride
  // along on the next generate call unless the user clears them.
  const [rerunParams, setRerunParams] = useState<Record<string, unknown> | null>(null);
  // Issue #15: first-use voice consent. Fetched once the sidecar is up;
  // a modal gate blocks the workspace until the user acknowledges.
  const [consent, setConsent] = useState<ConsentInfo | null>(null);
  const [consentAck, setConsentAck] = useState(false);
  const [consentBusy, setConsentBusy] = useState(false);
  const generateRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const installingRef = useRef<Record<string, boolean>>({});

  const selectedEngineInfo = engines.find((e) => e.id === selectedEngine) ?? null;
  // Voice design (issue #11): the engine picked in the design panel.
  const selectedDesignEngine =
    engines.find((e) => e.id === designEngineId) ?? null;

  // Re-derive parameter state from the selected engine's spec; parameters the
  // spec does not declare are simply not present here.
  useEffect(() => {
    const spec = selectedEngineInfo?.params ?? [];
    setEngineParams(
      Object.fromEntries(
        spec.filter((p) => p.default != null).map((p) => [p.name, String(p.default)]),
      ),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedEngine, engines]);

  async function refreshEngines() {
    if (!info) return;
    const res = await fetch(`${info.baseUrl}/engines`, {
      headers: { Authorization: `Bearer ${info.token}` },
    });
    if (res.ok) setEngines(((await res.json()) as { engines: EngineInfo[] }).engines);
  }

  async function saveKey(engineId: string) {
    if (!info) return;
    const key = (keyInputs[engineId] ?? "").trim();
    if (!key) {
      setKeyError("API Key 不能为空");
      return;
    }
    setKeyBusy((p) => ({ ...p, [engineId]: true }));
    setKeyError(null);
    try {
      const res = await fetch(`${info.baseUrl}/settings/keys`, {
        method: "PUT",
        headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ engine_id: engineId, key }),
      });
      if (!res.ok) {
        const detail = (await res.json().catch(() => null)) as { detail?: string } | null;
        setKeyError(detail?.detail ?? `保存 API Key 失败（HTTP ${res.status}）`);
        return;
      }
      setKeyInputs((p) => ({ ...p, [engineId]: "" }));
      await refreshEngines();
    } finally {
      setKeyBusy((p) => ({ ...p, [engineId]: false }));
    }
  }

  async function deleteKey(engineId: string) {
    if (!info) return;
    setKeyBusy((p) => ({ ...p, [engineId]: true }));
    setKeyError(null);
    try {
      const res = await fetch(`${info.baseUrl}/settings/keys/${engineId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${info.token}` },
      });
      if (!res.ok) {
        const detail = (await res.json().catch(() => null)) as { detail?: string } | null;
        setKeyError(detail?.detail ?? `删除 API Key 失败（HTTP ${res.status}）`);
        return;
      }
      await refreshEngines();
    } finally {
      setKeyBusy((p) => ({ ...p, [engineId]: false }));
    }
  }

  async function acknowledgeConsent() {
    if (!info || !consentAck || consentBusy) return;
    setConsentBusy(true);
    try {
      const res = await fetch(`${info.baseUrl}/consent`, {
        method: "PUT",
        headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ acknowledged: true }),
      });
      if (res.ok) setConsent((await res.json()) as ConsentInfo);
    } finally {
      setConsentBusy(false);
    }
  }

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

      // Issue #15: first-use voice consent gate state. A failed read must
      // not break startup (the sidecar may be mid-upgrade); the gate stays
      // open only when the sidecar positively reports an acknowledged state.
      try {
        const consentRes = await fetch(`${sidecarInfo.baseUrl}/consent`, { headers });
        if (consentRes.ok) setConsent((await consentRes.json()) as ConsentInfo);
      } catch {
        /* keep consent null; the workspace opens but consent stays unset */
      }

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
      let res: Response;
      try {
        res = await fetch(`${info.baseUrl}/transcription/local/status`, {
          headers: { Authorization: `Bearer ${info.token}` },
        });
      } catch {
        return; // sidecar restarting — keep polling
      }
      if (!res.ok) {
        // A non-ok status ends the install lifecycle; never spin forever.
        setLocalTransInstalling(false);
        setTransError(`查询本地转写安装状态失败（HTTP ${res.status}）`);
        return;
      }
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
    setTransError(null);
    const res = await fetch(`${info.baseUrl}/transcription/provider`, {
      method: "PUT",
      headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ provider }),
    });
    if (res.ok) {
      void refreshTransProviders();
    } else {
      setTransError(`切换转写提供方失败（HTTP ${res.status}）`);
    }
  }

  async function installLocalTranscriber() {
    if (!info || localTransInstalling) return;
    setLocalTransInstalling(true);
    setTransError(null);
    setLogsOpen(true);
    let res: Response;
    try {
      res = await fetch(`${info.baseUrl}/transcription/local/install`, {
        method: "POST",
        headers: { Authorization: `Bearer ${info.token}` },
      });
    } catch (err) {
      setLocalTransInstalling(false);
      setTransError(`发起本地转写安装失败：${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    if (!res.ok) {
      const detail = (await res.json().catch(() => null)) as { detail?: string } | null;
      setLocalTransInstalling(false);
      setTransError(detail?.detail ?? `发起本地转写安装失败（HTTP ${res.status}）`);
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

  async function createDesignedVoice() {
    if (!info || !designEngineId || designing) return;
    setDesigning(true);
    setDesignError(null);
    try {
      const res = await fetch(`${info.baseUrl}/voices/design`, {
        method: "POST",
        headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          engine_id: designEngineId,
          name: designName,
          voice_prompt: designPrompt,
          preview_text: designPreviewText,
        }),
      });
      if (!res.ok) {
        const detail = (await res.json().catch(() => null)) as { detail?: string } | null;
        setDesignError(detail?.detail ?? `设计音色失败（HTTP ${res.status}）`);
        return;
      }
      const voice = (await res.json()) as Voice;
      setDesignName("");
      setDesignPrompt("");
      await refreshVoices();
      setSelectedVoice(voice.id);
    } catch (err) {
      setDesignError(`设计音色失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setDesigning(false);
    }
  }

  // Issue #14: import a shared voice package. Bindings are dropped; the
  // user rebinds the imported voice on demand against their own engines.
  async function importVoicePackage(file: File) {
    if (!info || importingVoice) return;
    setImportingVoice(true);
    setVoiceError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${info.baseUrl}/voices/import`, {
        method: "POST",
        headers: { Authorization: `Bearer ${info.token}` },
        body,
      });
      if (!res.ok) {
        setVoiceError(await errorDetail(res, `导入音色包失败（HTTP ${res.status}）`));
        return;
      }
      const { voice } = (await res.json()) as { voice: Voice };
      await refreshVoices();
      setSelectedVoice(voice.id);
    } catch (err) {
      setVoiceError(`导入音色包失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setImportingVoice(false);
    }
  }

  async function backupLibrary() {
    if (!info || backupBusy) return;
    setBackupBusy(true);
    setBackupMessage(null);
    try {
      const stamp = new Date().toISOString().slice(0, 10);
      await downloadWithAuth(
        `${info.baseUrl}/backup`,
        info.token,
        `voice-library-backup-${stamp}.zip`,
      );
      setBackupMessage("备份已下载。请妥善保存备份文件。");
    } catch (err) {
      setBackupMessage(`备份失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBackupBusy(false);
    }
  }

  async function restoreLibrary(file: File) {
    if (!info || restoreBusy) return;
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
      const res = await fetch(`${info.baseUrl}/restore`, {
        method: "POST",
        headers: { Authorization: `Bearer ${info.token}` },
        body,
      });
      if (!res.ok) {
        setBackupMessage(await errorDetail(res, `恢复失败（HTTP ${res.status}）`));
        return;
      }
      const j = (await res.json()) as { voices: number; generations: number };
      setBackupMessage(
        `恢复完成：${j.voices} 个音色、${j.generations} 条历史记录。API Key 不在备份内，仍在系统钥匙串中。`,
      );
      await Promise.all([refreshVoices(), refreshEngines()]);
    } catch (err) {
      setBackupMessage(`恢复失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRestoreBusy(false);
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
    setJobHint(null);
    const payload = {
      engine_id: selectedEngine,
      text,
      ...(selectedVoice ? { voice_id: selectedVoice } : {}),
      ...(rerunParams || Object.keys(engineParams).length > 0
        ? {
            params: {
              ...engineParams,
              // Rerun carries only parameters the engine still declares —
              // undeclared parameters are never sent, only not rendered.
              ...Object.fromEntries(
                Object.entries(rerunParams ?? {}).filter(([k]) =>
                  (selectedEngineInfo?.params ?? []).some((p) => p.name === k),
                ),
              ),
            },
          }
        : {}),
    };
    // Issue #13: over the engine's per-request character limit the text is
    // auto-segmented and QUEUED instead of running synchronously — the job
    // panel below shows queue state, per-segment progress and cancellation.
    const maxChars = selectedEngineInfo?.capabilities?.max_chars_per_request ?? null;
    const longText = maxChars != null && text.length > maxChars;
    try {
      const res = await fetch(`${info.baseUrl}/generations${longText ? "/jobs" : ""}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${info.token}`, "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        setGenerateError(`生成请求失败（HTTP ${res.status}）`);
        return;
      }
      if (longText) {
        const job = (await res.json()) as GenerationJob;
        setJobHint(
          `全文 ${text.length} 字符，超过单次上限 ${maxChars}，已自动分段为 ${job.segment_count} 段并加入任务队列，可在下方「任务队列」查看进度或取消。`,
        );
        setRerunParams(null);
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
      {consent && !consent.acknowledged && (
        <div className="consent-overlay" role="dialog" aria-modal="true" aria-label="声音授权确认">
          <div className="consent-modal">
            <h2>声音授权确认</h2>
            <p>
              在使用声音复刻功能之前，请确认以下事项：
            </p>
            <ul>
              <li>你上传的参考音频中的声音，是你本人的声音，或你已获得该声音权利人的明确授权；</li>
              <li>生成的语音由人工智能合成，产物文件会带有 AI 生成内容的元数据标记，你应在传播时如实告知听众；</li>
              <li>你不会使用本工具伪造他人声音进行欺骗、冒充或侵犯他人权益的行为。</li>
            </ul>
            <label className="consent-check">
              <input
                type="checkbox"
                checked={consentAck}
                onChange={(e) => setConsentAck(e.target.checked)}
              />
              我已阅读并理解以上内容，确认拥有相关声音的使用授权
            </label>
            <button
              className="primary"
              disabled={!consentAck || consentBusy}
              onClick={() => void acknowledgeConsent()}
            >
              {consentBusy ? "提交中…" : "确认并开始使用"}
            </button>
          </div>
        </div>
      )}
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
        <div className="voice-create-row">
          <label>
            导入音色包（.zip）：
            <input
              type="file"
              accept=".zip"
              disabled={importingVoice}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void importVoicePackage(f);
                e.target.value = "";
              }}
            />
          </label>
          <span className="hint">
            导入后引擎绑定不会带来——需要时对导入的音色重新绑定引擎。
          </span>
        </div>
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
            <span className="hint">
              上传参考音频前请确认你拥有该声音的使用授权（首次使用时已确认）。
            </span>
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
        <div className="voice-design">
          <div className="voice-design-title">用文字描述创造音色（无需参考音频）</div>
          <div className="voice-create-row">
            <label>
              设计引擎：
              <select
                value={designEngineId}
                onChange={(e) => {
                  setDesignEngineId(e.target.value);
                  setDesignError(null);
                }}
              >
                <option value="">（选择引擎）</option>
                {engines.map((e) => {
                  const supported = e.capabilities.voice_design;
                  const reason = !supported
                    ? "不支持：引擎未声明音色设计能力"
                    : e.requires_key && !e.key_configured
                      ? "需先在「设置」页配置 API Key"
                      : "";
                  return (
                    <option key={e.id} value={e.id} disabled={!supported}>
                      {e.display_name}
                      {reason ? ` — ${reason}` : ""}
                    </option>
                  );
                })}
              </select>
            </label>
          </div>
          {selectedDesignEngine && !selectedDesignEngine.capabilities.voice_design && (
            <div className="hint">
              所选引擎不支持音色设计（能力未声明），请改用支持设计的引擎。
            </div>
          )}
          <div className="voice-create-row">
            <input
              placeholder="音色名称（必填）"
              value={designName}
              onChange={(e) => setDesignName(e.target.value)}
            />
            <input
              placeholder="声音描述，如：低沉缓慢的男声，适合纪录片旁白（必填）"
              value={designPrompt}
              onChange={(e) => setDesignPrompt(e.target.value)}
            />
            <input
              placeholder="试听文本（设计完成后用它生成预览样本）"
              value={designPreviewText}
              onChange={(e) => setDesignPreviewText(e.target.value)}
            />
            <button
              onClick={createDesignedVoice}
              disabled={
                designing ||
                !designName.trim() ||
                !designPrompt.trim() ||
                !designPreviewText.trim() ||
                !designEngineId ||
                !selectedDesignEngine?.capabilities.voice_design ||
                !!selectedDesignEngine?.requires_key &&
                !selectedDesignEngine.key_configured
              }
            >
              {designing ? "设计中…" : "设计音色"}
            </button>
          </div>
          {selectedDesignEngine?.requires_key && !selectedDesignEngine.key_configured && (
            <div className="hint">
              该设计引擎需要 API Key（BYOK）：请先在下方「设置」中保存后再设计。
            </div>
          )}
          {designError && <div className="error">{designError}</div>}
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
          {transError && <div className="error">{transError}</div>}
        </div>
      </section>

      <section ref={generateRef}>
        <h2>生成</h2>
        <div className="voice-picker">
          <label>
            使用音色：
            <select
              value={selectedVoice ?? ""}
              disabled={
                !!selectedEngineInfo &&
                selectedEngineInfo.capabilities.voice_cloning === false
              }
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
              <span className="hint">
                该引擎不支持声音复刻，音色选择已禁用（无参考音频可交给该引擎）。
              </span>
            )}
            {selectedVoice &&
              selectedEngine &&
              engines.find((e) => e.id === selectedEngine)?.capabilities
                .requires_reference_text && (
                <span className="hint">
                  该引擎需要参考文本：将自动转写所选音色的参考音频，无需手动输入。
                </span>
              )}
            {selectedEngineInfo && (
              <span className="hint training-note">
                上传内容是否用于训练：
                {selectedEngineInfo.capabilities.upload_used_for_training
                  ? "是（该引擎厂商声明会上传内容用于训练）"
                  : "否（该引擎厂商声明不上传内容用于训练）"}
                {selectedEngineInfo.data_usage_note
                  ? `　—　${selectedEngineInfo.data_usage_note}`
                  : ""}
              </span>
            )}
        </div>
        {(selectedEngineInfo?.params ?? []).map((p) => (
          <div className="param-row" key={p.name}>
            <label>
              {p.label}：
              {p.kind === "select" ? (
                <select
                  value={engineParams[p.name] ?? String(p.default ?? "")}
                  onChange={(e) =>
                    setEngineParams((prev) => ({ ...prev, [p.name]: e.target.value }))
                  }
                >
                  {p.choices.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type={p.kind === "number" ? "number" : "text"}
                  value={engineParams[p.name] ?? String(p.default ?? "")}
                  onChange={(e) =>
                    setEngineParams((prev) => ({ ...prev, [p.name]: e.target.value }))
                  }
                />
              )}
            </label>
            {p.help && <span className="hint">{p.help}</span>}
          </div>
        ))}

        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} />

        {normalized && normalized.changed && (
          <div className="normalize-preview">
            <div className="normalize-title">归一化预览（生成前引擎将收到以下文本）</div>
            <div className="normalize-text">{normalized.normalized}</div>
          </div>
        )}
        {selectedEngineInfo?.requires_key && !selectedEngineInfo.key_configured && (
          <div className="hint">
            云端引擎需要你自己的 API Key（BYOK）：请先在下方「设置」中保存，生成已禁用。
          </div>
        )}
        {selectedEngineInfo?.requires_key && selectedEngineInfo.key_configured && !selectedVoice && (
          <div className="hint">
            云端引擎必须使用音色：请先选择一个音色，首次生成时会自动把参考音频注册为云端音色。
          </div>
        )}
        <button
          className="primary"
          onClick={generate}
          disabled={
            generating ||
            !selectedEngine ||
            (!!selectedEngineInfo?.requires_key && !selectedEngineInfo.key_configured) ||
            (!!selectedEngineInfo?.requires_key && !selectedVoice) ||
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
        {jobHint && <div className="hint">{jobHint}</div>}
        {rerunParams && (
          <div className="rerun-params">
            重跑参数已载入：
            <code>{Object.keys(rerunParams).join("、")}</code>
            <button onClick={() => setRerunParams(null)}>清除</button>
          </div>
        )}

        {info && (
          <CompareSection
            baseUrl={info.baseUrl}
            token={info.token}
            voices={voices}
            engines={engines}
          />
        )}
      </section>

      <section>
        <h2>设置</h2>
        <div className="hint">
          云端引擎一律 BYOK（Bring Your Own Key）：API Key 只保存在本机系统钥匙串中，
          不落明文文件，也不经过任何第三方服务器。
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

      {info && (
        <JobsSection
          baseUrl={info.baseUrl}
          token={info.token}
          onSettled={() => setHistoryRefreshKey((k) => k + 1)}
        />
      )}

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
