import { useEffect, useState } from "react";
import type { EngineInfo, GenerationListResult, GenerationRecord } from "../api";
import { authHeaders } from "../client";
import { fmtCost, fmtDuration, fmtTime } from "../ui";

const HISTORY_PAGE_SIZE = 10;

export function HistorySection({
  baseUrl,
  token,
  mediaToken,
  engines,
  onRerun,
  refreshKey,
}: {
  baseUrl: string;
  token: string;
  mediaToken: string;
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
          headers: authHeaders(token),
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
      headers: authHeaders(token),
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
      <h2>生成历史</h2>
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
        <div className="history-empty">还没有生成记录。生成一次后，这里会留下完整档案。</div>
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
                    <audio controls preload="none" src={`${baseUrl}${rec.audio_url}?token=${encodeURIComponent(mediaToken)}`} />
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
