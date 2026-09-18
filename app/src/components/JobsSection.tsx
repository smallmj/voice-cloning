import { useEffect, useRef, useState } from "react";
import type { GenerationJob, JobStatus } from "../api";
import { apiJson } from "../client";
import { fmtTime, jobSettled } from "../ui";
import { Waveform } from "./Waveform";

const JOB_STATUS_LABELS: Record<JobStatus, string> = {
  queued: "排队中",
  running: "进行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

export function JobsSection({
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
        const data = await apiJson<{ jobs: GenerationJob[] }>(baseUrl, token, "/jobs?limit=20");
        if (!alive) return;
        setJobs(data.jobs);
        for (const j of data.jobs) {
          if (jobSettled(prevStatuses.current[j.id], j.status)) {
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
    try {
      await apiJson(baseUrl, token, `/jobs/${id}/cancel`, { method: "POST" });
      setCancelError(null);
    } catch (e) {
      setCancelError(e instanceof Error ? e.message : String(e));
    }
  }

  function progress(j: GenerationJob): string {
    const done = j.segments.filter((s) => s.status === "succeeded").length;
    return `${done}/${j.segment_count} 段`;
  }

  return (
    <section className="jobs-subsection">
      <h3>任务队列</h3>
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
                <Waveform url={`${baseUrl}${j.audio_url}`} token={token} />
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
