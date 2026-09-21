import { useMemo, useState } from "react";
import type { LogEvent, LogLevel } from "../api";
import { filterLogs, LOG_LEVELS, LOG_LEVEL_LABELS } from "../ui";
import { JobsSection } from "./JobsSection";

// Issue #36 / ADR-0013 2026 修订: the shared right sidebar — live logs on
// top, the job queue below — is rendered on every tab. Filtering is pure
// front-end over the buffered log stream (see ui.ts `filterLogs`).
export function SidePanel({
  className = "",
  width,
  baseUrl,
  token,
  logs,
  onSettled,
  onDragStart,
}: {
  className?: string;
  width: number;
  baseUrl: string;
  token: string;
  logs: LogEvent[];
  onSettled: () => void;
  onDragStart: (e: React.PointerEvent) => void;
}) {
  const [levels, setLevels] = useState<ReadonlySet<LogLevel>>(new Set());
  const [keyword, setKeyword] = useState("");
  const visible = useMemo(
    () => filterLogs(logs, { levels, keyword }),
    [logs, levels, keyword],
  );

  function toggleLevel(level: LogLevel) {
    setLevels((prev) => {
      const next = new Set(prev);
      if (next.has(level)) next.delete(level);
      else next.add(level);
      return next;
    });
  }

  return (
    <aside
      className={`side-panel ${className}`}
      style={{ width }}
      aria-label="实时日志与任务队列"
    >
      <div
        className="panel-drag"
        role="separator"
        aria-orientation="vertical"
        aria-label="拖动调整侧边栏宽度"
        onPointerDown={onDragStart}
      />
      <section className="panel-pane panel-logs">
        <div className="pane-head">
          <span className="pane-title">
            实时日志（{visible.length}/{logs.length}）
          </span>
          <span className="log-filters">
            {LOG_LEVELS.map((level) => (
              <label key={level} className={`log-level log-level-${level}`}>
                <input
                  type="checkbox"
                  checked={levels.has(level)}
                  onChange={() => toggleLevel(level)}
                />
                {LOG_LEVEL_LABELS[level]}
              </label>
            ))}
            <input
              className="log-keyword"
              type="search"
              placeholder="关键字筛选"
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
            />
          </span>
        </div>
        <pre className="pane-log-list">
          {visible.length === 0 ? (
            <span className="panel-empty">暂无匹配日志。</span>
          ) : (
            visible.map((l, i) => (
              // Keys ride ts+index: log frames carry no unique id.
              <div key={`${l.ts}-${i}`} className={`log-line log-line-${l.level}`}>
                <span className="log-level-tag">{LOG_LEVEL_LABELS[l.level]}</span>
                [{l.generation_id.slice(0, 6)}] {l.message}
              </div>
            ))
          )}
        </pre>
      </section>
      <section className="panel-pane panel-jobs">
        <JobsSection baseUrl={baseUrl} token={token} onSettled={onSettled} />
      </section>
    </aside>
  );
}
