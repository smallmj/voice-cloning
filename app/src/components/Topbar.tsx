/** ADR-0013 §2: the top bar carries global state — connection status,
 * current engine — and (spec #68 story 19) the shared right sidebar's
 * show/hide toggle lives at the TOP BAR's RIGHT edge, so the control sits
 * on the same side as the panel it opens. The open state itself persists
 * via /settings/ui (issue #36); this component only reports the intent. */
export function Topbar({
  statusText,
  connected,
  currentEngine,
  sidebarOpen,
  onToggleSidebar,
}: {
  statusText: string | null;
  connected: boolean;
  currentEngine: string | null;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}) {
  return (
    <header className="topbar">
      <div className="topbar-right">
        {currentEngine && (
          <span className="current-engine" title="当前引擎">
            {currentEngine}
          </span>
        )}
        {statusText !== null && (
          <span className={`status ${connected ? "ok" : "down"}`}>{statusText}</span>
        )}
        <button
          type="button"
          className={`sidebar-toggle ${sidebarOpen ? "active" : ""}`}
          aria-pressed={sidebarOpen}
          aria-label={sidebarOpen ? "隐藏侧边栏（实时日志与任务队列）" : "显示侧边栏（实时日志与任务队列）"}
          title="显示/隐藏侧边栏（实时日志与任务队列）"
          onClick={onToggleSidebar}
        >
          <span className="sidebar-toggle-icon" aria-hidden="true">
            {sidebarOpen ? "◨" : "◧"}
          </span>
          侧边栏
        </button>
      </div>
    </header>
  );
}
