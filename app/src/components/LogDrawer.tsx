import type { LogEvent } from "../api";

/** ADR-0013 §4: the log drawer never becomes its own section — it stays a
 * collapsible drawer inside the 生成 section. */
export function LogDrawer({
  logs,
  open,
  onToggle,
}: {
  logs: LogEvent[];
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <section className={`log-drawer ${open ? "open" : ""}`}>
      <button className="log-toggle" onClick={onToggle} aria-expanded={open}>
        {open ? "▾" : "▸"} 实时日志（{logs.length}）
      </button>
      {open && (
        <pre className="logs">
          {logs.map((l, i) => (
            <div key={i}>[{l.generation_id.slice(0, 6)}] {l.message}</div>
          ))}
        </pre>
      )}
    </section>
  );
}
