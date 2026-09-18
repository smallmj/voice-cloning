import React from "react";
import type { SectionId } from "../ui";

export function CapBadge({ label, on }: { label: string; on: boolean }) {
  return (
    <span className={`badge ${on ? "badge-on" : "badge-off"}`}>
      {label} {on ? "✓" : "✗"}
    </span>
  );
}

/** Issue #21: hard-coded directional copy（「下方『设置』」「上方『安装引擎』」）
 * becomes an explicit cross-section jump instead of a claim about layout. */
export function JumpLink({
  target,
  label,
  onNavigate,
}: {
  target: SectionId;
  label: string;
  onNavigate: (section: SectionId) => void;
}) {
  return (
    <button type="button" className="jump-link" onClick={() => onNavigate(target)}>
      {label}
    </button>
  );
}
