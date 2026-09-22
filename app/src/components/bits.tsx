import React from "react";
import { NONVERBAL_TAGS } from "../ui";
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

/** Issue #57: quick-insert buttons for the cookbook-canonical non-verbal
 * tags. Pure text-editing aid — engine-agnostic, no VoxCPM-specific API. */
export function NonverbalTagButtons({ onInsert }: { onInsert: (tag: string) => void }) {
  return (
    <div className="nv-tags" role="group" aria-label="插入非语言标签">
      {NONVERBAL_TAGS.map((tag) => (
        <button
          key={tag}
          type="button"
          className="nv-tag"
          title={`插入非语言标签 ${tag}`}
          onClick={() => onInsert(tag)}
        >
          {tag}
        </button>
      ))}
    </div>
  );
}
