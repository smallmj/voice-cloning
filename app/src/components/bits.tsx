import React, { useState } from "react";
import { filterTagGroups, groupTags, quickTags } from "../ui";
import type { SectionId } from "../ui";
import type { NonverbalTagInfo } from "../api";

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

/** Issue #68: reusable non-verbal tag control, driven entirely by the
 * engine-declared tag data from the sidecar. Renders NOTHING when `tags` is
 * empty/undefined — an engine without native tags must not offer them. When
 * tags exist: a quick-insert row (common tags) plus a searchable, category-
 * grouped dropdown of the full declared set. Pure text-editing aid — tags
 * are inserted verbatim, each engine's own syntax. */
export function NonverbalTagPicker({
  tags,
  onInsert,
}: {
  tags: NonverbalTagInfo[] | undefined | null;
  onInsert: (tag: string) => void;
}) {
  const [query, setQuery] = useState("");
  const list = tags ?? [];
  if (list.length === 0) return null;
  const groups = filterTagGroups(groupTags(list), query);
  const quick = quickTags(list);
  return (
    <div className="nv-tags" role="group" aria-label="插入非语言标签">
      {quick.map((tag) => (
        <button
          key={tag.text}
          type="button"
          className="nv-tag"
          title={`插入非语言标签 ${tag.text}（${tag.label}）`}
          onClick={() => onInsert(tag.text)}
        >
          {tag.text}
        </button>
      ))}
      <details className="nv-tag-picker">
        <summary title="全部非语言标签（可搜索）">
          全部标签（{list.length}）
        </summary>
        <div className="nv-tag-menu">
          <input
            type="search"
            className="nv-tag-search"
            placeholder="搜索标签…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {groups.length === 0 && <div className="hint">无匹配标签</div>}
          {groups.map((g) => (
            <div key={g.category} className="nv-tag-group">
              <div className="nv-tag-category">{g.category}</div>
              {g.tags.map((tag) => (
                <button
                  key={tag.text}
                  type="button"
                  className="nv-tag"
                  title={`${tag.label}（${tag.verification}）`}
                  onClick={() => onInsert(tag.text)}
                >
                  {tag.text}
                </button>
              ))}
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}
