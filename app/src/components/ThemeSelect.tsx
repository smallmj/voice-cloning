import type { ThemePref } from "../hooks";

/** ADR-0014: theme is a tri-state library setting (跟随系统 / 强制深色 /
 * 强制浅色) persisted in the sidecar's settings store — not localStorage.
 * Extracted as a component so the persistence contract is testable
 * independently of the rest of the settings page. */
const THEME_OPTIONS: { value: ThemePref; label: string }[] = [
  { value: "system", label: "跟随系统" },
  { value: "dark", label: "深色" },
  { value: "light", label: "浅色" },
];

export function ThemeSelect({
  value,
  onChange,
}: {
  value: ThemePref;
  onChange: (theme: ThemePref) => void;
}) {
  return (
    <label>
      主题：
      <select
        value={value}
        onChange={(e) => {
          const next = e.target.value;
          if (next === "system" || next === "dark" || next === "light") {
            onChange(next);
          }
        }}
      >
        {THEME_OPTIONS.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
