// ADR-0014: theme is a tri-state library setting (跟随系统 / 强制深色 /
// 强制浅色) persisted in the sidecar settings store via useUiPrefs →
// PUT /settings/ui. These component tests pin the tri-state control
// contract: exactly three options, every selection forwarded as a ThemePref
// (the SettingsSection wires onChange to updatePrefs({ theme })). The
// sidecar-side persistence/merging is covered by sidecar/tests/test_settings_ui.py.

// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { createElement } from "react";
import { describe, expect, it } from "vitest";
import { ThemeSelect } from "./components/ThemeSelect";
import type { ThemePref } from "./hooks";

(window as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function mountSelect(value: ThemePref, onChange: (t: ThemePref) => void) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(createElement(ThemeSelect, { value, onChange }));
  });
  const cleanup = async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
  };
  return { container, cleanup };
}

describe("ThemeSelect (ADR-0014 主题三态持久化)", () => {
  it("offers exactly the three tri-state options in a stable order", () => {
    const { container, cleanup } = mountSelect("system", () => {});
    try {
      const options = Array.from(container.querySelectorAll("option"));
      expect(options.map((o) => o.value)).toEqual(["system", "dark", "light"]);
      expect(options.map((o) => o.textContent)).toEqual(["跟随系统", "深色", "浅色"]);
    } finally {
      void cleanup();
    }
  });

  it("the stored pref is the selected value (no second source of truth)", async () => {
    for (const value of ["system", "dark", "light"] as const) {
      const { container, cleanup } = mountSelect(value, () => {});
      try {
        const select = container.querySelector("select") as HTMLSelectElement;
        expect(select.value).toBe(value);
      } finally {
        await cleanup();
      }
    }
  });

  it("every selection is forwarded as a ThemePref for updatePrefs({ theme })", () => {
    const seen: ThemePref[] = [];
    const { container, cleanup } = mountSelect("system", (t) => seen.push(t));
    try {
      const select = container.querySelector("select") as HTMLSelectElement;
      for (const next of ["dark", "light", "system"]) {
        act(() => {
          const setter = Object.getOwnPropertyDescriptor(
            HTMLSelectElement.prototype,
            "value",
          )!.set!;
          setter.call(select, next);
          select.dispatchEvent(new Event("change", { bubbles: true }));
        });
      }
      expect(seen).toEqual(["dark", "light", "system"]);
    } finally {
      void cleanup();
    }
  });
});
