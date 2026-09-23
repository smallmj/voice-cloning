// Spec #68 story 19 / ADR-0013 §2: the shared right sidebar's show/hide
// toggle lives at the TOP BAR's RIGHT edge — not in the left navigation
// footer (where it was before, on the opposite side from the panel). jsdom
// render tests assert render position, aria state and the toggle callback.
// The open state itself persists via /settings/ui (issue #36) — covered by
// the sidecar contract suite (test_settings_ui.py) and useUiPrefs.

// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { createElement, type ReactElement } from "react";
import { describe, expect, it } from "vitest";
import { Topbar } from "./components/Topbar";

(window as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function mount(ui: ReactElement): {
  html: () => string;
  query: (sel: string) => Element | null;
  click: (sel: string) => void;
  unmount: () => Promise<void>;
} {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(ui);
  });
  return {
    html: () => container.innerHTML,
    query: (sel) => container.querySelector(sel),
    click: (sel) => {
      const el = container.querySelector(sel);
      if (!el) throw new Error(`missing ${sel}`);
      act(() => {
        el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      });
    },
    unmount: async () => {
      await act(async () => {
        root.unmount();
      });
      container.remove();
    },
  };
}

const BASE = {
  statusText: "sidecar 已连接",
  connected: true,
  currentEngine: "IndexTTS-2.5",
  sidebarOpen: false,
  onToggleSidebar: () => {},
};

describe("Topbar (spec #68 story 19: 侧栏开关移至顶栏右侧)", () => {
  it("renders inside the topbar, with the toggle in the right-aligned cluster", async () => {
    const h = mount(createElement(Topbar, BASE));
    try {
      const topbar = h.query(".topbar");
      expect(topbar).not.toBeNull();
      const toggle = h.query(".topbar .sidebar-toggle");
      expect(toggle).not.toBeNull();
      // The right cluster hosts the toggle; the left nav contains none.
      expect(h.query(".topbar-right .sidebar-toggle")).not.toBeNull();
      expect(h.query(".side-nav .sidebar-toggle")).toBeNull();
    } finally {
      await h.unmount();
    }
  });

  it("reflects open state in aria-pressed and swaps the icon/aria-label", async () => {
    const closed = mount(createElement(Topbar, BASE));
    try {
      const btn = closed.query(".sidebar-toggle") as HTMLElement;
      expect(btn.getAttribute("aria-pressed")).toBe("false");
      expect(btn.getAttribute("aria-label")).toContain("显示侧边栏");
      expect(btn.textContent).toContain("◧");
    } finally {
      await closed.unmount();
    }
    const openState = mount(createElement(Topbar, { ...BASE, sidebarOpen: true }));
    try {
      const btn = openState.query(".sidebar-toggle") as HTMLElement;
      expect(btn.getAttribute("aria-pressed")).toBe("true");
      expect(btn.getAttribute("aria-label")).toContain("隐藏侧边栏");
      expect(btn.textContent).toContain("◨");
    } finally {
      await openState.unmount();
    }
  });

  it("clicking the toggle reports the intent (persistence stays in useUiPrefs)", async () => {
    let toggles = 0;
    const h = mount(
      createElement(Topbar, { ...BASE, onToggleSidebar: () => (toggles += 1) }),
    );
    try {
      h.click(".sidebar-toggle");
      expect(toggles).toBe(1);
    } finally {
      await h.unmount();
    }
  });

  it("carries global state: connection status and current engine", async () => {
    const h = mount(createElement(Topbar, BASE));
    try {
      expect(h.query(".topbar .status.ok")?.textContent).toBe("sidecar 已连接");
      expect(h.query(".topbar .current-engine")?.textContent).toBe("IndexTTS-2.5");
    } finally {
      void h.unmount();
    }
  });

  it("marks a failed connection with the down status style", async () => {
    const h = mount(
      createElement(Topbar, { ...BASE, connected: false, statusText: "连接失败" }),
    );
    try {
      expect(h.query(".topbar .status.down")?.textContent).toBe("连接失败");
    } finally {
      void h.unmount();
    }
  });
});
