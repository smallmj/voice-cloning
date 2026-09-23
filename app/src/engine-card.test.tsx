// Issue #68: design-only engines (voice_design without voice_cloning —
// e.g. qwen3-tts-vd-cloud) keep their engines-page card but cannot be
// SELECTED for generation; the card shows a「音色设计专用」badge instead.

// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { createElement, type ReactElement } from "react";
import { describe, expect, it } from "vitest";
import { EngineCard } from "./components/EngineCard";

(window as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function mount(ui: ReactElement): { html: () => string; click: (sel: string) => void } {
  const container = document.createElement("div");
  document.body.appendChild(container);
  act(() => {
    createRoot(container).render(ui);
  });
  return {
    html: () => container.innerHTML,
    click: (sel: string) => {
      const el = container.querySelector(sel);
      if (!el) throw new Error(`missing ${sel}`);
      act(() => {
        el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      });
    },
  };
}

function engine(voiceCloning: boolean) {
  return {
    id: "qwen3-tts-vd-cloud",
    display_name: "Qwen3-TTS 音色设计（阿里百炼 · 云端）",
    capabilities: {
      languages: ["zh"],
      voice_cloning: voiceCloning,
      voice_design: true,
      pronunciation_control: false,
      emotion: false,
      commercial_license: true,
      cross_device_use: true,
      upload_used_for_training: false,
      api_closed_loop: true,
    },
    installed: true,
  };
}

describe("EngineCard design-only (issue #68)", () => {
  it("design-only card shows the badge and never fires onSelect", () => {
    let selected = 0;
    const view = mount(
      createElement(EngineCard, {
        engine: engine(false),
        designOnly: true,
        selected: false,
        installStatus: null,
        installing: false,
        matrixEntry: null,
        onInstall: () => {},
        onSelect: () => {
          selected += 1;
        },
      }),
    );
    expect(view.html()).toContain("音色设计专用");
    view.click(".engine-card");
    expect(selected).toBe(0);
  });

  it("cloning engines stay selectable and carry no badge", () => {
    let selected = 0;
    const view = mount(
      createElement(EngineCard, {
        engine: engine(true),
        designOnly: false,
        selected: false,
        installStatus: null,
        installing: false,
        matrixEntry: null,
        onInstall: () => {},
        onSelect: () => {
          selected += 1;
        },
      }),
    );
    expect(view.html()).not.toContain("音色设计专用");
    view.click(".engine-card");
    expect(selected).toBe(1);
  });
});
