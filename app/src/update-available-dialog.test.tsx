// Startup 更新弹窗 tests (PR #63 review fix finding 4, spec #62 stories
// 7/8/10): jsdom render with a mocked window.voiceclone bridge, asserting
// title/notes, 去更新 navigation callback, 关闭, and the 跳过此版本
// checkbox → skipVersion. Precedent: update-card.test.tsx.

// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { createElement } from "react";
import { describe, expect, it } from "vitest";
import { UpdateAvailableDialog } from "./components/UpdateAvailableDialog";
import type { NewVersionInfo, UpdateEvent } from "./update-client";
import type { UpdateSettings, UpdaterResult } from "./updater-types";

(window as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const AVAILABLE: NewVersionInfo = {
  status: "available",
  latestTag: "v0.3.0",
  releaseNotes: "第一段更新说明\n\n第二段更新说明",
  installerUrl: "https://github.com/x/VoiceClone.Setup.0.3.0.exe",
  installerName: "VoiceClone.Setup.0.3.0.exe",
  effectiveChannel: "mirror",
  resolvedPrefix: "https://gh-proxy.com/",
  manifestSha512: "abc",
};

interface DialogHarness {
  html: () => string;
  pushNewVersion: (r: NewVersionInfo) => void;
  pushEvent: (e: UpdateEvent) => void;
  clickButton: (label: string) => Promise<void>;
  setCheckbox: (checked: boolean) => Promise<void>;
  skippedTags: string[];
  unmount: () => Promise<void>;
}

async function renderDialog(onGoToUpdate: () => void = () => {}): Promise<DialogHarness> {
  const skippedTags: string[] = [];
  let newVersionCb: ((r: NewVersionInfo) => void) | null = null;
  let eventCb: ((e: UpdateEvent) => void) | null = null;
  const settings: UpdateSettings = { channelMode: "auto", mirrorPrefix: "https://gh-proxy.com/", skippedTag: null };
  const bridge = {
    getUpdateStatus: async () => ({ currentVersion: "0.2.0", lastCheck: null, settings }),
    checkForUpdate: async (): Promise<UpdaterResult> => ({ status: "up-to-date", latestTag: "0", effectiveChannel: "mirror" }),
    saveUpdateSettings: async () => {},
    skipVersion: async (tag: string) => {
      skippedTags.push(tag);
    },
    startUpdateDownload: async () => {},
    cancelUpdateDownload: async () => {},
    onUpdateEvent: (cb: (e: UpdateEvent) => void) => {
      eventCb = cb;
      return () => {
        eventCb = null;
      };
    },
    onNewVersionAvailable: (cb: (r: NewVersionInfo) => void) => {
      newVersionCb = cb;
      return () => {
        newVersionCb = null;
      };
    },
  };
  (window as unknown as { voiceclone: unknown }).voiceclone = bridge;

  const container = document.createElement("div");
  document.body.appendChild(container);
  const root: Root = createRoot(container);
  await act(async () => {
    root.render(createElement(UpdateAvailableDialog, { onGoToUpdate }));
  });

  function findButton(label: string): HTMLButtonElement | null {
    return [...container.querySelectorAll("button")].find((b) => b.textContent === label) ?? null;
  }

  return {
    html: () => container.innerHTML,
    pushNewVersion: (r) => {
      act(() => {
        newVersionCb?.(r);
      });
    },
    pushEvent: (e) => {
      act(() => {
        eventCb?.(e);
      });
    },
    clickButton: async (label) => {
      const btn = findButton(label);
      if (!btn) throw new Error(`button not found: ${label}`);
      await act(async () => {
        btn.click();
        await Promise.resolve();
        await Promise.resolve();
      });
    },
    setCheckbox: async (checked) => {
      const box = container.querySelector<HTMLInputElement>("input[type=checkbox]");
      if (!box) throw new Error("checkbox not found");
      await act(async () => {
        box.click();
        if (box.checked !== checked) box.click();
        await Promise.resolve();
      });
    },
    get skippedTags() {
      return skippedTags;
    },
    unmount: async () => {
      await act(async () => {
        root.unmount();
      });
      container.remove();
      delete (window as { voiceclone?: unknown }).voiceclone;
    },
  };
}

describe("UpdateAvailableDialog (startup 更新弹窗, PR #63 review fix finding 4)", () => {
  it("renders nothing until a startup push arrives", async () => {
    const h = await renderDialog();
    expect(h.html()).toBe("");
    await h.unmount();
  });

  it("shows 版本号 + 更新日志摘要 with 去更新 / 关闭 / 跳过此版本", async () => {
    const h = await renderDialog();
    h.pushNewVersion(AVAILABLE);
    expect(h.html()).toContain("发现新版本 v0.3.0");
    expect(h.html()).toContain("第一段更新说明");
    expect(h.html()).toContain("去更新");
    expect(h.html()).toContain("关闭");
    expect(h.html()).toContain("跳过此版本");
    await h.unmount();
  });

  it("去更新 closes the dialog and navigates to the settings section", async () => {
    let navigated = 0;
    const h = await renderDialog(() => {
      navigated += 1;
    });
    h.pushNewVersion(AVAILABLE);
    await h.clickButton("去更新");
    expect(navigated).toBe(1);
    expect(h.html()).toBe("");
    await h.unmount();
  });

  it("关闭 with 跳过此版本 checked calls skipVersion(latestTag)", async () => {
    const h = await renderDialog();
    h.pushNewVersion(AVAILABLE);
    await h.setCheckbox(true);
    await h.clickButton("关闭");
    expect(h.skippedTags).toEqual(["v0.3.0"]);
    expect(h.html()).toBe("");
    await h.unmount();
  });

  it("关闭 without the checkbox does not skip", async () => {
    const h = await renderDialog();
    h.pushNewVersion(AVAILABLE);
    await h.clickButton("关闭");
    expect(h.skippedTags).toEqual([]);
    await h.unmount();
  });

  it("a dismissed tag is not shown again in the same app run", async () => {
    const h = await renderDialog();
    h.pushNewVersion(AVAILABLE);
    await h.clickButton("关闭");
    h.pushNewVersion(AVAILABLE);
    expect(h.html()).toBe("");
    await h.unmount();
  });
});
