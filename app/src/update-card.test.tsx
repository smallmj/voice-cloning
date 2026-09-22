// Issue #66: 「关于 / 更新」 card tests (spec #62, ADR-0020). Pure-logic
// cases (reducer, release-notes splitting, channel labels, settings
// derivation) plus jsdom interaction cases with a mocked window.voiceclone
// bridge (src/test-support/update-card-dom.ts) so click/change handlers and
// pushed progress events can be asserted. Precedent: nv-tags.test.tsx.

// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import {
  DEFAULT_MIRROR_PREFIX,
  RELEASES_PAGE_URL,
  type UpdateEvent,
  type UpdateSettings,
  effectiveChannelLabel,
  initialUpdateState,
  releaseNotesParagraphs,
  settingsForChannelChange,
  settingsForMirrorChange,
  updateReducer,
} from "./update-client";
import type { UpdaterResult } from "./updater-types";
import { renderCard, type CardHarness } from "./test-support/update-card-dom";


/** UpdaterAvailable fixture with the required fields filled. */
const avail = (
  over: Partial<Extract<UpdaterResult, { status: "available" }>> & { latestTag: string },
): Extract<UpdaterResult, { status: "available" }> => ({
  status: "available" as const,
  releaseNotes: "",
  installerUrl: "https://gh-proxy.com/x/VoiceClone-Setup-0.2.0.exe",
  installerName: "VoiceClone Setup 0.2.0.exe",
  manifestSha512: "abc",
  effectiveChannel: "mirror",
  ...over,
});

const BASE_SETTINGS: UpdateSettings = {
  channelMode: "auto",
  mirrorPrefix: DEFAULT_MIRROR_PREFIX,
  skippedTag: null,
};

async function open(
  overrides: Parameters<typeof renderCard>[0],
): Promise<{ h: CardHarness; html: string }> {
  const h = await renderCard(overrides);
  return { h, html: h.html() };
}

describe("update-client pure helpers (issue #66)", () => {
  it("splits a release body into trimmed plain-text paragraphs", () => {
    expect(releaseNotesParagraphs("第一段\n\n第二段\n\n\n第三段")).toEqual([
      "第一段",
      "第二段",
      "第三段",
    ]);
    expect(releaseNotesParagraphs("  \n\n  ")).toEqual([]);
  });

  it("labels effective channels (镜像 / GitHub 官方 / 自动)", () => {
    expect(effectiveChannelLabel("mirror")).toBe("镜像");
    expect(effectiveChannelLabel("official")).toBe("GitHub 官方");
    expect(effectiveChannelLabel("auto")).toBe("自动");
  });

  it("keeps other settings when the channel changes", () => {
    const prev = { ...BASE_SETTINGS, skippedTag: "v0.1.9" };
    expect(settingsForChannelChange(prev, "official")).toEqual({
      channelMode: "official",
      mirrorPrefix: DEFAULT_MIRROR_PREFIX,
      skippedTag: "v0.1.9",
    });
  });

  it("falls back to the default mirror prefix when the edit is emptied", () => {
    expect(settingsForMirrorChange(BASE_SETTINGS, "  ")).toEqual({
      ...BASE_SETTINGS,
      mirrorPrefix: DEFAULT_MIRROR_PREFIX,
    });
    expect(settingsForMirrorChange(BASE_SETTINGS, " https://my-proxy.example/ ").mirrorPrefix).toBe(
      "https://my-proxy.example/",
    );
  });
});

describe("updateReducer (issue #66)", () => {
  it("progress events update percent and byte counters", () => {
    let s = updateReducer(initialUpdateState, { type: "check-start" });
    s = updateReducer(s, {
      type: "check-result",
      result: avail({ latestTag: "v0.2.0", effectiveChannel: "mirror" }),
    });
    s = updateReducer(s, { type: "download-start" });
    expect(s.phase).toBe("downloading");
    s = updateReducer(s, { type: "progress", percent: 42, received: 42, total: 100 });
    expect(s.progress).toEqual({ percent: 42, received: 42, total: 100 });
    s = updateReducer(s, { type: "progress", percent: 100, received: 100, total: 100 });
    expect(s.progress?.percent).toBe(100);
    expect(s.phase).toBe("downloading");
  });

  it("done carries the manifest-missing warning; failed carries the message", () => {
    let s = updateReducer(initialUpdateState, { type: "done", warning: "manifest-missing" });
    expect(s.phase).toBe("done");
    expect(s.doneWarning).toBe("manifest-missing");
    s = updateReducer(s, { type: "failed", message: "网络错误" });
    expect(s.phase).toBe("failed");
    expect(s.failMessage).toBe("网络错误");
  });
});

describe("AboutUpdateCard (issue #66, jsdom)", () => {
  it("renders the up-to-date view from getUpdateStatus", async () => {
    const { h, html } = await open({
      status: {
        currentVersion: "0.1.9",
        lastCheck: { status: "up-to-date", latestTag: "0.1.9", effectiveChannel: "mirror" },
        settings: BASE_SETTINGS,
      },
    });
    expect(html).toContain("已是最新");
    expect(html).toContain("0.1.9");
    expect(html).toContain("检查更新");
    await h.unmount();
  });

  it("renders the available view: notes, effective channel, skip button", async () => {
    const { h, html } = await open({
      status: {
        currentVersion: "0.1.9",
        lastCheck: avail({
          latestTag: "0.2.0",
          releaseNotes: "修复了大文件转写\n\n新增历史导出",
          effectiveChannel: "mirror",
          installerName: "VoiceClone-0.2.0.exe",
          manifestSha512: "abc",
        }),
        settings: BASE_SETTINGS,
      },
    });
    expect(html).toContain("0.2.0");
    expect(html).toContain("修复了大文件转写");
    expect(html).toContain("新增历史导出");
    expect(html).toContain("实际生效渠道：镜像");
    expect(html).toContain("下载更新");
    expect(html).toContain("跳过此版本");
    await h.unmount();
  });

  it("renders release notes as text (no raw HTML injection)", async () => {
    const { h, html } = await open({
      status: {
        currentVersion: "0.1.9",
        lastCheck: avail({
          latestTag: "0.2.0",
          releaseNotes: "<script>alert(1)</script> 带标签的说明",
          effectiveChannel: "mirror",
        }),
        settings: BASE_SETTINGS,
      },
    });
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
    await h.unmount();
  });

  it("keeps unavailable results silent (no error banner)", async () => {
    const { h, html } = await open({
      status: {
        currentVersion: "0.1.9",
        lastCheck: {
          status: "unavailable",
          reason: "network-error",
          effectiveChannel: "mirror",
          attempts: [],
        },
        settings: BASE_SETTINGS,
      },
    });
    expect(html).not.toContain("更新失败");
    expect(html).not.toContain('class="error"');
    await h.unmount();
  });

  it("manual 检查更新 calls checkForUpdate(true) and renders the result", async () => {
    const calls: boolean[] = [];
    const { h } = await open({
      status: { currentVersion: "0.1.9", lastCheck: null, settings: BASE_SETTINGS },
      checkForUpdate: async (manual) => {
        calls.push(manual);
        return { status: "up-to-date", latestTag: "0.1.9", effectiveChannel: "mirror" };
      },
    });
    await h.clickButton("检查更新");
    expect(calls).toEqual([true]);
    expect(h.html()).toContain("已是最新");
    await h.unmount();
  });

  it("channel selector persists via saveUpdateSettings with derived args", async () => {
    const { h, html } = await open({
      status: { currentVersion: "0.1.9", lastCheck: null, settings: BASE_SETTINGS },
    });
    expect(html).toContain("更新渠道");
    await h.changeSelect("official");
    expect(h.savedSettings).toEqual([
      { channelMode: "official", mirrorPrefix: DEFAULT_MIRROR_PREFIX, skippedTag: null },
    ]);
    await h.unmount();
  });

  it("下载更新 starts the download and progress events update the bar", async () => {
    const { h, html } = await open({
      status: {
        currentVersion: "0.1.9",
        lastCheck: avail({
          latestTag: "0.2.0",
          effectiveChannel: "mirror",
          installerName: "VoiceClone-0.2.0.exe",
        }),
        settings: BASE_SETTINGS,
      },
    });
    expect(html).toContain("下载更新");
    await h.clickButton("下载更新");
    expect(h.downloadsStarted).toBe(1);
    const progress: Extract<UpdateEvent, { type: "progress" }> = {
      type: "progress",
      percent: 42,
      received: 420,
      total: 1000,
    };
    h.pushEvent(progress);
    expect(h.html()).toContain("42%");
    expect(h.html()).toContain("420 B");
    expect(h.html()).toContain("1000 B");
    await h.clickButton("取消下载");
    expect(h.downloadsCancelled).toBe(1);
    await h.unmount();
  });

  it("done shows the manifest-missing warning; failed shows message + release link", async () => {
    const { h } = await open({
      status: {
        currentVersion: "0.1.9",
        lastCheck: avail({
          latestTag: "0.2.0",
          effectiveChannel: "official",
          installerName: "VoiceClone-0.2.0.dmg",
        }),
        settings: BASE_SETTINGS,
      },
    });
    await h.clickButton("下载更新");
    h.pushEvent({ type: "done", warning: "manifest-missing" });
    expect(h.html()).toContain("已跳过 sha512 校验（清单缺失）");
    await h.unmount();

    const h2 = await renderCard({
      status: {
        currentVersion: "0.1.9",
        lastCheck: avail({
          latestTag: "0.2.0",
          effectiveChannel: "official",
          installerName: "VoiceClone-0.2.0.dmg",
        }),
        settings: BASE_SETTINGS,
      },
    });
    await h2.clickButton("下载更新");
    h2.pushEvent({ type: "failed", message: "镜像超时" });
    expect(h2.html()).toContain("更新失败");
    expect(h2.html()).toContain("镜像超时");
    expect(h2.html()).toContain("打开发布页");
    expect(h2.html()).toContain(RELEASES_PAGE_URL);
    await h2.unmount();
  });

  it("跳过此版本 calls skipVersion with the latest tag", async () => {
    const { h } = await open({
      status: {
        currentVersion: "0.1.9",
        lastCheck: avail({
          latestTag: "0.2.0",
          effectiveChannel: "mirror",
          installerName: "VoiceClone-0.2.0.exe",
        }),
        settings: BASE_SETTINGS,
      },
    });
    await h.clickButton("跳过此版本");
    expect(h.skippedTags).toEqual(["0.2.0"]);
    expect(h.html()).toContain("已跳过此版本");
    await h.unmount();
  });

  it("onNewVersionAvailable pushes refresh the card state", async () => {
    const { h } = await open({
      status: { currentVersion: "0.1.9", lastCheck: null, settings: BASE_SETTINGS },
    });
    h.pushNewVersion(avail({
      latestTag: "0.2.0",
      releaseNotes: "启动检测发现新版本",
      effectiveChannel: "mirror",
    }));
    expect(h.html()).toContain("0.2.0");
    expect(h.html()).toContain("启动检测发现新版本");
    await h.unmount();
  });
});
