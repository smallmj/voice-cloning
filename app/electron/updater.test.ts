// Module-level tests for the updater logic (issue #64, spec #62).
// Per the Testing Decisions: only external behavior is tested, with a fake
// injected fetch returning preset release JSON / timeouts / 404s.
import { describe, expect, it } from "vitest";
import {
  checkForUpdate,
  DEFAULT_MIRROR_PREFIX,
  isSuperseded,
  isNewerVersion,
  LATEST_RELEASE_API_URL,
  matchAssets,
  parseManifestSha512,
  resolveChannelChain,
  sha512Hex,
  verifySha512,
  type FetchLike,
  type LatestReleaseJson,
  type ReleaseAsset,
} from "./updater";

const API = LATEST_RELEASE_API_URL;
const INSTALLER: ReleaseAsset = { name: "VoiceClone.Setup.0.2.0.exe", browser_download_url: "https://github.com/smallmj/voice-cloning/releases/download/v0.2.0/VoiceClone.Setup.0.2.0.exe" };
const MAC_INSTALLER: ReleaseAsset = { name: "VoiceClone-0.2.0.dmg", browser_download_url: "https://github.com/smallmj/voice-cloning/releases/download/v0.2.0/VoiceClone-0.2.0.dmg" };
const MANIFEST: ReleaseAsset = {
  name: "latest.yml",
  browser_download_url: "https://github.com/smallmj/voice-cloning/releases/download/v0.2.0/latest.yml",
};
const MAC_MANIFEST: ReleaseAsset = {
  name: "latest-mac.yml",
  browser_download_url: "https://github.com/smallmj/voice-cloning/releases/download/v0.2.0/latest-mac.yml",
};
const SHA = "a".repeat(128);
const MANIFEST_YML = `version: 0.2.0\npath: VoiceClone.Setup.0.2.0.exe\nsha512: ${SHA}\n`;

function releaseJson(overrides: Partial<LatestReleaseJson> = {}): LatestReleaseJson {
  return { tag_name: "v0.2.0", body: "notes", assets: [INSTALLER, MANIFEST], ...overrides };
}

/** Fake fetch routing by exact URL; unmatched URLs 404. */
function fakeFetch(routes: Record<string, { json?: LatestReleaseJson; text?: string; status?: number; hangMs?: number }>): FetchLike {
  return async (url: string, init?: RequestInit) => {
    const route = routes[url];
    if (route?.hangMs !== undefined) {
      await new Promise((resolve) => setTimeout(resolve, route.hangMs));
      if (init?.signal?.aborted) throw new DOMException("The operation was aborted.", "AbortError");
      return new Response("{}", { status: 200 });
    }
    if (!route || (route.status ?? 200) >= 400) {
      return new Response("not found", { status: route?.status ?? 404 });
    }
    if (route.text !== undefined) return new Response(route.text, { status: 200 });
    return new Response(JSON.stringify(route.json ?? {}), { status: 200, headers: { "content-type": "application/json" } });
  };
}

const base = {
  currentVersion: "0.1.9",
  platform: "win32" as const,
};

describe("resolveChannelChain (更新渠道)", () => {
  it("auto (default) = mirror first, official fallback", () => {
    expect(resolveChannelChain("auto")).toEqual([
      { channel: "mirror", urlPrefix: DEFAULT_MIRROR_PREFIX },
      { channel: "official", urlPrefix: "" },
    ]);
  });

  it("official = official only", () => {
    expect(resolveChannelChain("official")).toEqual([{ channel: "official", urlPrefix: "" }]);
  });

  it("mirror = mirror first, official fallback", () => {
    expect(resolveChannelChain("mirror", "https://my-proxy.example/")).toEqual([
      { channel: "mirror", urlPrefix: "https://my-proxy.example/" },
      { channel: "official", urlPrefix: "" },
    ]);
  });

  it("empty mirror prefix drops the mirror entry", () => {
    expect(resolveChannelChain("mirror", "  ")).toEqual([{ channel: "official", urlPrefix: "" }]);
  });
});

describe("isSuperseded (跳过此版本)", () => {
  it("suppresses only when tags are equal", () => {
    expect(isSuperseded("v0.2.0", "v0.2.0")).toBe(true);
    expect(isSuperseded("v0.2.0", "v0.1.0")).toBe(false);
    expect(isSuperseded("v0.3.0", "v0.2.0")).toBe(false);
  });

  it("never suppresses without a skipped tag", () => {
    expect(isSuperseded("v0.2.0", null)).toBe(false);
    expect(isSuperseded("v0.2.0", undefined)).toBe(false);
    expect(isSuperseded("v0.2.0", "")).toBe(false);
  });
});

describe("isNewerVersion", () => {
  it("strict semver comparison", () => {
    expect(isNewerVersion("0.1.10", "0.1.9")).toBe(true);
    expect(isNewerVersion("0.1.9", "0.1.9")).toBe(false);
    expect(isNewerVersion("0.1.8", "0.1.9")).toBe(false);
  });

  it("rejects non-semver tags", () => {
    expect(isNewerVersion("not-a-version", "0.1.9")).toBe(false);
  });
});

describe("asset matching", () => {
  it("matches *.exe and latest.yml on win32", () => {
    const m = matchAssets([INSTALLER, MANIFEST], "win32");
    expect(m?.installer.name).toBe(INSTALLER.name);
    expect(m?.manifest?.name).toBe("latest.yml");
  });

  it("matches *.dmg and latest-mac.yml on darwin", () => {
    const m = matchAssets([MAC_INSTALLER, MAC_MANIFEST], "darwin");
    expect(m?.installer.name).toBe(MAC_INSTALLER.name);
    expect(m?.manifest?.name).toBe("latest-mac.yml");
  });

  it("platform mismatch (no matching installer) = no update for the channel", () => {
    expect(matchAssets([MAC_INSTALLER, MAC_MANIFEST], "win32")).toBeNull();
    expect(matchAssets([INSTALLER], "darwin")).toBeNull();
  });

  it("ignores blockmap files", () => {
    const blockmap: ReleaseAsset = { ...INSTALLER, name: `${INSTALLER.name}.blockmap` };
    expect(matchAssets([blockmap], "win32")).toBeNull();
  });
});

describe("sha512 verification", () => {
  it("verifies file bytes against the manifest hash", () => {
    const data = new TextEncoder().encode("installer-bytes");
    const good = sha512Hex(data);
    expect(verifySha512(data, good).ok).toBe(true);
    expect(verifySha512(data, SHA).ok).toBe(false);
  });

  it("missing manifest hash = warning, not failure", () => {
    const r = verifySha512(new Uint8Array([1]), null);
    expect(r.ok).toBe(false);
    expect(r.warning).toBe("manifest-missing");
  });
});

describe("parseManifestSha512", () => {
  it("extracts the sha512 field from latest.yml", () => {
    expect(parseManifestSha512(MANIFEST_YML)).toBe(SHA);
  });

  it("returns null when absent", () => {
    expect(parseManifestSha512("version: 0.2.0\npath: x.exe\n")).toBeNull();
  });
});

describe("checkForUpdate", () => {
  it("normal detection: available with effective channel and manifest hash", async () => {
    const fetch = fakeFetch({
      [`https://gh-proxy.com/${API}`]: { json: releaseJson() },
      [`https://gh-proxy.com/${MANIFEST.browser_download_url}`]: { text: MANIFEST_YML },
    });
    const r = await checkForUpdate({ ...base, fetch });
    expect(r.status).toBe("available");
    if (r.status !== "available") return;
    expect(r.effectiveChannel).toBe("mirror");
    expect(r.latestTag).toBe("v0.2.0");
    expect(r.releaseNotes).toBe("notes");
    expect(r.manifestSha512).toBe(SHA);
    expect(r.warning).toBeUndefined();
    expect(r.installerUrl).toBe(INSTALLER.browser_download_url);
  });

  it("mirror timeout falls back to official with effectiveChannel=official", async () => {
    const fetch = fakeFetch({
      [`https://gh-proxy.com/${API}`]: { hangMs: 50 },
      [API]: { json: releaseJson() },
      [`https://gh-proxy.com/${MANIFEST.browser_download_url}`]: { text: MANIFEST_YML },
      [MANIFEST.browser_download_url]: { text: MANIFEST_YML },
    });
    const r = await checkForUpdate({ ...base, fetch, timeoutMs: 20 });
    expect(r.status).toBe("available");
    if (r.status !== "available") return;
    expect(r.effectiveChannel).toBe("official");
  });

  it("probe timeout alone → unavailable, never throws", async () => {
    const fetch = fakeFetch({ [API]: { hangMs: 50 } });
    const r = await checkForUpdate({ ...base, fetch, timeoutMs: 20, channelMode: "official" });
    expect(r.status).toBe("unavailable");
    if (r.status !== "unavailable") return;
    expect(r.reason).toBe("timeout");
    expect(r.attempts).toEqual([{ channel: "official", reason: "timeout" }]);
  });

  it("404 / no release → unavailable with reason no-release", async () => {
    const r = await checkForUpdate({ ...base, fetch: fakeFetch({}), channelMode: "official" });
    expect(r.status).toBe("unavailable");
    if (r.status !== "unavailable") return;
    expect(r.reason).toBe("no-release");
  });

  it("mirror fails, official serves → correct effective channel", async () => {
    const fetch = fakeFetch({
      [`https://gh-proxy.com/${API}`]: { status: 502 },
      [API]: { json: releaseJson() },
      [MANIFEST.browser_download_url]: { text: MANIFEST_YML },
    });
    const r = await checkForUpdate({ ...base, fetch });
    expect(r.status).toBe("available");
    if (r.status !== "available") return;
    expect(r.effectiveChannel).toBe("official");
  });

  it("all channels fail → unavailable with attempt trail", async () => {
    const fetch = fakeFetch({ [`https://gh-proxy.com/${API}`]: { status: 502 }, [API]: { status: 502 } });
    const r = await checkForUpdate({ ...base, fetch });
    expect(r.status).toBe("unavailable");
    if (r.status !== "unavailable") return;
    expect(r.effectiveChannel).toBe("official");
    expect(r.attempts.map((a) => a.channel)).toEqual(["mirror", "official"]);
  });

  it("skipped tag equal to latest → status skipped", async () => {
    const fetch = fakeFetch({
      [API]: { json: releaseJson() },
      [MANIFEST.browser_download_url]: { text: MANIFEST_YML },
    });
    const r = await checkForUpdate({ ...base, fetch, channelMode: "official", skippedTag: "v0.2.0" });
    expect(r.status).toBe("skipped");
    if (r.status !== "skipped") return;
    expect(r.latestTag).toBe("v0.2.0");
  });

  it("skipped tag lower than latest → still available", async () => {
    const fetch = fakeFetch({
      [API]: { json: releaseJson() },
      [MANIFEST.browser_download_url]: { text: MANIFEST_YML },
    });
    const r = await checkForUpdate({ ...base, fetch, channelMode: "official", skippedTag: "v0.1.0" });
    expect(r.status).toBe("available");
  });

  it("same version as current → up-to-date", async () => {
    const fetch = fakeFetch({ [API]: { json: releaseJson() } });
    const r = await checkForUpdate({ ...base, fetch, channelMode: "official", currentVersion: "0.2.0" });
    expect(r.status).toBe("up-to-date");
    if (r.status !== "up-to-date") return;
    expect(r.effectiveChannel).toBe("official");
  });

  it("missing manifest → available with warning manifest-missing", async () => {
    const fetch = fakeFetch({ [API]: { json: releaseJson() } });
    const r = await checkForUpdate({ ...base, fetch, channelMode: "official" });
    expect(r.status).toBe("available");
    if (r.status !== "available") return;
    expect(r.manifestSha512).toBeNull();
    expect(r.warning).toBe("manifest-missing");
  });

  it("platform asset mismatch on every channel → unavailable", async () => {
    const fetch = fakeFetch({
      [`https://gh-proxy.com/${API}`]: { json: releaseJson({ assets: [MAC_INSTALLER, MAC_MANIFEST] }) },
      [API]: { json: releaseJson({ assets: [MAC_INSTALLER, MAC_MANIFEST] }) },
    });
    const r = await checkForUpdate({ ...base, fetch });
    expect(r.status).toBe("unavailable");
    if (r.status !== "unavailable") return;
    expect(r.reason).toBe("no-platform-asset");
    expect(r.attempts).toEqual([
      { channel: "mirror", reason: "no-platform-asset" },
      { channel: "official", reason: "no-platform-asset" },
    ]);
  });

  it("macOS detection works end-to-end on darwin", async () => {
    const fetch = fakeFetch({
      [API]: { json: releaseJson({ assets: [MAC_INSTALLER, MAC_MANIFEST] }) },
      [MAC_MANIFEST.browser_download_url]: { text: `version: 0.2.0\nsha512: ${SHA}\n` },
    });
    const r = await checkForUpdate({ ...base, fetch, platform: "darwin", channelMode: "official" });
    expect(r.status).toBe("available");
    if (r.status !== "available") return;
    expect(r.installerName).toBe(MAC_INSTALLER.name);
    expect(r.manifestSha512).toBe(SHA);
  });
});
