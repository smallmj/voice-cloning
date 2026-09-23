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
  resolveInstallerFileName,
  sanitizeMirrorPrefix,
  sha512Hex,
  verifySha512,
  verifySha512Hex,
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

  // Regression: the real v1.2.0 latest-mac.yml lists the mac.zip as the
  // primary file (top-level path/sha512 AND the first files: entry), while the
  // app downloads the *.dmg asset. Picking the FIRST sha512 in the manifest
  // compared the dmg bytes against the zip's hash — every macOS update failed
  // with 文件校验失败（sha512 不匹配）. The hash must be resolved per installer.
  const MAC_MANIFEST_YML = [
    "version: 1.2.0",
    "files:",
    "  - url: VoiceClone-1.2.0-arm64-mac.zip",
    `    sha512: ${"b".repeat(128)}`,
    "    size: 121002296",
    "path: VoiceClone-1.2.0-arm64-mac.zip",
    `sha512: ${"b".repeat(128)}`,
    "releaseDate: '2026-09-23T09:18:01.000Z'",
    "",
  ].join("\n");
  const MAC_MANIFEST_WITH_DMG = MAC_MANIFEST_YML.replace(
    "path: VoiceClone-1.2.0-arm64-mac.zip",
    ["path: VoiceClone-1.2.0-arm64.dmg", `sha512: ${"c".repeat(128)}`].join("\n"),
  );

  it("resolves the sha512 of the files: entry matching the installer name", () => {
    expect(parseManifestSha512(MAC_MANIFEST_YML, "VoiceClone-1.2.0-arm64.dmg")).toBeNull();
  });

  it("resolves the dmg entry when the manifest lists it", () => {
    expect(parseManifestSha512(MAC_MANIFEST_WITH_DMG, "VoiceClone-1.2.0-arm64.dmg")).toBe("c".repeat(128));
    expect(parseManifestSha512(MAC_MANIFEST_WITH_DMG, "VoiceClone-1.2.0-arm64-mac.zip")).toBe("b".repeat(128));
  });

  it("still matches the top-level path/sha512 for the primary installer", () => {
    expect(parseManifestSha512(MANIFEST_YML, "VoiceClone.Setup.0.2.0.exe")).toBe(SHA);
  });

  it("without an installer name keeps the first-sha512 behavior", () => {
    expect(parseManifestSha512(MAC_MANIFEST_YML)).toBe("b".repeat(128));
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
      [MAC_MANIFEST.browser_download_url]: {
        text: `version: 0.2.0\npath: VoiceClone-0.2.0.dmg\nsha512: ${SHA}\n`,
      },
    });
    const r = await checkForUpdate({ ...base, fetch, platform: "darwin", channelMode: "official" });
    expect(r.status).toBe("available");
    if (r.status !== "available") return;
    expect(r.installerName).toBe(MAC_INSTALLER.name);
    expect(r.manifestSha512).toBe(SHA);
  });
});

describe("sanitizeMirrorPrefix (PR #63 review fix: prefix validation)", () => {
  it("accepts https://host and https://host/path, trimmed, single trailing slash", () => {
    expect(sanitizeMirrorPrefix("https://gh-proxy.com/")).toBe("https://gh-proxy.com/");
    expect(sanitizeMirrorPrefix("  https://my.example  ")).toBe("https://my.example/");
    expect(sanitizeMirrorPrefix("https://my.example/prefix")).toBe("https://my.example/prefix/");
    expect(sanitizeMirrorPrefix("https://my.example/prefix//")).toBe("https://my.example/prefix/");
  });

  it("rejects file://, http://, scheme-relative and garbage → default", () => {
    for (const bad of ["file:///etc", "http://insecure.example/", "//evil.example/", "ftp://x/", "https://", "javascript:alert(1)", "   ", 42, null]) {
      expect(sanitizeMirrorPrefix(bad as unknown as string)).toBe(DEFAULT_MIRROR_PREFIX);
    }
  });

  it("falls back to a caller-provided fallback", () => {
    expect(sanitizeMirrorPrefix("nope", "https://fallback.example/")).toBe("https://fallback.example/");
  });
});

describe("resolveInstallerFileName (PR #63 review fix: traversal guard)", () => {
  it("accepts a plain installer name with the platform extension", () => {
    expect(resolveInstallerFileName("VoiceClone Setup 0.2.0.exe", "win32")).toBe("VoiceClone Setup 0.2.0.exe");
    expect(resolveInstallerFileName("VoiceClone-0.2.0.dmg", "darwin")).toBe("VoiceClone-0.2.0.dmg");
  });

  it("rejects path traversal and subdirectory names", () => {
    expect(resolveInstallerFileName("../../evil.exe", "win32")).toBeNull();
    expect(resolveInstallerFileName("sub/evil.exe", "win32")).toBeNull();
    expect(resolveInstallerFileName("..\\evil.dmg", "darwin")).toBeNull();
  });

  it("rejects a platform-extension mismatch", () => {
    expect(resolveInstallerFileName("VoiceClone.Setup.0.2.0.exe", "darwin")).toBeNull();
    expect(resolveInstallerFileName("VoiceClone-0.2.0.dmg", "win32")).toBeNull();
    expect(resolveInstallerFileName("latest.yml", "win32")).toBeNull();
    expect(resolveInstallerFileName("", "win32")).toBeNull();
  });
});

describe("verifySha512Hex (PR #63 review fix: incremental hashing)", () => {
  it("matches verifySha512 semantics, including the missing-manifest warning", () => {
    const data = new TextEncoder().encode("installer bytes");
    expect(verifySha512Hex(sha512Hex(data), sha512Hex(data)).ok).toBe(true);
    expect(verifySha512Hex("0".repeat(128), sha512Hex(data)).ok).toBe(false);
    expect(verifySha512Hex(sha512Hex(data), null)).toEqual({ ok: false, warning: "manifest-missing" });
  });
});

describe("available result carries the resolved channel prefix (PR #63 review fix: finding 6)", () => {
  it("mirror-served results expose the mirror prefix; official results the empty prefix", async () => {
    const fetch = fakeFetch({
      [API]: { json: releaseJson() },
      [MANIFEST.browser_download_url]: { text: MANIFEST_YML },
    });
    const r = await checkForUpdate({ ...base, fetch, channelMode: "official" });
    expect(r.status).toBe("available");
    if (r.status !== "available") return;
    expect(r.effectiveChannel).toBe("official");
    expect(r.resolvedPrefix).toBe("");

    const fetchMirror = fakeFetch({
      [`https://my.example/${API}`]: { json: releaseJson() },
      [`https://my.example/${MANIFEST.browser_download_url}`]: { text: MANIFEST_YML },
    });
    const r2 = await checkForUpdate({ ...base, fetch: fetchMirror, channelMode: "mirror", mirrorPrefix: "https://my.example/" });
    expect(r2.status).toBe("available");
    if (r2.status !== "available") return;
    expect(r2.resolvedPrefix).toBe("https://my.example/");
  });
});
