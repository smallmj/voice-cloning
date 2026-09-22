// Updater module for issue #64 (part of spec #62 / ADR-0020).
// ALL testable update logic lives here: 更新渠道 (Update Channel) resolution,
// semver comparison, release-asset platform matching, GitHub API probing with
// a 5s timeout, and sha512 verification against the electron-builder manifest.
// This module does NO IO except the injected fetch, and imports nothing from
// Electron — main.ts (ticket #65/#66) calls into it and owns all side effects.
import { createHash } from "node:crypto";
import * as path from "node:path";
import { gt as semverGt, valid as semverValid } from "semver";

/** GitHub repo queried for the latest stable release (ADR-0020 仓库常量). */
export const GITHUB_REPO = "smallmj/voice-cloning";

// DEFAULT_MIRROR_PREFIX + sanitizeMirrorPrefix live in ../src/updater-types so
// the vite renderer bundle can share them without importing node builtins
// from this module (vite externalizes those for browser builds).
import {
  DEFAULT_MIRROR_PREFIX,
  sanitizeMirrorPrefix,
} from "../src/updater-types";

export { DEFAULT_MIRROR_PREFIX, sanitizeMirrorPrefix };

/** GitHub API endpoint for the latest stable (non-prerelease) release. */
export const LATEST_RELEASE_API_URL = `https://api.github.com/repos/${GITHUB_REPO}/releases/latest`;

/** Timeout for one channel probe, in milliseconds (spec: 5s silent give-up). */
export const DEFAULT_TIMEOUT_MS = 5_000;

/** 更新渠道 (Update Channel) mode persisted in settings. */
export type ChannelMode = "auto" | "official" | "mirror";

/** A single channel in the resolved fallback chain. */
export type Channel = "mirror" | "official";

/** Minimal fetch signature the module depends on (Node/undici compatible). */
export type FetchLike = (url: string, init?: RequestInit) => Promise<Response>;

/** One entry of the resolved channel chain, with the prefix used to serve it. */
export interface ChannelEntry {
  channel: Channel;
  /** URL prefix applied to GitHub API and asset URLs for this channel. */
  urlPrefix: string;
}

/**
 * Resolve the 更新渠道 mode into an ordered fallback chain:
 * auto = [mirror, official]; official = [official]; mirror = [mirror, official].
 * An empty/whitespace mirror prefix makes the mirror entry unusable, so it is
 * dropped (mirror mode then degrades to the same chain as auto, per spec).
 */
export function resolveChannelChain(mode: ChannelMode = "auto", mirrorPrefix: string = DEFAULT_MIRROR_PREFIX): ChannelEntry[] {
  const prefix = mirrorPrefix.trim();
  const mirror: ChannelEntry[] = prefix === "" ? [] : [{ channel: "mirror", urlPrefix: prefix }];
  switch (mode) {
    case "official":
      return [{ channel: "official", urlPrefix: "" }];
    case "mirror":
      return [...mirror, { channel: "official", urlPrefix: "" }];
    case "auto":
    default:
      return [...mirror, { channel: "official", urlPrefix: "" }];
  }
}

/**
 * 跳过此版本 (Skip This Version) semantics: the latest tag is superseded by the
 * skipped tag ONLY when the two tags are equal. A newer latest tag always
 * alerts again, and a null skipped tag never suppresses anything.
 */
export function isSuperseded(latestTag: string, skippedTag: string | null | undefined): boolean {
  if (skippedTag == null || skippedTag === "") return false;
  return latestTag === skippedTag;
}

/** Strictly-newer comparison using the semver library (never hand-rolled). */
export function isNewerVersion(candidate: string, current: string): boolean {
  const c = semverValid(candidate);
  const cur = semverValid(current);
  if (c === null || cur === null) return false;
  return semverGt(c, cur);
}

/** sha512 of the given file bytes, hex-encoded (manifest comparison form). */
export function sha512Hex(data: Uint8Array): string {
  return createHash("sha512").update(data).digest("hex");
}

export interface VerifySha512Result {
  ok: boolean;
  /** Set when the expected hash was absent, so no verdict could be made. */
  warning?: "manifest-missing";
}

/**
 * Verify downloaded file bytes against the manifest sha512. A missing expected
 * hash is a warning, not a failure (spec: 无清单则警告但允许继续).
 */
export function verifySha512(data: Uint8Array, expectedSha512: string | null | undefined): VerifySha512Result {
  return verifySha512Hex(sha512Hex(data), expectedSha512);
}

/**
 * Same verdict for an already-computed hex digest (review fix PR #63 finding
 * 8: the download streams through an incremental hash, so main.ts never
 * re-reads the whole installer just to verify it).
 */
export function verifySha512Hex(digestHex: string, expectedSha512: string | null | undefined): VerifySha512Result {
  if (expectedSha512 == null || expectedSha512.trim() === "") {
    return { ok: false, warning: "manifest-missing" };
  }
  return { ok: digestHex.toLowerCase() === expectedSha512.trim().toLowerCase() };
}

/**
 * Extract the sha512 field from an electron-builder latest.yml /
 * latest-mac.yml manifest text. Parsed with a targeted regex instead of a YAML
 * dependency; returns null when absent (missing-manifest path).
 */
export function parseManifestSha512(manifestText: string): string | null {
  const match = manifestText.match(/^\s*sha512:\s*(\S+)\s*$/m);
  return match ? match[1] : null;
}

/** Platform key used for asset matching (process.platform style). */
export type UpdatePlatform = "win32" | "darwin";

/** The installer asset extension expected for each platform. */
function installerExtension(platform: UpdatePlatform): ".exe" | ".dmg" {
  return platform === "win32" ? ".exe" : ".dmg";
}

/** The checksum manifest file expected for each platform. */
export function manifestFileName(platform: UpdatePlatform): "latest.yml" | "latest-mac.yml" {
  return platform === "win32" ? "latest.yml" : "latest-mac.yml";
}

/**
 * Review fix (PR #63 finding 1): the installer name arrives from the release
 * JSON and is later joined under the temp dir — a hostile mirror could send
 * "../../Something" and escape the temp directory before the sha512 check
 * (the manifest comes through the same channel, so it provides no defense).
 * Returns the basename ONLY when it survives unmodified and carries the
 * platform installer extension; null means "treat the result as failed" so
 * the caller falls back to the release page in the browser.
 */
export function resolveInstallerFileName(installerName: string, platform: UpdatePlatform): string | null {
  if (typeof installerName !== "string" || installerName === "") return null;
  // Reject ANY path separator (both flavors: the file is joined on the local
  // OS, so a backslash is a separator on Windows even though basename here
  // runs with posix rules in tests).
  if (/[/\\]/.test(installerName)) return null;
  const fileName = path.basename(installerName);
  if (fileName !== installerName) return null; // separators / traversal in the raw name
  if (!fileName.toLowerCase().endsWith(installerExtension(platform))) return null;
  return fileName;
}

export interface ReleaseAsset {
  name: string;
  browser_download_url: string;
}

/**
 * Match the platform installer asset (*.exe on win / *.dmg on mac) and the
 * checksum manifest (latest.yml / latest-mac.yml) in a release asset list.
 * Returns null when the installer asset is absent (mismatch = no update
 * available for that channel); the manifest may legitimately be missing.
 */
export function matchAssets(
  assets: readonly ReleaseAsset[],
  platform: UpdatePlatform,
): { installer: ReleaseAsset; manifest: ReleaseAsset | null } | null {
  const ext = installerExtension(platform);
  const installer = assets.find((a) => a.name.toLowerCase().endsWith(ext) && !a.name.toLowerCase().endsWith(`${ext}.blockmap`));
  if (!installer) return null;
  const manifestName = manifestFileName(platform);
  const manifest = assets.find((a) => a.name.toLowerCase() === manifestName) ?? null;
  return { installer, manifest };
}

/** Minimal shape of the GitHub releases/latest JSON the module consumes. */
export interface LatestReleaseJson {
  tag_name?: unknown;
  body?: unknown;
  assets?: unknown;
}

interface ParsedRelease {
  tag: string;
  body: string;
  assets: ReleaseAsset[];
}

function parseReleaseJson(json: LatestReleaseJson): ParsedRelease | null {
  if (typeof json.tag_name !== "string" || json.tag_name === "") return null;
  const assets: ReleaseAsset[] = Array.isArray(json.assets)
    ? json.assets
        .filter(
          (a): a is ReleaseAsset =>
            typeof a === "object" && a !== null && typeof (a as ReleaseAsset).name === "string" && typeof (a as ReleaseAsset).browser_download_url === "string",
        )
        .map((a) => ({ name: a.name, browser_download_url: a.browser_download_url }))
    : [];
  return {
    tag: json.tag_name,
    body: typeof json.body === "string" ? json.body : "",
    assets,
  };
}

export type UnavailableReason =
  | "timeout"
  | "http-error"
  | "no-release"
  | "bad-payload"
  | "no-platform-asset"
  | "network-error";

export type EffectiveChannel = Channel | null;

/** A result that failed for every channel — always silent on the caller side. */
export interface UpdaterUnavailable {
  status: "unavailable";
  reason: UnavailableReason;
  /** Last channel that was tried, when any was reached. */
  effectiveChannel: EffectiveChannel;
  /** Per-channel failure detail, for debugging/UI display. */
  attempts: ReadonlyArray<{ channel: Channel; reason: UnavailableReason }>;
}

export interface UpdaterAvailable {
  status: "available";
  latestTag: string;
  releaseNotes: string;
  installerUrl: string;
  installerName: string;
  effectiveChannel: Channel;
  /**
   * URL prefix of the channel that actually served this result (review fix
   * PR #63 finding 6): the download must reuse the SAME channel prefix, not
   * re-read the (user-editable) settings value, which may have changed
   * between the check and the download click.
   */
  resolvedPrefix: string;
  /** sha512 from the manifest, null when the manifest was missing. */
  manifestSha512: string | null;
  /** Set when the checksum manifest was absent (warning, not failure). */
  warning?: "manifest-missing";
}

export interface UpdaterUpToDate {
  status: "up-to-date";
  latestTag: string;
  effectiveChannel: Channel;
}

export interface UpdaterSkipped {
  status: "skipped";
  latestTag: string;
  effectiveChannel: Channel;
}

export type UpdaterResult = UpdaterAvailable | UpdaterUnavailable | UpdaterUpToDate | UpdaterSkipped;

/** Everything the check needs, injected — the module persists nothing itself. */
export interface CheckForUpdateInput {
  /** Current app version (semver), from app metadata. */
  currentVersion: string;
  /** 更新渠道 mode; default "auto". */
  channelMode?: ChannelMode;
  /** Editable 镜像前缀; default DEFAULT_MIRROR_PREFIX. */
  mirrorPrefix?: string;
  /** 跳过此版本 tag persisted in settings, if any. */
  skippedTag?: string | null;
  /** Injected fetch (test seam; the only IO this module performs). */
  fetch: FetchLike;
  /** Probe timeout in ms; default 5000. */
  timeoutMs?: number;
  /** Platform for asset matching; default derived at call site in main.ts. */
  platform: UpdatePlatform;
}

type ProbeOutcome =
  | { ok: true; release: ParsedRelease }
  | { ok: false; reason: UnavailableReason };

async function probeChannel(fetch: FetchLike, entry: ChannelEntry, timeoutMs: number): Promise<ProbeOutcome> {
  // Mirror channel note (review fix PR #63 finding 5, verified empirically
  // 2026): gh-proxy-class services DO proxy `https://api.github.com/...` REST
  // traffic (gh-proxy.com returns real release JSON, HTTP 200), while they do
  // NOT proxy `github.com/<repo>/releases.atom` (404). So splicing the mirror
  // prefix in front of the REST API URL is correct here, and no alternate
  // metadata parser (atom / HTML scraping) is needed. Asset downloads splice
  // the same prefix, which is the traffic gh-proxy is designed for.
  const url = `${entry.urlPrefix}${LATEST_RELEASE_API_URL}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    let response: Response;
    try {
      response = await fetch(url, {
        signal: controller.signal,
        headers: { Accept: "application/vnd.github+json", "User-Agent": GITHUB_REPO },
      });
    } catch (error) {
      return { ok: false, reason: isAbortError(error) ? "timeout" : "network-error" };
    }
    if (!response.ok) {
      // 404 = no stable release published yet — the normal early state.
      return { ok: false, reason: response.status === 404 ? "no-release" : "http-error" };
    }
    let json: unknown;
    try {
      json = await response.json();
    } catch {
      return { ok: false, reason: "bad-payload" };
    }
    const release = parseReleaseJson(json as LatestReleaseJson);
    return release ? { ok: true, release } : { ok: false, reason: "bad-payload" };
  } finally {
    clearTimeout(timer);
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

/** Fetch a small text resource (the checksum manifest) with the same timeout. */
async function fetchText(fetch: FetchLike, url: string, timeoutMs: number): Promise<string | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) return null;
    return await response.text();
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Check for an update: walk the resolved channel chain, take the first channel
 * that yields a matching platform asset, and report the 实际生效渠道 (the
 * channel that actually served). Never throws — every failure path resolves to
 * a typed "unavailable" result.
 */
export async function checkForUpdate(input: CheckForUpdateInput): Promise<UpdaterResult> {
  const chain = resolveChannelChain(input.channelMode, input.mirrorPrefix);
  const timeoutMs = input.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const attempts: Array<{ channel: Channel; reason: UnavailableReason }> = [];
  let lastChannel: EffectiveChannel = null;

  for (const entry of chain) {
    lastChannel = entry.channel;
    const outcome = await probeChannel(input.fetch, entry, timeoutMs);
    if (!outcome.ok) {
      attempts.push({ channel: entry.channel, reason: outcome.reason });
      continue; // 静默兜底: silent fallback to the next channel.
    }
    const { release } = outcome;
    const matched = matchAssets(release.assets, input.platform);
    if (!matched) {
      attempts.push({ channel: entry.channel, reason: "no-platform-asset" });
      continue;
    }
    if (!isNewerVersion(release.tag, input.currentVersion)) {
      return { status: "up-to-date", latestTag: release.tag, effectiveChannel: entry.channel };
    }
    if (isSuperseded(release.tag, input.skippedTag)) {
      return { status: "skipped", latestTag: release.tag, effectiveChannel: entry.channel };
    }
    // Fetch the checksum manifest through the same channel so its sha512 is
    // ready for the download stage (#65). A missing/unfetchable manifest is a
    // warning, not a failure (spec: 无清单则警告但允许继续).
    const manifestText = matched.manifest ? await fetchText(input.fetch, `${entry.urlPrefix}${matched.manifest.browser_download_url}`, timeoutMs) : null;
    const manifestSha512 = manifestText === null ? null : parseManifestSha512(manifestText);
    const warning: "manifest-missing" | undefined = manifestText === null || manifestSha512 === null ? "manifest-missing" : undefined;
    return {
      status: "available",
      latestTag: release.tag,
      releaseNotes: release.body,
      installerUrl: matched.installer.browser_download_url,
      installerName: matched.installer.name,
      effectiveChannel: entry.channel,
      resolvedPrefix: entry.urlPrefix,
      manifestSha512,
      warning,
    };
  }

  return {
    status: "unavailable",
    reason: attempts.at(-1)?.reason ?? "network-error",
    effectiveChannel: lastChannel,
    attempts,
  };
}
