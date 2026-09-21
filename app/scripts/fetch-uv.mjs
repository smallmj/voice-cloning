// Fetch the pinned uv binary for one or more target platforms into
// build/uv/<platform>-<arch>/ so electron-builder can bundle it as an
// extraResource (issue #29: uv must ship with the app, referenced by
// absolute path — never resolved from PATH at launch time).
//
// UV_VERSION + UV_SHA256 here MUST stay in sync with
// sidecar/voiceclone_sidecar/runtime/uvman.py (both reference issue #29;
// app/tests/version-sync.test.mjs enforces the version half).
//
// Usage: node scripts/fetch-uv.mjs [darwin-arm64 darwin-x64 win32-x64 ...]
//        (no args = current platform+arch)
import { mkdirSync, createWriteStream, existsSync, renameSync, readdirSync, statSync } from "fs";
import { readFile } from "fs/promises";
import { pipeline } from "stream/promises";
import { Readable } from "stream";
import { createHash } from "crypto";
import * as path from "path";
import { execFileSync } from "child_process";

const UV_VERSION = "0.12.10";
const BASE = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}`;
// Same semantics as uvman.py: the env var is a FULL URL of the archive (not
// a {path} template). Mirrors must serve the exact pinned archive.
const MIRROR_ENV = "VOICECLONE_UV_DOWNLOAD_URL";
// Official sha256 digests of uv-<triple>.tar.gz for UV_VERSION (from
// astral-sh/versions uv.ndjson). Every download is verified against these —
// a tampered or mis-mirrored archive fails the fetch.
const UV_SHA256 = {
  "aarch64-apple-darwin": "51c6170e8e3a01cef9f33b94f582b7b81ac65046f55d40afb35f9cff5a68c179",
  "x86_64-apple-darwin": "5296d5aa2b9143360405eea866f8ef4d5dc8986b164eb0dc35e8f876a9304d30",
  "x86_64-pc-windows-msvc": "f65744f94072152b1f86ba2aace4d01f1124d9a8ecb235805039e3718c36cac2",
  "aarch64-pc-windows-msvc": "ee985c51c0c9c1f82267a5d80f959b34a7ff888c109182bd3b2b35c4661bbcde",
  "x86_64-unknown-linux-gnu": "173d95a0c32d18c896c46ba6fafbf3cf9c14ab74b033f81b76c883ef492a976b",
  "aarch64-unknown-linux-gnu": "9ff6b9d4665edcdd3a88dcc73cd1eb641754deb927f14e8c62ebfde6bf4f5f5e",
};

const MACHINES = { arm64: "aarch64", x64: "x86_64" };
const OSES = {
  darwin: "apple-darwin",
  win32: "pc-windows-msvc",
  linux: "unknown-linux-gnu",
};

function tripleFor(platform, arch) {
  const machine = MACHINES[arch];
  const osPart = OSES[platform];
  if (!machine || !osPart) throw new Error(`unsupported target: ${platform}-${arch}`);
  return `${machine}-${osPart}`;
}

function defaultTargets() {
  return [`${process.platform}-${process.arch}`];
}

async function download(urls, dest) {
  for (const url of urls) {
    try {
      console.log(`downloading ${url}`);
      const res = await fetch(url);
      if (!res.ok) {
        console.warn(`  -> HTTP ${res.status}, trying next source`);
        continue;
      }
      await pipeline(Readable.fromWeb(res.body), createWriteStream(dest));
      return;
    } catch (err) {
      console.warn(`  -> ${err.message}, trying next source`);
    }
  }
  throw new Error(`all sources failed for ${dest}`);
}

async function fetchUv(target) {
  const [platform, arch] = target.split("-");
  const triple = tripleFor(platform, arch);
  const exe = platform === "win32" ? "uv.exe" : "uv";
  const outDir = path.resolve("build", "uv", target);
  const outPath = path.join(outDir, exe);
  if (existsSync(outPath)) {
    console.log(`${outPath} already present, skipping`);
    return outPath;
  }
  mkdirSync(outDir, { recursive: true });
  const archiveName = `uv-${triple}.tar.gz`;
  const archive = path.join(outDir, archiveName);
  const mirror = process.env[MIRROR_ENV];
  const urls = mirror ? [mirror, `${BASE}/${archiveName}`] : [`${BASE}/${archiveName}`];
  await download(urls, archive);
  const hash = createHash("sha256").update(await readFile(archive)).digest("hex");
  const expected = UV_SHA256[triple];
  if (!expected) throw new Error(`no pinned sha256 for ${triple}; add it to UV_SHA256`);
  if (hash !== expected) {
    execFileSync("rm", ["-f", archive]);
    throw new Error(
      `sha256 mismatch for ${archiveName}: got ${hash}, expected ${expected} (mirror serving a tampered or wrong archive?)`,
    );
  }
  console.log(`  sha256 verified ${hash}`);
  // Extract just the uv binary from the tar.gz.
  const tmp = path.join(outDir, "_extract");
  mkdirSync(tmp, { recursive: true });
  execFileSync("tar", ["-xzf", archive, "-C", tmp]);
  const found = findFile(tmp, exe);
  if (!found) throw new Error(`${exe} not found inside ${archive}`);
  renameSync(found, outPath);
  if (platform !== "win32") execFileSync("chmod", ["755", outPath]);
  execFileSync("rm", ["-rf", tmp, archive]);
  return outPath;
}

function findFile(dir, name) {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) {
      const hit = findFile(full, name);
      if (hit) return hit;
    } else if (entry === name) {
      return full;
    }
  }
  return null;
}

const targets = process.argv.slice(2).length ? process.argv.slice(2) : defaultTargets();
for (const target of targets) {
  const p = await fetchUv(target);
  console.log(`fetched: ${p}`);
}
