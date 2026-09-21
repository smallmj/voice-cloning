// Fetch the pinned uv binary for one or more target platforms into
// build/uv/<platform>-<arch>/ so electron-builder can bundle it as an
// extraResource (issue #29: uv must ship with the app, referenced by
// absolute path — never resolved from PATH at launch time).
//
// UV_VERSION here MUST stay in sync with
// sidecar/voiceclone_sidecar/runtime/uvman.py (both reference issue #29).
//
// Usage: node scripts/fetch-uv.mjs [darwin-arm64 darwin-x64 win32-x64 ...]
//        (no args = current platform+arch)
import { mkdirSync, createWriteStream, existsSync, renameSync, readdirSync, statSync } from "fs";
import { readFile } from "fs/promises";
import { pipeline } from "stream/promises";
import { Readable } from "stream";
import { createHash } from "crypto";
import * as path from "path";
import * as os from "os";
import { execFileSync } from "child_process";

const UV_VERSION = "0.12.10";
const BASE = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}`;
const MIRROR_ENV = "VOICECLONE_UV_DOWNLOAD_URL"; // str.format(path=) template

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
  const template = process.env[MIRROR_ENV];
  const urls = template
    ? [template.replace("{path}", archiveName), `${BASE}/${archiveName}`]
    : [`${BASE}/${archiveName}`];
  await download(urls, archive);
  const hash = createHash("sha256").update(await readFile(archive)).digest("hex");
  console.log(`  sha256 ${hash}`);
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
