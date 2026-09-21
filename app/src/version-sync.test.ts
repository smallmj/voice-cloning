// The pinned uv version must not drift apart between the two places that
// reference it (issue #29): the Python runtime downloader and the Electron
// packaging script. This test parses both sources and fails loudly if they
// disagree.
import { readFileSync } from "fs";
import * as path from "path";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(__dirname, "..", "..");
const fetchUvSource = readFileSync(path.join(repoRoot, "app", "scripts", "fetch-uv.mjs"), "utf8");
const uvmanSource = readFileSync(
  path.join(repoRoot, "sidecar", "voiceclone_sidecar", "runtime", "uvman.py"),
  "utf8",
);

const fetchUvVersion = fetchUvSource.match(/const UV_VERSION = "([^"]+)"/)?.[1];
const uvmanVersion = uvmanSource.match(/^UV_VERSION = "([^"]+)"/m)?.[1];

describe("pinned uv version sync (issue #29)", () => {
  it("both sources declare a version", () => {
    expect(fetchUvVersion, "fetch-uv.mjs UV_VERSION").toBeTruthy();
    expect(uvmanVersion, "uvman.py UV_VERSION").toBeTruthy();
  });

  it("fetch-uv.mjs and uvman.py pin the same uv version", () => {
    expect(fetchUvVersion).toBe(uvmanVersion);
  });

  it("the pinned version is a concrete x.y.z (never floating)", () => {
    expect(uvmanVersion).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
