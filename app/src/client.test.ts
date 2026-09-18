import { describe, expect, it, vi } from "vitest";
import { apiJson, authHeaders, errorDetail } from "./client";

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: async () => body,
  } as unknown as Response;
}

describe("authHeaders", () => {
  it("adds the bearer header and preserves extras", () => {
    expect(authHeaders("t", { "Content-Type": "application/json" })).toEqual({
      Authorization: "Bearer t",
      "Content-Type": "application/json",
    });
  });
});

describe("errorDetail", () => {
  it("prefers the sidecar's detail message", async () => {
    const res = jsonResponse({ detail: "引擎未安装" }, false, 400);
    await expect(errorDetail(res, "fallback")).resolves.toBe("引擎未安装");
  });

  it("falls back when the body is not JSON", async () => {
    const res = {
      ok: false,
      status: 500,
      json: async () => {
        throw new Error("not json");
      },
    } as unknown as Response;
    await expect(errorDetail(res, "fallback")).resolves.toBe("fallback");
  });
});

describe("apiJson", () => {
  it("sends the bearer header and parses JSON on success", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ engines: [] }));
    vi.stubGlobal("fetch", fetchMock);
    const data = await apiJson<{ engines: unknown[] }>("http://x", "tok", "/engines");
    expect(data).toEqual({ engines: [] });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://x/engines");
    expect(init.headers.Authorization).toBe("Bearer tok");
    vi.unstubAllGlobals();
  });

  it("throws the sidecar detail on non-ok responses (API contract)", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ detail: "theme 必须是 system/dark/light 之一" }, false, 422));
    vi.stubGlobal("fetch", fetchMock);
    await expect(apiJson("http://x", "tok", "/settings/ui", { method: "PUT" })).rejects.toThrow(
      "theme 必须是 system/dark/light 之一",
    );
    vi.unstubAllGlobals();
  });
});
