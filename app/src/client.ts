// Issue #21: `api.ts` stays types-only. This module is the single place that
// knows how to reach the sidecar over fetch — every component used to hand-
// write the Authorization header ~30 times.

import type { MediaToken } from "./api";

export function authHeaders(token: string, extra?: Record<string, string>): Record<string, string> {
  return { Authorization: `Bearer ${token}`, ...extra };
}

export async function errorDetail(res: Response, fallback: string): Promise<string> {
  try {
    const j = (await res.json()) as { detail?: string };
    return j.detail ?? fallback;
  } catch {
    return fallback;
  }
}

/** GET/POST/PUT/DELETE JSON against the sidecar; throws with the sidecar's
 * `detail` message when the response is not ok. */
export async function apiJson<T>(
  baseUrl: string,
  token: string,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: authHeaders(token, {
      "Content-Type": "application/json",
      ...(init?.headers as Record<string, string> | undefined),
    }),
  });
  if (!res.ok) throw new Error(await errorDetail(res, `请求失败（HTTP ${res.status}）`));
  return (await res.json()) as T;
}

/** Issue #14: download a sidecar file (voice package / library backup) with
 * bearer auth, then hand it to the browser's normal save flow. */
export async function downloadWithAuth(url: string, token: string, filename: string): Promise<void> {
  const res = await fetch(url, { headers: authHeaders(token) });
  if (!res.ok) throw new Error(await errorDetail(res, `下载失败（HTTP ${res.status}）`));
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** Same as downloadWithAuth but issues the request with POST (e.g. the
 * whole-library backup endpoint returns the zip as a POST response). */
export async function downloadPostWithAuth(url: string, token: string, filename: string): Promise<void> {
  const res = await fetch(url, { method: "POST", headers: authHeaders(token) });
  if (!res.ok) throw new Error(await errorDetail(res, `下载失败（HTTP ${res.status}）`));
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/** Issue #19: fetch the short-TTL, media-scoped token used by <img> /
 * <audio> elements and the log WebSocket. */
export async function fetchMediaToken(baseUrl: string, token: string): Promise<MediaToken> {
  const res = await fetch(`${baseUrl}/media-token`, { headers: authHeaders(token) });
  if (!res.ok) throw new Error(`media-token HTTP ${res.status}`);
  return (await res.json()) as MediaToken;
}
