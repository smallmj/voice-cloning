// Issue #21: the data-fetching layer above `client.ts`. `api.ts` stays
// types-only; these hooks own the auth header, polling lifecycles and
// (per ADR-0014) UI-preference persistence against the sidecar.

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ConsentInfo,
  EngineInstallStatus,
  EngineInfo,
  LogEvent,
  TranscriptionProviders,
  Voice,
} from "./api";
import { apiJson, fetchMediaToken } from "./client";
import {
  appendLog,
  clampSidebarWidth,
  isThemePref,
  parseLogEvent,
  SIDEBAR_DEFAULT_WIDTH,
  type ThemePref,
} from "./ui";

export type { ThemePref };

function sanitizeEngineParams(raw: unknown): Record<string, Record<string, string>> {
  if (!raw || typeof raw !== "object") return {};
  const out: Record<string, Record<string, string>> = {};
  for (const [engine, params] of Object.entries(raw as Record<string, unknown>)) {
    if (!params || typeof params !== "object") continue;
    out[engine] = Object.fromEntries(
      Object.entries(params as Record<string, unknown>)
        .filter(([, v]) => typeof v === "string" || typeof v === "number" || typeof v === "boolean")
        .map(([k, v]) => [k, String(v)]),
    );
  }
  return out;
}

export interface UiPrefs {
  theme: ThemePref;
  engine_id: string | null;
  // Issue #23 / ADR-0018 decision 6: per-engine last-used generation
  // parameters, remembered with the library (backup/restore covers them).
  engine_params: Record<string, Record<string, string>>;
  // Issue #36: shared right sidebar (logs + job queue) persisted state.
  sidebar_open: boolean;
  sidebar_width: number;
}

function sanitizeSidebarWidth(raw: unknown): number {
  // Mirrors the server clamp; a corrupt value falls back to the default.
  const width = typeof raw === "number" ? raw : NaN;
  return Number.isFinite(width) ? clampSidebarWidth(width) : SIDEBAR_DEFAULT_WIDTH;
}

/** GET/PUT /settings/ui — theme + engine selection persist with the library
 * (ADR-0013 §6, ADR-0014 §7), not in localStorage. */
export function useUiPrefs(baseUrl: string | null, token: string | null) {
  const [prefs, setPrefs] = useState<UiPrefs>({
    theme: "system",
    engine_id: null,
    engine_params: {},
    sidebar_open: true,
    sidebar_width: SIDEBAR_DEFAULT_WIDTH,
  });
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!baseUrl || !token) return;
    let cancelled = false;
    apiJson<UiPrefs>(baseUrl, token, "/settings/ui")
      .then((p) => {
        if (cancelled) return;
        setPrefs({
          theme: isThemePref(p.theme) ? p.theme : "system",
          engine_id: typeof p.engine_id === "string" ? p.engine_id : null,
          engine_params: sanitizeEngineParams(p.engine_params),
          sidebar_open: p.sidebar_open !== false,
          sidebar_width: sanitizeSidebarWidth(p.sidebar_width),
        });
        setReady(true);
      })
      .catch(() => {
        /* sidecar not ready yet — defaults apply */
        if (!cancelled) setReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, token]);

  const update = useCallback(
    (partial: Partial<UiPrefs>) => {
      setPrefs((prev) => ({ ...prev, ...partial }));
      if (!baseUrl || !token) return;
      apiJson<UiPrefs>(baseUrl, token, "/settings/ui", {
        method: "PUT",
        body: JSON.stringify(partial),
      })
        .then((p) => {
          setPrefs((prev) => ({
            theme: isThemePref(p.theme) ? p.theme : prev.theme,
            engine_id:
              typeof p.engine_id === "string" && p.engine_id ? p.engine_id : null,
            engine_params: sanitizeEngineParams(p.engine_params),
            sidebar_open: p.sidebar_open !== false,
            sidebar_width: sanitizeSidebarWidth(p.sidebar_width),
          }));
        })
        .catch(() => {
          /* keep the optimistic value; retried on next change */
        });
    },
    [baseUrl, token],
  );

  return { prefs, ready, update };
}

export function useEngines(baseUrl: string | null, token: string | null) {
  const [engines, setEngines] = useState<EngineInfo[]>([]);
  const reload = useCallback(async () => {
    if (!baseUrl || !token) return;
    try {
      const data = await apiJson<{ engines: EngineInfo[] }>(baseUrl, token, "/engines");
      setEngines(data.engines);
    } catch {
      /* keep the last list */
    }
  }, [baseUrl, token]);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { engines, setEngines, reload };
}

export function useVoices(baseUrl: string | null, token: string | null) {
  const [voices, setVoices] = useState<Voice[]>([]);
  useEffect(() => {
    if (!baseUrl || !token) return;
    let cancelled = false;
    apiJson<{ voices: Voice[] }>(baseUrl, token, "/voices")
      .then((d) => {
        if (!cancelled) setVoices(d.voices);
      })
      .catch(() => {
        /* sidecar restarting — keep the last list */
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, token]);
  return { voices, setVoices };
}

export function useTranscriptionProviders(baseUrl: string | null, token: string | null) {
  const [providers, setProviders] = useState<TranscriptionProviders | null>(null);
  const reload = useCallback(async () => {
    if (!baseUrl || !token) return;
    try {
      setProviders(
        await apiJson<TranscriptionProviders>(baseUrl, token, "/transcription/providers"),
      );
    } catch {
      /* keep the last snapshot */
    }
  }, [baseUrl, token]);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { providers, reload };
}

/** Issue #15 consent gate state, read once the sidecar is up. A failed read
 * must not break startup. */
export function useConsent(baseUrl: string | null, token: string | null) {
  const [consent, setConsent] = useState<ConsentInfo | null>(null);
  useEffect(() => {
    if (!baseUrl || !token) return;
    let cancelled = false;
    apiJson<ConsentInfo>(baseUrl, token, "/consent")
      .then((c) => {
        if (!cancelled) setConsent(c);
      })
      .catch(() => {
        /* keep consent null; the workspace opens but consent stays unset */
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, token]);
  return { consent, setConsent };
}

// Issue #19: short-TTL media-scoped token for <img>/<audio>/WS URLs,
// refreshed comfortably before expiry.
export function useMediaToken(baseUrl: string | null, token: string | null): string {
  const [mediaToken, setMediaToken] = useState<string>("");
  useEffect(() => {
    if (!baseUrl || !token) return;
    let cancelled = false;
    let timer: number | null = null;
    const schedule = (ms: number) => {
      timer = window.setTimeout(() => void run(), ms);
    };
    const run = async () => {
      try {
        const mt = await fetchMediaToken(baseUrl, token);
        if (cancelled) return;
        setMediaToken(mt.media_token);
        schedule(Math.max(10_000, (mt.expires_in_seconds - 60) * 1000));
      } catch {
        // Retry soon; media stays broken only as long as the fetch does.
        if (!cancelled) schedule(10_000);
      }
    };
    void run();
    return () => {
      cancelled = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }, [baseUrl, token]);
  return mediaToken;
}

export interface InstallWatcher {
  statuses: Record<string, EngineInstallStatus>;
  watch: (id: string) => void;
}

/** Per-engine install status: one immediate pass, then poll only engines that
 * are mid-install. The interval is assigned synchronously in the effect body,
 * so StrictMode cleanup always reaches it (the old async-IIFE version could
 * never cancel its 1s poller). */
export function useInstallStatuses(
  baseUrl: string | null,
  token: string | null,
  engines: EngineInfo[],
): InstallWatcher {
  const [statuses, setStatuses] = useState<Record<string, EngineInstallStatus>>({});
  const watched = useRef<Set<string>>(new Set());

  const watch = useCallback((id: string) => {
    watched.current.add(id);
  }, []);

  useEffect(() => {
    if (!baseUrl || !token || engines.length === 0) return;
    const fetchStatus = async (id: string) => {
      try {
        const status = await apiJson<EngineInstallStatus>(
          baseUrl,
          token,
          `/engines/${id}/status`,
        );
        setStatuses((prev) => ({ ...prev, [id]: status }));
        // A settled install leaves the watch set — no polling forever.
        if (!status.installing) watched.current.delete(id);
      } catch {
        /* sidecar restarting — try again on the next tick */
      }
    };
    for (const e of engines) void fetchStatus(e.id);
    const timer = setInterval(() => {
      for (const id of [...watched.current]) void fetchStatus(id);
    }, 1000);
    return () => clearInterval(timer);
  }, [baseUrl, token, engines]);

  return { statuses, watch };
}

/** Issue #21 topbar badge: how many long-text jobs are queued/running. */
export function useActiveJobCount(baseUrl: string | null, token: string | null): number {
  const [count, setCount] = useState(0);
  useEffect(() => {
    if (!baseUrl || !token) return;
    let alive = true;
    const poll = async () => {
      try {
        const data = await apiJson<{ jobs: { status: string }[] }>(
          baseUrl,
          token,
          "/jobs?limit=100",
        );
        if (!alive) return;
        setCount(
          data.jobs.filter((j) => j.status === "queued" || j.status === "running").length,
        );
      } catch {
        /* sidecar restarting — keep the last count */
      }
    };
    void poll();
    const timer = setInterval(poll, 2000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [baseUrl, token]);
  return count;
}

/** Live log stream. Reconnects with a fixed backoff when the sidecar drops
 * the socket; malformed frames are dropped via `parseLogEvent`. */
export function useLogStream(baseUrl: string | null, mediaToken: string): LogEvent[] {
  const [logs, setLogs] = useState<LogEvent[]>([]);
  useEffect(() => {
    if (!baseUrl || !mediaToken) return;
    let ws: WebSocket | null = null;
    let retry: number | null = null;
    let closed = false;
    const connect = () => {
      if (closed) return;
      try {
        ws = new WebSocket(
          `${baseUrl.replace("http", "ws")}/ws/logs?token=${encodeURIComponent(mediaToken)}`,
        );
      } catch {
        retry = window.setTimeout(connect, 3000);
        return;
      }
      ws.onmessage = (ev) => {
        const event = parseLogEvent(typeof ev.data === "string" ? ev.data : "");
        if (event) setLogs((prev) => appendLog(prev, event));
      };
      ws.onclose = () => {
        if (!closed) retry = window.setTimeout(connect, 3000);
      };
      ws.onerror = () => {
        ws?.close();
      };
    };
    connect();
    return () => {
      closed = true;
      if (retry != null) window.clearTimeout(retry);
      ws?.close();
    };
  }, [baseUrl, mediaToken]);
  return logs;
}
