import { useEffect, useMemo, useRef } from "react";
import WaveSurfer from "wavesurfer.js";

/**
 * Issue #21 lifecycle fix: the old version took a `headers` object literal
 * prop — a fresh reference on every parent render — and re-ran its effect on
 * every change. Combined with the 2s job poller above it, the waveform was
 * destroyed and rebuilt every 2 seconds, resetting playback mid-stream.
 * The headers object is now memoized here and only changes when the token
 * does.
 */
export function Waveform({ url, token }: { url: string; token: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const fetchParams = useMemo(
    () => ({ headers: { Authorization: `Bearer ${token}` } }),
    [token],
  );

  useEffect(() => {
    if (!containerRef.current) return;
    // ADR-0014: no one-off colors — resolve the theme tokens at draw time
    // (canvas cannot use var(), and the value must re-resolve per theme).
    const styles = getComputedStyle(document.documentElement);
    const waveColor = styles.getPropertyValue("--wave-base").trim() || "#5c6bc0";
    const progressColor = styles.getPropertyValue("--accent").trim() || "#82b1ff";
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor,
      progressColor,
      height: 80,
      url,
      fetchParams,
    });
    wsRef.current = ws;
    return () => {
      ws.destroy();
      wsRef.current = null;
    };
  }, [url, fetchParams]);

  return (
    <div className="waveform">
      <div ref={containerRef} />
      <button
        onClick={() => {
          if (wsRef.current) wsRef.current.playPause();
        }}
      >
        ▶ / ⏸
      </button>
    </div>
  );
}
