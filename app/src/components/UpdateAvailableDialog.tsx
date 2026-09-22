// Startup 更新弹窗 (spec #62 stories 7/8/10, review fix PR #63 finding 4):
// when the main process detects a new version at startup it pushes
// updater:new-version; this lightweight dialog shows 版本号 + 更新日志摘要
// with 「去更新」(jump to the settings section) / 「关闭」 and a
// 「跳过此版本」 checkbox (skipVersion). It renders ONLY for startup pushes —
// manual 检查更新 results stay inside the AboutUpdateCard — and never re-shows
// a tag the user already dismissed in this app run.
//
// Access to the bridge goes through ../update-client (guarded, single
// module); styling follows the existing consent-overlay/consent-modal pattern
// and design tokens.

import { useEffect, useRef, useState } from "react";
import {
  onNewVersionAvailable,
  releaseNotesParagraphs,
  skipVersion,
  type NewVersionInfo,
} from "../update-client";

/** How many release-note paragraphs the collapsed summary shows. */
const MAX_NOTE_PARAGRAPHS = 6;

export function UpdateAvailableDialog({ onGoToUpdate }: { onGoToUpdate: () => void }) {
  const [available, setAvailable] = useState<NewVersionInfo | null>(null);
  const [skipChecked, setSkipChecked] = useState(false);
  const [skipBusy, setSkipBusy] = useState(false);
  // "not previously dismissed in this app run" — a plain module-of-instance
  // Set keeps dismissed tags; a ref avoids re-subscription on state changes.
  const dismissedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    // Real unsubscribe (review fix PR #63 finding 3): App remounts must not
    // stack listeners.
    return onNewVersionAvailable((result) => {
      if (dismissedRef.current.has(result.latestTag)) return;
      dismissedRef.current.add(result.latestTag);
      setSkipChecked(false);
      setAvailable(result);
    });
  }, []);

  function close() {
    setAvailable(null);
  }

  async function handleClose() {
    if (skipChecked && available) {
      setSkipBusy(true);
      try {
        await skipVersion(available.latestTag);
      } finally {
        setSkipBusy(false);
      }
    }
    close();
  }

  if (!available) return null;
  const notes = releaseNotesParagraphs(available.releaseNotes ?? "").slice(0, MAX_NOTE_PARAGRAPHS);

  return (
    <div className="consent-overlay" role="dialog" aria-modal="true" aria-label="发现新版本">
      <div className="consent-modal" data-testid="update-available-dialog">
        <h2>发现新版本 {available.latestTag}</h2>
        {notes.length > 0 ? (
          <div className="update-notes">
            {notes.map((p, i) => (
              <p key={i}>{p}</p>
            ))}
          </div>
        ) : (
          <p className="hint">此版本未提供更新日志。</p>
        )}
        <label className="consent-check">
          <input
            type="checkbox"
            checked={skipChecked}
            onChange={(e) => setSkipChecked(e.target.checked)}
          />
          跳过此版本
        </label>
        <div className="key-row">
          <button
            className="primary"
            onClick={() => {
              close();
              onGoToUpdate();
            }}
          >
            去更新
          </button>
          <button disabled={skipBusy} onClick={() => void handleClose()}>
            关闭
          </button>
        </div>
      </div>
    </div>
  );
}
