import { useState } from "react";
import type { EngineInfo } from "../api";

// ---------------------------------------------------------------------------
// Vendor-level BYOK keys (issue #53): one input per VENDOR, not per engine.
// The two Qwen3-TTS cloud engines (复刻 / 音色设计) talk to the same 阿里百炼
// account, so forcing the user to paste the same key twice was pure friction;
// MiniMax gets the same treatment for consistency. Saving writes the key to
// every engine of the vendor through the existing per-engine keychain
// endpoints — the backend key store stays per-engine, the UI just stops
// making the user repeat themselves.
// ---------------------------------------------------------------------------

export function VendorKeysSection({
  engines,
  onSaveKey,
  onDeleteKey,
  keyBusy,
  keyError,
}: {
  engines: EngineInfo[];
  onSaveKey: (engineIds: string[], key: string) => Promise<void>;
  onDeleteKey: (engineIds: string[]) => Promise<void>;
  keyBusy: boolean;
  keyError: string | null;
}) {
  const [drafts, setDrafts] = useState<Record<string, string>>({});

  // Engines of one vendor, in stable display order; keyed engines only.
  const vendors = new Map<string, EngineInfo[]>();
  for (const e of engines) {
    if (!e.requires_key || !e.vendor_label) continue;
    const list = vendors.get(e.vendor_label) ?? [];
    list.push(e);
    vendors.set(e.vendor_label, list);
  }
  if (vendors.size === 0) return null;

  return (
    <div className="vendor-keys" id="vendor-keys">
      <h3>厂商 API Key</h3>
      <div className="hint">
        每个厂商只需填一次 Key，同厂商的引擎共用（云端转写也使用对应厂商的 Key）。
        Key 只保存在本机系统钥匙串中（BYOK），不会进入备份或上传。
      </div>
      {[...vendors.entries()].map(([vendor, list]) => {
        const configured = list.some((e) => e.key_configured);
        const ids = list.map((e) => e.id);
        const draft = drafts[vendor] ?? "";
        return (
          <div className="vendor-key-row" key={vendor}>
            <div className="settings-engine-head">
              <strong>{vendor}</strong>
              <span className={`badge ${configured ? "badge-on" : "badge-off"}`}>
                {configured ? "已配置" : "未配置"}
              </span>
            </div>
            <div className="hint">
              作用于：{list.map((e) => e.display_name).join("、")}
            </div>
            <div className="key-row">
              <input
                type="password"
                autoComplete="off"
                placeholder={`粘贴 ${vendor} API Key`}
                value={draft}
                onChange={(ev) => setDrafts({ ...drafts, [vendor]: ev.target.value })}
              />
              <button
                disabled={keyBusy || !draft.trim()}
                onClick={() =>
                  void onSaveKey(ids, draft.trim()).then(() =>
                    setDrafts((p) => ({ ...p, [vendor]: "" })),
                  )
                }
              >
                {keyBusy ? "保存中…" : "保存"}
              </button>
              {configured && (
                <button disabled={keyBusy} onClick={() => void onDeleteKey(ids)}>
                  清除
                </button>
              )}
            </div>
            {list[0]?.billing_note && <div className="hint">{list[0].billing_note}</div>}
            {list[0]?.data_usage_note && <div className="hint">{list[0].data_usage_note}</div>}
          </div>
        );
      })}
      {keyError && <div className="error">{keyError}</div>}
    </div>
  );
}
