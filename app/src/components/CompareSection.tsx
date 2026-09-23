import { useRef, useState } from "react";
import type { CompareSession, EngineInfo, PreferenceProfile, Voice } from "../api";
import { apiJson } from "../client";
import { commonNonverbalTags, insertAtCursor } from "../ui";
import { NonverbalTagPicker } from "./bits";

const COMPARE_TEXT_TYPES = [
  { value: "general", label: "通用" },
  { value: "narration", label: "旁白/朗读" },
  { value: "dialogue", label: "对话" },
  { value: "news", label: "新闻" },
  { value: "poetry", label: "诗歌/文学" },
] as const;

const TEXT_TYPE_LABELS: Record<string, string> = Object.fromEntries(
  COMPARE_TEXT_TYPES.map((t) => [t.value, t.label]),
);

// Issue #45: 偏好画像降级为盲听结果页内的一行只读汇总（按语种 × 文本类型 × 引擎）。
function profileSummary(profile: PreferenceProfile | null, engines: EngineInfo[]): string {
  if (!profile) return "";
  return profile.cells
    .map((cell) => {
      const rows = cell.engines
        .map((row) => {
          const name =
            engines.find((x) => x.id === row.engine_id)?.display_name ?? row.engine_id;
          return `${name} 平均 ${row.average_score}（${row.score_count} 次）`;
        })
        .join("、");
      return `${cell.language === "zh" ? "中文" : "英文"}·${
        TEXT_TYPE_LABELS[cell.text_type] ?? cell.text_type
      }：${rows}`;
    })
    .join("；");
}

export function CompareSection({
  baseUrl,
  token,
  mediaToken,
  voices,
  engines,
}: {
  baseUrl: string;
  token: string;
  mediaToken: string;
  voices: Voice[];
  engines: EngineInfo[];
}) {
  const [voiceId, setVoiceId] = useState("");
  const [text, setText] = useState(
    "同一音色、同一文本，交给多个引擎并排生成。The comparison is blind until you reveal it.",
  );
  const [picked, setPicked] = useState<Record<string, boolean>>({});
  const [textType, setTextType] = useState("general");
  const [session, setSession] = useState<CompareSession | null>(null);
  const [scores, setScores] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [profile, setProfile] = useState<PreferenceProfile | null>(null);
  // Issue #68: caret-accurate tag insertion into the compare textarea.
  const compareTextRef = useRef<HTMLTextAreaElement | null>(null);

  const pickedIds = engines.filter((e) => picked[e.id]).map((e) => e.id);
  // Issue #68 (US6): the compare textarea offers the same tag control, but
  // only markers EVERY picked engine natively understands (intersection);
  // with no/mixed-tag picks it stays hidden rather than lie.
  const pickedEngines = engines.filter((e) => picked[e.id]);
  const commonTags = commonNonverbalTags(pickedEngines);

  async function runCompare() {
    setBusy(true);
    setError(null);
    try {
      const data = await apiJson<CompareSession>(baseUrl, token, "/compare", {
        method: "POST",
        body: JSON.stringify({
          voice_id: voiceId,
          text,
          engine_ids: pickedIds,
          text_type: textType,
        }),
      });
      setSession(data);
      setScores(Object.fromEntries(data.entries.map((e) => [e.label, e.score ?? 0])));
      await refreshProfile();
    } catch (err) {
      setError(`创建对比失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  async function reloadSession(reveal: boolean) {
    if (!session) return;
    try {
      setSession(
        await apiJson<CompareSession>(
          baseUrl,
          token,
          `/compare/${session.id}?reveal=${reveal ? "true" : "false"}`,
        ),
      );
    } catch {
      /* keep the current session */
    }
  }

  async function submitScores() {
    if (!session) return;
    setBusy(true);
    setError(null);
    try {
      const data = await apiJson<CompareSession>(baseUrl, token, `/compare/${session.id}/scores`, {
        method: "POST",
        body: JSON.stringify({ scores }),
      });
      setSession(data);
      await refreshProfile();
    } catch (err) {
      setError(`提交评分失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  }

  async function refreshProfile() {
    try {
      setProfile(await apiJson<PreferenceProfile>(baseUrl, token, "/preferences"));
    } catch {
      /* keep the last profile */
    }
  }

  const allScored = !!session && session.scored;
  const canReveal = !!session && !allScored;

  return (
    <div className="compare-mode">
      <h3>对比盲听</h3>
      <div className="hint">
        同一音色、同一文本交给多个引擎并排生成；所有结果先做响度归一化（LUFS −16），
        再以 A/B/C 盲标呈现。评分沉淀为本机偏好画像，仅作参考，不影响默认引擎或任何自动路由。
        此处与「单条生成」互不共享状态。
      </div>
      <div className="voice-picker">
        <label>
          对比音色：
          <select value={voiceId} onChange={(e) => setVoiceId(e.target.value)}>
            <option value="">（请选择音色）</option>
            {voices.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          文本类型：
          <select value={textType} onChange={(e) => setTextType(e.target.value)}>
            {COMPARE_TEXT_TYPES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="param-row">
        {engines.map((e) => (
          <label key={e.id} className="compare-engine-toggle">
            <input
              type="checkbox"
              checked={!!picked[e.id]}
              onChange={(ev) =>
                setPicked((p) => ({ ...p, [e.id]: ev.target.checked }))
              }
            />
            {e.display_name}
          </label>
        ))}
      </div>
      {commonTags.length > 0 && (
        <NonverbalTagPicker
          tags={commonTags}
          onInsert={(tag) => {
            const el = compareTextRef.current;
            const result = insertAtCursor(text, tag, el?.selectionStart, el?.selectionEnd);
            setText(result.text);
            requestAnimationFrame(() => {
              const node = compareTextRef.current;
              if (node) {
                node.focus();
                node.setSelectionRange(result.cursor, result.cursor);
              }
            });
          }}
        />
      )}
      <textarea ref={compareTextRef} value={text} onChange={(e) => setText(e.target.value)} rows={3} />
      <button
        className="primary"
        onClick={runCompare}
        disabled={busy || !voiceId || pickedIds.length < 2 || !text.trim()}
      >
        {busy ? "生成中…" : `并排生成（已选 ${pickedIds.length} 个引擎）`}
      </button>
      {error && <div className="error">{error}</div>}

      {session && (
        <div className="compare-session">
          <div className="compare-head">
            <strong>
              盲听（{session.language === "zh" ? "中文" : "英文"} ·{" "}
              {TEXT_TYPE_LABELS[session.text_type] ?? session.text_type} · 目标 {session.target_lufs} LUFS）
            </strong>
            {canReveal && (
              <button onClick={() => void reloadSession(true)}>揭示引擎身份</button>
            )}
            {!allScored && (
              <button
                onClick={() => void submitScores()}
                disabled={busy || !session.entries.every((e) => (scores[e.label] ?? 0) > 0)}
                title="为每个对比项打分后可提交"
              >
                提交评分
              </button>
            )}
            {allScored && <span className="badge badge-on">已评分 · 身份已揭示</span>}
          </div>
          <div className="compare-grid">
            {session.entries.map((e) => (
              <div className="compare-card" key={e.label}>
                <div className="compare-card-head">
                  <strong>{e.label}</strong>
                  <span className="hint">
                    {e.engine_id
                      ? engines.find((x) => x.id === e.engine_id)?.display_name ?? e.engine_id
                      : "引擎已隐藏"}
                  </span>
                </div>
                <audio
                  controls
                  preload="none"
                  src={`${baseUrl}${e.normalized_audio_url}?token=${encodeURIComponent(mediaToken)}`}
                />
                <div className="hint">
                  原始 {e.original_lufs ?? "—"} LUFS → 归一化 {e.achieved_lufs ?? "—"} LUFS
                  （增益 {e.gain_db != null && e.gain_db > 0 ? "+" : ""}
                  {e.gain_db ?? "—"} dB{e.peak_limited ? "，峰值受限" : ""}）
                </div>
                <div className="score-row">
                  {[1, 2, 3, 4, 5].map((n) => (
                    <button
                      key={n}
                      className={`score-star ${(scores[e.label] ?? 0) >= n ? "on" : ""}`}
                      onClick={() => setScores((p) => ({ ...p, [e.label]: n }))}
                      title={`${n} 分`}
                    >
                      ★
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
          {(session.failed_count ?? 0) > 0 && (
            <div className="hint">
              {session.failed_count} 个引擎未能生成（未安装或缺少 API Key），已跳过。
            </div>
          )}
          {session.failed.length > 0 && (
            <div className="error">
              {session.failed.map((f) => `${f.engine_id}：${f.error}`).join("；")}
            </div>
          )}
          <div className="pref-summary hint">
            {profile && profile.cells.length > 0 ? (
              <>
                <strong>偏好画像（只读）：</strong>
                {profileSummary(profile, engines)}。
              </>
            ) : (
              "偏好画像（只读）：暂无评分数据。完成一次对比盲听并评分后，会在此按语种 × 文本类型 × 引擎汇总平均分与次数。它仅作参考，不改变默认引擎，也不参与任何自动路由。"
            )}
          </div>
        </div>
      )}
    </div>
  );
}
