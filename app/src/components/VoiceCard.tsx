import { useState } from "react";
import type { ReferenceAnalysis, Voice } from "../api";
import { apiJson, downloadWithAuth } from "../client";
import { DIAG_LABELS } from "../labels";
import type { SectionId } from "../ui";
import { JumpLink } from "./bits";

export function VoiceCard({
  voice,
  baseUrl,
  token,
  mediaToken,
  selected,
  onSelect,
  onDelete,
  onTranscribed,
  onNavigate,
}: {
  voice: Voice;
  baseUrl: string;
  token: string;
  mediaToken: string;
  selected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onTranscribed: (voice: Voice) => void;
  onNavigate: (section: SectionId) => void;
}) {
  const ref = voice.reference;
  const designed = voice.origin === "designed";
  const [analysis, setAnalysis] = useState<ReferenceAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [cardError, setCardError] = useState<string | null>(null);

  async function exportPackage() {
    setExporting(true);
    setCardError(null);
    try {
      const safeName = (voice.name || voice.id).replace(/[\\/:*?"<>|\s]+/g, "_");
      await downloadWithAuth(
        `${baseUrl}/voices/${voice.id}/export`,
        token,
        `${safeName}.voice.zip`,
      );
    } catch (e) {
      setCardError(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting(false);
    }
  }

  async function diagnose() {
    setAnalyzing(true);
    setCardError(null);
    try {
      const analysis = await apiJson<ReferenceAnalysis>(
        baseUrl,
        token,
        `/voices/${voice.id}/diagnose`,
      );
      setAnalysis(analysis);
    } catch (e) {
      setCardError(e instanceof Error ? e.message : String(e));
    } finally {
      setAnalyzing(false);
    }
  }

  async function transcribe() {
    setTranscribing(true);
    setCardError(null);
    try {
      const updated = await apiJson<Voice>(baseUrl, token, `/voices/${voice.id}/transcribe`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      onTranscribed(updated);
    } catch (e) {
      setCardError(e instanceof Error ? e.message : String(e));
    } finally {
      setTranscribing(false);
    }
  }

  return (
    <div
      className={`voice-card ${selected ? "selected" : ""}`}
      onClick={onSelect}
      role="button"
      aria-pressed={selected}
      tabIndex={0}
      onKeyDown={(ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          onSelect();
        }
      }}
    >
      <div className="voice-head">
        {voice.avatar ? (
          <img
            className="voice-avatar"
            src={`${baseUrl}/voices/${voice.id}/avatar?token=${encodeURIComponent(mediaToken)}`}
            alt={voice.name}
          />
        ) : (
          <div className="voice-avatar placeholder">{voice.name.slice(0, 1)}</div>
        )}
        <div className="voice-meta">
          <strong>{voice.name}</strong>{" "}
          <span className={`badge ${designed ? "badge-design" : "badge-clone"}`}>
            {designed ? "文字设计" : "参考音频复刻"}
          </span>
          <div className="voice-ref-info">
            {ref
              ? `${ref.format.toUpperCase()} · ${ref.duration_seconds.toFixed(1)}s · ${(
                  ref.size_bytes / 1024 / 1024
                ).toFixed(1)}MB`
              : "尚无参考样本"}
          </div>
        </div>
        <button
          className="voice-delete"
          title="删除音色（连带删除全部绑定与本地文件）"
          onClick={(ev) => {
            ev.stopPropagation();
            onDelete();
          }}
        >
          删除
        </button>
      </div>
      {voice.description && <div className="voice-desc">{voice.description}</div>}
      {Object.values(voice.bindings).some((b) => b.status === "unactivated") && (
        // Issue #27: a MiniMax cloud voice that could not be activated is
        // deleted by the vendor after 7 days — this must be prominently
        // visible on the voice/engine management page, not only on the
        // generate page.
        <div className="voice-placeholder-warning">
          ⚠️ 有引擎绑定尚未激活：云端音色 7 天不使用会被厂商删除，请立即用该引擎生成一次以激活。
          {Object.entries(voice.bindings)
            .filter(([, b]) => b.status === "unactivated")
            .map(([engineId, b]) => `${engineId}${b.activation_error || b.error ? `（${b.activation_error || b.error}）` : ""}`)
            .join("、")}
        </div>
      )}
      {Object.keys(voice.bindings).length > 0 && (
        <div className="voice-bindings">
          绑定：
          {Object.entries(voice.bindings)
            .map(
              ([engineId, b]) =>
                `${engineId}（${b.status}${b.error ? `：${b.error}` : ""}）`,
            )
            .join("、")}
        </div>
      )}
      {ref?.transcript && !ref.transcript_placeholder && (
        <div className="voice-transcript">
          转写文本：<code>{ref.transcript}</code>
        </div>
      )}
      {ref?.transcript && ref.transcript_placeholder && (
        <div className="voice-placeholder-warning" onClick={(ev) => ev.stopPropagation()}>
          ⚠️ 这段转写文本是早期版本内置测试引擎写入的占位文本，不是真实转写结果。
          需要参考文本的引擎会把它当成参考文本使用，影响复刻质量。
          请先安装并使用本地转写，再点击「重新转写」覆盖。
          <JumpLink target="settings" label="打开转写设置" onNavigate={onNavigate} />
        </div>
      )}
      <div className="voice-tools" onClick={(ev) => ev.stopPropagation()}>
        {ref ? (
          <>
            <button onClick={() => void diagnose()} disabled={analyzing}>
              {analyzing ? "诊断中…" : "诊断"}
            </button>
            <button onClick={() => void transcribe()} disabled={transcribing}>
              {transcribing ? "转写中…" : ref.transcript ? "重新转写" : "转写"}
            </button>
            <button onClick={() => void exportPackage()} disabled={exporting} title="导出为音色包（.zip），可分享给他人">
              {exporting ? "导出中…" : "导出"}
            </button>
          </>
        ) : (
          <span className="hint">设计失败遗留的音色，请删除后重新设计。</span>
        )}
      </div>
      {cardError && <div className="error">{cardError}</div>}
      {analysis && (
        <div className="voice-diagnostics" onClick={(ev) => ev.stopPropagation()}>
          {analysis.diagnostics.map((d) => (
            <div key={d.id} className={`diagnostic diag-${d.status}`}>
              <span className="diag-label">{DIAG_LABELS[d.id] ?? d.id}</span>
              <span className="diag-message">{d.message}</span>
              <span className="diag-advice">建议：{d.advice}</span>
            </div>
          ))}
        </div>
      )}
      {ref && (
        <audio
          controls
          preload="none"
          src={`${baseUrl}/voices/${voice.id}/reference?token=${encodeURIComponent(mediaToken)}`}
          onClick={(ev) => ev.stopPropagation()}
        />
      )}
    </div>
  );
}
