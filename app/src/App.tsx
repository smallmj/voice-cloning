import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  ConsentInfo,
  GenerationJob,
  GenerationRecord,
  NormalizeResult,
  SidecarInfo,
  Voice,
} from "./api";
import { HistorySection } from "./components/HistorySection";
import { SidePanel } from "./components/SidePanel";
import { EnginesSection } from "./sections/EnginesSection";
import { GenerateSection } from "./sections/GenerateSection";
import { SettingsSection } from "./sections/SettingsSection";
import { VoicesSection } from "./sections/VoicesSection";
import { apiJson, authHeaders, errorDetail } from "./client";
import {
  buildRerunState,
  clampSidebarWidth,
  resolveTheme,
  SECTION_IDS,
  SECTION_LABELS,
  sendableParams,
  type GenerateTabId,
  type SectionId,
} from "./ui";
import { useActiveJobCount, useConsent, useEngines, useInstallStatuses, useLogStream, useMediaToken, useUiPrefs, useVoices } from "./hooks";

const DEFAULT_TEXT = "你好，世界。这是一次端到端生成测试。Hello, world!";

// ADR-0013: 五区（生成 / 引擎 / 音色库 / 历史 / 设置）+ 左侧导航；
// 主动作「生成」是默认落点。跨区行为（重跑、刷新历史、展开日志）
// 一律经由 App 的显式回调，不再依赖"同一滚动容器"。
export default function App() {
  const [info, setInfo] = useState<SidecarInfo | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [active, setActive] = useState<SectionId>("generate");
  // Issue #43: which generate sub-tab is showing. Kept in App state (not
  // persisted — no requirement) so the rerun jump lands on 「单条生成」.
  const [generateTab, setGenerateTab] = useState<GenerateTabId>("single");

  const baseUrl = info?.baseUrl ?? null;
  const token = info?.token ?? null;
  const { engines, reload: reloadEngines } = useEngines(baseUrl, token);
  const { voices, setVoices } = useVoices(baseUrl, token);
  const { consent, setConsent } = useConsent(baseUrl, token);
  const mediaToken = useMediaToken(baseUrl, token);
  const { prefs, ready: prefsReady, update: updatePrefs } = useUiPrefs(baseUrl, token);
  // Issue #44: the in-memory log buffer's cap is a software-level setting.
  const logs = useLogStream(baseUrl, mediaToken, prefs.log_buffer);
  const activeJobs = useActiveJobCount(baseUrl, token);

  const navigate = useCallback((section: SectionId) => setActive(section), []);

  // --- theme application (ADR-0014) -----------------------------------------
  const systemDark = useSystemPrefersDark();
  useEffect(() => {
    document.documentElement.dataset.theme = resolveTheme(prefs.theme, systemDark);
  }, [prefs.theme, systemDark]);

  // --- issue #44: UI scale + font-size override ------------------------------
  // Scale rides the root element's `zoom` (Chromium treats root zoom like
  // page zoom, so viewport units follow the scale); a font-size override
  // rewrites the --fs-base variable. Both take effect immediately.
  useEffect(() => {
    const root = document.documentElement;
    if (prefs.ui_scale === 1) root.style.removeProperty("zoom");
    else root.style.zoom = String(prefs.ui_scale);
    if (prefs.font_size == null) root.style.removeProperty("--fs-base");
    else root.style.setProperty("--fs-base", `${prefs.font_size}px`);
  }, [prefs.ui_scale, prefs.font_size]);

  // --- generate panel state (written by history rerun → explicit rewrite) ---
  const [selectedEngine, setSelectedEngine] = useState<string | null>(null);
  const [text, setText] = useState(DEFAULT_TEXT);
  const [record, setRecord] = useState<GenerationRecord | null>(null);
  // Issue #36: the log drawer is gone — logs live in the shared right
  // sidebar; its open state and width persist via /settings/ui. While the
  // user drags the divider, dragWidth shadows the stored value.
  const [dragWidth, setDragWidth] = useState<number | null>(null);
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [jobHint, setJobHint] = useState<string | null>(null);
  const [normalized, setNormalized] = useState<NormalizeResult | null>(null);
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [creatingVoice, setCreatingVoice] = useState(false);
  const [importingVoice, setImportingVoice] = useState(false);
  const [designing, setDesigning] = useState(false);
  const [designError, setDesignError] = useState<string | null>(null);
  const [newVoice, setNewVoiceState] = useState<{
    name: string;
    desc: string;
    file: File | null;
    avatar: File | null;
  }>({ name: "", desc: "", file: null, avatar: null });
  const [design, setDesignState] = useState<{
    engineId: string;
    name: string;
    prompt: string;
    previewText: string;
  }>({ engineId: "", name: "", prompt: "", previewText: "你好，很高兴认识你。" });
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [rerunHint, setRerunHint] = useState<string | null>(null);
  const [engineParams, setEngineParams] = useState<Record<string, string>>({});
  const [rerunParams, setRerunParams] = useState<Record<string, unknown> | null>(null);
  const [installing, setInstalling] = useState<Record<string, boolean>>({});
  const [consentAck, setConsentAck] = useState(false);
  const [consentBusy, setConsentBusy] = useState(false);
  const installingRef = useRef<Record<string, boolean>>({});

  const selectedEngineInfo = useMemo(
    () => engines.find((e) => e.id === selectedEngine) ?? null,
    [engines, selectedEngine],
  );
  const installWatcher = useInstallStatuses(baseUrl, token, engines);

  // Issue #21: engine selection persists via /settings/ui (ADR-0013 §6).
  // Hydrate once the engine list first arrives; afterwards every user
  // selection is written back.
  const hydratedRef = useRef(false);
  useEffect(() => {
    if (engines.length === 0 || !prefsReady || hydratedRef.current) return;
    hydratedRef.current = true;
    const saved = prefs.engine_id;
    setSelectedEngine((current) => {
      if (current) return current;
      if (saved && engines.some((e) => e.id === saved)) return saved;
      return engines[0].id;
    });
  }, [engines, prefs.engine_id, prefsReady]);

  const selectEngine = useCallback(
    (id: string) => {
      setSelectedEngine(id);
      updatePrefs({ engine_id: id });
    },
    [updatePrefs],
  );

  // Re-derive parameter state from the selected engine's spec; parameters the
  // spec does not declare are simply not present here. ADR-0018 decision 6:
  // the engine's last-used values (remembered in the settings store) win over
  // spec defaults, so each engine picks up where the user left off.
  useEffect(() => {
    const spec = selectedEngineInfo?.params ?? [];
    const defaults = Object.fromEntries(
      spec.filter((p) => p.exposed && p.default != null).map((p) => [p.name, String(p.default)]),
    );
    const remembered = (selectedEngine && prefs.engine_params[selectedEngine]) || {};
    setEngineParams({ ...defaults, ...remembered });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedEngine, engines, prefs.engine_params]);

  // Startup: sidecar handshake + consent is handled by hooks; surface hard
  // connection failures from the electron bridge.
  useEffect(() => {
    window.voiceclone.onSidecarError((message) => setConnectionError(message));
    (async () => {
      const sidecarInfo = await window.voiceclone.getSidecarInfo();
      if (!sidecarInfo) return;
      setInfo(sidecarInfo);
    })();
  }, []);

  // Issue #21 lifecycle fix: a pending install is cleared as soon as its
  // status settles, instead of inside the poller.
  useEffect(() => {
    for (const [id, busy] of Object.entries(installing)) {
      const status = installWatcher.statuses[id];
      if (busy && status && !status.installing) {
        setInstalling((p) => ({ ...p, [id]: false }));
        installingRef.current[id] = false;
      }
    }
  }, [installWatcher.statuses, installing]);

  // 归一化预览：文本变化后防抖请求 sidecar 的 /normalize（与生成同一层）。
  useEffect(() => {
    if (!baseUrl || !token) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${baseUrl}/normalize`, {
          method: "POST",
          headers: authHeaders(token, { "Content-Type": "application/json" }),
          body: JSON.stringify({ text }),
          signal: controller.signal,
        });
        if (res.ok) setNormalized((await res.json()) as NormalizeResult);
      } catch {
        /* aborted or sidecar restarting — keep the last preview */
      }
    }, 300);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [baseUrl, token, text]);

  async function acknowledgeConsent() {
    if (!baseUrl || !token || !consentAck || consentBusy) return;
    setConsentBusy(true);
    try {
      setConsent(
        await apiJson<ConsentInfo>(baseUrl, token, "/consent", {
          method: "PUT",
          body: JSON.stringify({ acknowledged: true }),
        }),
      );
    } catch {
      /* the gate stays open; the user can retry */
    } finally {
      setConsentBusy(false);
    }
  }

  async function installEngine(id: string) {
    if (!baseUrl || !token || installing[id]) return;
    setInstalling((p) => ({ ...p, [id]: true }));
    installingRef.current[id] = true;
    installWatcher.watch(id);
    // ADR-0013 2026 修订: the install log surfaces in the shared right
    // sidebar; opening it from the 引擎 section is an explicit cross-section
    // action (and the choice persists).
    updatePrefs({ sidebar_open: true });
    try {
      await fetch(`${baseUrl}/engines/${id}/install`, {
        method: "POST",
        headers: authHeaders(token),
      });
      // Completion is observed by the shared status poller.
    } catch {
      setInstalling((p) => ({ ...p, [id]: false }));
      installingRef.current[id] = false;
    }
  }

  async function createVoice() {
    if (!baseUrl || !token || !newVoice.file || creatingVoice) return;
    setCreatingVoice(true);
    setVoiceError(null);
    try {
      const form = new FormData();
      form.append("name", newVoice.name);
      form.append("description", newVoice.desc);
      form.append("file", newVoice.file);
      if (newVoice.avatar) form.append("avatar", newVoice.avatar);
      const res = await fetch(`${baseUrl}/voices`, {
        method: "POST",
        headers: authHeaders(token),
        body: form,
      });
      if (!res.ok) {
        setVoiceError(`创建音色失败（HTTP ${res.status}）`);
        return;
      }
      const voice = (await res.json()) as Voice;
      setNewVoiceState({ name: "", desc: "", file: null, avatar: null });
      setVoices((prev) => [...prev, voice]);
      setSelectedVoice(voice.id);
    } catch (err) {
      setVoiceError(`创建音色失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCreatingVoice(false);
    }
  }

  async function createDesignedVoice() {
    if (!baseUrl || !token || !design.engineId || designing) return;
    setDesigning(true);
    setDesignError(null);
    try {
      const voice = await apiJson<Voice>(baseUrl, token, "/voices/design", {
        method: "POST",
        body: JSON.stringify({
          engine_id: design.engineId,
          name: design.name,
          voice_prompt: design.prompt,
          preview_text: design.previewText,
        }),
      });
      setDesignState((d) => ({ ...d, name: "", prompt: "" }));
      setVoices((prev) => [...prev, voice]);
      setSelectedVoice(voice.id);
    } catch (err) {
      setDesignError(`设计音色失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setDesigning(false);
    }
  }

  // Issue #14: import a shared voice package.
  async function importVoicePackage(file: File) {
    if (!baseUrl || !token || importingVoice) return;
    setImportingVoice(true);
    setVoiceError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${baseUrl}/voices/import`, {
        method: "POST",
        headers: authHeaders(token),
        body,
      });
      if (!res.ok) {
        setVoiceError(`导入音色包失败（HTTP ${res.status}）`);
        return;
      }
      const { voice } = (await res.json()) as { voice: Voice };
      setVoices((prev) => [...prev, voice]);
      setSelectedVoice(voice.id);
    } catch (err) {
      setVoiceError(`导入音色包失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setImportingVoice(false);
    }
  }

  async function deleteVoice(id: string) {
    if (!baseUrl || !token) return;
    if (!window.confirm("删除该音色？其全部引擎绑定与本地参考音频文件都会被删除。")) return;
    const res = await fetch(`${baseUrl}/voices/${id}`, {
      method: "DELETE",
      headers: authHeaders(token),
    });
    if (res.ok) {
      setVoices((prev) => prev.filter((v) => v.id !== id));
      setSelectedVoice((prev) => (prev === id ? null : prev));
    } else {
      setVoiceError(`删除音色失败（HTTP ${res.status}）`);
    }
  }

  // Issue #37: upload a generation-time audio input (情感参考音频). The
  // sidecar answers with a bare filename inside its audio dir; that name is
  // stored as the param value and the pipeline resolves it server-side.
  async function uploadEngineAudio(name: string, file: File) {
    if (!baseUrl || !token) return;
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch(`${baseUrl}/uploads/audio`, {
        method: "POST",
        headers: authHeaders(token), // no Content-Type: the browser sets the multipart boundary
        body,
      });
      if (!res.ok) throw new Error(await errorDetail(res, `HTTP ${res.status}`));
      const data = (await res.json()) as { file: string };
      setEngineParams((prev) => ({ ...prev, [name]: data.file }));
    } catch (err) {
      setGenerateError(`情感参考音频上传失败：${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function generate() {
    if (!baseUrl || !token || !selectedEngine || generating) return;
    setGenerating(true);
    setRecord(null);
    setGenerateError(null);
    setJobHint(null);
    // ADR-0018 decision 6: remember the values actually used, per engine.
    if (Object.keys(engineParams).length > 0) {
      updatePrefs({
        engine_params: { ...prefs.engine_params, [selectedEngine]: engineParams },
      });
    }
    const sendable = sendableParams(selectedEngineInfo?.params, engineParams);
    const payload = {
      engine_id: selectedEngine,
      text,
      ...(selectedVoice ? { voice_id: selectedVoice } : {}),
      ...(rerunParams || Object.keys(sendable).length > 0
        ? {
            params: {
              // Only parameters the engine still declares AND exposes are
              // ever sent (ADR-0018: exposed:false is data, not rendering).
              ...sendable,
              ...Object.fromEntries(
                Object.entries(rerunParams ?? {}).filter(([k]) =>
                  (selectedEngineInfo?.params ?? []).some((p) => p.name === k && p.exposed),
                ),
              ),
            },
          }
        : {}),
    };
    // Issue #13: over the engine's per-request character limit the text is
    // auto-segmented and QUEUED instead of running synchronously.
    const maxChars = selectedEngineInfo?.capabilities?.max_chars_per_request ?? null;
    const longText = maxChars != null && text.length > maxChars;
    try {
      const data = await apiJson<GenerationRecord | GenerationJob>(
        baseUrl,
        token,
        `/generations${longText ? "/jobs" : ""}`,
        { method: "POST", body: JSON.stringify(payload) },
      );
      if (longText) {
        const job = data as GenerationJob;
        setJobHint(
          `全文 ${text.length} 字符，超过单次上限 ${maxChars}，已自动分段为 ${job.segment_count} 段并加入任务队列，可在右侧「任务队列」窗格查看进度或取消。`,
        );
        setRerunParams(null);
        return;
      }
      setRecord(data as GenerationRecord);
      setRerunParams(null);
      updatePrefs({ sidebar_open: true });
      setHistoryRefreshKey((k) => k + 1);
    } catch (err) {
      setGenerateError(`生成请求失败：${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setGenerating(false);
    }
  }

  // Issue #36: drag-resize the shared sidebar. While dragging, dragWidth
  // shadows the stored width; on release the final width persists via
  // /settings/ui so the geometry survives a restart.
  function startSidebarDrag(e: React.PointerEvent, startWidth: number) {
    e.preventDefault();
    const startX = e.clientX;
    const onMove = (ev: PointerEvent) =>
      setDragWidth(clampSidebarWidth(startWidth + (startX - ev.clientX)));
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      setDragWidth(null);
      const width = clampSidebarWidth(startWidth + (startX - ev.clientX));
      if (width !== prefs.sidebar_width) updatePrefs({ sidebar_width: width });
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  // ADR-0013 §5: rerun loads the record back into the 生成 panel and jumps
  // there — no scrollIntoView against a possibly-unmounted target.
  function rerun(historyRecord: GenerationRecord) {
    setRerunHint(null);
    const state = buildRerunState(historyRecord, voices);
    setRerunParams(state.params);
    setSelectedEngine(state.engineId);
    setText(state.text);
    setSelectedVoice(state.voiceId);
    if (state.hint) setRerunHint(state.hint);
    setGenerateTab("single"); // rerun targets the single-generation pane only
    setActive("generate");
  }

  const commonSectionProps = info
    ? { baseUrl: info.baseUrl, token: info.token }
    : null;

  return (
    <div className="app">
      {consent && !consent.acknowledged && (
        <div className="consent-overlay" role="dialog" aria-modal="true" aria-label="声音授权确认">
          <div className="consent-modal">
            <h2>声音授权确认</h2>
            <p>在使用声音复刻功能之前，请确认以下事项：</p>
            <ul>
              <li>你上传的参考音频中的声音，是你本人的声音，或你已获得该声音权利人的明确授权；</li>
              <li>生成的语音由人工智能合成，产物文件会带有 AI 生成内容的元数据标记，你应在传播时如实告知听众；</li>
              <li>你不会使用本工具伪造他人声音进行欺骗、冒充或侵犯他人权益的行为。</li>
            </ul>
            <label className="consent-check">
              <input
                type="checkbox"
                checked={consentAck}
                onChange={(e) => setConsentAck(e.target.checked)}
              />
              我已阅读并理解以上内容，确认拥有相关声音的使用授权
            </label>
            <button
              className="primary"
              disabled={!consentAck || consentBusy}
              onClick={() => void acknowledgeConsent()}
            >
              {consentBusy ? "提交中…" : "确认并开始使用"}
            </button>
          </div>
        </div>
      )}

      <div className="shell">
        <nav className="side-nav" aria-label="主导航">
          <div className="side-nav-title">声音复刻工作台</div>
          <ul>
            {SECTION_IDS.map((id) => (
              <li key={id}>
                <button
                  className={`nav-item ${active === id ? "active" : ""}`}
                  aria-current={active === id ? "page" : undefined}
                  onClick={() => setActive(id)}
                >
                  <span className="nav-icon" aria-hidden="true">
                    {
                      {
                        generate: "▶",
                        engines: "⚙",
                        voices: "🎙",
                        history: "≡",
                        settings: "✳",
                      }[id]
                    }
                  </span>
                  {SECTION_LABELS[id]}
                  {id === "history" && activeJobs > 0 && (
                    <span className="nav-badge" title="进行中的分段任务">
                      {activeJobs}
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
          <div className="side-nav-footer">
            {/* Issue #36: explicit show/hide for the shared right sidebar. */}
            <button
              className={`sidebar-toggle ${prefs.sidebar_open ? "active" : ""}`}
              aria-pressed={prefs.sidebar_open}
              title="显示/隐藏侧边栏（实时日志与任务队列）"
              onClick={() => updatePrefs({ sidebar_open: !prefs.sidebar_open })}
            >
              <span aria-hidden="true">◧</span> 侧边栏
            </button>
            <span className={`status ${info ? "ok" : "down"}`}>
              {connectionError ?? (info ? "sidecar 已连接" : "sidecar 连接中…")}
            </span>
            {selectedEngineInfo && (
              <span className="current-engine" title="当前引擎">
                {selectedEngineInfo.display_name}
              </span>
            )}
          </div>
        </nav>

        <main className="content">
          {!info ? (
            <div className="hint">正在连接 sidecar…</div>
          ) : !commonSectionProps ? null : active === "generate" ? (
            <GenerateSection
              baseUrl={commonSectionProps.baseUrl}
              token={commonSectionProps.token}
              mediaToken={mediaToken}
              tab={generateTab}
              onTabChange={setGenerateTab}
              engines={engines}
              selectedEngine={selectedEngine}
              selectedEngineInfo={selectedEngineInfo}
              installStatus={installWatcher.statuses}
              voices={voices}
              selectedVoice={selectedVoice}
              onSelectVoice={setSelectedVoice}
              text={text}
              onTextChange={setText}
              engineParams={engineParams}
              onEngineParamChange={(name, value) =>
                setEngineParams((prev) => ({ ...prev, [name]: value }))
              }
              onEngineAudioUpload={(name, file) =>
                void uploadEngineAudio(name, file)
              }
              normalized={normalized}
              record={record}
              generating={generating}
              generateError={generateError}
              rerunHint={rerunHint}
              jobHint={jobHint}
              rerunParams={rerunParams}
              onClearRerunParams={() => setRerunParams(null)}
              onGenerate={() => void generate()}
              onNavigate={navigate}
            />
          ) : active === "engines" ? (
            <EnginesSection
              baseUrl={commonSectionProps.baseUrl}
              token={commonSectionProps.token}
              engines={engines}
              selectedEngine={selectedEngine}
              installStatus={installWatcher.statuses}
              installing={installing}
              onInstall={(id) => void installEngine(id)}
              onSelect={selectEngine}
              onNavigate={navigate}
              refreshEngines={reloadEngines}
            />
          ) : active === "voices" ? (
            <VoicesSection
              baseUrl={commonSectionProps.baseUrl}
              token={commonSectionProps.token}
              mediaToken={mediaToken}
              engines={engines}
              voices={voices}
              selectedVoice={selectedVoice}
              onNavigate={navigate}
              onSelectVoice={setSelectedVoice}
              onTranscribed={(updated) =>
                setVoices((prev) => prev.map((x) => (x.id === updated.id ? updated : x)))
              }
              createVoice={() => void createVoice()}
              createDesignedVoice={() => void createDesignedVoice()}
              importVoicePackage={(f) => void importVoicePackage(f)}
              deleteVoice={(id) => void deleteVoice(id)}
              creatingVoice={creatingVoice}
              importingVoice={importingVoice}
              newVoice={newVoice}
              setNewVoice={(patch) => setNewVoiceState((prev) => ({ ...prev, ...patch }))}
              design={{ ...design, busy: designing }}
              setDesign={(patch) => setDesignState((prev) => ({ ...prev, ...patch }))}
              voiceError={voiceError}
              designError={designError}
            />
          ) : active === "history" ? (
            // Issue #36: the inline job queue moved to the shared right
            // sidebar; the history section is the record list only.
            <HistorySection
              baseUrl={commonSectionProps.baseUrl}
              token={commonSectionProps.token}
              mediaToken={mediaToken}
              engines={engines}
              onRerun={rerun}
              refreshKey={historyRefreshKey}
            />
          ) : (
            <SettingsSection
              baseUrl={commonSectionProps.baseUrl}
              token={commonSectionProps.token}
              prefs={prefs}
              updatePrefs={updatePrefs}
              onNavigate={navigate}
            />
          )}
        </main>

        {/* Issue #36: five tabs share one right sidebar — logs on top, job
        queue below. It stays mounted while hidden so job polling and the
        history-refresh callback keep running. */}
        {commonSectionProps && (
          <SidePanel
            className={prefs.sidebar_open ? "" : "closed"}
            width={dragWidth ?? prefs.sidebar_width}
            baseUrl={commonSectionProps.baseUrl}
            token={commonSectionProps.token}
            logs={logs}
            onSettled={() => setHistoryRefreshKey((k) => k + 1)}
            onDragStart={(e) => startSidebarDrag(e, dragWidth ?? prefs.sidebar_width)}
          />
        )}
      </div>
    </div>
  );
}

function useSystemPrefersDark(): boolean {
  const [dark, setDark] = useState(() =>
    typeof window !== "undefined" && window.matchMedia
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
      : true,
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (e: MediaQueryListEvent) => setDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return dark;
}
