import { describe, expect, it } from "vitest";
import type { GenerationRecord, LogEvent } from "./api";
import {
  appendLog,
  engineDetailParagraphs,
  voiceDesignEngines,
  buildRerunState,
  isSectionId,
  jobSettled,
  LOG_BUFFER_LIMIT,
  parseLogEvent,
  resolveTheme,
  SECTION_IDS,
} from "./ui";

function rec(overrides: Partial<GenerationRecord> = {}): GenerationRecord {
  return {
    id: "g1",
    engine_id: "fake",
    text: "hello",
    params: {},
    voice_id: null,
    status: "succeeded",
    logs: [],
    created_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("five sections + navigation (ADR-0013)", () => {
  it("defines exactly the five sections with generate first", () => {
    expect(SECTION_IDS).toEqual(["generate", "engines", "voices", "history", "settings"]);
  });

  it("validates persisted section ids", () => {
    expect(isSectionId("generate")).toBe(true);
    expect(isSectionId("bogus")).toBe(false);
    expect(isSectionId(null)).toBe(false);
  });
});

describe("theme resolution (ADR-0014)", () => {
  it("explicit themes win over the system", () => {
    expect(resolveTheme("dark", false)).toBe("dark");
    expect(resolveTheme("light", true)).toBe("light");
  });

  it("system follows the OS", () => {
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
  });
});

describe("cross-section rerun (history → generate)", () => {
  it("carries engine, text and stored params, dropping server-side ref_audio", () => {
    const state = buildRerunState(
      rec({ engine_id: "qwen", text: "全文", params: { speed: "1.2", ref_audio: "/x.wav" } }),
      [{ id: "v1" }],
    );
    expect(state.engineId).toBe("qwen");
    expect(state.text).toBe("全文");
    expect(state.params).toEqual({ speed: "1.2" });
    expect(state.hint).toBeNull();
  });

  it("keeps a still-existing voice selected", () => {
    const state = buildRerunState(rec({ voice_id: "v1" }), [{ id: "v1" }]);
    expect(state.voiceId).toBe("v1");
    expect(state.hint).toBeNull();
  });

  it("clears a deleted voice and says so", () => {
    const state = buildRerunState(rec({ voice_id: "gone" }), [{ id: "v1" }]);
    expect(state.voiceId).toBeNull();
    expect(state.hint).toContain("原音色已删除");
  });

  it("returns null params when nothing was stored", () => {
    expect(buildRerunState(rec(), []).params).toBeNull();
  });
});

describe("cross-section: job settle → history refresh", () => {
  it("only fires for queued/running jobs reaching a terminal state", () => {
    expect(jobSettled("running", "succeeded")).toBe(true);
    expect(jobSettled("running", "failed")).toBe(true);
    expect(jobSettled("queued", "cancelled")).toBe(true); // cancels also refresh history
    expect(jobSettled("succeeded", "failed")).toBe(false);
    expect(jobSettled("running", "queued")).toBe(false);
    expect(jobSettled(undefined, "succeeded")).toBe(false); // first sighting
  });
});

describe("log stream", () => {
  const ev: LogEvent = { type: "log", generation_id: "abc123", message: "hi", ts: 1, level: "info" };

  it("caps the buffer at 500 entries", () => {
    let logs: LogEvent[] = [];
    for (let i = 0; i < LOG_BUFFER_LIMIT + 50; i++) {
      logs = appendLog(logs, { ...ev, message: String(i) });
    }
    expect(logs).toHaveLength(LOG_BUFFER_LIMIT);
    expect(logs[logs.length - 1].message).toBe(String(LOG_BUFFER_LIMIT + 49));
  });

  it("parses well-formed frames", () => {
    expect(parseLogEvent(JSON.stringify(ev))).toEqual(ev);
  });

  it("drops malformed frames instead of throwing", () => {
    expect(parseLogEvent("not json")).toBeNull();
    expect(parseLogEvent(JSON.stringify({ type: "log" }))).toBeNull();
    expect(parseLogEvent("null")).toBeNull();
  });

  it("parses the level, defaulting unknown/missing levels to info (issue #36)", () => {
    expect(parseLogEvent(JSON.stringify({ ...ev, level: "error" }))?.level).toBe("error");
    expect(parseLogEvent(JSON.stringify(ev))?.level).toBe("info");
    expect(parseLogEvent(JSON.stringify({ ...ev, level: "bogus" }))?.level).toBe("info");
  });
});

// --- issue #36: log filtering + sidebar geometry -------------------------------

import { clampSidebarWidth, filterLogs, SIDEBAR_MAX_WIDTH, SIDEBAR_MIN_WIDTH } from "./ui";

function logEv(level: LogEvent["level"], message: string): LogEvent {
  return { type: "log", generation_id: "abc123", message, ts: 1, level };
}

describe("log filtering (issue #36)", () => {
  const logs = [
    logEv("info", "分段 1 完成"),
    logEv("warn", "云端音色已失效，正在重建绑定"),
    logEv("error", "generation failed: timeout"),
  ];

  it("passes everything when no level is selected and no keyword given", () => {
    expect(filterLogs(logs, { levels: new Set(), keyword: "" })).toHaveLength(3);
  });

  it("filters by level", () => {
    const only = filterLogs(logs, { levels: new Set(["error"]), keyword: "" });
    expect(only.map((l) => l.level)).toEqual(["error"]);
  });

  it("matches the keyword case-insensitively across levels", () => {
    const hits = filterLogs(logs, { levels: new Set(), keyword: "FAILED" });
    expect(hits).toHaveLength(1);
    expect(hits[0].level).toBe("error");
  });

  it("combines level + keyword", () => {
    expect(
      filterLogs(logs, { levels: new Set(["warn", "error"]), keyword: "分段" }),
    ).toHaveLength(0);
  });

  it("trims whitespace around the keyword", () => {
    expect(filterLogs(logs, { levels: new Set(), keyword: "  重建绑定 " })).toHaveLength(1);
  });
});

describe("sidebar width clamp (issue #36)", () => {
  it("keeps widths inside the draggable range", () => {
    expect(clampSidebarWidth(400)).toBe(400);
    expect(clampSidebarWidth(10)).toBe(SIDEBAR_MIN_WIDTH);
    expect(clampSidebarWidth(99999)).toBe(SIDEBAR_MAX_WIDTH);
  });

  it("falls back to the default for non-finite values", () => {
    expect(clampSidebarWidth(NaN)).toBe(360);
    expect(clampSidebarWidth(Infinity)).toBe(SIDEBAR_MAX_WIDTH);
  });
});

// --- issue #23 / ADR-0018: parameter layers -----------------------------------

import { sendableParams, splitParamLayers } from "./ui";
import type { ParamSpecInfo } from "./api";

function spec(overrides: Partial<ParamSpecInfo>): ParamSpecInfo {
  return {
    name: "p",
    label: "P",
    kind: "text",
    layer: "engine",
    group: "",
    wire_path: "",
    default: null,
    choices: [],
    help: "",
    unit: "",
    min: null,
    max: null,
    step: null,
    integer: false,
    min_open: false,
    max_open: false,
    max_from: null,
    max_length: null,
    max_items: null,
    items: [],
    exposed: true,
    not_exposed_reason: null,
    applies_to: { engine: "e", model: "m", mode: "cloning" },
    ignored_when: [],
    wire_map: false,
    ...overrides,
  };
}

describe("parameter layers (issue #23 / ADR-0018)", () => {
  it("splits canonical, engine-specific and hidden specs", () => {
    const layers = splitParamLayers([
      spec({ name: "speed", layer: "canonical" }),
      spec({ name: "fake_mode" }),
      spec({ name: "speed_dead", exposed: false, not_exposed_reason: "no-op" }),
      spec({ name: "pronunciation", layer: "canonical", kind: "textarea" }),
    ]);
    expect(layers.canonical.map((p) => p.name)).toEqual(["speed", "pronunciation"]);
    expect(layers.engine.map((p) => p.name)).toEqual(["fake_mode"]);
    expect(layers.hidden.map((p) => p.name)).toEqual(["speed_dead"]);
  });

  it("tolerates a missing params list", () => {
    expect(splitParamLayers(undefined)).toEqual({ canonical: [], engine: [], hidden: [] });
  });

  it("only sends exposed, declared parameters", () => {
    const params = [
      spec({ name: "speed" }),
      spec({ name: "speed_dead", exposed: false, not_exposed_reason: "no-op" }),
    ];
    expect(sendableParams(params, { speed: "1.2", speed_dead: "3", gone: "x" })).toEqual({
      speed: "1.2",
    });
  });
});

// --- issue #37: ignored_when mode linkage --------------------------------------

import { visibleParamSpecs } from "./ui";

describe("visibleParamSpecs (issue #37 ignored_when)", () => {
  const specs = [
    spec({ name: "emo_mode", kind: "select", choices: ["与参考音频相同", "情感向量"], default: "与参考音频相同" }),
    spec({ name: "emo_vector", ignored_when: ["emo_mode!=情感向量"] }),
  ];

  it("hides specs whose ignored_when predicates all hold", () => {
    expect(visibleParamSpecs(specs, {}).map((p) => p.name)).toEqual(["emo_mode"]);
  });

  it("shows the gated spec once the mode matches", () => {
    expect(visibleParamSpecs(specs, { emo_mode: "情感向量" }).map((p) => p.name)).toEqual([
      "emo_mode",
      "emo_vector",
    ]);
  });

  it("never hides on an unknown predicate", () => {
    const weird = [
      spec({ name: "x", ignored_when: ["???garbage"] }),
    ];
    expect(visibleParamSpecs(weird, {}).map((p) => p.name)).toEqual(["x"]);
  });

  it("keeps specs without ignored_when always visible", () => {
    expect(visibleParamSpecs([spec({ name: "temperature" })], {})).toHaveLength(1);
  });
});

// --- install progress rendering (issue #35) -----------------------------------

import { formatBytes, progressPercent } from "./install-progress";

describe("progressPercent", () => {
  it("computes percent from bytes", () => {
    expect(progressPercent({ file: "w", done_bytes: 50, total_bytes: 200 })).toBe(25);
  });

  it("returns null when the total size is unknown", () => {
    expect(progressPercent({ file: "w", done_bytes: 50, total_bytes: null })).toBeNull();
    expect(progressPercent({ file: "w", done_bytes: 50, total_bytes: 0 })).toBeNull();
  });

  it("clamps at 100", () => {
    expect(progressPercent({ file: "w", done_bytes: 300, total_bytes: 200 })).toBe(100);
  });
});

describe("formatBytes", () => {
  it("formats human-readable byte counts", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(formatBytes(1.5 * 1024 * 1024 * 1024)).toBe("1.50 GB");
  });
});

// ---------------------------------------------------------------------------
// Issue #40: engines-page card rework helpers.
// ---------------------------------------------------------------------------

describe("voice design engine filter (issue #40)", () => {
  function engine(id: string, voiceDesign: boolean, keyConfigured = true) {
    return {
      id,
      display_name: id,
      capabilities: {
        languages: ["zh"],
        voice_cloning: true,
        voice_design: voiceDesign,
        pronunciation_control: false,
        emotion: false,
        commercial_license: false,
        cross_device_use: false,
        upload_used_for_training: false,
        api_closed_loop: true,
      },
      requires_key: true,
      key_configured: keyConfigured,
    };
  }

  it("keeps only engines that declare voice_design", () => {
    const engines = [engine("a", true), engine("b", false), engine("c", true, false)];
    expect(voiceDesignEngines(engines).map((e) => e.id)).toEqual(["a", "c"]);
  });

  it("returns an empty list when no engine supports design", () => {
    expect(voiceDesignEngines([engine("b", false)])).toEqual([]);
  });
});

describe("engine card detail paragraphs (issue #40)", () => {
  const engine = {
    id: "qwen",
    display_name: "Qwen3-TTS",
    capabilities: {
      languages: ["zh", "en"],
      voice_cloning: true,
      voice_design: false,
      pronunciation_control: false,
      emotion: false,
      commercial_license: false,
      cross_device_use: false,
      upload_used_for_training: false,
      api_closed_loop: true,
    },
    billing_note: "按字符计费",
    data_usage_note: "不上传训练",
  };

  it("orders role → capability declaration → notes → matrix notes", () => {
    const paragraphs = engineDetailParagraphs(engine, {
      engine_id: "qwen",
      display_name: "Qwen3-TTS",
      kind: "cloud",
      role: "BYOK 云端复刻档",
      notes: "回归覆盖归一化文本。",
    });
    expect(paragraphs[0]).toBe("BYOK 云端复刻档");
    expect(paragraphs[1]).toContain("能力声明");
    expect(paragraphs[1]).toContain("复刻 ✓");
    expect(paragraphs[1]).toContain("音色设计 ✗");
    expect(paragraphs[1]).toContain("zh/en");
    expect(paragraphs[2]).toContain("计费口径：按字符计费");
    expect(paragraphs[3]).toContain("数据与训练：不上传训练");
    expect(paragraphs[4]).toBe("回归覆盖归一化文本。");
  });

  it("skips absent matrix entry and notes without placeholders", () => {
    const paragraphs = engineDetailParagraphs(engine, null);
    expect(paragraphs).toHaveLength(3);
    expect(paragraphs.join("\n")).not.toMatch(/undefined|null/);
  });
});
