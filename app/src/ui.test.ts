import { describe, expect, it } from "vitest";
import type { GenerationRecord, LogEvent } from "./api";
import {
  appendLog,
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
  const ev: LogEvent = { type: "log", generation_id: "abc123", message: "hi", ts: 1 };

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
    expect(
      sendableParams(params, { speed: "1.2", speed_dead: "3", gone: "x" }),
    ).toEqual({ speed: "1.2" });
  });
});
