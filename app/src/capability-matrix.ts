import { useEffect, useState } from "react";
import { apiJson } from "./client";

// ---------------------------------------------------------------------------
// 能力矩阵（issue #16）：固化的矩阵数据，界面直接驱动。
// Shared by the engine cards (issue #40 「详情」) and the matrix table.
// ---------------------------------------------------------------------------

export interface MatrixEvidence {
  field: string;
  value: string;
  verification: "measured" | "verified" | "vendor" | "unverified" | "unknown";
  source: string;
  date?: string;
}

export interface MatrixPerf {
  env: string;
  rtf: number | null;
  peak_vram_bytes: number | null;
  verification: MatrixEvidence["verification"];
  source?: string;
  note?: string;
}

export interface MatrixEngine {
  engine_id: string;
  display_name: string;
  kind: "builtin" | "local" | "cloud";
  role?: string;
  evidence?: MatrixEvidence[];
  performance?: MatrixPerf[];
  notes?: string;
  runtime?: { capabilities: Record<string, unknown> } | null;
  regression_summary?: {
    ok: number;
    failed: number;
    rtf_mean: number | null;
  } | null;
}

export interface CapabilityMatrix {
  updated_at: string;
  categories: { id: string; label: string }[];
  engines: MatrixEngine[];
}

export const VERIFY_LABELS: Record<MatrixEvidence["verification"], string> = {
  measured: "实测",
  verified: "已核实",
  vendor: "厂商口径",
  unverified: "未核实",
  unknown: "未知",
};

export function matrixEntryFor(
  matrix: CapabilityMatrix | null,
  engineId: string,
): MatrixEngine | null {
  return matrix?.engines.find((e) => e.engine_id === engineId) ?? null;
}

export function useCapabilityMatrix(
  baseUrl: string,
  token: string,
): CapabilityMatrix | null {
  const [matrix, setMatrix] = useState<CapabilityMatrix | null>(null);
  useEffect(() => {
    apiJson<CapabilityMatrix>(baseUrl, token, "/capability-matrix")
      .then(setMatrix)
      .catch(() => {
        /* sidecar not ready yet */
      });
  }, [baseUrl, token]);
  return matrix;
}
