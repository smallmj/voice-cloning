import type { Capabilities } from "./api";

export const CAP_LABELS: Record<keyof Capabilities, string> = {
  languages: "语种",
  voice_cloning: "复刻",
  voice_design: "音色设计",
  pronunciation_control: "发音控制",
  emotion: "情感",
  commercial_license: "商业许可",
  cross_device_use: "跨设备",
  upload_used_for_training: "上传用于训练",
  api_closed_loop: "API 闭环",
  requires_reference_text: "需参考文本",
  // Rendered as its own control (issue #68), never as a capability badge.
  nonverbal_tags: "非语言标签",
  // Never rendered as a badge — the settings matrix lists its keys explicitly.
  max_chars_per_request: "单次字符上限",
};

export const STEP_LABELS: Record<string, string> = {
  python: "Python 运行时",
  venv: "引擎独立环境",
  packages: "依赖安装",
  weights: "引擎权重下载",
};

export const DIAG_LABELS: Record<string, string> = {
  snr: "信噪比",
  speaker: "说话人",
  clipping: "削波",
  silence: "静音段",
};
