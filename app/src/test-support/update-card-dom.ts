// Shared jsdom render harness for the AboutUpdateCard tests (issue #66).
// Mocks the window.voiceclone updater bridge, returns markup/snapshot
// helpers and event-push access so interaction cases can assert call args
// and pushed-event state transitions.

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { createElement } from "react";
import { AboutUpdateCard } from "../components/AboutUpdateCard";
import {
  type UpdateEvent,
  type UpdateSettings,
  type UpdateStatus,
  type UpdaterResult,
  type NewVersionInfo,
} from "../update-client";

(window as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

export interface CardMockOverrides {
  status?: UpdateStatus | null;
  checkForUpdate?: (manual: boolean) => Promise<UpdaterResult>;
  saveUpdateSettings?: (s: UpdateSettings) => Promise<void>;
  skipVersion?: (tag: string) => Promise<void>;
  startUpdateDownload?: () => Promise<void>;
  cancelUpdateDownload?: () => Promise<void>;
}

export interface CardHarness {
  html: () => string;
  clickButton: (label: string) => Promise<void>;
  changeSelect: (value: string) => Promise<void>;
  pushEvent: (e: UpdateEvent) => void;
  pushNewVersion: (r: NewVersionInfo) => void;
  savedSettings: UpdateSettings[];
  skippedTags: string[];
  downloadsStarted: number;
  downloadsCancelled: number;
  /** PR #63 review fix: listener bookkeeping for the unsubscribe test. */
  subscriptionsAdded: number;
  subscriptionsRemoved: number;
  unmount: () => Promise<void>;
}

export async function renderCard(overrides: CardMockOverrides = {}): Promise<CardHarness> {
  const savedSettings: UpdateSettings[] = [];
  const skippedTags: string[] = [];
  let downloadsStarted = 0;
  let downloadsCancelled = 0;
  let eventCb: ((e: UpdateEvent) => void) | null = null;
  let newVersionCb: ((r: NewVersionInfo) => void) | null = null;
  // PR #63 review fix: subscription bookkeeping so tests can assert that
  // unmount actually removes the listeners (no stacking on remount).
  let subscriptionsAdded = 0;
  let subscriptionsRemoved = 0;

  const bridge = {
    getUpdateStatus: async () => overrides.status ?? null,
    checkForUpdate:
      overrides.checkForUpdate ??
      (async () => ({ status: "up-to-date" as const, latestTag: "0", effectiveChannel: "mirror" })),
    saveUpdateSettings: async (s: UpdateSettings) => {
      savedSettings.push(s);
      overrides.saveUpdateSettings?.(s);
    },
    skipVersion: async (tag: string) => {
      skippedTags.push(tag);
      overrides.skipVersion?.(tag);
    },
    startUpdateDownload: async () => {
      downloadsStarted += 1;
      overrides.startUpdateDownload?.();
    },
    cancelUpdateDownload: async () => {
      downloadsCancelled += 1;
      overrides.cancelUpdateDownload?.();
    },
    onUpdateEvent: (cb: (e: UpdateEvent) => void) => {
      eventCb = cb;
      subscriptionsAdded += 1;
      // Real unsubscribe (review fix PR #63 finding 3), mirroring preload.
      return () => {
        if (eventCb === cb) eventCb = null;
        subscriptionsRemoved += 1;
      };
    },
    onNewVersionAvailable: (cb: (r: NewVersionInfo) => void) => {
      newVersionCb = cb;
      subscriptionsAdded += 1;
      return () => {
        if (newVersionCb === cb) newVersionCb = null;
        subscriptionsRemoved += 1;
      };
    },
  };
  (window as unknown as { voiceclone: unknown }).voiceclone = bridge;

  const container = document.createElement("div");
  document.body.appendChild(container);
  const root: Root = createRoot(container);
  await act(async () => {
    root.render(createElement(AboutUpdateCard));
  });
  // Flush the getUpdateStatus promise chain.
  await act(async () => {
    await Promise.resolve();
  });

  function findButton(label: string): HTMLButtonElement | null {
    return [...container.querySelectorAll("button")].find((b) => b.textContent === label) ?? null;
  }

  return {
    html: () => container.innerHTML,
    clickButton: async (label) => {
      const btn = findButton(label);
      if (!btn) throw new Error(`button not found: ${label}`);
      await act(async () => {
        btn.click();
        await Promise.resolve();
        await Promise.resolve();
      });
    },
    changeSelect: async (value) => {
      const select = container.querySelector<HTMLSelectElement>("select");
      if (!select) throw new Error("no select found");
      await act(async () => {
        const proto = Object.getPrototypeOf(select);
        const desc = Object.getOwnPropertyDescriptor(proto, "value");
        desc?.set?.call(select, value);
        select.dispatchEvent(new Event("change", { bubbles: true }));
        await Promise.resolve();
      });
    },
    pushEvent: (e) => {
      act(() => {
        eventCb?.(e);
      });
    },
    pushNewVersion: (r) => {
      act(() => {
        newVersionCb?.(r);
      });
    },
    get savedSettings() {
      return savedSettings;
    },
    get skippedTags() {
      return skippedTags;
    },
    get downloadsStarted() {
      return downloadsStarted;
    },
    get downloadsCancelled() {
      return downloadsCancelled;
    },
    get subscriptionsAdded() {
      return subscriptionsAdded;
    },
    get subscriptionsRemoved() {
      return subscriptionsRemoved;
    },
    unmount: async () => {
      await act(async () => {
        root.unmount();
      });
      container.remove();
      delete (window as { voiceclone?: unknown }).voiceclone;
    },
  };
}
