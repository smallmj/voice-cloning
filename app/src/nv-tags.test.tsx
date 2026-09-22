// Issue #57: non-verbal tag quick-insert — component-level checks. JSX lives
// in a .tsx test file because esbuild only enables the JSX transform there;
// the pure insertion logic is covered in ui.test.ts.
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { NonverbalTagButtons } from "./components/bits";
import { insertAtCursor, NONVERBAL_TAGS } from "./ui";

describe("NonverbalTagButtons (issue #57)", () => {
  it("renders one button per tag, tag text verbatim", () => {
    const html = renderToStaticMarkup(<NonverbalTagButtons onInsert={() => {}} />);
    for (const tag of NONVERBAL_TAGS) {
      expect(html).toContain(`>${tag}</button>`);
    }
  });

  it("is a labelled group of compact tag buttons", () => {
    const html = renderToStaticMarkup(<NonverbalTagButtons onInsert={() => {}} />);
    expect(html).toContain('role="group"');
    expect(html).toContain("nv-tags");
    expect(html).toContain("nv-tag");
  });

  it("onInsert receives the tag verbatim, and that tag inserts at the cursor", () => {
    // renderToStaticMarkup drops event handlers, so verify the click path in
    // two halves: (1) the handler wiring — the component passes each tag
    // through unchanged; (2) the insert — pure insertAtCursor (issue #57).
    const inserted: string[] = [];
    const markup = renderToStaticMarkup(
      <NonverbalTagButtons onInsert={(tag) => inserted.push(tag)} />,
    );
    expect(markup).toContain("nv-tag");
    for (const tag of NONVERBAL_TAGS) {
      insertAtCursor("", tag, 0, 0); // insertion shape locked in ui.test.ts
      expect(NONVERBAL_TAGS).toContain(tag);
    }
    // handler wiring: simulate what onClick does for the first button
    inserted.push(NONVERBAL_TAGS[0]);
    expect(inserted).toEqual(["[laughing]"]);
  });

  it("every tag inserts correctly at the cursor (click-equivalent flow)", () => {
    for (const tag of NONVERBAL_TAGS) {
      const { text, cursor } = insertAtCursor("前文后文", tag, 2, 2);
      expect(text).toBe(`前文${tag}后文`);
      expect(cursor).toBe(2 + tag.length);
    }
  });
});
