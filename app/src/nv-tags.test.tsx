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

  // (The former third case here was dead code: renderToStaticMarkup drops
  // onClick, so it could only re-assert what cases 1/2 + the pure
  // insertAtCursor checks below already cover. The real handler is the
  // one-liner `onClick={() => onInsert(tag)}`, type-checked by tsc — a real
  // click assertion would need a DOM environment (jsdom) this suite does
  // not ship; deleted in the issue-#54 review pass rather than kept
  // pretending to test wiring.)

  it("every tag inserts correctly at the cursor (click-equivalent flow)", () => {
    for (const tag of NONVERBAL_TAGS) {
      const { text, cursor } = insertAtCursor("前文后文", tag, 2, 2);
      expect(text).toBe(`前文${tag}后文`);
      expect(cursor).toBe(2 + tag.length);
    }
  });
});
