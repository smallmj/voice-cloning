// Issue #68: engine-declared non-verbal tag picker — component-level checks.
// JSX lives in a .tsx test file because esbuild only enables the JSX
// transform there; the pure grouping/filtering/insertion logic is covered in
// ui.test.ts.
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { NonverbalTagPicker } from "./components/bits";
import { insertAtCursor } from "./ui";
import type { NonverbalTagInfo } from "./api";

const VOX_TAGS: NonverbalTagInfo[] = [
  { text: "[laughing]", category: "笑叹", label: "笑", verification: "measured" },
  { text: "[sigh]", category: "笑叹", label: "叹气", verification: "measured" },
  { text: "[breath]", category: "呼吸停顿", label: "呼吸", verification: "vendor" },
  { text: "[Uhm]", category: "呼吸停顿", label: "迟疑嗯", verification: "vendor" },
];
const MINIMAX_TAGS: NonverbalTagInfo[] = [
  { text: "(laughs)", category: "笑叹", label: "笑声", verification: "vendor" },
  { text: "<#1#>", category: "停顿", label: "停顿 1 秒", verification: "vendor" },
];

describe("NonverbalTagPicker (issue #68)", () => {
  it("renders NOTHING for engines that declare no tags (or absent data)", () => {
    expect(renderToStaticMarkup(<NonverbalTagPicker tags={[]} onInsert={() => {}} />)).toBe("");
    expect(renderToStaticMarkup(<NonverbalTagPicker tags={undefined} onInsert={() => {}} />)).toBe(
      "",
    );
  });

  it("renders quick buttons + searchable category dropdown when tags exist", () => {
    const html = renderToStaticMarkup(<NonverbalTagPicker tags={VOX_TAGS} onInsert={() => {}} />);
    // Quick row: the common tags verbatim.
    expect(html).toContain(">[laughing]</button>");
    expect(html).toContain(">[sigh]</button>");
    expect(html).toContain(">[breath]</button>");
    // Dropdown shell with search input and the declared count.
    expect(html).toContain("nv-tag-picker");
    expect(html).toContain('type="search"');
    expect(html).toContain("全部标签（4）");
    // The full categorized set lives inside the collapsed dropdown markup.
    expect(html).toContain('title="迟疑嗯（vendor）"');
    expect(html).toContain("呼吸停顿");
  });

  it("renders per-engine syntax verbatim (VoxCPM2 brackets vs MiniMax pause)", () => {
    const vox = renderToStaticMarkup(<NonverbalTagPicker tags={VOX_TAGS} onInsert={() => {}} />);
    const mm = renderToStaticMarkup(<NonverbalTagPicker tags={MINIMAX_TAGS} onInsert={() => {}} />);
    expect(vox).not.toContain("(laughs)");
    expect(mm).toContain(">(laughs)</button>");
    expect(mm).toContain("&lt;#1#&gt;");
    expect(mm).toContain("停顿");
  });

  it("every declared tag inserts correctly at the cursor (click-equivalent flow)", () => {
    for (const tag of [...VOX_TAGS, ...MINIMAX_TAGS]) {
      const { text, cursor } = insertAtCursor("前文后文", tag.text, 2, 2);
      expect(text).toBe(`前文${tag.text}后文`);
      expect(cursor).toBe(2 + tag.text.length);
    }
  });
});
