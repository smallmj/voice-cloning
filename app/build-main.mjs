// Bundle the Electron main + preload with esbuild (CJS, node platform).
import * as esbuild from "esbuild";
import { mkdirSync } from "fs";

mkdirSync("dist-electron", { recursive: true });

await esbuild.build({
  entryPoints: ["electron/main.ts", "electron/preload.ts"],
  outdir: "dist-electron",
  bundle: true,
  platform: "node",
  format: "cjs",
  target: "node20",
  external: ["electron"],
  sourcemap: "inline",
});

console.log("main + preload bundled to dist-electron/");
