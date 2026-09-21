import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: ["dist/**", "dist-electron/**", "release/**", "build/**", "node_modules/**"],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    // Node build scripts at the repo top level of app/.
    files: ["*.mjs", "scripts/**/*.mjs"],
    languageOptions: {
      globals: { ...globals.node },
    },
  },
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
    },
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // The renderer intentionally re-derives parameter state from engine
      // specs (ADR-0018); the two `react-hooks/exhaustive-deps` suppressions
      // in the source are deliberate and now validated by this config.
      // v7 strict additions: the existing data-fetch hooks set state after an
      // await inside effects by design; adopting these is a separate refactor.
      "react-hooks/set-state-in-effect": "off",
      "react-hooks/refs": "off",
      // Full-width spaces inside Chinese template literals are intentional.
      "no-irregular-whitespace": ["error", { skipStrings: true, skipTemplates: true }],
    },
  },
  {
    files: ["electron/**"],
    languageOptions: {
      globals: { ...globals.node },
    },
  },
);
