#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const records = readFileSync(join(root, "manifest.jsonl"), "utf8")
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const titleCase = (value) => value
  .split(/[\s_-]+/)
  .filter(Boolean)
  .map((word) => `${word[0].toUpperCase()}${word.slice(1)}`)
  .join(" ");

const tones = {
  human_hands_visible: "green",
  human_body_hidden: "blue",
  nonhuman_organic: "accent",
  nonhuman_mechanical: "blue",
};

const presets = records.map((record) => ({
  id: `lingbot-testset-20260810-${record.case_id.replace("_", "-")}`,
  name: titleCase(record.subject_detail.split("/").at(-1)),
  tone: tones[record.category] || "green",
  size: "1280x704",
  fps: 24,
  prompt: record.prompt,
  referenceUrl: `./assets/presets/lingbot_testset_20_20260810/${record.first_frame}`,
  mime: "image/png",
  source: "LingBot reviewed testset 20260810",
  metadata: {
    caseId: record.case_id,
    category: record.category,
    trajectoryId: record.trajectory_id,
    actionFamily: record.action_family_pattern,
  },
}));

const output = `// Generated from manifest.jsonl by build_presets.mjs.\n`
  + `(function exposeLingBotTestset(root, factory) {\n`
  + `  const presets = factory();\n`
  + `  if (typeof module === "object" && module.exports) module.exports = presets;\n`
  + `  if (root) root.LINGBOT_TESTSET_20_20260810 = presets;\n`
  + `})(typeof globalThis !== "undefined" ? globalThis : this, function buildPresets() {\n`
  + `  return ${JSON.stringify(presets, null, 2)};\n`
  + `});\n`;

writeFileSync(join(root, "presets.js"), output);
console.log(`generated ${presets.length} presets`);
