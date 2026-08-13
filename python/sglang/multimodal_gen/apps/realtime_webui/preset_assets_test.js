const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = __dirname;
const appJs = fs.readFileSync(path.join(root, "app.js"), "utf8");
const assetRoot = path.join(root, "assets", "presets", "v1");
const assetNames = [
  "reactor-dragon-ride.png",
  "reactor-misted-kingdom.png",
  "reactor-storm-crossing.jpg",
  "reactor-citadel-approach.jpg",
  "reactor-spring-valley.png",
  "reactor-reef-patrol.jpg",
  "reactor-alpine-run.jpg",
  "reactor-ice-kayak.jpg",
  "reactor-penguin-colony.jpg",
  "reactor-mars-mountain.jpg",
  "reactor-seaside-adventurer.png",
  "reactor-roman-chariot.png",
  "reactor-asylum-corridor.jpg",
  "lingbot-example-00-dragon-dolly.jpg",
];

assert.ok(
  appJs.includes('const PRESET_ASSET_BASE_URL = "./assets/presets/v1";'),
  "preset references should use the same-origin asset root",
);

for (const externalHost of [
  "reactor.inc/lingbot-world-fast-v1",
  "raw.githubusercontent.com/robbyant/lingbot-world",
  "upload.wikimedia.org",
  "mzstatic.com",
]) {
  assert.ok(
    !appJs.includes(externalHost),
    `app.js should not depend on ${externalHost}`,
  );
}

for (const name of assetNames) {
  const assetPath = path.join(assetRoot, name);
  assert.ok(appJs.includes(name), `${name} should be referenced by a preset`);
  assert.ok(fs.existsSync(assetPath), `${name} should be packaged with the WebUI`);
  assert.ok(fs.statSync(assetPath).size > 0, `${name} should not be empty`);
}

assert.strictEqual(assetNames.length, 14);

const featuredPresetMatch = appJs.match(
  /const FEATURED_PRESET_NAMES = \[([\s\S]*?)\];/,
);
assert.ok(featuredPresetMatch, "featured preset ordering must be explicit");
const featuredPresetNames = [
  ...featuredPresetMatch[1].matchAll(/"([^"]+)"/g),
].map((match) => match[1]);
assert.deepStrictEqual(featuredPresetNames, [
  "Misted Kingdom",
  "Penguin Colony",
  "Seaside Adventurer",
  "Dragon Ride",
  "Spring Valley",
]);
assert.match(
  appJs,
  /\.\.\.FEATURED_PRESET_NAMES\.map\([\s\S]*?\.\.\.reactorPresets\.filter/,
  "featured presets must render before all remaining presets",
);

assert.ok(
  !appJs.includes("globalThis.LINGBOT_TESTSET_20_20260810"),
  "presets without packaged first-frame images must not be exposed",
);
console.log("preset asset contract checks passed");
