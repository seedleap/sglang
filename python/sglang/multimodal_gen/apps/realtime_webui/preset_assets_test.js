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
  "lingbot-example-01-stone-orbit.jpg",
  "lingbot-example-02-urban-tilt.jpg",
  "lingbot-example-03-lake-scout.jpg",
  "artwork-ziggy-stardust.jpg",
  "artwork-plastic-beach.jpg",
  "artwork-plastic-ono-band.jpg",
  "artwork-kid-a.jpg",
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

assert.strictEqual(assetNames.length, 21);

const testsetModule = path.join(root, "assets", "presets", "lingbot_testset_20_20260810", "presets.js");
assert.ok(fs.existsSync(testsetModule), "the 20260810 LingBot testset preset module should be packaged");
const testsetPresets = require(testsetModule);
assert.strictEqual(testsetPresets.length, 20, "the complete 20-case LingBot testset should be available");
for (const preset of testsetPresets) {
  assert.match(preset.id, /^lingbot-testset-20260810-case-\d{2}$/);
  assert.ok(preset.prompt.length > 80, `${preset.id} should include its complete prompt`);
  assert.strictEqual(preset.size, "1280x704");
  assert.strictEqual(preset.fps, 24);
  assert.match(preset.referenceUrl, /^\.\/assets\/presets\/lingbot_testset_20_20260810\/images\//);
  const imageName = path.basename(preset.referenceUrl);
  const imagePath = path.join(path.dirname(testsetModule), "images", imageName);
  assert.ok(fs.existsSync(imagePath), `${preset.id} image should be packaged with the WebUI`);
  assert.ok(fs.statSync(imagePath).size > 0, `${preset.id} image should not be empty`);
}
assert.ok(
  appJs.includes("globalThis.LINGBOT_TESTSET_20_20260810"),
  "app.js should append the packaged LingBot testset presets",
);
console.log("preset asset contract checks passed");
