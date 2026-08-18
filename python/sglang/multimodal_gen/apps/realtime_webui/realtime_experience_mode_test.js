const assert = require("node:assert/strict");

const mode = require("./realtime_experience_mode.js");

assert.equal(mode.isZingOnly(), false);
assert.equal(mode.isZingOnly({}), false);
assert.equal(mode.isZingOnly({ zingOnly: false }), false);
assert.equal(mode.isZingOnly({ zingOnly: "true" }), false);
assert.equal(mode.isZingOnly({ zingOnly: 1 }), false);
assert.equal(mode.isZingOnly({ zingOnly: true }), true);

const requested = ["lingbot2", "minwm", "happyoyster"];
assert.deepEqual(
  mode.selectedModelKeys({}, requested),
  requested,
  "the default mode must preserve the existing model selection",
);
assert.deepEqual(
  mode.selectedModelKeys({ zingOnly: false }, requested),
  requested,
  "explicit false must preserve the existing model selection",
);
assert.deepEqual(
  mode.selectedModelKeys({ zingOnly: true }, requested),
  ["minwm"],
  "Zing-only mode must fail closed to the MinWM backend",
);
assert.deepEqual(mode.recordingVariants({}), ["comparison", "zing"]);
assert.deepEqual(mode.recordingVariants({ zingOnly: false }), ["comparison", "zing"]);
assert.deepEqual(mode.recordingVariants({ zingOnly: true }), ["zing"]);

function fakeElement(dataset = {}) {
  return {
    dataset: { ...dataset },
    hidden: false,
    textContent: "original",
    attributes: {},
    setAttribute(name, value) {
      this.attributes[name] = String(value);
    },
  };
}

const hidden = fakeElement();
const copy = fakeElement({ zingOnlyCopy: "Zing copy" });
const aria = fakeElement({ zingOnlyAriaLabel: "Zing label" });
const title = fakeElement({ zingOnlyTitle: "Zing title" });
const minwm = fakeElement({ modelKey: "minwm" });
const lingbot2 = fakeElement({ modelKey: "lingbot2" });
const happyoyster = fakeElement({ modelKey: "happyoyster" });
const elements = new Map([
  ["[data-zing-only-hide]", [hidden]],
  ["[data-zing-only-copy]", [copy]],
  ["[data-zing-only-aria-label]", [aria]],
  ["[data-zing-only-title]", [title]],
  ["[data-model-key]", [minwm, lingbot2, happyoyster]],
]);
const documentRef = {
  documentElement: { dataset: {} },
  title: "World Studio · 实时模型对比",
  querySelectorAll(selector) {
    return elements.get(selector) || [];
  },
};

assert.equal(mode.applyToDocument(documentRef, { zingOnly: true }), true);
assert.equal(documentRef.documentElement.dataset.realtimeExperience, "zing-only");
assert.equal(documentRef.documentElement.dataset.realtimeExperienceReady, "true");
assert.equal(documentRef.title, "World Studio · Zing 实时世界");
assert.equal(hidden.hidden, true);
assert.equal(hidden.attributes["aria-hidden"], "true");
assert.equal(copy.textContent, "Zing copy");
assert.equal(aria.attributes["aria-label"], "Zing label");
assert.equal(title.attributes.title, "Zing title");
assert.equal(minwm.hidden, false, "the Zing player must remain visible");
assert.equal(lingbot2.hidden, true);
assert.equal(happyoyster.hidden, true);

const untouched = {
  documentElement: { dataset: {} },
  title: "dual",
  querySelectorAll() {
    throw new Error("default mode must not mutate the DOM");
  },
};
assert.equal(mode.applyToDocument(untouched, {}), false);
assert.deepEqual(untouched.documentElement.dataset, {});
assert.equal(untouched.title, "dual");

console.log("realtime experience mode tests passed");
