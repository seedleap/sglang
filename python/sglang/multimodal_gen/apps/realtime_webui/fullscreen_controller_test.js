const assert = require("assert");

const { createFullscreenController } = require("./fullscreen_controller.js");

function createEventTarget() {
  const listeners = new Map();
  return {
    addEventListener(type, listener) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(listener);
    },
    removeEventListener(type, listener) {
      listeners.get(type)?.delete(listener);
    },
    emit(type) {
      for (const listener of listeners.get(type) || []) listener();
    },
    listenerCount(type) {
      return listeners.get(type)?.size || 0;
    },
  };
}

function createFixture() {
  const documentEvents = createEventTarget();
  const buttonEvents = createEventTarget();
  const documentRef = {
    ...documentEvents,
    fullscreenElement: null,
    exitCount: 0,
    async exitFullscreen() {
      this.exitCount += 1;
    },
  };
  const target = {
    requestCount: 0,
    async requestFullscreen() {
      this.requestCount += 1;
    },
  };
  const button = {
    ...buttonEvents,
    attributes: {},
    title: "",
    setAttribute(name, value) {
      this.attributes[name] = value;
    },
  };
  return { button, documentRef, target };
}

async function run() {
  const fixture = createFixture();
  const controller = createFullscreenController(fixture);

  assert.equal(fixture.documentRef.listenerCount("fullscreenchange"), 1);
  assert.equal(fixture.button.listenerCount("click"), 1);
  assert.equal(fixture.button.attributes["aria-pressed"], "false");
  assert.equal(fixture.button.title, "Enter fullscreen comparison");
  assert.equal(
    fixture.button.attributes["aria-label"],
    "Enter fullscreen comparison",
  );

  await controller.toggle();
  assert.equal(fixture.target.requestCount, 1);

  fixture.documentRef.fullscreenElement = fixture.target;
  fixture.documentRef.emit("fullscreenchange");
  assert.equal(fixture.button.attributes["aria-pressed"], "true");
  assert.equal(fixture.button.title, "Exit fullscreen comparison");

  await controller.toggle();
  assert.equal(fixture.documentRef.exitCount, 1);

  fixture.documentRef.fullscreenElement = null;
  fixture.documentRef.emit("fullscreenchange");
  assert.equal(fixture.button.attributes["aria-pressed"], "false");

  controller.destroy();
  assert.equal(fixture.documentRef.listenerCount("fullscreenchange"), 0);
  assert.equal(fixture.button.listenerCount("click"), 0);

  const rejected = createFixture();
  const errors = [];
  rejected.target.requestFullscreen = async () => {
    throw new Error("permission denied");
  };
  const rejectedController = createFullscreenController({
    ...rejected,
    onError: (error) => errors.push(error.message),
  });
  await rejectedController.toggle();
  assert.deepEqual(errors, ["permission denied"]);
  rejectedController.destroy();

  console.log("fullscreen controller ok");
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
