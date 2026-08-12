const assert = require("assert");

const { DualModelController } = require("./dual_model_controller.js");

class FakeSession {
  constructor(key, { failConnect = false, sendOk = true } = {}) {
    this.key = key;
    this.failConnect = failConnect;
    this.sendOk = sendOk;
    this.connectCalls = [];
    this.events = [];
    this.closeCalls = [];
  }

  async connect(init, url) {
    this.connectCalls.push({ init, url });
    if (this.failConnect) throw new Error(`${this.key} unavailable`);
  }

  sendEvent(event) {
    this.events.push(event);
    return this.sendOk;
  }

  close(reason) {
    this.closeCalls.push(reason);
  }
}

async function main() {
  const minwm = new FakeSession("minwm");
  const lingbot2 = new FakeSession("lingbot2");
  const controller = new DualModelController({
    sessions: { minwm, lingbot2 },
    backends: {
      minwm: { model: "minwm-model", wsUrl: "/backends/minwm/generate" },
      lingbot2: {
        model: "lingbot2-model",
        wsUrl: "/backends/lingbot2/generate",
        transformInit: (init) => ({
          ...init,
          size: init.size === "1280x704" ? "1280x720" : init.size,
          realtime_causal_sink_size: 3,
          realtime_causal_kv_cache_num_frames: 12,
        }),
      },
    },
    now: () => 1234,
  });

  await controller.connect({
    prompt: "shared prompt",
    size: "1280x704",
    realtime_causal_sink_size: 9,
    realtime_causal_kv_cache_num_frames: 18,
    trace_id: "trace",
  });
  assert.equal(minwm.connectCalls.length, 1);
  assert.equal(lingbot2.connectCalls.length, 1);
  assert.equal(minwm.connectCalls[0].init.model, "minwm-model");
  assert.equal(lingbot2.connectCalls[0].init.model, "lingbot2-model");
  assert.equal(minwm.connectCalls[0].init.prompt, "shared prompt");
  assert.equal(minwm.connectCalls[0].init.size, "1280x704");
  assert.equal(minwm.connectCalls[0].init.realtime_causal_sink_size, 9);
  assert.equal(minwm.connectCalls[0].init.realtime_causal_kv_cache_num_frames, 18);
  assert.equal(lingbot2.connectCalls[0].init.size, "1280x720");
  assert.equal(lingbot2.connectCalls[0].init.realtime_causal_sink_size, 3);
  assert.equal(lingbot2.connectCalls[0].init.realtime_causal_kv_cache_num_frames, 12);
  assert.equal(minwm.connectCalls[0].url, "/backends/minwm/generate");
  assert.equal(lingbot2.connectCalls[0].url, "/backends/lingbot2/generate");
  assert.notEqual(
    minwm.connectCalls[0].init.trace_id,
    lingbot2.connectCalls[0].init.trace_id,
    "each backend should retain an independently queryable trace",
  );

  const firstEvent = controller.sendEvent("camera_actions", { transitions: [{ actions: ["w"] }] });
  const secondEvent = controller.sendEvent("prompt", "updated prompt");
  assert.equal(firstEvent.eventId, 1);
  assert.deepEqual(firstEvent.sent, { minwm: true, lingbot2: true });
  assert.equal(secondEvent.eventId, 2);
  assert.deepEqual(secondEvent.sent, { minwm: true, lingbot2: true });
  assert.deepEqual(minwm.events, lingbot2.events, "both sessions must receive identical envelopes");
  assert.equal(minwm.events[0].event_id, 1);
  assert.equal(minwm.events[0].client_sent_perf_ms, 1234);

  await controller.reconnect("lingbot2");
  assert.equal(lingbot2.connectCalls.length, 2, "only the failed peer should reconnect");
  assert.equal(minwm.connectCalls.length, 1, "recovery must not restart the healthy peer");
  assert.match(lingbot2.connectCalls[1].init.trace_id, /:lingbot2:retry1$/);
  assert.deepEqual(
    lingbot2.events.slice(-2).map((event) => event.event_id),
    [1, 2],
    "recovery should replay the latest action and prompt state in event order",
  );

  const failingMinwm = new FakeSession("minwm");
  const failingLingbot = new FakeSession("lingbot2", { failConnect: true });
  const failingController = new DualModelController({
    sessions: { minwm: failingMinwm, lingbot2: failingLingbot },
    backends: {
      minwm: { model: "minwm", wsUrl: "/minwm" },
      lingbot2: { model: "lingbot2", wsUrl: "/lingbot2" },
    },
  });
  const partialReport = await failingController.connect({ trace_id: "failed" });
  assert.deepEqual(partialReport.connected, ["minwm"]);
  assert.deepEqual(partialReport.failed.map((item) => item.key), ["lingbot2"]);
  assert.equal(failingMinwm.closeCalls.length, 0, "healthy peer remains live after partial startup");
  const partialEvent = failingController.sendEvent("prompt", "minwm only");
  assert.deepEqual(partialEvent.sent, { minwm: true });
  assert.equal(failingMinwm.events.length, 1, "healthy peer keeps receiving events");

  const allFailedController = new DualModelController({
    sessions: {
      minwm: new FakeSession("minwm", { failConnect: true }),
      lingbot2: new FakeSession("lingbot2", { failConnect: true }),
    },
    backends: {
      minwm: { model: "minwm", wsUrl: "/minwm" },
      lingbot2: { model: "lingbot2", wsUrl: "/lingbot2" },
    },
  });
  await assert.rejects(
    () => allFailedController.connect({ trace_id: "all-failed" }),
    /no realtime model connected/,
  );

  const t2vMinwm = new FakeSession("minwm");
  const t2vLingbot = new FakeSession("lingbot2", { failConnect: true });
  const t2vController = new DualModelController({
    sessions: { minwm: t2vMinwm, lingbot2: t2vLingbot },
    backends: {
      minwm: { model: "minwm", wsUrl: "/minwm" },
      lingbot2: {
        model: "lingbot2",
        wsUrl: "/lingbot2",
        enabled: (init) => init.generation_mode !== "t2v",
      },
    },
    now: () => 5678,
  });
  await t2vController.connect({ trace_id: "t2v", generation_mode: "t2v" });
  assert.equal(t2vMinwm.connectCalls.length, 1, "MinWM should serve T2V");
  assert.equal(t2vLingbot.connectCalls.length, 0, "LingBot2 should stay disconnected for T2V");
  assert.equal(t2vLingbot.closeCalls.length, 1, "disabled peer should clear stale state");
  t2vController.sendEvent("camera_actions", { transitions: [{ actions: ["w"] }] });
  assert.equal(t2vMinwm.events.length, 1, "active T2V backend should receive shared input");
  assert.equal(t2vLingbot.events.length, 0, "disabled T2V backend should not receive input");

  console.log("dual model controller ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
