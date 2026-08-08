const assert = require("assert");

const { DualModelController } = require("./dual_model_controller.js");

class FakeSession {
  constructor(key, { failConnect = false } = {}) {
    this.key = key;
    this.failConnect = failConnect;
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
    return true;
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
  assert.equal(firstEvent, 1);
  assert.equal(secondEvent, 2);
  assert.deepEqual(minwm.events, lingbot2.events, "both sessions must receive identical envelopes");
  assert.equal(minwm.events[0].event_id, 1);
  assert.equal(minwm.events[0].client_sent_perf_ms, 1234);

  const failingMinwm = new FakeSession("minwm");
  const failingLingbot = new FakeSession("lingbot2", { failConnect: true });
  const failingController = new DualModelController({
    sessions: { minwm: failingMinwm, lingbot2: failingLingbot },
    backends: {
      minwm: { model: "minwm", wsUrl: "/minwm" },
      lingbot2: { model: "lingbot2", wsUrl: "/lingbot2" },
    },
  });
  await assert.rejects(() => failingController.connect({ trace_id: "failed" }), /lingbot2 unavailable/);
  assert.equal(failingMinwm.closeCalls.length, 1, "successful peer is closed after partial startup");

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
