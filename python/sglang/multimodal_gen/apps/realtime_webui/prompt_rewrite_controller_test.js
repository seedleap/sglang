const assert = require("assert");

const { PromptRewriteController } = require("./prompt_rewrite_controller.js");

async function main() {
  const rewriteCalls = [];
  const sends = [];
  const timers = [];
  const cleared = [];
  const controller = new PromptRewriteController({
    rewrite: async (request) => {
      rewriteCalls.push(request);
      return request.instruction === "下雪"
        ? { prompt: "persistent snow world", change_type: "persistent" }
        : { prompt: "one-time jump", change_type: "one_time" };
    },
    sendPrompt: (prompt, metadata) => {
      sends.push({ prompt, metadata });
      return sends.length;
    },
    setTimer: (callback, delay) => {
      const timer = { callback, delay };
      timers.push(timer);
      return timer;
    },
    clearTimer: (timer) => cleared.push(timer),
  });

  controller.beginSession("initial world description");
  const persistent = await controller.submit("下雪");
  assert.equal(persistent.change_type, "persistent");
  assert.deepEqual(rewriteCalls[0], {
    instruction: "下雪",
    previous_prompt: "initial world description",
  });
  assert.equal(sends[0].prompt, "persistent snow world");
  assert.equal(sends[0].metadata.trigger, "user");
  assert.equal(sends[0].metadata.instruction, "下雪");

  const preparedSends = [];
  const preparedTimers = [];
  const preparedController = new PromptRewriteController({
    rewrite: async () => {
      throw new Error("prepared prompts must not call the rewriter");
    },
    sendPrompt: (prompt, metadata) => {
      preparedSends.push({ prompt, metadata });
      return preparedSends.length;
    },
    setTimer: (callback, delay) => {
      const timer = { callback, delay };
      preparedTimers.push(timer);
      return timer;
    },
    clearTimer: () => {},
  });
  preparedController.beginSession("prepared baseline");
  const prepared = preparedController.submitPrepared(
    { prompt: "prepared skill action", change_type: "one_time" },
    "召唤飞船",
    { trigger: "skill", skillName: "召唤飞船" },
  );
  assert.equal(prepared.change_type, "one_time");
  assert.equal(preparedSends[0].prompt, "prepared skill action");
  assert.equal(preparedSends[0].metadata.phase, "prepared");
  assert.equal(preparedSends[0].metadata.trigger, "skill");
  assert.equal(preparedSends[0].metadata.skillName, "召唤飞船");
  assert.equal(preparedTimers[0].delay, 10000);
  preparedTimers[0].callback();
  assert.equal(preparedSends[1].prompt, "prepared baseline");

  const oneTime = await controller.submit("跳一下");
  assert.equal(oneTime.change_type, "one_time");
  assert.deepEqual(rewriteCalls[1], {
    instruction: "跳一下",
    previous_prompt: "persistent snow world",
  });
  assert.equal(sends[1].prompt, "one-time jump");
  assert.equal(timers[0].delay, 10000);
  timers[0].callback();
  assert.equal(sends[2].prompt, "persistent snow world");
  assert.equal(sends[2].metadata.phase, "restore");
  assert.equal(sends[2].metadata.trigger, "rule");
  assert.equal(sends[2].metadata.rule, "one_time_timeout_restore");
  assert.equal(sends[2].metadata.afterMs, 10000);

  await controller.submit("再跳一下");
  const staleTimer = timers[1];
  await controller.submit("下雪");
  assert.ok(cleared.includes(staleTimer), "a new direction cancels the old restore");
  staleTimer.callback();
  assert.equal(
    sends.filter((entry) => entry.metadata.phase === "restore").length,
    1,
    "a stale timer must not restore an older prompt",
  );

  const failingSends = [];
  const failingTimers = [];
  const failingController = new PromptRewriteController({
    rewrite: async ({ instruction }) => {
      if (instruction === "失败") throw new Error("rewrite failed");
      return { prompt: "temporary action", change_type: "one_time" };
    },
    sendPrompt: (prompt, metadata) => {
      failingSends.push({ prompt, metadata });
      return failingSends.length;
    },
    setTimer: (callback) => {
      const timer = { callback };
      failingTimers.push(timer);
      return timer;
    },
    clearTimer: () => {},
  });
  failingController.beginSession("failure baseline");
  await failingController.submit("动作");
  await assert.rejects(() => failingController.submit("失败"), /rewrite failed/);
  failingTimers[0].callback();
  assert.equal(failingSends.at(-1).prompt, "failure baseline");
  assert.equal(failingSends.at(-1).metadata.phase, "restore");

  await controller.submit("再跳一下");
  const endedTimer = timers[2];
  controller.endSession();
  endedTimer.callback();
  assert.equal(
    sends.filter((entry) => entry.metadata.phase === "restore").length,
    1,
    "closing a session cancels pending restoration",
  );

  console.log("prompt rewrite controller ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
