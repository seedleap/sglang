const assert = require("assert");

const {
  DEFAULT_GOAL_MIN_PLAY_SECONDS,
  GOAL_ACHIEVEMENT_DELAY_MS,
  WorldRulesController,
  normalizeWorldRulesDraft,
} = require("./world_rules_controller.js");

async function main() {
  assert.deepEqual(normalizeWorldRulesDraft({}), { skills: [], goal: null });
  assert.deepEqual(
    normalizeWorldRulesDraft({ goal: { probability: "0.3" } }),
    { skills: [], goal: null },
    "a probability alone must not make optional rules required",
  );
  assert.throws(
    () => normalizeWorldRulesDraft({ goal: { name: "徽章", instruction: "出现徽章", probability: 1.2 } }),
    /0–1/,
  );
  assert.throws(
    () => normalizeWorldRulesDraft({ goal: { input: "出现徽章", min_play_seconds: -1 } }),
    /最早触发时间/,
  );
  assert.equal(
    normalizeWorldRulesDraft({ goal: { input: "出现徽章" } }).goal.min_play_seconds,
    DEFAULT_GOAL_MIN_PLAY_SECONDS,
    "legacy saved goals should receive the default minimum play time",
  );
  assert.deepEqual(
    normalizeWorldRulesDraft({ skills: [{ id: "fly", name: "召唤飞船" }] }),
    { skills: [{ id: "fly", input: "召唤飞船" }], goal: null },
    "a legacy label-only skill should migrate into the single rule input",
  );

  const completionCalls = [];
  const dispatches = [];
  const timers = [];
  const achievements = [];
  const goalResults = [];
  const controller = new WorldRulesController({
    completeRule: async (request) => {
      completionCalls.push(request);
      return {
        name: request.kind === "goal"
          ? "星光徽章"
          : request.input.includes("天气") ? "暴雪" : "召唤飞船",
        prompt: `prepared:${request.input}`,
        change_type: request.input.includes("天气") ? "persistent" : "one_time",
      };
    },
    dispatchPrepared: async (prepared, metadata) => {
      dispatches.push({ prepared, metadata });
      return { event_id: dispatches.length, change_type: prepared.change_type };
    },
    random: () => 0.1,
    setTimer: (callback, delay) => {
      const timer = { callback, delay };
      timers.push(timer);
      return timer;
    },
    clearTimer: () => {},
    onAchievement: (goal) => achievements.push(goal.name),
    onGoalResult: (result, goal) => goalResults.push({ result, goal }),
  });

  const prepared = await controller.prepare({
    skills: [
      { id: "fly", input: "召唤一艘飞船" },
      { id: "snow", input: "让天气变成暴雪" },
    ],
    goal: { min_play_seconds: 12, probability: 0.5, input: "出现一枚星光徽章" },
  }, "A rider explores a valley.");
  assert.equal(completionCalls.length, 3);
  assert.ok(completionCalls.every((request) => request.previous_prompt === "A rider explores a valley."));
  assert.deepEqual(completionCalls.map((request) => request.kind), ["skill", "skill", "goal"]);
  assert.equal(prepared.skills[0].name, "召唤飞船");
  assert.equal(prepared.goal.name, "星光徽章");
  assert.equal(prepared.goal.min_play_seconds, 12);
  assert.equal(prepared.skills[0].prepared.change_type, "one_time");
  assert.equal(prepared.skills[1].prepared.change_type, "persistent");

  controller.activate(prepared);
  const skillResult = await controller.triggerSkill("fly");
  assert.equal(skillResult.event_id, 1);
  assert.equal(dispatches[0].metadata.trigger, "skill");
  assert.equal(dispatches[0].metadata.skillName, "召唤飞船");
  assert.equal(dispatches.length, 1, "skills must not roll the goal probability");

  controller.startSession();
  assert.equal(timers[0].delay, 12000, "goal timing starts when gameplay becomes visible");
  assert.equal(dispatches.length, 1, "the goal prompt must wait for its configured play time");
  await timers[0].callback();
  assert.equal(dispatches[1].metadata.rule, "goal_time_probability");
  assert.equal(dispatches[1].metadata.source, "elapsed_time");
  assert.equal(dispatches[1].metadata.minPlaySeconds, 12);
  assert.equal(dispatches[1].metadata.goalName, "星光徽章");
  assert.equal(goalResults[0].result.triggered, true);
  assert.equal(timers[1].delay, GOAL_ACHIEVEMENT_DELAY_MS);
  assert.deepEqual(achievements, []);
  timers[1].callback();
  assert.deepEqual(achievements, ["星光徽章"]);
  assert.equal(dispatches.length, 2, "one goal may only trigger once per world session");
  controller.endSession();

  const missedTimers = [];
  const missedDispatches = [];
  const missedResults = [];
  const missedController = new WorldRulesController({
    completeRule: async (request) => ({
      name: "隐藏宝箱",
      prompt: `prepared:${request.input}`,
      change_type: "one_time",
    }),
    dispatchPrepared: async (...args) => {
      missedDispatches.push(args);
      return { event_id: 1, change_type: "one_time" };
    },
    random: () => 0.9,
    setTimer: (callback, delay) => {
      const timer = { callback, delay };
      missedTimers.push(timer);
      return timer;
    },
    clearTimer: () => {},
    onGoalResult: (result) => missedResults.push(result),
  });
  const missedPrepared = await missedController.prepare({
    goal: { min_play_seconds: 3, probability: 0.2, input: "出现隐藏宝箱" },
  }, "A traveler explores a forest.");
  missedController.activate(missedPrepared);
  missedController.startSession();
  assert.equal(missedTimers[0].delay, 3000);
  await missedTimers[0].callback();
  assert.equal(missedDispatches.length, 0, "a missed roll must not send the goal prompt");
  assert.equal(missedResults[0].triggered, false);
  assert.equal(missedController.snapshot().goalAttempted, true);
  missedController.startSession();
  assert.equal(missedTimers.length, 1, "the timed probability is evaluated only once per world");

  console.log("world rules controller ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
