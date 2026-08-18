const assert = require("assert");

const {
  DEFAULT_GOAL_MIN_PLAY_SECONDS,
  DEFAULT_SHARED_SKILL_COOLDOWN_MS,
  GOAL_ACHIEVEMENT_DELAY_MS,
  WorldRulesController,
  normalizeWorldRulesDraft,
} = require("./world_rules_controller.js");

async function main() {
  assert.deepEqual(normalizeWorldRulesDraft({}), { skills: [], goals: [] });
  assert.deepEqual(
    normalizeWorldRulesDraft({ goal: { probability: "0.3" } }),
    { skills: [], goals: [] },
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
  assert.throws(
    () => normalizeWorldRulesDraft({
      goals: Array.from({ length: 10 }, (_, index) => ({ input: `目标 ${index + 1}` })),
    }),
    /最多可以配置 9 个目标/,
  );
  assert.equal(
    normalizeWorldRulesDraft({ goal: { input: "出现徽章" } }).goals[0].min_play_seconds,
    DEFAULT_GOAL_MIN_PLAY_SECONDS,
    "legacy saved goals should receive the default minimum play time",
  );
  assert.deepEqual(
    normalizeWorldRulesDraft({ skills: [{ id: "fly", name: "召唤飞船" }] }),
    { skills: [{ id: "fly", input: "召唤飞船" }], goals: [] },
    "a legacy label-only skill should migrate into the single rule input",
  );

  const completionCalls = [];
  const dispatches = [];
  const timers = [];
  const achievements = [];
  const goalResults = [];
  let nowMs = 1000;
  const controller = new WorldRulesController({
    completeRule: async (request) => {
      completionCalls.push(request);
      return {
        name: request.kind === "goal"
          ? request.input.includes("传送门") ? "传送门" : "星光徽章"
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
    now: () => nowMs,
    onAchievement: (goal) => achievements.push(goal.name),
    onGoalResult: (result, goal) => goalResults.push({ result, goal }),
  });

  const prepared = await controller.prepare({
    skills: [
      { id: "fly", input: "召唤一艘飞船" },
      { id: "snow", input: "让天气变成暴雪" },
    ],
    goals: [
      { id: "badge", min_play_seconds: 12, probability: 0.5, input: "出现一枚星光徽章" },
      { id: "portal", min_play_seconds: 4, probability: 0.5, input: "打开一扇传送门" },
    ],
  }, "A rider explores a valley.");
  assert.equal(completionCalls.length, 4);
  assert.ok(completionCalls.every((request) => request.previous_prompt === "A rider explores a valley."));
  assert.deepEqual(completionCalls.map((request) => request.kind), ["skill", "skill", "goal", "goal"]);
  assert.equal(prepared.skills[0].name, "召唤飞船");
  assert.equal(prepared.goals[0].name, "星光徽章");
  assert.equal(prepared.goals[0].min_play_seconds, 12);
  assert.equal(prepared.goals[1].name, "传送门");
  assert.equal(prepared.skills[0].prepared.change_type, "one_time");
  assert.equal(prepared.skills[1].prepared.change_type, "persistent");

  controller.activate(prepared);
  const skillResult = await controller.triggerSkill("fly");
  assert.equal(skillResult.event_id, 1);
  assert.equal(dispatches[0].metadata.trigger, "skill");
  assert.equal(dispatches[0].metadata.skillName, "召唤飞船");
  assert.equal(dispatches.length, 1, "skills must not roll the goal probability");
  assert.equal(timers[0].delay, DEFAULT_SHARED_SKILL_COOLDOWN_MS);
  assert.equal(controller.snapshot().skillCooldownRemainingMs, 10000);
  const blockedSkillResult = await controller.triggerSkill("snow");
  assert.equal(blockedSkillResult.ignored, true);
  assert.equal(blockedSkillResult.reason, "shared_cooldown");
  assert.equal(dispatches.length, 1, "all skills should share the same cooldown");
  nowMs += DEFAULT_SHARED_SKILL_COOLDOWN_MS;
  timers[0].callback();
  assert.equal(controller.snapshot().skillCooldownActive, false);

  controller.startSession();
  assert.equal(timers[1].delay, 12000, "goal timing starts when gameplay becomes visible");
  assert.equal(timers[2].delay, 4000, "each goal should keep its own timed trigger");
  assert.equal(dispatches.length, 1, "the goal prompt must wait for its configured play time");
  await timers[2].callback();
  assert.equal(dispatches[1].metadata.rule, "goal_time_probability");
  assert.equal(dispatches[1].metadata.source, "elapsed_time");
  assert.equal(dispatches[1].metadata.goalId, "portal");
  assert.equal(dispatches[1].metadata.minPlaySeconds, 4);
  assert.equal(dispatches[1].metadata.goalName, "传送门");
  assert.equal(goalResults[0].result.triggered, true);
  assert.equal(timers[3].delay, GOAL_ACHIEVEMENT_DELAY_MS);
  assert.deepEqual(achievements, []);
  timers[3].callback();
  assert.deepEqual(achievements, ["传送门"]);
  await timers[1].callback();
  assert.equal(dispatches[2].metadata.goalId, "badge");
  assert.equal(dispatches[2].metadata.goalName, "星光徽章");
  assert.equal(goalResults[1].result.triggered, true);
  assert.equal(timers[4].delay, GOAL_ACHIEVEMENT_DELAY_MS);
  timers[4].callback();
  assert.deepEqual(achievements, ["传送门", "星光徽章"]);
  await timers[1].callback();
  assert.equal(dispatches.length, 3, "each goal may only trigger once per world session");
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
    goals: [{ id: "chest", min_play_seconds: 3, probability: 0.2, input: "出现隐藏宝箱" }],
  }, "A traveler explores a forest.");
  missedController.activate(missedPrepared);
  missedController.startSession();
  assert.equal(missedTimers[0].delay, 3000);
  await missedTimers[0].callback();
  assert.equal(missedDispatches.length, 0, "a missed roll must not send the goal prompt");
  assert.equal(missedResults[0].triggered, false);
  assert.equal(missedController.snapshot().goalAttempted, true);
  assert.equal(missedController.snapshot().goals[0].attempted, true);
  missedController.startSession();
  assert.equal(missedTimers.length, 1, "the timed probability is evaluated only once per world");

  console.log("world rules controller ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
