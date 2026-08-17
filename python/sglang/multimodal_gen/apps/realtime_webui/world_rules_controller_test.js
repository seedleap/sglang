const assert = require("assert");

const {
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

  const rewriteCalls = [];
  const dispatches = [];
  const timers = [];
  const achievements = [];
  const controller = new WorldRulesController({
    rewrite: async (request) => {
      rewriteCalls.push(request);
      return {
        prompt: `prepared:${request.instruction}`,
        change_type: request.instruction.includes("天气") ? "persistent" : "one_time",
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
  });

  const prepared = await controller.prepare({
    skills: [
      { id: "fly", name: "召唤飞船", instruction: "召唤一艘飞船" },
      { id: "snow", name: "暴雪", instruction: "让天气变成暴雪" },
    ],
    goal: { name: "星光徽章", probability: 0.5, instruction: "出现一枚星光徽章" },
  }, "A rider explores a valley.");
  assert.equal(rewriteCalls.length, 3);
  assert.ok(rewriteCalls.every((request) => request.previous_prompt === "A rider explores a valley."));
  assert.equal(prepared.skills[0].prepared.change_type, "one_time");
  assert.equal(prepared.skills[1].prepared.change_type, "persistent");

  controller.activate(prepared);
  const skillResult = await controller.triggerSkill("fly");
  assert.equal(skillResult.event_id, 1);
  assert.equal(dispatches[0].metadata.trigger, "skill");
  assert.equal(dispatches[0].metadata.skillName, "召唤飞船");
  assert.equal(dispatches[1].metadata.rule, "goal_probability");
  assert.equal(dispatches[1].metadata.goalName, "星光徽章");
  assert.equal(skillResult.goal.triggered, true);
  assert.equal(timers[0].delay, GOAL_ACHIEVEMENT_DELAY_MS);
  assert.deepEqual(achievements, []);
  timers[0].callback();
  assert.deepEqual(achievements, ["星光徽章"]);

  await controller.noteUserPromptSuccess();
  assert.equal(dispatches.length, 2, "one goal may only trigger once per world session");
  controller.endSession();

  console.log("world rules controller ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
