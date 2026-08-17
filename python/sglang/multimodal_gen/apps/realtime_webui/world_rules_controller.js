(function (global) {
  const MAX_WORLD_SKILLS = 9;
  const DEFAULT_GOAL_PROBABILITY = 0.2;
  const GOAL_ACHIEVEMENT_DELAY_MS = 5000;

  function normalizedText(value) {
    return String(value || "").trim();
  }

  function normalizeProbability(value) {
    if (value === "" || value == null) return DEFAULT_GOAL_PROBABILITY;
    const probability = Number(value);
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
      throw new Error("目标触发概率必须是 0–1 之间的数字");
    }
    return probability;
  }

  function normalizeWorldRulesDraft(draft = {}) {
    const skills = (Array.isArray(draft.skills) ? draft.skills : [])
      .map((skill, index) => ({
        id: normalizedText(skill?.id) || `skill-${index + 1}`,
        input: normalizedText(skill?.input)
          || normalizedText(skill?.instruction)
          || normalizedText(skill?.name),
      }))
      .filter((skill) => skill.input);
    if (skills.length > MAX_WORLD_SKILLS) {
      throw new Error(`最多可以配置 ${MAX_WORLD_SKILLS} 个技能`);
    }

    const rawGoal = draft.goal || {};
    const goalInput = normalizedText(rawGoal.input)
      || normalizedText(rawGoal.instruction)
      || normalizedText(rawGoal.name);
    let goal = null;
    if (goalInput) {
      goal = {
        probability: normalizeProbability(rawGoal.probability),
        input: goalInput,
      };
    }
    return { skills, goal };
  }

  function normalizePreparedResult(result) {
    const prompt = normalizedText(result?.prompt);
    const changeType = normalizedText(result?.change_type);
    if (!prompt) throw new Error("Prompt 润色结果为空");
    if (changeType !== "persistent" && changeType !== "one_time") {
      throw new Error("Prompt 润色结果缺少有效的持续类型");
    }
    return { prompt, change_type: changeType };
  }

  function normalizeCompletedRule(result) {
    const name = normalizedText(result?.name);
    if (!name) throw new Error("规则补全结果缺少按键名称或奖励名称");
    return {
      name: name.slice(0, 28),
      prepared: normalizePreparedResult(result),
    };
  }

  class WorldRulesController {
    constructor({
      completeRule,
      dispatchPrepared,
      random = Math.random,
      setTimer = global.setTimeout.bind(global),
      clearTimer = global.clearTimeout.bind(global),
      achievementDelayMs = GOAL_ACHIEVEMENT_DELAY_MS,
      onAchievement = () => {},
      onStateChange = () => {},
    }) {
      this.completeRule = completeRule;
      this.dispatchPrepared = dispatchPrepared;
      this.random = random;
      this.setTimer = setTimer;
      this.clearTimer = clearTimer;
      this.achievementDelayMs = achievementDelayMs;
      this.onAchievement = onAchievement;
      this.onStateChange = onStateChange;
      this.activeRules = { skills: [], goal: null };
      this.goalTriggered = false;
      this.goalPending = false;
      this.achievementTimer = null;
      this.pendingSkillIds = new Set();
    }

    async prepare(draft, previousPrompt) {
      const normalized = normalizeWorldRulesDraft(draft);
      const basePrompt = normalizedText(previousPrompt);
      if ((normalized.skills.length || normalized.goal) && !basePrompt) {
        throw new Error("世界描述为空，无法预先润色规则 Prompt");
      }
      const skillPromise = Promise.all(normalized.skills.map(async (skill) => ({
        ...skill,
        ...normalizeCompletedRule(await this.completeRule({
          kind: "skill",
          input: skill.input,
          previous_prompt: basePrompt,
        })),
        instruction: skill.input,
      })));
      const goalPromise = normalized.goal
        ? this.completeRule({
          kind: "goal",
          input: normalized.goal.input,
          previous_prompt: basePrompt,
        }).then((result) => ({
          ...normalized.goal,
          ...normalizeCompletedRule(result),
          instruction: normalized.goal.input,
        }))
        : Promise.resolve(null);
      const [skills, goal] = await Promise.all([skillPromise, goalPromise]);
      return { skills, goal };
    }

    activate(preparedRules = {}) {
      this.endSession();
      this.activeRules = {
        skills: Array.isArray(preparedRules.skills) ? preparedRules.skills : [],
        goal: preparedRules.goal || null,
      };
      this.onStateChange(this.snapshot());
      return this.snapshot();
    }

    endSession() {
      if (this.achievementTimer !== null) this.clearTimer(this.achievementTimer);
      this.achievementTimer = null;
      this.goalTriggered = false;
      this.goalPending = false;
      this.pendingSkillIds.clear();
      this.activeRules = { skills: [], goal: null };
      this.onStateChange(this.snapshot());
    }

    async triggerSkill(skillId) {
      const skill = this.activeRules.skills.find((item) => item.id === skillId);
      if (!skill) throw new Error("这个技能不在当前世界中");
      if (this.pendingSkillIds.has(skill.id)) return { ignored: true };
      this.pendingSkillIds.add(skill.id);
      this.onStateChange(this.snapshot());
      try {
        const result = await this.dispatchPrepared(skill.prepared, {
          trigger: "skill",
          skillId: skill.id,
          skillName: skill.name,
          instruction: skill.instruction,
        });
        if (result?.ignored) return result;
        try {
          const goal = await this.maybeTriggerGoal("skill");
          return { ...result, goal };
        } catch (goalError) {
          return { ...result, goalError };
        }
      } finally {
        this.pendingSkillIds.delete(skill.id);
        this.onStateChange(this.snapshot());
      }
    }

    async noteUserPromptSuccess() {
      return this.maybeTriggerGoal("live_direction");
    }

    async maybeTriggerGoal(source = "unknown") {
      const goal = this.activeRules.goal;
      if (!goal || this.goalTriggered || this.goalPending) return { triggered: false };
      const roll = Number(this.random());
      if (goal.probability <= 0 || roll >= goal.probability) {
        return { triggered: false, roll };
      }
      this.goalPending = true;
      this.onStateChange(this.snapshot());
      try {
        const result = await this.dispatchPrepared(goal.prepared, {
          trigger: "rule",
          rule: "goal_probability",
          goalName: goal.name,
          probability: goal.probability,
          source,
          instruction: goal.instruction,
        });
        if (result?.ignored) return { triggered: false, ignored: true, roll };
        this.goalTriggered = true;
        this.achievementTimer = this.setTimer(() => {
          this.achievementTimer = null;
          this.onAchievement(goal);
        }, this.achievementDelayMs);
        return { triggered: true, roll, result };
      } finally {
        this.goalPending = false;
        this.onStateChange(this.snapshot());
      }
    }

    snapshot() {
      return {
        skills: this.activeRules.skills.map((skill) => ({
          ...skill,
          pending: this.pendingSkillIds.has(skill.id),
        })),
        goal: this.activeRules.goal,
        goalTriggered: this.goalTriggered,
        goalPending: this.goalPending,
      };
    }
  }

  global.WorldRulesController = WorldRulesController;
  global.normalizeWorldRulesDraft = normalizeWorldRulesDraft;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      DEFAULT_GOAL_PROBABILITY,
      GOAL_ACHIEVEMENT_DELAY_MS,
      MAX_WORLD_SKILLS,
      WorldRulesController,
      normalizeWorldRulesDraft,
    };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
