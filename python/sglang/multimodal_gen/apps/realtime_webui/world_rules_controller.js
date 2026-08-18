(function (global) {
  const MAX_WORLD_SKILLS = 9;
  const MAX_WORLD_GOALS = 9;
  const DEFAULT_GOAL_PROBABILITY = 0.2;
  const DEFAULT_GOAL_MIN_PLAY_SECONDS = 10;
  const MAX_GOAL_MIN_PLAY_SECONDS = 3600;
  const DEFAULT_SHARED_SKILL_COOLDOWN_MS = 10000;
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

  function normalizeMinPlaySeconds(value) {
    if (value === "" || value == null) return DEFAULT_GOAL_MIN_PLAY_SECONDS;
    const seconds = Number(value);
    if (
      !Number.isFinite(seconds)
      || seconds < 0
      || seconds > MAX_GOAL_MIN_PLAY_SECONDS
    ) {
      throw new Error(`目标最早触发时间必须是 0–${MAX_GOAL_MIN_PLAY_SECONDS} 秒之间的数字`);
    }
    return seconds;
  }

  function normalizeGoal(rawGoal, index) {
    const goalInput = normalizedText(rawGoal?.input)
      || normalizedText(rawGoal?.instruction)
      || normalizedText(rawGoal?.name);
    if (!goalInput) return null;
    return {
      id: normalizedText(rawGoal?.id) || `goal-${index + 1}`,
      probability: normalizeProbability(rawGoal?.probability),
      min_play_seconds: normalizeMinPlaySeconds(
        rawGoal?.min_play_seconds ?? rawGoal?.minPlaySeconds,
      ),
      input: goalInput,
    };
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

    const rawGoals = Array.isArray(draft.goals) ? draft.goals : (
      draft.goal ? [draft.goal] : []
    );
    const usedGoalIds = new Set();
    let nextGeneratedGoalId = 1;
    const reserveGeneratedGoalIdFloor = (id) => {
      const match = /^goal-(\d+)$/.exec(id);
      if (!match) return;
      nextGeneratedGoalId = Math.max(nextGeneratedGoalId, Number(match[1]) + 1);
    };
    const goals = rawGoals
      .map((goal, index) => normalizeGoal(goal, index))
      .filter(Boolean)
      .map((goal) => {
        let id = goal.id;
        if (usedGoalIds.has(id)) {
          do {
            id = `goal-${nextGeneratedGoalId}`;
            nextGeneratedGoalId += 1;
          } while (usedGoalIds.has(id));
        }
        reserveGeneratedGoalIdFloor(id);
        usedGoalIds.add(id);
        return { ...goal, id };
      });
    if (goals.length > MAX_WORLD_GOALS) {
      throw new Error(`最多可以配置 ${MAX_WORLD_GOALS} 个目标`);
    }
    return { skills, goals };
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
      now = () => Date.now(),
      skillCooldownMs = DEFAULT_SHARED_SKILL_COOLDOWN_MS,
      achievementDelayMs = GOAL_ACHIEVEMENT_DELAY_MS,
      onAchievement = () => {},
      onGoalResult = () => {},
      onGoalError = () => {},
      onStateChange = () => {},
    }) {
      this.completeRule = completeRule;
      this.dispatchPrepared = dispatchPrepared;
      this.random = random;
      this.setTimer = setTimer;
      this.clearTimer = clearTimer;
      this.now = now;
      this.skillCooldownMs = Math.max(0, Number(skillCooldownMs) || 0);
      this.achievementDelayMs = achievementDelayMs;
      this.onAchievement = onAchievement;
      this.onGoalResult = onGoalResult;
      this.onGoalError = onGoalError;
      this.onStateChange = onStateChange;
      this.activeRules = { skills: [], goals: [] };
      this.goalTriggeredIds = new Set();
      this.goalAttemptedIds = new Set();
      this.goalPendingIds = new Set();
      this.goalTriggerTimers = new Map();
      this.achievementTimers = new Map();
      this.skillCooldownTimer = null;
      this.skillCooldownDeadlineMs = 0;
      this.sessionGeneration = 0;
      this.sessionStarted = false;
      this.pendingSkillIds = new Set();
    }

    async prepare(draft, previousPrompt) {
      const normalized = normalizeWorldRulesDraft(draft);
      const basePrompt = normalizedText(previousPrompt);
      if ((normalized.skills.length || normalized.goals.length) && !basePrompt) {
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
      const goalPromise = Promise.all(normalized.goals.map(async (goal) => ({
        ...goal,
        ...normalizeCompletedRule(await this.completeRule({
          kind: "goal",
          input: goal.input,
          previous_prompt: basePrompt,
        })),
        instruction: goal.input,
      })));
      const [skills, goals] = await Promise.all([skillPromise, goalPromise]);
      return { skills, goals };
    }

    activate(preparedRules = {}) {
      this.endSession();
      const legacyGoal = preparedRules.goal ? [preparedRules.goal] : [];
      this.activeRules = {
        skills: Array.isArray(preparedRules.skills) ? preparedRules.skills : [],
        goals: Array.isArray(preparedRules.goals) ? preparedRules.goals : legacyGoal,
      };
      this.onStateChange(this.snapshot());
      return this.snapshot();
    }

    startSession() {
      this.clearGoalTimers();
      this.sessionStarted = true;
      const goals = this.activeRules.goals;
      if (!goals.length) {
        this.onStateChange(this.snapshot());
        return this.snapshot();
      }
      const generation = this.sessionGeneration;
      goals.forEach((goal) => {
        if (this.goalAttemptedIds.has(goal.id) || this.goalTriggeredIds.has(goal.id)) return;
        const delayMs = Math.max(0, Number(goal.min_play_seconds || 0) * 1000);
        const timer = this.setTimer(async () => {
          this.goalTriggerTimers.delete(goal.id);
          try {
            const result = await this.maybeTriggerGoal(goal.id, "elapsed_time", generation);
            if (generation === this.sessionGeneration && !result?.canceled) {
              this.onGoalResult(result, goal);
            }
          } catch (error) {
            if (generation === this.sessionGeneration) this.onGoalError(error, goal);
          }
        }, delayMs);
        this.goalTriggerTimers.set(goal.id, timer);
      });
      this.onStateChange(this.snapshot());
      return this.snapshot();
    }

    endSession() {
      this.sessionGeneration += 1;
      this.clearGoalTimers();
      this.achievementTimers.forEach((timer) => this.clearTimer(timer));
      if (this.skillCooldownTimer !== null) this.clearTimer(this.skillCooldownTimer);
      this.achievementTimers.clear();
      this.skillCooldownTimer = null;
      this.skillCooldownDeadlineMs = 0;
      this.goalTriggeredIds.clear();
      this.goalAttemptedIds.clear();
      this.goalPendingIds.clear();
      this.sessionStarted = false;
      this.pendingSkillIds.clear();
      this.activeRules = { skills: [], goals: [] };
      this.onStateChange(this.snapshot());
    }

    clearGoalTimers() {
      this.goalTriggerTimers.forEach((timer) => this.clearTimer(timer));
      this.goalTriggerTimers.clear();
    }

    skillCooldownRemainingMs() {
      return Math.max(0, this.skillCooldownDeadlineMs - Number(this.now()));
    }

    beginSharedSkillCooldown() {
      if (this.skillCooldownTimer !== null) this.clearTimer(this.skillCooldownTimer);
      this.skillCooldownTimer = null;
      if (this.skillCooldownMs <= 0) {
        this.skillCooldownDeadlineMs = 0;
        return;
      }
      const generation = this.sessionGeneration;
      this.skillCooldownDeadlineMs = Number(this.now()) + this.skillCooldownMs;
      this.skillCooldownTimer = this.setTimer(() => {
        if (generation !== this.sessionGeneration) return;
        this.skillCooldownTimer = null;
        this.skillCooldownDeadlineMs = 0;
        this.onStateChange(this.snapshot());
      }, this.skillCooldownMs);
      this.onStateChange(this.snapshot());
    }

    async triggerSkill(skillId) {
      const skill = this.activeRules.skills.find((item) => item.id === skillId);
      if (!skill) throw new Error("这个技能不在当前世界中");
      const cooldownRemainingMs = this.skillCooldownRemainingMs();
      if (cooldownRemainingMs > 0) {
        return {
          ignored: true,
          reason: "shared_cooldown",
          remaining_ms: cooldownRemainingMs,
        };
      }
      if (this.pendingSkillIds.has(skill.id)) return { ignored: true };
      this.pendingSkillIds.add(skill.id);
      this.beginSharedSkillCooldown();
      try {
        const result = await this.dispatchPrepared(skill.prepared, {
          trigger: "skill",
          skillId: skill.id,
          skillName: skill.name,
          instruction: skill.instruction,
        });
        return result;
      } finally {
        this.pendingSkillIds.delete(skill.id);
        this.onStateChange(this.snapshot());
      }
    }

    async maybeTriggerGoal(goalId, source = "elapsed_time", generation = this.sessionGeneration) {
      const goal = this.activeRules.goals.find((item) => item.id === goalId);
      if (
        generation !== this.sessionGeneration
        || !goal
        || this.goalTriggeredIds.has(goal.id)
        || this.goalAttemptedIds.has(goal.id)
        || this.goalPendingIds.has(goal.id)
      ) {
        return { triggered: false, canceled: generation !== this.sessionGeneration };
      }
      this.goalAttemptedIds.add(goal.id);
      const roll = Number(this.random());
      if (goal.probability <= 0 || roll >= goal.probability) {
        this.onStateChange(this.snapshot());
        return { triggered: false, roll };
      }
      this.goalPendingIds.add(goal.id);
      this.onStateChange(this.snapshot());
      try {
        const result = await this.dispatchPrepared(goal.prepared, {
          trigger: "rule",
          rule: "goal_time_probability",
          goalId: goal.id,
          goalName: goal.name,
          probability: goal.probability,
          minPlaySeconds: goal.min_play_seconds,
          source,
          instruction: goal.instruction,
        });
        if (
          generation !== this.sessionGeneration
          || goal !== this.activeRules.goals.find((item) => item.id === goal.id)
        ) {
          return { triggered: false, canceled: true, roll };
        }
        if (result?.ignored) return { triggered: false, ignored: true, roll };
        this.goalTriggeredIds.add(goal.id);
        if (this.achievementTimers.has(goal.id)) {
          this.clearTimer(this.achievementTimers.get(goal.id));
        }
        const achievementTimer = this.setTimer(() => {
          this.achievementTimers.delete(goal.id);
          this.onAchievement(goal);
        }, this.achievementDelayMs);
        this.achievementTimers.set(goal.id, achievementTimer);
        return { triggered: true, roll, result };
      } finally {
        this.goalPendingIds.delete(goal.id);
        this.onStateChange(this.snapshot());
      }
    }

    snapshot() {
      const skillCooldownRemainingMs = this.skillCooldownRemainingMs();
      const goals = this.activeRules.goals.map((goal) => ({
        ...goal,
        pending: this.goalPendingIds.has(goal.id),
        attempted: this.goalAttemptedIds.has(goal.id),
        triggered: this.goalTriggeredIds.has(goal.id),
        scheduled: this.goalTriggerTimers.has(goal.id),
      }));
      return {
        skills: this.activeRules.skills.map((skill) => ({
          ...skill,
          pending: this.pendingSkillIds.has(skill.id),
        })),
        goals,
        goal: goals[0] || null,
        goalTriggered: goals.some((goal) => goal.triggered),
        goalAttempted: goals.some((goal) => goal.attempted),
        goalPending: goals.some((goal) => goal.pending),
        goalScheduled: goals.some((goal) => goal.scheduled),
        sessionStarted: this.sessionStarted,
        sharedSkillCooldownMs: this.skillCooldownMs,
        skillCooldownRemainingMs,
        skillCooldownActive: skillCooldownRemainingMs > 0,
      };
    }
  }

  global.WorldRulesController = WorldRulesController;
  global.normalizeWorldRulesDraft = normalizeWorldRulesDraft;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      DEFAULT_GOAL_PROBABILITY,
      DEFAULT_GOAL_MIN_PLAY_SECONDS,
      DEFAULT_SHARED_SKILL_COOLDOWN_MS,
      GOAL_ACHIEVEMENT_DELAY_MS,
      MAX_GOAL_MIN_PLAY_SECONDS,
      MAX_WORLD_GOALS,
      MAX_WORLD_SKILLS,
      WorldRulesController,
      normalizeWorldRulesDraft,
    };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
