(function (global) {
  const PERSISTENT = "persistent";
  const ONE_TIME = "one_time";

  class PromptRewriteController {
    constructor({
      rewrite,
      sendPrompt,
      restoreDelayMs = 10000,
      setTimer = global.setTimeout.bind(global),
      clearTimer = global.clearTimeout.bind(global),
    }) {
      this.rewrite = rewrite;
      this.sendPrompt = sendPrompt;
      this.restoreDelayMs = restoreDelayMs;
      this.setTimer = setTimer;
      this.clearTimer = clearTimer;
      this.persistentPrompt = "";
      this.restoreTimer = null;
      this.sessionGeneration = 0;
      this.requestGeneration = 0;
    }

    beginSession(worldDescription) {
      const prompt = String(worldDescription || "").trim();
      if (!prompt) throw new Error("world description must not be empty");
      this.endSession();
      this.persistentPrompt = prompt;
      this.sessionGeneration += 1;
    }

    endSession() {
      this.requestGeneration += 1;
      this.sessionGeneration += 1;
      this._cancelRestore();
      this.persistentPrompt = "";
    }

    async submit(instruction) {
      const normalizedInstruction = String(instruction || "").trim();
      if (!normalizedInstruction) throw new Error("instruction must not be empty");
      if (!this.persistentPrompt) throw new Error("enter a world before sending a direction");

      const shouldRestoreOnFailure = this.restoreTimer !== null;
      this._cancelRestore();
      const sessionGeneration = this.sessionGeneration;
      const requestGeneration = ++this.requestGeneration;
      const previousPrompt = this.persistentPrompt;
      let result;
      try {
        result = await this.rewrite({
          instruction: normalizedInstruction,
          previous_prompt: previousPrompt,
        });
      } catch (error) {
        if (
          shouldRestoreOnFailure
          && sessionGeneration === this.sessionGeneration
          && requestGeneration === this.requestGeneration
        ) {
          this.sendPrompt(previousPrompt, {
            changeType: PERSISTENT,
            phase: "restore",
          });
        }
        throw error;
      }
      if (
        sessionGeneration !== this.sessionGeneration
        || requestGeneration !== this.requestGeneration
      ) {
        return { ignored: true };
      }

      const prompt = String(result?.prompt || "").trim();
      const changeType = String(result?.change_type || "");
      if (!prompt) throw new Error("prompt rewriter returned an empty prompt");
      if (changeType !== PERSISTENT && changeType !== ONE_TIME) {
        throw new Error("prompt rewriter returned an invalid change type");
      }
      const eventId = this.sendPrompt(prompt, {
        changeType,
        phase: "rewrite",
      });
      if (!eventId) throw new Error("no model session is connected");
      if (changeType === PERSISTENT) {
        this.persistentPrompt = prompt;
      } else {
        const restorePrompt = previousPrompt;
        let restoreTimer = null;
        restoreTimer = this.setTimer(() => {
          if (this.restoreTimer !== restoreTimer) return;
          this.restoreTimer = null;
          if (sessionGeneration !== this.sessionGeneration) return;
          this.sendPrompt(restorePrompt, {
            changeType: PERSISTENT,
            phase: "restore",
          });
        }, this.restoreDelayMs);
        this.restoreTimer = restoreTimer;
      }
      return { prompt, change_type: changeType, event_id: eventId };
    }

    _cancelRestore() {
      if (this.restoreTimer !== null) this.clearTimer(this.restoreTimer);
      this.restoreTimer = null;
    }
  }

  global.PromptRewriteController = PromptRewriteController;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { PromptRewriteController, PERSISTENT, ONE_TIME };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
