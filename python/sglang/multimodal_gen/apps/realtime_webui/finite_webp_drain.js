(function attachFiniteWebpDrain(global) {
  function normalizedActivity(activity = {}) {
    return {
      pendingHeader: Boolean(activity.pendingHeader),
      decodeInProgress: Boolean(activity.decodeInProgress),
      pendingDecodeBatches: Math.max(0, Number(activity.pendingDecodeBatches || 0)),
      decodeQueueLength: Math.max(0, Number(activity.decodeQueueLength || 0)),
      queuedDecodeFrames: Math.max(0, Number(activity.queuedDecodeFrames || 0)),
      queuedDecodeBytes: Math.max(0, Number(activity.queuedDecodeBytes || 0)),
      playbackQueueFrames: Math.max(0, Number(activity.playbackQueueFrames || 0)),
      renderActivity: Math.max(0, Number(activity.renderActivity || 0)),
    };
  }

  function playbackIsDrained(activity = {}) {
    const state = normalizedActivity(activity);
    return (
      !state.pendingHeader &&
      !state.decodeInProgress &&
      state.pendingDecodeBatches === 0 &&
      state.decodeQueueLength === 0 &&
      state.queuedDecodeFrames === 0 &&
      state.queuedDecodeBytes === 0 &&
      state.playbackQueueFrames === 0 &&
      state.renderActivity === 0
    );
  }

  function shouldDrainFiniteWebpClose({
    closeCode,
    finiteSession,
    opened,
    expectedClose,
    clearQueueOnClose,
    serverError,
    transportError,
    decodeErrors,
    sessionLifetimeClose,
    pendingHeader,
  } = {}) {
    return (
      (Number(closeCode) === 1000 || Number(closeCode) === 1001) &&
      Boolean(finiteSession) &&
      Boolean(opened) &&
      !expectedClose &&
      !clearQueueOnClose &&
      !serverError &&
      !transportError &&
      Number(decodeErrors || 0) === 0 &&
      !sessionLifetimeClose &&
      !pendingHeader
    );
  }

  function finitePlaybackIntegrity({ decodeErrors, decodedFrames, renderedFrames } = {}) {
    const errors = Math.max(0, Number(decodeErrors || 0));
    const decoded = Math.max(0, Number(decodedFrames || 0));
    const rendered = Math.max(0, Number(renderedFrames || 0));
    if (errors > 0) {
      return { complete: false, reason: `${errors} frame decode errors` };
    }
    if (decoded <= 0 || rendered !== decoded) {
      return {
        complete: false,
        reason: `rendered ${rendered} of ${decoded} decoded frames`,
      };
    }
    return { complete: true, reason: "" };
  }

  class FiniteWebpDrainController {
    constructor({ onBegin = null, onComplete = null } = {}) {
      this.onBegin = onBegin;
      this.onComplete = onComplete;
      this.activeDrain = null;
      this.lastActivity = normalizedActivity();
    }

    begin(context = {}) {
      const epoch = Number(context.epoch);
      if (!Number.isFinite(epoch)) throw new Error("finite WebP drain requires an epoch");
      this.activeDrain = { ...context, epoch };
      this.lastActivity = normalizedActivity();
      this.onBegin?.(this.snapshot());
      return this.snapshot();
    }

    cancel() {
      const cancelled = this.activeDrain;
      this.activeDrain = null;
      this.lastActivity = normalizedActivity();
      return cancelled;
    }

    isActive(epoch = undefined) {
      if (!this.activeDrain) return false;
      return epoch === undefined || Number(epoch) === this.activeDrain.epoch;
    }

    completeIfDrained(epoch, activity = {}) {
      if (!this.isActive(epoch)) return false;
      this.lastActivity = normalizedActivity(activity);
      if (!playbackIsDrained(this.lastActivity)) return false;
      const completed = { ...this.activeDrain };
      const finalActivity = { ...this.lastActivity };
      this.activeDrain = null;
      this.onComplete?.(completed, finalActivity);
      return true;
    }

    snapshot() {
      return {
        active: Boolean(this.activeDrain),
        ...(this.activeDrain || {}),
        activity: { ...this.lastActivity },
      };
    }
  }

  const api = {
    FiniteWebpDrainController,
    finitePlaybackIntegrity,
    normalizedActivity,
    playbackIsDrained,
    shouldDrainFiniteWebpClose,
  };
  global.SGLangFiniteWebpDrain = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
