(function (global) {
  class SessionLifetimeGuard {
    constructor({
      durationMs,
      onExpire,
      setTimer = global.setTimeout.bind(global),
      clearTimer = global.clearTimeout.bind(global),
    }) {
      this.durationMs = durationMs;
      this.onExpire = onExpire;
      this.setTimer = setTimer;
      this.clearTimer = clearTimer;
      this.timer = null;
      this.generation = 0;
    }

    start() {
      this.cancel();
      const generation = ++this.generation;
      this.timer = this.setTimer(() => {
        if (generation !== this.generation) return;
        this.timer = null;
        this.onExpire();
      }, this.durationMs);
    }

    cancel() {
      this.generation += 1;
      if (this.timer !== null) this.clearTimer(this.timer);
      this.timer = null;
    }
  }

  global.SessionLifetimeGuard = SessionLifetimeGuard;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { SessionLifetimeGuard };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
