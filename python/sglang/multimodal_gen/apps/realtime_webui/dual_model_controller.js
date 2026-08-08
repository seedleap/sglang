(function (global) {
  class DualModelController {
    constructor({ sessions, backends, now = () => performance.now() }) {
      this.sessions = sessions;
      this.backends = backends;
      this.now = now;
      this.nextEventId = 1;
      this.activeKeys = new Set();
    }

    async connect(baseInit) {
      this.nextEventId = 1;
      this.activeKeys.clear();
      const entries = [];
      for (const [key, session] of Object.entries(this.sessions)) {
        const backend = this.backends[key];
        if (!backend) throw new Error(`missing backend configuration for ${key}`);
        const enabled = typeof backend.enabled === "function"
          ? backend.enabled(baseInit, key)
          : backend.enabled !== false;
        if (enabled) {
          entries.push([key, session]);
          this.activeKeys.add(key);
          continue;
        }
        if (typeof session.setUnavailable === "function") {
          session.setUnavailable(backend.unavailableReason || "Unavailable for this mode");
        } else {
          session.close("disabled for request");
        }
      }
      const results = await Promise.allSettled(entries.map(([key, session]) => {
        const backend = this.backends[key];
        const traceBase = baseInit.trace_id || `trace-${Date.now()}`;
        const model = typeof backend.model === "function"
          ? backend.model(baseInit, key)
          : backend.model;
        let init = {
          ...baseInit,
          model,
          trace_id: key === "minwm" ? traceBase : `${traceBase}:${key}`,
        };
        if (typeof backend.transformInit === "function") {
          init = backend.transformInit(init, key);
        }
        const wsUrl = typeof backend.wsUrl === "function"
          ? backend.wsUrl(init, key)
          : backend.wsUrl;
        return session.connect(init, wsUrl);
      }));
      const failure = results.find((result) => result.status === "rejected");
      if (!failure) return;
      for (const [, session] of entries) session.close("dual model startup failed");
      this.activeKeys.clear();
      throw failure.reason;
    }

    sendEvent(kind, payload) {
      const eventId = this.nextEventId++;
      const sentAt = this.now();
      const envelope = {
        type: "event",
        kind,
        payload,
        event_id: eventId,
        client_sent_perf_ms: sentAt,
        client_sent_epoch_ms: Date.now(),
      };
      for (const key of this.activeKeys) this.sessions[key]?.sendEvent(envelope);
      return eventId;
    }

    close(reason = "dual model session closed") {
      for (const session of Object.values(this.sessions)) session.close(reason);
      this.activeKeys.clear();
    }
  }

  global.DualModelController = DualModelController;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { DualModelController };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
