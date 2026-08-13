(function (global) {
  class DualModelController {
    constructor({ sessions, backends, now = () => performance.now() }) {
      this.sessions = sessions;
      this.backends = backends;
      this.now = now;
      this.nextEventId = 1;
      this.activeKeys = new Set();
      this.connectionGeneration = 0;
      this.connectionTemplates = new Map();
      this.latestEvents = new Map();
      this.reconnectCounts = new Map();
    }

    async connect(baseInit) {
      this.connectionGeneration += 1;
      const generation = this.connectionGeneration;
      this.nextEventId = 1;
      this.activeKeys.clear();
      this.connectionTemplates.clear();
      this.latestEvents.clear();
      this.reconnectCounts.clear();
      const entries = [];
      const backgroundEntries = [];
      for (const [key, session] of Object.entries(this.sessions)) {
        const backend = this.backends[key];
        if (!backend) throw new Error(`missing backend configuration for ${key}`);
        const enabled = typeof backend.enabled === "function"
          ? backend.enabled(baseInit, key)
          : backend.enabled !== false;
        if (enabled) {
          (backend.nonBlocking ? backgroundEntries : entries).push([key, session]);
          continue;
        }
        if (typeof session.setUnavailable === "function") {
          session.setUnavailable(backend.unavailableReason || "Unavailable for this mode");
        } else {
          session.close("disabled for request");
        }
      }
      const connectOne = async (key, { replayLatest = false } = {}) => {
        await this.connectKey(key, baseInit, { reconnect: false });
        if (generation !== this.connectionGeneration) {
          this.sessions[key]?.close("stale connection");
          return false;
        }
        this.activeKeys.add(key);
        if (replayLatest) {
          const replayEvents = [...this.latestEvents.values()].sort(
            (left, right) => Number(left.event_id || 0) - Number(right.event_id || 0),
          );
          for (const envelope of replayEvents) this.sessions[key]?.sendEvent(envelope);
        }
        return true;
      };
      const results = await Promise.allSettled(entries.map(([key]) => connectOne(key)));
      const report = {
        connected: [],
        failed: [],
        pending: backgroundEntries.map(([key]) => key),
      };
      results.forEach((result, index) => {
        const [key] = entries[index];
        if (result.status === "fulfilled" && result.value) {
          report.connected.push(key);
        } else {
          report.failed.push({ key, error: result.reason || new Error("stale connection") });
        }
      });
      for (const [key] of backgroundEntries) {
        void connectOne(key, { replayLatest: true }).catch(() => {
          this.activeKeys.delete(key);
        });
      }
      if (!report.connected.length && entries.length) {
        throw new Error(
          `no realtime model connected: ${report.failed
            .map(({ key, error }) => `${key}: ${error?.message || error}`)
            .join("; ")}`,
        );
      }
      return report;
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
      this.latestEvents.set(kind, envelope);
      const sent = {};
      for (const key of this.activeKeys) {
        sent[key] = Boolean(this.sessions[key]?.sendEvent(envelope));
      }
      return { eventId, sent };
    }

    close(reason = "dual model session closed") {
      this.connectionGeneration += 1;
      for (const session of Object.values(this.sessions)) session.close(reason);
      this.activeKeys.clear();
      this.connectionTemplates.clear();
      this.latestEvents.clear();
      this.reconnectCounts.clear();
    }

    async reconnect(key) {
      const template = this.connectionTemplates.get(key);
      if (!template) throw new Error(`missing reconnect template for ${key}`);
      const generation = this.connectionGeneration;
      this.activeKeys.delete(key);
      await this.connectKey(key, template.baseInit, { reconnect: true });
      if (generation !== this.connectionGeneration) {
        this.sessions[key]?.close("stale reconnect");
        return false;
      }
      this.activeKeys.add(key);
      const replayEvents = [...this.latestEvents.values()].sort(
        (left, right) => Number(left.event_id || 0) - Number(right.event_id || 0),
      );
      for (const envelope of replayEvents) this.sessions[key]?.sendEvent(envelope);
      return true;
    }

    async connectKey(key, baseInit, { reconnect }) {
      const backend = this.backends[key];
      const session = this.sessions[key];
      if (!backend || !session) throw new Error(`missing backend session for ${key}`);
      const traceBase = baseInit.trace_id || `trace-${Date.now()}`;
      const model = typeof backend.model === "function"
        ? backend.model(baseInit, key)
        : backend.model;
      const reconnectCount = reconnect
        ? Number(this.reconnectCounts.get(key) || 0) + 1
        : 0;
      this.reconnectCounts.set(key, reconnectCount);
      let init = {
        ...baseInit,
        model,
        trace_id: key === "minwm"
          ? traceBase
          : `${traceBase}:${key}${reconnectCount ? `:retry${reconnectCount}` : ""}`,
      };
      if (typeof backend.transformInit === "function") {
        init = backend.transformInit(init, key);
      }
      const wsUrl = typeof backend.wsUrl === "function"
        ? backend.wsUrl(init, key)
        : backend.wsUrl;
      this.connectionTemplates.set(key, { baseInit: { ...baseInit } });
      return session.connect(init, wsUrl);
    }
  }

  global.DualModelController = DualModelController;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { DualModelController };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
