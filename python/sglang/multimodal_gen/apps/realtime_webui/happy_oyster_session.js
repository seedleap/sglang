(function (global) {
  const POLL_INTERVAL_MS = 3000;
  const WORLD_READY_TIMEOUT_MS = 180000;

  function isRecord(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }

  function activeActions(payload) {
    const transitions = Array.isArray(payload?.transitions) ? payload.transitions : [];
    const final = transitions[transitions.length - 1];
    return new Set(Array.isArray(final?.actions) ? final.actions : []);
  }

  function translateCameraActions(payload) {
    const actions = activeActions(payload);
    const vertical = actions.has("w") ? "Front" : actions.has("s") ? "Back" : "";
    const horizontal = actions.has("a") ? "Left" : actions.has("d") ? "Right" : "";
    const translation = vertical && horizontal
      ? `${vertical}_${horizontal}`
      : vertical || horizontal || "None";
    const rotation = actions.has("i")
      ? "Mouse_Up"
      : actions.has("k")
        ? "Mouse_Down"
        : actions.has("j")
          ? "Mouse_Left"
          : actions.has("l")
            ? "Mouse_Right"
            : "None";
    const interaction = actions.has("space")
      ? "Jump"
      : actions.has("shift")
        ? "Sprint"
        : actions.has("ctrl") || actions.has("c")
          ? "Crouch"
          : actions.has("f")
            ? "Attack"
            : "None";
    return { translation, rotation, interaction };
  }

  async function readJson(response) {
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      throw new Error(payload?.error || payload?.message || `HTTP ${response.status}`);
    }
    return payload;
  }

  class HappyOysterSession {
    constructor({ video, root = null, overlay = null, fetchImpl = global.fetch, onState = () => {}, onError = () => {} }) {
      this.video = video;
      this.root = root;
      this.overlay = overlay;
      this.fetchImpl = fetchImpl;
      this.onState = onState;
      this.onError = onError;
      this.engine = null;
      this.travel = null;
      this.prepared = null;
      this.preparationError = null;
      this.unsubscribers = [];
      this.connected = false;
      this.epoch = 0;
      this.pendingCommand = Promise.resolve();
      this.setState("idle");
    }

    async configured() {
      const response = await this.fetchImpl("./api/happyoyster/config", { cache: "no-store" });
      return readJson(response);
    }

    async prepare({ prompt, firstFrame, firstFrameMimeType = "image/png", perspective = "third_person" }) {
      await this.close("replace session", { notify: false });
      const epoch = ++this.epoch;
      this.preparationError = null;
      this.setState("preparing");
      try {
        this.setState("preparing", { message: "正在检查快乐生蚝 API…" });
        const config = await this.configured();
        if (!config.enabled) throw new Error("快乐生蚝 API 未配置");
        let firstFrameUrl = "";
        if (firstFrame?.byteLength) {
          this.setState("preparing", { message: "正在上传首帧…" });
          const response = await this.fetchImpl("./api/happyoyster/share-image", {
            method: "POST",
            headers: { "Content-Type": firstFrameMimeType || "image/png" },
            body: firstFrame,
          });
          firstFrameUrl = (await readJson(response)).url || "";
        }
        this.setState("preparing", { message: "正在创建快乐生蚝 World…" });
        const created = await readJson(await this.fetchImpl("./api/happyoyster/worlds", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, firstFrameUrl, perspective }),
        }));
        const worldId = created?.encryptedWorldId;
        if (!worldId) throw new Error("快乐生蚝创建 World 未返回 encryptedWorldId");
        this.setState("preparing", { message: "快乐生蚝 World 构建中…" });
        await this.waitUntilReady(worldId, epoch);
        this.setState("preparing", { message: "正在获取 RTC 连接凭证…" });
        const prepared = await readJson(await this.fetchImpl("./api/happyoyster/prepare", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ encryptedWorldId: worldId }),
        }));
        if (epoch !== this.epoch) throw new Error("快乐生蚝会话已被替换");
        this.prepared = { ...prepared, epoch };
        this.setState("ready", { message: "World 已就绪，正在连接视频流…" });
        return true;
      } catch (error) {
        if (epoch === this.epoch) {
          this.preparationError = error instanceof Error ? error : new Error(String(error));
          this.fail(this.preparationError);
        }
        throw error;
      }
    }

    async connect(init = null) {
      if (!this.prepared && init) await this.prepare(init);
      if (!this.prepared) {
        throw this.preparationError || new Error("快乐生蚝 World 尚未准备完成");
      }
      const prepared = this.prepared;
      this.prepared = null;
      const epoch = prepared.epoch;
      if (epoch !== this.epoch) throw new Error("快乐生蚝会话已被替换");
      this.setState("connecting", { message: "正在连接快乐生蚝视频流…" });
      try {
        const sdk = global.HappyOysterSDK;
        if (!sdk?.HappyOysterEngine) throw new Error("快乐生蚝 Web SDK 加载失败");
        const engine = new sdk.HappyOysterEngine({
          apiBaseUrl: prepared.apiBaseUrl,
          token: prepared.token,
          logLevel: "warn",
          streamReadyTimeout: 20000,
        });
        const travel = engine.createTravel({ ticket: prepared.ticket, videoElement: this.video });
        if (this.video) {
          this.video.defaultMuted = true;
          this.video.muted = true;
        }
        this.engine = engine;
        this.travel = travel;
        this.unsubscribers = [
          travel.on("statusChanged", (status) => {
            if (epoch !== this.epoch) return;
            if (status === "running") this.setState("live");
            if (status === "completed") this.setState("closed");
          }),
          travel.onError((error) => {
            if (epoch !== this.epoch) return;
            this.fail(error);
          }),
        ];
        await travel.start();
        if (epoch !== this.epoch) {
          await travel.end().catch(() => {});
          return false;
        }
        this.connected = true;
        this.setState("live");
        return true;
      } catch (error) {
        if (epoch === this.epoch) this.fail(error);
        throw error;
      }
    }

    async waitUntilReady(worldId, epoch) {
      const deadline = Date.now() + WORLD_READY_TIMEOUT_MS;
      while (Date.now() < deadline) {
        if (epoch !== this.epoch) throw new Error("快乐生蚝会话已取消");
        const url = `./api/happyoyster/worlds/build-status?encryptedWorldId=${encodeURIComponent(worldId)}`;
        const status = await readJson(await this.fetchImpl(url, { cache: "no-store" }));
        if (status?.status === "ready") return status;
        if (status?.status === "failed") throw new Error(status.message || "快乐生蚝 World 构建失败");
        await new Promise((resolve) => global.setTimeout(resolve, POLL_INTERVAL_MS));
      }
      throw new Error("快乐生蚝 World 构建超时");
    }

    sendEvent(envelope) {
      if (!this.connected || !this.travel) return false;
      if (envelope.kind === "camera_actions") {
        const command = translateCameraActions(envelope.payload);
        this.pendingCommand = this.pendingCommand
          .catch(() => {})
          .then(() => this.travel?.sendCommand(command))
          .catch((error) => this.fail(error));
        return true;
      }
      if (envelope.kind === "prompt") {
        return false;
      }
      return envelope.kind === "heartbeat";
    }

    async close(reason = "session closed", { notify = true } = {}) {
      this.epoch += 1;
      this.connected = false;
      this.prepared = null;
      this.preparationError = null;
      const travel = this.travel;
      this.travel = null;
      this.engine = null;
      for (const unsubscribe of this.unsubscribers.splice(0)) unsubscribe?.();
      if (travel) await travel.end().catch(() => {});
      if (this.video) {
        this.video.pause?.();
        this.video.srcObject = null;
        this.video.removeAttribute?.("src");
        this.video.load?.();
      }
      if (notify) this.setState("closed", { reason });
    }

    setUnavailable(reason = "Unavailable") {
      void this.close(reason, { notify: false });
      this.setState("unavailable", { reason });
    }

    setState(state, details = {}) {
      if (this.root) this.root.dataset.sessionState = state;
      if (this.overlay) {
        const visible = ["preparing", "ready", "connecting", "error", "unavailable"].includes(state);
        this.overlay.setAttribute("aria-hidden", visible ? "false" : "true");
      }
      this.onState(state, details);
    }

    fail(error) {
      const normalized = error instanceof Error ? error : new Error(String(error));
      this.connected = false;
      this.setState("error", { message: normalized.message });
      this.onError(normalized, "happyoyster");
    }
  }

  global.HappyOysterSession = HappyOysterSession;
  global.translateHappyOysterCameraActions = translateCameraActions;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { HappyOysterSession, translateHappyOysterCameraActions };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
