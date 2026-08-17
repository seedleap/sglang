(function (global) {
  const OPEN = 1;
  const MIME_TYPE = 'video/mp4; codecs="avc1.4D401F"';
  const PLAYBACK_MODES = new Set(["live", "adaptive", "timeline", "smooth_timeline"]);

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function normalizePlaybackMode(mode) {
    const normalized = String(mode || "live").toLowerCase();
    return PLAYBACK_MODES.has(normalized) ? normalized : "live";
  }

  function isFiniteRequest(init = {}) {
    const maxChunks = Number(init.max_chunks);
    if (Number.isFinite(maxChunks) && maxChunks > 0) return true;
    const generationMode = String(init.generation_mode || "").toLowerCase();
    const numFrames = Number(init.num_frames);
    const textToVideo = generationMode === "t2v"
      || (!generationMode && init.first_frame == null);
    return textToVideo && Number.isFinite(numFrames) && numFrames > 0;
  }

  function bytesToDataUrl(value, mimeType = "image/png") {
    if (typeof value === "string") return Promise.resolve(value);
    const bytes = value instanceof Uint8Array
      ? value
      : value instanceof ArrayBuffer
        ? new Uint8Array(value)
        : null;
    if (!bytes) return Promise.resolve(value);
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(reader.error || new Error("first frame conversion failed"));
      reader.readAsDataURL(new Blob([bytes], { type: mimeType }));
    });
  }

  function waitForSourceOpen(mediaSource, timeoutMs) {
    if (mediaSource.readyState === "open") return Promise.resolve();
    return new Promise((resolve, reject) => {
      const timer = global.setTimeout(() => {
        cleanup();
        reject(new Error("H.264 MSE source open timed out"));
      }, timeoutMs);
      const cleanup = () => {
        global.clearTimeout(timer);
        mediaSource.removeEventListener("sourceopen", handleOpen);
        mediaSource.removeEventListener("sourceclose", handleClose);
      };
      const handleOpen = () => { cleanup(); resolve(); };
      const handleClose = () => { cleanup(); reject(new Error("H.264 MSE source closed")); };
      mediaSource.addEventListener("sourceopen", handleOpen, { once: true });
      mediaSource.addEventListener("sourceclose", handleClose, { once: true });
    });
  }

  function resultFromError(error) {
    const value = String(error?.code || error?.name || error?.message || "").toLowerCase();
    if (value.includes("timeout")) return "timeout";
    if (value.includes("cancel")) return "cancelled";
    return "error";
  }

  class H264WebSocketSession {
    constructor({
      video,
      overlay = null,
      root = null,
      endpoint = "/api/h264ws",
      startupTimeoutMs = 30000,
      liveEdgeTargetMs = 80,
      liveEdgeSeekThresholdMs = 420,
      mode = "live",
      targetFps = 24,
      smoothTimelineStartupBufferMs = 450,
      smoothTimelineTargetLagMs = 650,
      smoothTimelineMaxLagMs = 1800,
      smoothTimelinePlaybackRateMin = 0.94,
      smoothTimelinePlaybackRateMax = 1.1,
      smoothTimelinePlaybackRateGain = 0.16,
      smoothTimelinePlaybackRateSlewPerSecond = 0.18,
      drainPlaybackGraceMs = 2000,
      drainPlaybackMaxWaitMs = 120000,
      onState = () => {},
      onPlayable = () => {},
      onPresentedFrame = () => {},
      onStats = () => {},
      onError = () => {},
      WebSocketImpl = global.WebSocket,
      MediaSourceImpl = global.MediaSource,
      metricFlushMs = 250,
      maxMetricBatch = 32,
      documentRef = global.document,
    }) {
      if (!video) throw new Error("H264WebSocketSession requires a video element");
      this.video = video;
      this.overlay = overlay;
      this.root = root;
      this.endpoint = endpoint;
      this.startupTimeoutMs = startupTimeoutMs;
      this.liveEdgeTargetMs = Math.max(0, Number(liveEdgeTargetMs) || 0);
      this.liveEdgeSeekThresholdMs = Math.max(
        this.liveEdgeTargetMs,
        Number(liveEdgeSeekThresholdMs) || 420,
      );
      this.playbackMode = normalizePlaybackMode(mode);
      this.targetFps = Math.max(1, Number(targetFps) || 24);
      this.smoothTimelineStartupBufferMs = Math.max(
        0,
        Number(smoothTimelineStartupBufferMs) || 0,
      );
      this.smoothTimelineTargetLagMs = Math.max(
        this.smoothTimelineStartupBufferMs,
        Number(smoothTimelineTargetLagMs) || 650,
      );
      this.smoothTimelineMaxLagMs = Math.max(
        this.smoothTimelineTargetLagMs,
        Number(smoothTimelineMaxLagMs) || 1800,
      );
      this.smoothTimelinePlaybackRateMin = clamp(
        Number(smoothTimelinePlaybackRateMin) || 0.94,
        0.5,
        1,
      );
      this.smoothTimelinePlaybackRateMax = clamp(
        Number(smoothTimelinePlaybackRateMax) || 1.1,
        1,
        2.5,
      );
      this.smoothTimelinePlaybackRateGain = clamp(
        Number(smoothTimelinePlaybackRateGain) || 0.16,
        0.01,
        1,
      );
      this.smoothTimelinePlaybackRateSlewPerSecond = clamp(
        Number(smoothTimelinePlaybackRateSlewPerSecond) || 0.18,
        0.01,
        2.5,
      );
      this.drainPlaybackGraceMs = Math.max(0, Number(drainPlaybackGraceMs) || 0);
      this.drainPlaybackMaxWaitMs = Math.max(
        this.drainPlaybackGraceMs,
        Number(drainPlaybackMaxWaitMs) || 120000,
      );
      this.onState = onState;
      this.onPlayable = onPlayable;
      this.onPresentedFrame = onPresentedFrame;
      this.onStats = onStats;
      this.onError = onError;
      this.WebSocketImpl = WebSocketImpl;
      this.MediaSourceImpl = MediaSourceImpl;
      this.metricFlushMs = Math.max(0, Number(metricFlushMs) || 0);
      this.maxMetricBatch = Math.max(1, Number(maxMetricBatch) || 32);
      this.documentRef = documentRef;
      this.socket = null;
      this.mediaSource = null;
      this.sourceBuffer = null;
      this.objectUrl = "";
      this.generation = 0;
      this.state = "idle";
      this.playable = false;
      this.expectedClose = false;
      this.streamCompleteReceived = false;
      this.streamCompleteDetails = null;
      this.transportClosed = false;
      this.drainFinished = false;
      this.drainDetails = {};
      this.drainTimer = 0;
      this.drainPlayFailureReported = false;
      this.appendQueue = [];
      this.appendQueueBytes = 0;
      this.activeAppendItem = null;
      this.pendingPayloadTimings = [];
      this.mediaBatches = [];
      this.presentedSequence = 0;
      this.lastPresentedEventId = 0;
      this.controlSentEpochByEvent = new Map();
      this.lastBridgeClockOffsetMs = 0;
      this.bytesReceived = 0;
      this.framesPresented = 0;
      this.presentedSamples = [];
      this.sourceSamples = [];
      this.deliverySamples = [];
      this.networkSample = null;
      this.statsTimer = 0;
      this.frameCallback = 0;
      this.metricBuffer = [];
      this.metricFlushTimer = 0;
      this.lastRenderedChunk = null;
      this.traceId = "";
      this.mediaFps = 24;
      this.finiteRequest = false;
      this.playbackStarted = false;
      this.lastPlaybackPolicyAt = 0;
      this.playbackAckEnabled = false;
      this.lastStats = {};
      this.handlePlayable = () => this._markPlayable();
      this.handleTimeUpdate = () => {
        if (this.playbackMode === "smooth_timeline") this._maintainLiveEdge();
        this._markPlayable();
        this._maybeEndMediaStream();
      };
      this.handleEnded = () => this._completeDrain();
      this.handleVisibilityChange = () => {
        this.lastPlaybackPolicyAt = 0;
        if (this._isDocumentHidden()) {
          if (this.playbackMode === "smooth_timeline") this._setPlaybackRate(1);
          return;
        }
        if (this.playbackMode === "smooth_timeline") this._maintainLiveEdge();
      };
      for (const name of ["loadeddata", "playing", "resize"]) {
        this.video.addEventListener(name, this.handlePlayable);
      }
      this.video.addEventListener("timeupdate", this.handleTimeUpdate);
      this.video.addEventListener("ended", this.handleEnded);
      this.documentRef?.addEventListener?.("visibilitychange", this.handleVisibilityChange);
    }

    get active() {
      return Boolean(this.socket || this.mediaSource);
    }

    get connected() {
      return Boolean(this.socket?.readyState === OPEN && this.playable);
    }

    get bufferedAmount() {
      return Number(this.socket?.bufferedAmount || 0);
    }

    snapshot() {
      return {
        ...this.lastStats,
        state: this.state,
        connected: this.connected,
        mode: this.playbackMode,
        finitePlayback: this.finiteRequest,
        playbackStarted: this.playbackStarted,
        playbackRate: Number(this.video.playbackRate || 1),
      };
    }

    configure(options = {}) {
      const previousMode = this.playbackMode;
      if (options.mode != null) this.playbackMode = normalizePlaybackMode(options.mode);
      if (options.targetFps != null) {
        this.targetFps = Math.max(1, Number(options.targetFps) || this.targetFps);
      }
      if (options.smoothTimelinePlaybackRateMax != null) {
        this.smoothTimelinePlaybackRateMax = clamp(
          Number(options.smoothTimelinePlaybackRateMax) || 1,
          1,
          2.5,
        );
      }
      if (options.smoothTimelineStartupBufferMs != null) {
        this.smoothTimelineStartupBufferMs = Math.max(
          0,
          Number(options.smoothTimelineStartupBufferMs) || 0,
        );
      }
      if (options.smoothTimelineTargetLagMs != null) {
        this.smoothTimelineTargetLagMs = Math.max(
          this.smoothTimelineStartupBufferMs,
          Number(options.smoothTimelineTargetLagMs) || this.smoothTimelineTargetLagMs,
        );
      }
      if (options.smoothTimelineMaxLagMs != null) {
        this.smoothTimelineMaxLagMs = Math.max(
          this.smoothTimelineTargetLagMs,
          Number(options.smoothTimelineMaxLagMs) || this.smoothTimelineMaxLagMs,
        );
      }
      this.smoothTimelineTargetLagMs = Math.max(
        this.smoothTimelineStartupBufferMs,
        this.smoothTimelineTargetLagMs,
      );
      this.smoothTimelineMaxLagMs = Math.max(
        this.smoothTimelineTargetLagMs,
        this.smoothTimelineMaxLagMs,
      );
      if (this.playbackMode !== "smooth_timeline") {
        this.playbackStarted = this.playable;
        this._setPlaybackRate(1);
      } else if (previousMode !== "smooth_timeline") {
        // Do not interrupt an already-playing stream when the user changes the
        // policy at runtime. New sessions still observe the startup buffer.
        this.playbackStarted = this.playable;
        this.lastPlaybackPolicyAt = 0;
      }
      if (this.active) this._maintainLiveEdge();
      return this.snapshot();
    }

    async connect(init) {
      await this.close("replace H.264 WebSocket session", { emitState: false });
      if (!this.WebSocketImpl || !this.MediaSourceImpl) {
        throw new Error("This browser does not support WebSocket MSE playback");
      }
      if (!this.MediaSourceImpl.isTypeSupported?.(MIME_TYPE)) {
        throw new Error(`Browser does not support ${MIME_TYPE}`);
      }
      const generation = ++this.generation;
      this.expectedClose = false;
      this.playable = false;
      this.streamCompleteReceived = false;
      this.streamCompleteDetails = null;
      this.transportClosed = false;
      this.drainFinished = false;
      this.drainDetails = {};
      this._clearDrainTimer();
      this.drainPlayFailureReported = false;
      this.traceId = String(init.trace_id || "");
      this.mediaFps = Math.max(1, Number(init.fps || 24));
      this.targetFps = Math.max(1, Number(this.targetFps || this.mediaFps));
      this.finiteRequest = isFiniteRequest(init);
      this.playbackStarted = this.playbackMode !== "smooth_timeline";
      this.lastPlaybackPolicyAt = 0;
      this._setPlaybackRate(1);
      this.playbackAckEnabled = init.playback_ack_enabled === true;
      this.appendQueue = [];
      this.appendQueueBytes = 0;
      this.activeAppendItem = null;
      this.pendingPayloadTimings = [];
      this.mediaBatches = [];
      this.presentedSequence = 0;
      this.lastPresentedEventId = 0;
      this.controlSentEpochByEvent.clear();
      this.bytesReceived = 0;
      this.framesPresented = 0;
      this.presentedSamples = [];
      this.sourceSamples = [];
      this.deliverySamples = [];
      this.networkSample = null;
      this.lastStats = {};
      this.lastRenderedChunk = null;
      this._clearMetricFlushTimer();
      this.metricBuffer = [];
      this._setState("connecting", { codec: "h264", protocol: "websocket" });

      const mediaSource = new this.MediaSourceImpl();
      this.mediaSource = mediaSource;
      this.objectUrl = URL.createObjectURL(mediaSource);
      this.video.srcObject = null;
      this.video.src = this.objectUrl;
      this.video.hidden = true;
      await waitForSourceOpen(mediaSource, this.startupTimeoutMs);
      if (generation !== this.generation) throw new Error("H.264 WebSocket startup canceled");
      const sourceBuffer = mediaSource.addSourceBuffer(MIME_TYPE);
      this.sourceBuffer = sourceBuffer;
      sourceBuffer.addEventListener("updateend", () => {
        if (generation !== this.generation) return;
        this._handleAppendEnd();
        this._maintainLiveEdge();
        this._appendNext();
      });
      sourceBuffer.addEventListener("error", () => {
        if (generation !== this.generation) return;
        this._fail(new Error("H.264 MSE SourceBuffer error"));
      });

      const firstFrame = await bytesToDataUrl(init.first_frame);
      return new Promise((resolve, reject) => {
        const protocol = global.location?.protocol === "https:" ? "wss:" : "ws:";
        const host = global.location?.host || "localhost";
        const endpoint = this.endpoint.startsWith("ws")
          ? this.endpoint
          : `${protocol}//${host}${this.endpoint.startsWith("/") ? "" : "/"}${this.endpoint}`;
        const socket = new this.WebSocketImpl(endpoint);
        this.socket = socket;
        socket.binaryType = "arraybuffer";
        let settled = false;
        const timer = global.setTimeout(() => {
          if (settled || generation !== this.generation) return;
          settled = true;
          reject(new Error("H.264 WebSocket startup timed out"));
          void this.close("startup timeout", { emitState: false });
        }, this.startupTimeoutMs);
        socket.onopen = () => {
          if (generation !== this.generation) return;
          const serializeStartedAt = performance.now();
          const payload = JSON.stringify({ ...init, first_frame: firstFrame });
          socket.send(payload);
          this._queueMetric(
            "client_serialize_queue",
            performance.now() - serializeStartedAt,
            { codec: "h264", scope: "request" },
          );
        };
        socket.onmessage = (message) => {
          if (generation !== this.generation) return;
          if (typeof message.data !== "string") {
            this._queueMedia(message.data);
            return;
          }
          try {
            const event = JSON.parse(message.data);
            if (event.type === "status" && event.state === "connected") {
              if (!settled) {
                settled = true;
                global.clearTimeout(timer);
                resolve(event);
              }
              return;
            }
            if (event.type === "error") throw new Error(event.message || "H.264 WebSocket error");
            this._handleMetadata(event);
          } catch (error) {
            if (!settled) {
              settled = true;
              global.clearTimeout(timer);
              reject(error);
            }
            this._fail(error);
          }
        };
        socket.onerror = () => {
          if (generation !== this.generation) return;
          const error = new Error("H.264 WebSocket transport error");
          if (!settled) {
            settled = true;
            global.clearTimeout(timer);
            reject(error);
          }
          this._fail(error);
        };
        socket.onclose = (event) => {
          if (generation !== this.generation) return;
          global.clearTimeout(timer);
          this.socket = null;
          if (this.expectedClose) return;
          const normalClose = event.code === 1000 || event.code === 1001;
          if (!settled) {
            const error = new Error(
              `H.264 WebSocket closed before startup (${event.code}): ${event.reason || "unknown"}`,
            );
            reject(error);
            this._fail(error);
            return;
          }
          if (normalClose && this.state !== "error") {
            this._beginDrain({
              code: event.code,
              reason: event.reason || "generation complete",
              transportClosed: true,
            });
            return;
          }
          const error = new Error(
            `H.264 WebSocket closed (${event.code}): ${event.reason || "unknown"}`,
          );
          this._fail(error);
        };
      });
    }

    sendEvent(envelope) {
      if (!this.socket || this.socket.readyState !== OPEN) return false;
      const eventId = Number(envelope.event_id || 0);
      if (eventId > 0 && ["camera_actions", "prompt", "scene_cut"].includes(envelope.kind)) {
        this.controlSentEpochByEvent.set(
          eventId,
          Number(envelope.client_sent_epoch_ms || Date.now()),
        );
        while (this.controlSentEpochByEvent.size > 64) {
          this.controlSentEpochByEvent.delete(this.controlSentEpochByEvent.keys().next().value);
        }
      }
      const serializeStartedAt = performance.now();
      const payload = JSON.stringify({ ...envelope, trace_id: this.traceId });
      this.socket.send(payload);
      this._queueMetric(
        "client_serialize_queue",
        performance.now() - serializeStartedAt,
        { codec: "h264", scope: "request" },
      );
      return true;
    }

    _queueMetric(stage, durationMs, {
      result = "success",
      codec = "h264",
      scope = "request",
    } = {}) {
      const duration = Number(durationMs);
      if (!Number.isFinite(duration) || duration < 0) return;
      if (!this.socket || this.socket.readyState !== OPEN) return;
      this.metricBuffer.push({
        stage,
        duration_ms: duration,
        result,
        codec,
        scope,
      });
      if (this.metricBuffer.length >= this.maxMetricBatch || this.metricFlushMs === 0) {
        this._flushClientMetrics();
        return;
      }
      if (this.metricFlushTimer) return;
      this.metricFlushTimer = global.setTimeout(() => {
        this.metricFlushTimer = 0;
        this._flushClientMetrics();
      }, this.metricFlushMs);
    }

    _flushClientMetrics() {
      if (!this.metricBuffer.length) return;
      if (!this.socket || this.socket.readyState !== OPEN) return;
      const events = this.metricBuffer.splice(0, this.maxMetricBatch);
      const payload = events.length === 1
        ? { type: "client_metric", ...events[0] }
        : { type: "client_metric_batch", events };
      try {
        this.socket.send(JSON.stringify(payload));
      } catch {
        this.metricBuffer.unshift(...events);
      }
    }

    _clearMetricFlushTimer() {
      if (!this.metricFlushTimer) return;
      global.clearTimeout(this.metricFlushTimer);
      this.metricFlushTimer = 0;
    }

    async close(reason = "H.264 WebSocket session closed", { emitState = true } = {}) {
      const hadSession = this.active;
      this.expectedClose = true;
      this.generation += 1;
      for (const item of this.appendQueue) {
        this._queueMetric(
          "client_receive_queue",
          Math.max(0, performance.now() - Number(item.receivedAtMs || performance.now())),
          { result: "cancelled", codec: "h264", scope: "frame" },
        );
      }
      this._flushClientMetrics();
      this._clearMetricFlushTimer();
      this._stopMonitoring();
      this._clearDrainTimer();
      const socket = this.socket;
      this.socket = null;
      try { socket?.close?.(1000, String(reason).slice(0, 120)); } catch {}
      try { this.video.pause?.(); } catch {}
      this._setPlaybackRate(1);
      this.video.removeAttribute("src");
      this.video.load?.();
      this.sourceBuffer = null;
      this.mediaSource = null;
      if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = "";
      this.appendQueue = [];
      this.appendQueueBytes = 0;
      this.activeAppendItem = null;
      this.pendingPayloadTimings = [];
      this.mediaBatches = [];
      this.controlSentEpochByEvent.clear();
      this.playable = false;
      this.finiteRequest = false;
      this.playbackStarted = false;
      this.lastPlaybackPolicyAt = 0;
      this.streamCompleteReceived = false;
      this.streamCompleteDetails = null;
      this.transportClosed = true;
      this.drainFinished = true;
      this.drainDetails = {};
      if (emitState && hadSession) this._setState("closed", { reason });
    }

    setUnavailable(reason = "Unavailable for this mode") {
      void this.close(reason, { emitState: false });
      this._setState("unavailable", { reason });
    }

    _queueMedia(value) {
      const data = value instanceof ArrayBuffer
        ? new Uint8Array(value)
        : value instanceof Uint8Array
          ? value
          : null;
      if (!data?.byteLength) return;
      const receivedAtMs = performance.now();
      const payloadTiming = this.pendingPayloadTimings.shift() || {};
      const serverSentEpochMs = Number(payloadTiming.serverSentEpochMs || 0);
      const webSocketDownlinkMs = serverSentEpochMs
        ? Math.max(
            0,
            Date.now() - serverSentEpochMs + this.lastBridgeClockOffsetMs,
          )
        : 0;
      this.bytesReceived += data.byteLength;
      this.appendQueue.push({
        data,
        receivedAtMs,
        webSocketDownlinkMs,
        payloadSequence: Number(payloadTiming.sequence || 0),
      });
      this.appendQueueBytes += data.byteLength;
      if (webSocketDownlinkMs > 0) {
        this._queueMetric("client_downlink", webSocketDownlinkMs, {
          codec: "h264",
          scope: "frame",
        });
        this._emitStats({ lastWebSocketDownlinkMs: webSocketDownlinkMs });
      }
      this._appendNext();
    }

    _appendNext() {
      const sourceBuffer = this.sourceBuffer;
      if (!sourceBuffer || sourceBuffer.updating || !this.appendQueue.length) return;
      const item = this.appendQueue.shift();
      this.appendQueueBytes -= item.data.byteLength;
      item.appendStartedAtMs = performance.now();
      this.activeAppendItem = item;
      try {
        sourceBuffer.appendBuffer(item.data);
        this._queueMetric(
          "client_receive_queue",
          Math.max(0, item.appendStartedAtMs - item.receivedAtMs),
          { codec: "h264", scope: "frame" },
        );
      } catch (error) {
        this._queueMetric(
          "client_receive_queue",
          Math.max(0, performance.now() - item.receivedAtMs),
          {
            result: resultFromError(error),
            codec: "h264",
            scope: "frame",
          },
        );
        this.activeAppendItem = null;
        this._fail(error);
      }
    }

    _handleAppendEnd() {
      const item = this.activeAppendItem;
      if (item) {
        this.activeAppendItem = null;
        const completedAtMs = performance.now();
        const appendedMetadata = this.mediaBatches.find(
          (metadata) => !metadata.appendCompletedAtMs,
        );
        if (appendedMetadata) appendedMetadata.appendCompletedAtMs = completedAtMs;
        this._queueMetric(
          "client_video_decode",
          Math.max(0, completedAtMs - item.appendStartedAtMs),
          { codec: "h264", scope: "frame" },
        );
        this._emitStats({
          lastMseQueueMs: Math.max(0, item.appendStartedAtMs - item.receivedAtMs),
          lastMseAppendMs: Math.max(0, completedAtMs - item.appendStartedAtMs),
        });
      }
      // updateend also fires for SourceBuffer.remove(). In that case there is
      // no active append item, but a pending finite drain must still advance.
      this._maybeEndMediaStream();
    }

    _beginDrain(details = {}) {
      if (this.expectedClose || this.drainFinished || this.state === "error") return;
      this.playbackStarted = true;
      this.lastPlaybackPolicyAt = 0;
      this._setPlaybackRate(1);
      this.transportClosed = this.transportClosed || details.transportClosed === true;
      this.drainDetails = { ...this.drainDetails, ...details };
      if (this.state !== "draining") {
        this._setState("draining", {
          ...this.drainDetails,
          complete: this.streamCompleteReceived,
        });
      }
      this._maybeEndMediaStream();
    }

    _maybeEndMediaStream() {
      if (this.state !== "draining" || this.drainFinished) return;
      if (this.appendQueue.length || this.activeAppendItem || this.sourceBuffer?.updating) return;
      const mediaSource = this.mediaSource;
      if (mediaSource?.readyState === "open") {
        try {
          mediaSource.endOfStream();
        } catch (error) {
          this._fail(error);
          return;
        }
      }
      const buffered = this.sourceBuffer?.buffered;
      const end = buffered?.length ? buffered.end(buffered.length - 1) : 0;
      const current = Number(this.video.currentTime || 0);
      if (!buffered?.length || this.video.ended || current >= end - 0.01) {
        this._completeDrain();
        return;
      }
      // Keep the MediaSource and video element alive until the browser presents
      // every already-appended frame.  The `ended` event finalizes the session.
      this._scheduleDrainTimeout(end, current);
      this._attemptDrainPlayback();
    }

    _scheduleDrainTimeout(end, current) {
      if (this.drainTimer || this.drainFinished || this.state !== "draining") return;
      const remainingMs = Math.max(0, (end - current) * 1000);
      const waitMs = Math.min(
        this.drainPlaybackMaxWaitMs,
        Math.max(this.drainPlaybackGraceMs, remainingMs + this.drainPlaybackGraceMs),
      );
      const generation = this.generation;
      this.drainTimer = global.setTimeout(() => {
        this.drainTimer = 0;
        if (generation !== this.generation || this.state !== "draining") return;
        const buffered = this.sourceBuffer?.buffered;
        const latestEnd = buffered?.length ? buffered.end(buffered.length - 1) : 0;
        const latestCurrent = Number(this.video.currentTime || 0);
        const remainingPlaybackMs = Math.max(0, (latestEnd - latestCurrent) * 1000);
        this._completeDrain({
          drainTimedOut: remainingPlaybackMs > 10,
          remainingPlaybackMs,
        });
      }, waitMs);
    }

    _attemptDrainPlayback() {
      let result;
      try {
        result = this.video.play?.();
      } catch (error) {
        this._noteDrainPlaybackFailure(error);
        return;
      }
      if (result && typeof result.catch === "function") {
        void result.catch((error) => this._noteDrainPlaybackFailure(error));
      }
    }

    _noteDrainPlaybackFailure(error) {
      if (this.state !== "draining" || this.drainFinished) return;
      const playbackError = error?.message || String(error || "video.play() rejected");
      this.drainDetails = { ...this.drainDetails, playbackError };
      this._emitStats({ drainPlaybackError: playbackError });
      if (this.drainPlayFailureReported) return;
      this.drainPlayFailureReported = true;
      this._setState("draining", {
        ...this.drainDetails,
        complete: this.streamCompleteReceived,
      });
    }

    _clearDrainTimer() {
      if (this.drainTimer) global.clearTimeout(this.drainTimer);
      this.drainTimer = 0;
    }

    _stopMonitoring() {
      if (this.statsTimer) global.clearInterval(this.statsTimer);
      this.statsTimer = 0;
      if (this.frameCallback && typeof this.video.cancelVideoFrameCallback === "function") {
        try { this.video.cancelVideoFrameCallback(this.frameCallback); } catch {}
      }
      this.frameCallback = 0;
    }

    _completeDrain(details = {}) {
      if (this.state !== "draining" || this.drainFinished) return;
      this.drainFinished = true;
      this._clearDrainTimer();
      this._stopMonitoring();
      this._setState("closed", {
        ...this.drainDetails,
        ...details,
        complete: this.streamCompleteReceived,
        terminal: this.streamCompleteDetails,
      });
    }

    _maintainLiveEdge() {
      const sourceBuffer = this.sourceBuffer;
      if (!sourceBuffer || !sourceBuffer.buffered?.length) return;
      const end = sourceBuffer.buffered.end(sourceBuffer.buffered.length - 1);
      const current = Number(this.video.currentTime || 0);
      const leadMs = Math.max(0, (end - current) * 1000);
      let playbackBufferMs = leadMs;
      const draining = this.state === "draining";
      const preservingTail = this.finiteRequest || draining;
      const smoothTimeline = this.playbackMode === "smooth_timeline";
      const fullTimeline = this.playbackMode === "timeline";
      const hidden = this._isDocumentHidden();

      if (hidden) {
        this.lastPlaybackPolicyAt = 0;
        this._setPlaybackRate(1);
        this._emitPlaybackStats(playbackBufferMs, {
          playbackBuffering: smoothTimeline && !this.playbackStarted,
          playbackPolicySuspended: true,
        });
        return;
      }

      if (
        !preservingTail
        && !smoothTimeline
        && !fullTimeline
        && leadMs > this.liveEdgeSeekThresholdMs
      ) {
        const liveEdge = Math.max(0, end - this.liveEdgeTargetMs / 1000);
        this.video.currentTime = liveEdge;
        playbackBufferMs = Math.max(0, (end - liveEdge) * 1000);
      }

      if (smoothTimeline && !this.playbackStarted && !draining) {
        if (leadMs < this.smoothTimelineStartupBufferMs) {
          try { this.video.pause?.(); } catch {}
          this._setPlaybackRate(1);
          this._emitPlaybackStats(playbackBufferMs, { playbackBuffering: true });
          return;
        }
        this.playbackStarted = true;
        this.lastPlaybackPolicyAt = 0;
      }

      if (draining || fullTimeline || !smoothTimeline) {
        this._setPlaybackRate(1);
      } else {
        this._updateSmoothTimelinePlaybackRate(leadMs);
      }

      if (!preservingTail && !sourceBuffer.updating && current > 6) {
        const historyToKeepSeconds = smoothTimeline || fullTimeline ? 10 : 3;
        const removeEnd = current - historyToKeepSeconds;
        if (removeEnd > 0 && sourceBuffer.buffered.start(0) < removeEnd) {
          try { sourceBuffer.remove(0, removeEnd); } catch {}
        }
      }
      this._attemptPlayback();
      this._markPlayable();
      this._emitPlaybackStats(playbackBufferMs, {
        playbackBuffering: false,
      });
    }

    _updateSmoothTimelinePlaybackRate(leadMs) {
      const targetLagMs = this.smoothTimelineTargetLagMs;
      const maxLagMs = this.smoothTimelineMaxLagMs;
      const errorRatio = (leadMs - targetLagMs) / Math.max(250, targetLagMs);
      let desiredRate = 1 + errorRatio * this.smoothTimelinePlaybackRateGain;
      if (leadMs >= maxLagMs) desiredRate = this.smoothTimelinePlaybackRateMax;
      desiredRate = clamp(
        desiredRate,
        this.smoothTimelinePlaybackRateMin,
        this.smoothTimelinePlaybackRateMax,
      );
      const now = performance.now();
      const elapsedSeconds = this.lastPlaybackPolicyAt
        ? clamp((now - this.lastPlaybackPolicyAt) / 1000, 0, 0.5)
        : 0.25;
      this.lastPlaybackPolicyAt = now;
      const currentRate = Number(this.video.playbackRate || 1);
      const maxStep = this.smoothTimelinePlaybackRateSlewPerSecond * elapsedSeconds;
      this._setPlaybackRate(clamp(
        desiredRate,
        currentRate - maxStep,
        currentRate + maxStep,
      ));
    }

    _setPlaybackRate(rate) {
      try {
        this.video.playbackRate = clamp(Number(rate) || 1, 0.5, 2.5);
      } catch {}
    }

    _attemptPlayback() {
      try {
        const result = this.video.play?.();
        if (result && typeof result.catch === "function") void result.catch(() => {});
      } catch {}
    }

    _isDocumentHidden() {
      return this.documentRef?.hidden === true;
    }

    _emitPlaybackStats(playbackBufferMs, extra = {}) {
      this._emitStats({
        mseBufferMs: playbackBufferMs,
        playbackBufferMs,
        appendQueueBytes: this.appendQueueBytes,
        playbackMode: this.playbackMode,
        playbackRate: Number(this.video.playbackRate || 1),
        playbackTargetLagMs: this.smoothTimelineTargetLagMs,
        playbackMaxLagMs: this.smoothTimelineMaxLagMs,
        finitePlayback: this.finiteRequest,
        playbackPolicySuspended: false,
        ...extra,
      });
    }

    _handleMetadata(event) {
      if (event.type === "stream_complete") {
        this.streamCompleteReceived = true;
        this.streamCompleteDetails = { ...event };
        this._beginDrain({
          reason: "generation complete",
          terminal: this.streamCompleteDetails,
        });
      } else if (event.type === "media_batch") {
        const receivedAt = performance.now();
        const frameCount = Math.max(1, Number(event.num_frames || 1));
        for (let index = 0; index < frameCount; index += 1) {
          this.deliverySamples.push(receivedAt);
          if (!event.repeated_frame) this.sourceSamples.push(receivedAt);
        }
        this.deliverySamples = this.deliverySamples.filter(
          (sample) => receivedAt - sample < 1000,
        );
        this.sourceSamples = this.sourceSamples.filter(
          (sample) => receivedAt - sample < 1000,
        );
        this.mediaBatches.push({
          sourceFrameIndex: Number(event.first_frame_index || 0),
          chunkIndex: Number(event.chunk_index || 0),
          eventId: Number(event.event_id || 0),
          frameBatchIndex: Number(event.frame_batch_index || 0),
          isFinalFrameBatch: Boolean(event.is_final_frame_batch),
          bridgeEncodedEpochMs: Number(event.bridge_encoded_epoch_ms || 0),
          bridgeEncodeStartedEpochMs: Number(
            event.bridge_encode_started_epoch_ms || 0,
          ),
          bridgeQueueMs: Number(event.bridge_queue_ms || 0),
          bridgeEncoderFeedMs: Number(event.bridge_encoder_feed_ms || 0),
          metadataReceivedAtMs: performance.now(),
          appendCompletedAtMs: 0,
        });
        if (this.mediaBatches.length > 1024) this.mediaBatches.shift();
        this._emitStats({
          lastBridgeQueueMs: Number(event.bridge_queue_ms || 0),
          lastBridgeEncoderFeedMs: Number(event.bridge_encoder_feed_ms || 0),
          droppedFrames: Number(event.dropped_frames || 0),
          startupDroppedFrames: Number(event.startup_dropped_frames || 0),
          serverFps: this.sourceSamples.length,
          deliveryFps: this.deliverySamples.length,
          sourceFps: this.mediaFps,
        });
      } else if (event.type === "media_encode_timing") {
        const frameIndex = Number(event.first_frame_index || 0);
        const metadata = this.mediaBatches.find(
          (item) => Number(item.sourceFrameIndex || 0) === frameIndex,
        );
        if (metadata) {
          metadata.bridgeEncodedEpochMs = Number(
            event.bridge_encoded_epoch_ms || metadata.bridgeEncodedEpochMs || 0,
          );
          metadata.bridgeEncoderFeedMs = Number(
            event.bridge_encoder_feed_ms || 0,
          );
        }
        this._emitStats({
          lastBridgeEncoderFeedMs: Number(event.bridge_encoder_feed_ms || 0),
        });
      } else if (event.type === "media_payload") {
        this.pendingPayloadTimings.push({
          sequence: Number(event.sequence || 0),
          numBytes: Number(event.num_bytes || 0),
          serverSentEpochMs: Number(event.server_sent_epoch_ms || 0),
        });
        if (this.pendingPayloadTimings.length > 1024) {
          this.pendingPayloadTimings.shift();
        }
      } else if (event.type === "chunk_telemetry") {
        this._emitStats({ chunkTelemetry: { ...event } });
      } else if (event.type === "control_ack") {
        const clientSentEpochMs = Number(event.client_sent_epoch_ms || 0);
        const serverReceivedEpochMs = Number(
          event.server_received_epoch_ms || event.bridge_received_epoch_ms || 0,
        );
        const serverSentEpochMs = Number(event.server_sent_epoch_ms || 0);
        const clientReceivedEpochMs = Date.now();
        const serverProcessingMs = serverReceivedEpochMs && serverSentEpochMs
          ? Math.max(0, serverSentEpochMs - serverReceivedEpochMs)
          : 0;
        const roundTripMs = clientSentEpochMs
          ? Math.max(0, clientReceivedEpochMs - clientSentEpochMs)
          : 0;
        if (clientSentEpochMs && serverReceivedEpochMs && serverSentEpochMs) {
          this.lastBridgeClockOffsetMs = (
            (serverReceivedEpochMs - clientSentEpochMs)
            + (serverSentEpochMs - clientReceivedEpochMs)
          ) / 2;
        }
        const controlKind = String(event.kind || "");
        const isInteractiveControl = Number(event.event_id || 0) > 0
          && ["camera_actions", "prompt", "scene_cut"].includes(controlKind);
        if (clientSentEpochMs && isInteractiveControl) {
          this._queueMetric(
            "client_uplink",
            Math.max(0, (roundTripMs - serverProcessingMs) / 2),
            { codec: "h264", scope: "request" },
          );
        }
        this._emitStats({
          ...(clientSentEpochMs && isInteractiveControl
            ? { lastInputUplinkMs: Math.max(0, (roundTripMs - serverProcessingMs) / 2) }
            : {}),
          controlBridgeRoundTripMs: roundTripMs,
          bridgeClockOffsetMs: this.lastBridgeClockOffsetMs,
        });
      }
    }

    _startPresentedFrameMonitor() {
      if (typeof this.video.requestVideoFrameCallback !== "function") return;
      const generation = this.generation;
      const handle = (now, presentation = {}) => {
        if (generation !== this.generation || this.expectedClose) return;
        const mediaTime = Number(presentation.mediaTime);
        const targetFrameIndex = Number.isFinite(mediaTime)
          ? Math.max(0, Math.round(mediaTime * this.mediaFps))
          : this.presentedSequence;
        while (
          this.mediaBatches.length > 1
          && Number(this.mediaBatches[1].sourceFrameIndex || 0) <= targetFrameIndex
        ) {
          this.mediaBatches.shift();
        }
        const metadata = this.mediaBatches[0] || {};
        this.presentedSequence = targetFrameIndex + 1;
        const eventId = Number(metadata.eventId || 0);
        const isFirstForEvent = eventId > this.lastPresentedEventId;
        if (isFirstForEvent) this.lastPresentedEventId = eventId;
        this.framesPresented += 1;
        this.presentedSamples.push(Number(now || performance.now()));
        this.presentedSamples = this.presentedSamples.filter(
          (sample) => Number(now || performance.now()) - sample < 1000,
        );
        this.lastRenderedChunk = Number(metadata.chunkIndex ?? this.lastRenderedChunk ?? 0);
        const sentEventIds = Array.from(this.controlSentEpochByEvent.keys())
          .filter((id) => id <= eventId)
          .sort((left, right) => left - right);
        const sentEpochMs = sentEventIds.length
          ? this.controlSentEpochByEvent.get(sentEventIds[0])
          : 0;
        const stats = {
          frames: this.framesPresented,
          framesDecoded: this.framesPresented,
          lastChunk: this.lastRenderedChunk,
          lastPresentedMediaEventId: eventId,
          lastBridgeQueueMs: Number(metadata.bridgeQueueMs || 0),
          lastBridgeEncoderFeedMs: Number(metadata.bridgeEncoderFeedMs || 0),
          lastEncodeToPresentMs: metadata.bridgeEncodedEpochMs
            ? Math.max(
                0,
                Date.now() - metadata.bridgeEncodedEpochMs + this.lastBridgeClockOffsetMs,
              )
            : 0,
          lastPresentedAfterMetadataMs: metadata.metadataReceivedAtMs
            ? Math.max(0, Number(now || performance.now()) - metadata.metadataReceivedAtMs)
            : 0,
          ...(sentEpochMs && isFirstForEvent
            ? { lastPresentedControlToVideoMs: Math.max(0, Date.now() - sentEpochMs) }
            : {}),
        };
        this._emitStats(stats);
        if (metadata.appendCompletedAtMs || metadata.metadataReceivedAtMs) {
          this._queueMetric(
            "client_render_wait",
            Math.max(
              0,
              Number(now || performance.now())
                - Number(metadata.appendCompletedAtMs || metadata.metadataReceivedAtMs),
            ),
            { codec: "h264", scope: "frame" },
          );
        }
        this.onPresentedFrame({
          presentedAt: Number(now || 0),
          width: Number(presentation.width || this.video.videoWidth || 0),
          height: Number(presentation.height || this.video.videoHeight || 0),
          chunkIndex: this.lastRenderedChunk,
          eventId,
        });
        if (sentEpochMs && isFirstForEvent) {
          for (const id of sentEventIds) this.controlSentEpochByEvent.delete(id);
        }
        if (this.playbackAckEnabled) this._sendPlaybackAck();
        this.frameCallback = this.video.requestVideoFrameCallback(handle);
      };
      this.frameCallback = this.video.requestVideoFrameCallback(handle);
    }

    _sendPlaybackAck() {
      if (!this.socket || this.socket.readyState !== OPEN) return;
      const now = performance.now();
      if (now - Number(this.lastPlaybackAckAt || 0) < 50) return;
      this.lastPlaybackAckAt = now;
      this.socket.send(JSON.stringify({
        type: "event",
        kind: "playback_ack",
        trace_id: this.traceId,
        payload: {
          last_received_chunk: this.lastRenderedChunk,
          last_rendered_chunk: this.lastRenderedChunk,
          last_rendered_event_id: this.lastPresentedEventId,
          playable: this.playable,
        },
      }));
    }

    _startStats() {
      if (this.statsTimer) global.clearInterval(this.statsTimer);
      const sample = () => {
        const now = performance.now();
        this.deliverySamples = this.deliverySamples.filter(
          (receivedAt) => now - receivedAt < 1000,
        );
        this.sourceSamples = this.sourceSamples.filter(
          (receivedAt) => now - receivedAt < 1000,
        );
        let receiveMbps = 0;
        if (this.networkSample && now > this.networkSample.at) {
          receiveMbps = Math.max(
            0,
            (this.bytesReceived - this.networkSample.bytes) * 8
              / ((now - this.networkSample.at) / 1000) / 1_000_000,
          );
        }
        this.networkSample = { at: now, bytes: this.bytesReceived };
        const buffered = this.sourceBuffer?.buffered;
        const end = buffered?.length ? buffered.end(buffered.length - 1) : 0;
        const bufferMs = Math.max(0, (end - Number(this.video.currentTime || 0)) * 1000);
        this._emitStats({
          bytesReceived: this.bytesReceived,
          bytes: this.bytesReceived,
          receiveMbps,
          bufferMs,
          playbackBufferMs: bufferMs,
          appendQueueBytes: this.appendQueueBytes,
          queueFrames: this.mediaBatches.length,
          renderFps: this.presentedSamples.length,
          sourceFps: this.mediaFps,
          serverFps: this.sourceSamples.length,
          deliveryFps: this.deliverySamples.length,
          protocol: "websocket",
          codec: "h264",
          playbackMode: this.playbackMode,
          playbackRate: Number(this.video.playbackRate || 1),
          finitePlayback: this.finiteRequest,
        });
      };
      sample();
      this.statsTimer = global.setInterval(sample, 1000);
    }

    _markPlayable() {
      if (this.playable || this.drainFinished || this.expectedClose) return;
      if (
        this.playbackMode === "smooth_timeline"
        && !this.playbackStarted
        && this.state !== "draining"
      ) return;
      if (Number(this.video.readyState || 0) < 2) return;
      if (!Number(this.video.videoWidth || 0) || !Number(this.video.videoHeight || 0)) return;
      this.playable = true;
      this.video.hidden = false;
      const details = {
        codec: "h264",
        protocol: "websocket",
        width: this.video.videoWidth,
        height: this.video.videoHeight,
      };
      // A short finite stream can finish transport/append before loadeddata.
      // Mark it playable without overwriting the terminal draining state.
      if (this.state !== "draining") this._setState("live", details);
      this.onPlayable(details);
      this._startPresentedFrameMonitor();
      this._startStats();
    }

    _emitStats(partial) {
      this.lastStats = { ...this.lastStats, ...partial };
      this.onStats({ ...this.lastStats });
    }

    _setState(state, details = {}) {
      this.state = state;
      if (this.root) this.root.dataset.sessionState = state;
      if (this.overlay?.style) {
        this.overlay.style.display = state === "connecting" && !this.playable ? "grid" : "none";
      }
      this.onState(state, details);
    }

    _fail(error) {
      if (this.expectedClose) return;
      this._flushClientMetrics();
      this._clearMetricFlushTimer();
      this.drainFinished = true;
      this._clearDrainTimer();
      this._stopMonitoring();
      this._setPlaybackRate(1);
      this._setState("error", { message: error.message || String(error) });
      this.onError(error);
    }
  }

  global.H264WebSocketSession = H264WebSocketSession;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { H264WebSocketSession };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
