(function (global) {
  const OPEN = 1;
  const MIME_TYPE = 'video/mp4; codecs="avc1.4D401F"';

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

  function closeDetails(event) {
    const code = Number(event?.code || 0);
    const reason = String(event?.reason || "");
    return {
      code,
      reason,
      wasClean: Boolean(event?.wasClean),
      message: `H.264 WebSocket closed (${code || "unknown"}): ${reason || "unknown"}`,
    };
  }

  function isTerminalCloseReason(reason) {
    const value = String(reason || "").toLowerCase();
    return [
      "maximum session lifetime reached",
      "generation complete",
      "session idle timeout",
      "session closed",
      "session closed by client",
      "disabled for request",
    ].some((token) => value.includes(token));
  }

  function resolveEndpoint(configuredEndpoint, location = global.location) {
    const endpoint = String(configuredEndpoint || "").trim();
    const protocol = location?.protocol === "https:" ? "wss:" : "ws:";
    const host = location?.host || "localhost";
    if (endpoint.startsWith("ws")) return endpoint;
    try {
      const url = new URL(endpoint || "/", `${protocol}//${host}/`);
      if (url.protocol === "https:") url.protocol = "wss:";
      if (url.protocol === "http:") url.protocol = "ws:";
      if (url.protocol === "ws:" || url.protocol === "wss:") return url.toString();
    } catch {
      // Fall through to the simple path join below.
    }
    return `${protocol}//${host}${endpoint.startsWith("/") ? "" : "/"}${endpoint}`;
  }

  function firstNumeric(...values) {
    for (const value of values) {
      if (value === undefined || value === null || value === "") continue;
      const number = Number(value);
      if (Number.isFinite(number)) return number;
    }
    return 0;
  }

  class H264WebSocketSession {
    constructor({
      video,
      overlay = null,
      root = null,
      endpoint = "/backends/minwm/v1/realtime_video/generate",
      startupTimeoutMs = 30000,
      liveEdgeTargetMs = 80,
      liveEdgeSeekThresholdMs = 420,
      onState = () => {},
      onPlayable = () => {},
      onPresentedFrame = () => {},
      onStats = () => {},
      onError = () => {},
      WebSocketImpl = global.WebSocket,
      MediaSourceImpl = global.MediaSource,
      metricFlushMs = 250,
      maxMetricBatch = 32,
      directGateway = true,
      packMessage = null,
      unpackMessage = null,
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
      this.onState = onState;
      this.onPlayable = onPlayable;
      this.onPresentedFrame = onPresentedFrame;
      this.onStats = onStats;
      this.onError = onError;
      this.WebSocketImpl = WebSocketImpl;
      this.MediaSourceImpl = MediaSourceImpl;
      this.metricFlushMs = Math.max(0, Number(metricFlushMs) || 0);
      this.maxMetricBatch = Math.max(1, Number(maxMetricBatch) || 32);
      this.directGateway = directGateway === true;
      this.packMessage = packMessage;
      this.unpackMessage = unpackMessage;
      if (this.directGateway && (!this.packMessage || !this.unpackMessage)) {
        throw new Error("Direct H.264 Gateway mode requires MessagePack codecs");
      }
      this.socket = null;
      this.mediaSource = null;
      this.sourceBuffer = null;
      this.objectUrl = "";
      this.generation = 0;
      this.state = "idle";
      this.playable = false;
      this.startupConnected = false;
      this.expectedClose = false;
      this.appendQueue = [];
      this.appendQueueBytes = 0;
      this.activeAppendItem = null;
      this.pendingPayloadTimings = [];
      this.mediaBatches = [];
      this.presentedSequence = 0;
      this.lastPresentedEventId = 0;
      this.controlSentEpochByEvent = new Map();
      this.lastH264ClockOffsetMs = 0;
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
      this.playbackAckEnabled = false;
      this.lastStats = {};
      this.handlePlayable = () => this._markPlayable();
      for (const name of ["loadeddata", "playing", "resize", "timeupdate"]) {
        this.video.addEventListener(name, this.handlePlayable);
      }
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
      return { ...this.lastStats, state: this.state, connected: this.connected };
    }

    async connect(init, endpointOverride = "") {
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
      this.startupConnected = false;
      this.traceId = String(init.trace_id || "");
      this.mediaFps = Math.max(1, Number(init.fps || 24));
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

      const firstFrame = this.directGateway
        ? (init.first_frame instanceof ArrayBuffer
          ? new Uint8Array(init.first_frame)
          : init.first_frame)
        : await bytesToDataUrl(init.first_frame);
      return new Promise((resolve, reject) => {
        const endpoint = resolveEndpoint(endpointOverride || this.endpoint);
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
          const directInit = this.directGateway
            // JPEG is an opt-in sentinel understood by the dedicated VAE
            // deployment. The VAE rewrites it to raw locally, then emits
            // H.264/fMP4; no JPEG or RGB crosses the network.
            ? { ...init, first_frame: firstFrame, realtime_output_format: "jpeg" }
            : { ...init, first_frame: firstFrame };
          const payload = this.directGateway
            ? this.packMessage(directInit)
            : JSON.stringify(directInit);
          socket.send(payload);
          this._queueMetric(
            "client_serialize_queue",
            performance.now() - serializeStartedAt,
            { codec: "h264", scope: "request" },
          );
        };
        socket.onmessage = (message) => {
          if (generation !== this.generation) return;
          if (!this.directGateway && typeof message.data !== "string") {
            this._queueMedia(message.data);
            return;
          }
          try {
            const event = this.directGateway
              ? this.unpackMessage(new Uint8Array(message.data))
              : JSON.parse(message.data);
            if (
              (event.type === "status" && event.state === "connected")
              || event.type === "session_ready"
              || (this.directGateway && event.type === "media_init")
            ) {
              this.startupConnected = true;
              if (!settled) {
                settled = true;
                global.clearTimeout(timer);
                resolve(event);
              }
              return;
            }
            if (event.type === "error") {
              throw new Error(
                event.message || event.content || event.reason || "H.264 WebSocket error",
              );
            }
            if (event.type === "media_payload" && event.payload) {
              this._handleMetadata(event);
              this._queueMedia(event.payload);
              return;
            }
            this._handleMetadata(event);
          } catch (error) {
            const phase = settled ? "runtime" : "startup";
            if (!settled) {
              settled = true;
              global.clearTimeout(timer);
              reject(error);
            }
            error.h264Phase = phase;
            this._fail(error, {
              phase,
              recoverable: phase === "runtime",
            });
          }
        };
        socket.onerror = () => {
          if (generation !== this.generation) return;
          const error = new Error("H.264 WebSocket transport error");
          error.h264Phase = settled ? "runtime" : "startup";
          if (!settled) {
            settled = true;
            global.clearTimeout(timer);
            reject(error);
          }
          this._fail(error, { phase: error.h264Phase, recoverable: error.h264Phase === "runtime" });
        };
        socket.onclose = (event) => {
          if (generation !== this.generation) return;
          global.clearTimeout(timer);
          settled = this._handleSocketClose(event, { settled, reject });
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
      const outgoing = { ...envelope, trace_id: this.traceId };
      const payload = this.directGateway
        ? this.packMessage(outgoing)
        : JSON.stringify(outgoing);
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
        this.socket.send(
          this.directGateway ? this.packMessage(payload) : JSON.stringify(payload),
        );
      } catch {
        this.metricBuffer.unshift(...events);
      }
    }

    _clearMetricFlushTimer() {
      if (!this.metricFlushTimer) return;
      global.clearTimeout(this.metricFlushTimer);
      this.metricFlushTimer = 0;
    }

    _stopTimersAndFlushMetrics() {
      if (this.statsTimer) global.clearInterval(this.statsTimer);
      this.statsTimer = 0;
      this._flushClientMetrics();
      this._clearMetricFlushTimer();
    }

    _handleSocketClose(event, { settled = true, reject = () => {} } = {}) {
      const details = closeDetails(event);
      this.socket = null;
      if (this.expectedClose) return settled;

      if (settled && isTerminalCloseReason(details.reason)) {
        this._stopTimersAndFlushMetrics();
        this._setState("closed", details);
        return true;
      }

      const error = new Error(details.message);
      error.h264CloseCode = details.code;
      error.h264CloseReason = details.reason;
      error.h264WasClean = details.wasClean;
      error.h264Phase = settled ? "runtime" : "startup";
      if (!settled) {
        reject(error);
      }
      this._fail(error, {
        phase: error.h264Phase,
        recoverable: error.h264Phase === "runtime",
        ...details,
      });
      return true;
    }

    async close(reason = "H.264 WebSocket session closed", { emitState = true } = {}) {
      const hadSession = this.active;
      this.expectedClose = true;
      this.generation += 1;
      if (this.statsTimer) global.clearInterval(this.statsTimer);
      this.statsTimer = 0;
      for (const item of this.appendQueue) {
        this._queueMetric(
          "client_receive_queue",
          Math.max(0, performance.now() - Number(item.receivedAtMs || performance.now())),
          { result: "cancelled", codec: "h264", scope: "frame" },
        );
      }
      this._flushClientMetrics();
      this._clearMetricFlushTimer();
      if (this.frameCallback && typeof this.video.cancelVideoFrameCallback === "function") {
        try { this.video.cancelVideoFrameCallback(this.frameCallback); } catch {}
      }
      this.frameCallback = 0;
      const socket = this.socket;
      this.socket = null;
      try { socket?.close?.(1000, String(reason).slice(0, 120)); } catch {}
      try { this.video.pause?.(); } catch {}
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
      this.startupConnected = false;
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
            Date.now() - serverSentEpochMs + this.lastH264ClockOffsetMs,
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
      if (!item) return;
      this.activeAppendItem = null;
      const completedAtMs = performance.now();
      const appendedMetadata = this.mediaBatches.find((metadata) => !metadata.appendCompletedAtMs);
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

    _maintainLiveEdge() {
      const sourceBuffer = this.sourceBuffer;
      if (!sourceBuffer || !sourceBuffer.buffered?.length) return;
      const end = sourceBuffer.buffered.end(sourceBuffer.buffered.length - 1);
      const current = Number(this.video.currentTime || 0);
      const leadMs = Math.max(0, (end - current) * 1000);
      let playbackBufferMs = leadMs;
      if (leadMs > this.liveEdgeSeekThresholdMs) {
        const liveEdge = Math.max(0, end - this.liveEdgeTargetMs / 1000);
        this.video.currentTime = liveEdge;
        playbackBufferMs = Math.max(0, (end - liveEdge) * 1000);
      }
      if (!sourceBuffer.updating && current > 6) {
        const removeEnd = current - 3;
        if (removeEnd > 0 && sourceBuffer.buffered.start(0) < removeEnd) {
          try { sourceBuffer.remove(0, removeEnd); } catch {}
        }
      }
      void this.video.play?.().catch(() => {});
      this._markPlayable();
      this._emitStats({
        mseBufferMs: playbackBufferMs,
        playbackBufferMs,
        appendQueueBytes: this.appendQueueBytes,
      });
    }

    _handleMetadata(event) {
      if (event.type === "media_batch") {
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
          h264EncodedEpochMs: firstNumeric(
            event.h264_encoded_epoch_ms,
            event.bridge_encoded_epoch_ms,
          ),
          h264EncodeStartedEpochMs: firstNumeric(
            event.h264_encode_started_epoch_ms,
            event.bridge_encode_started_epoch_ms,
          ),
          h264QueueMs: firstNumeric(event.h264_queue_ms, event.bridge_queue_ms),
          h264EncoderFeedMs: firstNumeric(
            event.h264_encoder_feed_ms,
            event.bridge_encoder_feed_ms,
          ),
          metadataReceivedAtMs: performance.now(),
          appendCompletedAtMs: 0,
        });
        if (this.mediaBatches.length > 1024) this.mediaBatches.shift();
        this._emitStats({
          lastH264QueueMs: firstNumeric(event.h264_queue_ms, event.bridge_queue_ms),
          lastH264EncoderFeedMs: firstNumeric(
            event.h264_encoder_feed_ms,
            event.bridge_encoder_feed_ms,
          ),
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
          metadata.h264EncodedEpochMs = firstNumeric(
            event.h264_encoded_epoch_ms,
            event.bridge_encoded_epoch_ms,
            metadata.h264EncodedEpochMs,
          );
          metadata.h264EncoderFeedMs = firstNumeric(
            event.h264_encoder_feed_ms,
            event.bridge_encoder_feed_ms,
          );
        }
        this._emitStats({
          lastH264EncoderFeedMs: firstNumeric(
            event.h264_encoder_feed_ms,
            event.bridge_encoder_feed_ms,
          ),
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
          this.lastH264ClockOffsetMs = (
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
          controlServerRoundTripMs: roundTripMs,
          h264ClockOffsetMs: this.lastH264ClockOffsetMs,
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
          lastH264QueueMs: Number(metadata.h264QueueMs || 0),
          lastH264EncoderFeedMs: Number(metadata.h264EncoderFeedMs || 0),
          lastEncodeToPresentMs: metadata.h264EncodedEpochMs
            ? Math.max(
                0,
                Date.now() - metadata.h264EncodedEpochMs + this.lastH264ClockOffsetMs,
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
      const acknowledgement = {
        type: "event",
        kind: "playback_ack",
        trace_id: this.traceId,
        payload: {
          last_received_chunk: this.lastRenderedChunk,
          last_rendered_chunk: this.lastRenderedChunk,
          last_rendered_event_id: this.lastPresentedEventId,
          playable: this.playable,
        },
      };
      this.socket.send(
        this.directGateway
          ? this.packMessage(acknowledgement)
          : JSON.stringify(acknowledgement),
      );
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
        });
      };
      sample();
      this.statsTimer = global.setInterval(sample, 1000);
    }

    _markPlayable() {
      if (this.playable) return;
      if (Number(this.video.readyState || 0) < 2) return;
      if (!Number(this.video.videoWidth || 0) || !Number(this.video.videoHeight || 0)) return;
      this.playable = true;
      this.video.hidden = false;
      this._setState("live", {
        codec: "h264",
        protocol: "websocket",
        width: this.video.videoWidth,
        height: this.video.videoHeight,
      });
      this.onPlayable({
        codec: "h264",
        protocol: "websocket",
        width: this.video.videoWidth,
        height: this.video.videoHeight,
      });
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
        this.overlay.style.display = ["connecting", "recovering"].includes(state) && !this.playable
          ? "grid"
          : "none";
      }
      this.onState(state, details);
    }

    _fail(error, details = {}) {
      if (this.expectedClose) return;
      this._stopTimersAndFlushMetrics();
      const phase = details.phase || (this.startupConnected ? "runtime" : "startup");
      const recoverable = details.recoverable === true || phase === "runtime";
      this._setState(recoverable ? "recovering" : "error", {
        ...details,
        phase,
        message: error.message || String(error),
      });
      this.onError(error, { ...details, phase, recoverable });
    }
  }

  H264WebSocketSession.isTerminalCloseReason = isTerminalCloseReason;
  H264WebSocketSession.resolveEndpoint = resolveEndpoint;

  global.H264WebSocketSession = H264WebSocketSession;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { H264WebSocketSession };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
