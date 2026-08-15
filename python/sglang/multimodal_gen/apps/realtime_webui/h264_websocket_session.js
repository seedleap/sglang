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

  class H264WebSocketSession {
    constructor({
      video,
      overlay = null,
      root = null,
      endpoint = "/api/h264ws",
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
      this.socket = null;
      this.mediaSource = null;
      this.sourceBuffer = null;
      this.objectUrl = "";
      this.generation = 0;
      this.state = "idle";
      this.playable = false;
      this.expectedClose = false;
      this.appendQueue = [];
      this.appendQueueBytes = 0;
      this.mediaBatches = [];
      this.presentedSequence = 0;
      this.lastPresentedEventId = 0;
      this.controlSentEpochByEvent = new Map();
      this.lastBridgeClockOffsetMs = 0;
      this.bytesReceived = 0;
      this.framesPresented = 0;
      this.presentedSamples = [];
      this.networkSample = null;
      this.statsTimer = 0;
      this.frameCallback = 0;
      this.lastRenderedChunk = null;
      this.traceId = "";
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
      this.traceId = String(init.trace_id || "");
      this.playbackAckEnabled = init.playback_ack_enabled === true;
      this.appendQueue = [];
      this.appendQueueBytes = 0;
      this.mediaBatches = [];
      this.presentedSequence = 0;
      this.lastPresentedEventId = 0;
      this.controlSentEpochByEvent.clear();
      this.bytesReceived = 0;
      this.framesPresented = 0;
      this.presentedSamples = [];
      this.networkSample = null;
      this.lastStats = {};
      this.lastRenderedChunk = null;
      this._setState("connecting", { codec: "h264", protocol: "websocket" });

      const mediaSource = new this.MediaSourceImpl();
      this.mediaSource = mediaSource;
      this.objectUrl = URL.createObjectURL(mediaSource);
      this.video.srcObject = null;
      this.video.src = this.objectUrl;
      this.video.hidden = false;
      await waitForSourceOpen(mediaSource, this.startupTimeoutMs);
      if (generation !== this.generation) throw new Error("H.264 WebSocket startup canceled");
      const sourceBuffer = mediaSource.addSourceBuffer(MIME_TYPE);
      this.sourceBuffer = sourceBuffer;
      sourceBuffer.addEventListener("updateend", () => {
        if (generation !== this.generation) return;
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
          socket.send(JSON.stringify({ ...init, first_frame: firstFrame }));
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
          if (!this.expectedClose) {
            const error = new Error(
              `H.264 WebSocket closed (${event.code}): ${event.reason || "unknown"}`,
            );
            if (!settled) reject(error);
            this._fail(error);
          }
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
      this.socket.send(JSON.stringify({ ...envelope, trace_id: this.traceId }));
      return true;
    }

    async close(reason = "H.264 WebSocket session closed", { emitState = true } = {}) {
      const hadSession = this.active;
      this.expectedClose = true;
      this.generation += 1;
      if (this.statsTimer) global.clearInterval(this.statsTimer);
      this.statsTimer = 0;
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
      this.mediaBatches = [];
      this.controlSentEpochByEvent.clear();
      this.playable = false;
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
      this.bytesReceived += data.byteLength;
      this.appendQueue.push(data);
      this.appendQueueBytes += data.byteLength;
      this._appendNext();
    }

    _appendNext() {
      const sourceBuffer = this.sourceBuffer;
      if (!sourceBuffer || sourceBuffer.updating || !this.appendQueue.length) return;
      const data = this.appendQueue.shift();
      this.appendQueueBytes -= data.byteLength;
      try {
        sourceBuffer.appendBuffer(data);
      } catch (error) {
        this._fail(error);
      }
    }

    _maintainLiveEdge() {
      const sourceBuffer = this.sourceBuffer;
      if (!sourceBuffer || !sourceBuffer.buffered?.length) return;
      const end = sourceBuffer.buffered.end(sourceBuffer.buffered.length - 1);
      const current = Number(this.video.currentTime || 0);
      const leadMs = Math.max(0, (end - current) * 1000);
      if (leadMs > this.liveEdgeSeekThresholdMs) {
        this.video.currentTime = Math.max(0, end - this.liveEdgeTargetMs / 1000);
      }
      if (!sourceBuffer.updating && current > 6) {
        const removeEnd = current - 3;
        if (removeEnd > 0 && sourceBuffer.buffered.start(0) < removeEnd) {
          try { sourceBuffer.remove(0, removeEnd); } catch {}
        }
      }
      void this.video.play?.().catch(() => {});
      this._markPlayable();
      this._emitStats({ mseBufferMs: leadMs, appendQueueBytes: this.appendQueueBytes });
    }

    _handleMetadata(event) {
      if (event.type === "media_batch") {
        this.mediaBatches.push({
          chunkIndex: Number(event.chunk_index || 0),
          eventId: Number(event.event_id || 0),
          frameBatchIndex: Number(event.frame_batch_index || 0),
          isFinalFrameBatch: Boolean(event.is_final_frame_batch),
          bridgeEncodedEpochMs: Number(event.bridge_encoded_epoch_ms || 0),
          bridgeQueueMs: Number(event.bridge_queue_ms || 0),
          bridgeEncoderFeedMs: Number(event.bridge_encoder_feed_ms || 0),
          metadataReceivedAtMs: performance.now(),
        });
        if (this.mediaBatches.length > 1024) this.mediaBatches.shift();
        this._emitStats({
          lastBridgeQueueMs: Number(event.bridge_queue_ms || 0),
          lastBridgeEncoderFeedMs: Number(event.bridge_encoder_feed_ms || 0),
          droppedFrames: Number(event.dropped_frames || 0),
        });
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
        this._emitStats({
          lastInputUplinkMs: Math.max(0, (roundTripMs - serverProcessingMs) / 2),
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
        const metadata = this.mediaBatches.shift() || {};
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
          lastPresentedTransportMs: metadata.bridgeEncodedEpochMs
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
          queueFrames: this.mediaBatches.length,
          renderFps: this.presentedSamples.length,
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
        this.overlay.style.display = state === "connecting" && !this.playable ? "grid" : "none";
      }
      this.onState(state, details);
    }

    _fail(error) {
      if (this.expectedClose) return;
      this._setState("error", { message: error.message || String(error) });
      this.onError(error);
    }
  }

  global.H264WebSocketSession = H264WebSocketSession;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { H264WebSocketSession };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
