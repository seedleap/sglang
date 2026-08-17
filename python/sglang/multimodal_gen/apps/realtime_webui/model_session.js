(function (global) {
  const DEFAULT_WORKER_URL = "./decoder_worker.js?v=realtime-production-gateway-v17";

  function closeFrame(frame) {
    const image = frame?.image;
    if (image && typeof ImageData !== "undefined" && image instanceof ImageData) return;
    image?.close?.();
  }

  function cameraActionHasActiveMotion(payload) {
    const transitions = payload?.transitions || [];
    const finalTransition = transitions[transitions.length - 1];
    return Array.isArray(finalTransition?.actions) && finalTransition.actions.length > 0;
  }

  class RealtimeModelSession {
    constructor({
      key,
      canvas,
      overlay = null,
      root = null,
      pack,
      unpack,
      WebSocketCtor = global.WebSocket,
      PlaybackController = global.RealtimePlaybackController,
      decodeBatch = null,
      workerUrl = DEFAULT_WORKER_URL,
      requestFrame = (callback) => global.requestAnimationFrame(callback),
      now = () => performance.now(),
      setTimer = (callback, delayMs) => global.setTimeout(callback, delayMs),
      clearTimer = (timer) => global.clearTimeout(timer),
      startupMinChunk = 0,
      startupTimeoutMs = 12000,
      stallTimeoutMs = 7000,
      maxDecodeQueueBatches = 4,
      onState = () => {},
      onStats = () => {},
      onFrame = () => {},
      onError = () => {},
    }) {
      this.key = key;
      this.canvas = canvas;
      this.overlay = overlay;
      this.root = root;
      this.pack = pack;
      this.unpack = unpack;
      this.WebSocketCtor = WebSocketCtor;
      this.decodeBatchOverride = decodeBatch;
      this.workerUrl = workerUrl;
      this.requestFrame = requestFrame;
      this.now = now;
      this.setTimer = setTimer;
      this.clearTimer = clearTimer;
      this.startupMinChunk = Math.max(0, Number(startupMinChunk) || 0);
      this.startupTimeoutMs = Math.max(0, Number(startupTimeoutMs) || 0);
      this.stallTimeoutMs = Math.max(0, Number(stallTimeoutMs) || 0);
      this.maxDecodeQueueBatches = Math.max(1, Number(maxDecodeQueueBatches) || 4);
      this.awaitingStableFrame = false;
      this.mediaWatchdogTimer = null;
      this.hasVisibleFrame = false;
      this.onState = onState;
      this.onStats = onStats;
      this.onFrame = onFrame;
      this.onError = onError;
      this.ctx = canvas.getContext("2d", { alpha: false });
      this.scratch = typeof document === "undefined" ? null : document.createElement("canvas");
      this.scratchCtx = this.scratch?.getContext("2d", { alpha: false }) || null;
      this.playback = new PlaybackController({
        mode: "adaptive",
        targetFps: 24,
        lowLatencyPlayback: true,
        holdForTargetLead: true,
        targetLeadChunkRatio: 0.7,
        minTargetLeadMs: 260,
        maxTargetLeadMs: 900,
        startLeadChunkRatio: 0.45,
        minStartLeadMs: 220,
        resumeLeadChunkRatio: 0.45,
        minResumeLeadMs: 180,
        maxResumeLeadMs: 650,
        maxDeliveryLeadBoostMs: 0,
        realtimeMaxBufferMs: 1100,
        realtimeMaxBufferChunks: 2,
        realtimeMaxFrameAgeMs: 1800,
      });
      this.socket = null;
      this.pendingHeader = null;
      this.traceId = "";
      this.epoch = 0;
      this.renderScheduled = false;
      this.renderSamples = [];
      this.decodeQueue = [];
      this.decodeInProgress = false;
      this.worker = null;
      this.decodeRequests = new Map();
      this.decodeRequestId = 1;
      this.playbackAckTimer = null;
      this.playbackAckEnabled = false;
      this.controlSentEpochByEvent = new Map();
      this.lastNetworkSample = null;
      this.stats = {
        frames: 0,
        bytes: 0,
        renderedFrames: 0,
        lastChunk: null,
        lastReceivedChunk: null,
        lastReceivedFrameBatchIndex: null,
        frameBatchGapCount: 0,
        lastEventId: 0,
        lastSentEventId: 0,
        lastAppliedEventId: 0,
        lastDecodeMs: 0,
        lastDisplayLagMs: 0,
        lastRenderedEventId: 0,
        lastControlToVideoMs: 0,
        lastDownlinkMs: 0,
        receiveMbps: 0,
        chunkTelemetry: null,
        lastInputUplinkMs: 0,
        controlRoundTripMs: 0,
        serverClockOffsetMs: 0,
      };
      this.setState("idle");
    }

    configure({ mode, targetFps, smoothTimelinePlaybackRateMax } = {}) {
      if (mode) this.playback.setMode?.(mode);
      if (targetFps) this.playback.setTargetFps?.(targetFps);
      if (smoothTimelinePlaybackRateMax) {
        this.playback.setSmoothTimelinePlaybackRateMax?.(smoothTimelinePlaybackRateMax);
      }
    }

    connect(init, url) {
      this.close("replace session", { notify: false });
      this.epoch += 1;
      const epoch = this.epoch;
      this.traceId = init.trace_id || "";
      this.playbackAckEnabled = init.playback_ack_enabled === true;
      this.pendingHeader = null;
      this.decodeQueue = [];
      this.decodeInProgress = false;
      this.renderSamples = [];
      this.controlSentEpochByEvent.clear();
      this.lastNetworkSample = null;
      this.awaitingStableFrame = this.startupMinChunk > 0;
      this.stats = {
        frames: 0,
        bytes: 0,
        renderedFrames: 0,
        lastChunk: null,
        lastReceivedChunk: null,
        lastReceivedFrameBatchIndex: null,
        frameBatchGapCount: 0,
        lastEventId: 0,
        lastSentEventId: 0,
        lastAppliedEventId: 0,
        lastDecodeMs: 0,
        lastDisplayLagMs: 0,
        lastRenderedEventId: 0,
        lastControlToVideoMs: 0,
        lastDownlinkMs: 0,
        receiveMbps: 0,
        chunkTelemetry: null,
        lastInputUplinkMs: 0,
        controlRoundTripMs: 0,
        serverClockOffsetMs: 0,
      };
      for (const frame of this.playback.clear?.() || []) closeFrame(frame);
      this.playback.reset?.({ targetFps: init.fps || 24 });
      this.worker?.postMessage?.({ type: "reset" });
      this.setState("connecting");

      return new Promise((resolve, reject) => {
        const socket = new this.WebSocketCtor(url);
        this.socket = socket;
        socket.binaryType = "arraybuffer";
        let opened = false;
        socket.onopen = () => {
          if (epoch !== this.epoch) return;
          opened = true;
          socket.send(this.pack({
            ...init,
            playback_ack_enabled: this.playbackAckEnabled,
          }));
          this.armMediaWatchdog(epoch, "startup");
          if (!this.awaitingStableFrame) this.setState("live");
          this.scheduleRender();
          resolve();
        };
        socket.onmessage = (event) => {
          if (epoch !== this.epoch) return;
          try {
            this.receive(event.data, epoch);
          } catch (error) {
            this.fail(error);
          }
        };
        socket.onerror = () => {
          if (epoch !== this.epoch) return;
          const error = new Error(`${this.key} websocket transport error`);
          this.onError(error, this.key);
          if (!opened) reject(error);
        };
        socket.onclose = (event) => {
          if (epoch !== this.epoch) return;
          this.clearMediaWatchdog();
          if (this.socket === socket) this.socket = null;
          this.setState(event.code === 1000 ? "closed" : "error", {
            code: event.code,
            reason: event.reason || "",
          });
          if (!opened) {
            reject(new Error(`${this.key} closed before startup (${event.code})`));
          } else {
            const normalClose = event.code === 1000 || event.code === 1001;
            const error = new Error(
              `${this.key} websocket closed (${event.code}): ${event.reason || "unknown"}`,
            );
            error.code = normalClose ? "UNEXPECTED_MEDIA_CLOSE" : "MEDIA_SOCKET_CLOSED";
            this.onError(error, this.key);
          }
        };
      });
    }

    sendEvent(envelope) {
      if (!this.socket || this.socket.readyState !== this.WebSocketCtor.OPEN) return false;
      this.socket.send(this.pack({ ...envelope, trace_id: this.traceId }));
      this.stats.lastSentEventId = Number(envelope.event_id || this.stats.lastSentEventId);
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
      this.playback.noteInputEvent?.(envelope.event_id, this.now(), {
        cutoverMode: envelope.kind === "prompt"
          ? "prompt"
          : envelope.kind === "camera_actions" && cameraActionHasActiveMotion(envelope.payload)
            ? "motion"
            : "settle",
      });
      return true;
    }

    close(reason = "session closed", { notify = true } = {}) {
      this.clearMediaWatchdog();
      if (this.playbackAckTimer) this.clearTimer(this.playbackAckTimer);
      this.playbackAckTimer = null;
      this.epoch += 1;
      const socket = this.socket;
      this.socket = null;
      this.pendingHeader = null;
      this.decodeQueue = [];
      this.controlSentEpochByEvent.clear();
      this.lastNetworkSample = null;
      for (const frame of this.playback.clear?.() || []) closeFrame(frame);
      this.worker?.postMessage?.({ type: "reset" });
      if (socket && socket.readyState !== this.WebSocketCtor.CLOSED) {
        socket.close(1000, reason.slice(0, 120));
      }
      if (notify) this.setState("closed", { reason });
    }

    setUnavailable(reason = "Unavailable for this mode") {
      this.close(reason, { notify: false });
      this.clearCanvas();
      this.setState("unavailable", { reason });
    }

    clearCanvas() {
      const width = this.canvas.width || 1280;
      const height = this.canvas.height || 720;
      this.ctx.fillStyle = "#11140f";
      this.ctx.fillRect(0, 0, width, height);
      this.hasVisibleFrame = false;
    }

    receive(data, epoch) {
      if (this.pendingHeader) {
        const header = this.pendingHeader;
        this.pendingHeader = null;
        header.__received_at = this.now();
        this.enqueueDecode(header, data, epoch);
        return;
      }
      const packed = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
      const message = this.unpack(packed);
      message.__received_at = this.now();
      if (message.type === "error") {
        if ((message.content || "") === "invalid event") return;
        const error = new Error(message.content || `${this.key} server error`);
        error.reason = message.reason || "";
        error.retryAfterS = Number(message.retry_after_s || 0);
        throw error;
      }
      if (message.type === "control_ack" && message.stage === "worker") {
        const clientSentEpochMs = Number(message.client_sent_epoch_ms || 0);
        const serverReceivedEpochMs = Number(message.server_received_epoch_ms || 0);
        const serverSentEpochMs = Number(message.server_sent_epoch_ms || 0);
        const clientReceivedEpochMs = Date.now();
        if (clientSentEpochMs && serverReceivedEpochMs && serverSentEpochMs) {
          const serverProcessingMs = Math.max(0, serverSentEpochMs - serverReceivedEpochMs);
          const roundTripMs = Math.max(0, clientReceivedEpochMs - clientSentEpochMs);
          this.stats.controlRoundTripMs = roundTripMs;
          this.stats.lastInputUplinkMs = Math.max(0, (roundTripMs - serverProcessingMs) / 2);
          this.stats.serverClockOffsetMs = (
            (serverReceivedEpochMs - clientSentEpochMs)
            + (serverSentEpochMs - clientReceivedEpochMs)
          ) / 2;
          this.emitStats();
        }
      } else if (message.type === "chunk_telemetry") {
        this.stats.chunkTelemetry = { ...message };
        this.emitStats();
      } else if (message.type === "frame_batch") {
        const payload = message.payload;
        delete message.payload;
        if (payload !== undefined) {
          this.enqueueDecode(message, payload, epoch);
        } else {
          this.pendingHeader = message;
        }
      } else if (message.type === "frame_batch_header") {
        this.pendingHeader = message;
      }
    }

    enqueueDecode(header, payload, epoch) {
      this.observeFrameBatch(header);
      const eventId = Number(header.event_id || 0);
      if (this.stats.lastSentEventId > 0 && eventId >= this.stats.lastSentEventId) {
        this.decodeQueue = this.decodeQueue.filter(
          (item) => Number(item.header?.event_id || 0) >= eventId,
        );
      }
      this.decodeQueue.push({ header, payload, epoch });
      this.trimDecodeQueue();
      this.pumpDecode();
    }

    trimDecodeQueue() {
      while (this.decodeQueue.length > this.maxDecodeQueueBatches) {
        this.decodeQueue.shift();
      }
    }

    observeFrameBatch(header) {
      const chunkIndex = Number(header.chunk_index || 0);
      const frameBatchIndex = Number(header.frame_batch_index || 0);
      if (this.stats.lastReceivedChunk === null) {
        this.stats.frameBatchGapCount += Math.max(0, frameBatchIndex);
      } else if (chunkIndex === this.stats.lastReceivedChunk) {
        const expected = Number(this.stats.lastReceivedFrameBatchIndex || 0) + 1;
        if (frameBatchIndex > expected) {
          this.stats.frameBatchGapCount += frameBatchIndex - expected;
        }
      } else if (chunkIndex > this.stats.lastReceivedChunk) {
        this.stats.frameBatchGapCount += Math.max(0, frameBatchIndex);
        if (chunkIndex > this.stats.lastReceivedChunk + 1) {
          this.stats.frameBatchGapCount += chunkIndex - this.stats.lastReceivedChunk - 1;
        }
      }
      this.stats.lastReceivedChunk = chunkIndex;
      this.stats.lastReceivedFrameBatchIndex = frameBatchIndex;
      const serverSentEpochMs = Number(header.server_sent_epoch_ms || 0);
      if (serverSentEpochMs > 0) {
        this.stats.lastDownlinkMs = Math.max(
          0,
          Date.now() - serverSentEpochMs + Number(this.stats.serverClockOffsetMs || 0),
        );
      }
      this.schedulePlaybackAck();
    }

    async pumpDecode() {
      if (this.decodeInProgress) return;
      const item = this.decodeQueue.shift();
      if (!item) return;
      this.decodeInProgress = true;
      try {
        const startedAt = this.now();
        const frames = await this.decodeBatch(item.header, item.payload);
        if (item.epoch !== this.epoch) {
          frames.forEach(closeFrame);
          return;
        }
        const decodedAt = this.now();
        const decodeMs = decodedAt - startedAt;
        const prepared = frames.map((frame) => ({
          ...frame,
          eventId: Number(frame.eventId ?? item.header.event_id ?? 0),
          receivedAt: frame.receivedAt || item.header.__received_at,
          decodedAt,
          decodeMs: frame.decodeMs ?? decodeMs,
        }));
        const result = this.playback.enqueueDecodedFrames(item.header, prepared, decodedAt);
        for (const dropped of result.droppedFrames || []) closeFrame(dropped);
        const bytes = Number(item.payload?.byteLength || item.payload?.size || item.payload?.length || 0);
        this.stats.frames += Number(item.header.num_frames || prepared.length);
        this.stats.bytes += bytes;
        const networkNow = this.now();
        if (this.lastNetworkSample && networkNow > this.lastNetworkSample.at) {
          const elapsedSeconds = (networkNow - this.lastNetworkSample.at) / 1000;
          this.stats.receiveMbps = Math.max(
            0,
            (this.stats.bytes - this.lastNetworkSample.bytes) * 8 / elapsedSeconds / 1_000_000,
          );
        }
        this.lastNetworkSample = { at: networkNow, bytes: this.stats.bytes };
        this.stats.lastChunk = Number(item.header.chunk_index || 0);
        this.stats.lastEventId = Number(item.header.event_id || this.stats.lastEventId);
        this.stats.lastAppliedEventId = Number(
          item.header.event_id || this.stats.lastAppliedEventId,
        );
        this.stats.lastDecodeMs = decodeMs;
        if (prepared.length) this.armMediaWatchdog(item.epoch, "stall");
        this.emitStats();
        this.scheduleRender();
      } catch (error) {
        this.fail(error);
      } finally {
        this.decodeInProgress = false;
        if (this.decodeQueue.length) this.pumpDecode();
      }
    }

    async decodeBatch(header, payload) {
      if (this.decodeBatchOverride) return this.decodeBatchOverride(header, payload);
      const worker = this.ensureWorker();
      const bytes = payload instanceof ArrayBuffer
        ? payload
        : payload.buffer.slice(payload.byteOffset || 0, (payload.byteOffset || 0) + payload.byteLength);
      const id = this.decodeRequestId++;
      const startedAt = this.now();
      return new Promise((resolve, reject) => {
        this.decodeRequests.set(id, {
          resolve: (message) => {
            const decodedAt = this.now();
            resolve(message.frames.map((frame) => ({
              image: message.frame_type === "bitmap"
                ? frame
                : new ImageData(new Uint8ClampedArray(frame), message.width, message.height),
              chunk: message.chunk,
              receivedAt: header.__received_at,
              decodedAt,
              decodeMs: decodedAt - startedAt,
            })));
          },
          reject,
        });
        worker.postMessage({
          type: "decode",
          header: { ...header, __decode_id: id },
          payload: bytes,
        }, [bytes]);
      });
    }

    ensureWorker() {
      if (this.worker) return this.worker;
      this.worker = new Worker(this.workerUrl);
      this.worker.onmessage = (event) => {
        const message = event.data;
        const request = this.decodeRequests.get(message.id);
        if (!request) return;
        this.decodeRequests.delete(message.id);
        if (message.type === "error") request.reject(new Error(message.message || "decode failed"));
        else request.resolve(message);
      };
      this.worker.onerror = (event) => {
        const error = new Error(event.message || "decode worker failed");
        for (const request of this.decodeRequests.values()) request.reject(error);
        this.decodeRequests.clear();
        this.worker?.terminate?.();
        this.worker = null;
        this.fail(error);
      };
      return this.worker;
    }

    scheduleRender() {
      if (this.renderScheduled) return;
      this.renderScheduled = true;
      this.requestFrame((now) => this.render(now));
    }

    render(now) {
      this.renderScheduled = false;
      const decision = this.playback.render(now, {
        hasPendingInput: this.decodeInProgress || this.decodeQueue.length > 0 || Boolean(this.socket),
      });
      for (const dropped of decision.droppedFrames || []) closeFrame(dropped);
      if (decision.action === "draw") {
        const frame = decision.frame;
        if (this.awaitingStableFrame && Number(frame.chunk || 0) < this.startupMinChunk) {
          closeFrame(frame);
          if (this.socket || this.decodeInProgress || this.decodeQueue.length || this.snapshot().queueFrames) {
            this.scheduleRender();
          }
          return;
        }
        this.awaitingStableFrame = false;
        this.drawFrame(frame.image);
        this.renderSamples.push(now);
        this.renderSamples = this.renderSamples.filter((sample) => now - sample < 1000);
        this.stats.renderedFrames += 1;
        this.stats.lastChunk = Number(frame.chunk ?? this.stats.lastChunk ?? 0);
        this.stats.lastDisplayLagMs = now - Number(frame.receivedAt || now);
        this.stats.lastRenderedEventId = Number(
          frame.eventId || this.stats.lastRenderedEventId || 0,
        );
        const appliedEventId = this.stats.lastRenderedEventId;
        const pendingEventIds = Array.from(this.controlSentEpochByEvent.keys())
          .filter((eventId) => eventId <= appliedEventId)
          .sort((left, right) => left - right);
        if (pendingEventIds.length) {
          const sentEpochMs = this.controlSentEpochByEvent.get(pendingEventIds[0]);
          this.stats.lastControlToVideoMs = Math.max(0, Date.now() - sentEpochMs);
          for (const eventId of pendingEventIds) this.controlSentEpochByEvent.delete(eventId);
        }
        this.onFrame({
          key: this.key,
          chunk: this.stats.lastChunk,
          eventId: frame.eventId,
          decodeMs: frame.decodeMs,
          displayLagMs: this.stats.lastDisplayLagMs,
        });
        this.emitStats();
        this.schedulePlaybackAck();
      }
      if (this.socket || this.decodeInProgress || this.decodeQueue.length || this.snapshot().queueFrames) {
        this.scheduleRender();
      }
    }

    drawFrame(image) {
      let source = image;
      const isImageData = typeof ImageData !== "undefined" && image instanceof ImageData;
      if (isImageData && this.scratch && this.scratchCtx) {
        this.scratch.width = image.width;
        this.scratch.height = image.height;
        this.scratchCtx.putImageData(image, 0, 0);
        source = this.scratch;
      }
      if (this.canvas.width !== image.width || this.canvas.height !== image.height) {
        this.canvas.width = image.width;
        this.canvas.height = image.height;
      }
      this.ctx.imageSmoothingEnabled = true;
      this.ctx.imageSmoothingQuality = "medium";
      if (isImageData && source === image) this.ctx.putImageData(image, 0, 0);
      else this.ctx.drawImage(source, 0, 0, image.width, image.height);
      if (!isImageData) image.close?.();
      this.hasVisibleFrame = true;
      this.setState("live");
    }

    schedulePlaybackAck() {
      if (
        !this.playbackAckEnabled
        || this.playbackAckTimer
        || !this.socket
        || this.socket.readyState !== this.WebSocketCtor.OPEN
      ) {
        return;
      }
      this.playbackAckTimer = this.setTimer(() => {
        this.playbackAckTimer = null;
        if (!this.socket || this.socket.readyState !== this.WebSocketCtor.OPEN) return;
        this.socket.send(this.pack({
          type: "event",
          kind: "playback_ack",
          trace_id: this.traceId,
          payload: {
            last_received_chunk: this.stats.lastReceivedChunk,
            last_rendered_chunk: this.stats.lastChunk,
            last_rendered_event_id: this.stats.lastRenderedEventId,
            playable: this.hasVisibleFrame,
          },
        }));
      }, 50);
    }

    setState(state, details = {}) {
      if (this.root) this.root.dataset.sessionState = state;
      if (this.overlay?.style) {
        this.overlay.style.display =
          (state === "connecting" && !this.hasVisibleFrame) || state === "unavailable"
            ? "grid"
            : "none";
      }
      const message = this.overlay?.querySelector?.(".preview-unavailable-text");
      if (message && state === "unavailable") message.textContent = details.reason || "Unavailable";
      this.canvas.setAttribute?.("aria-busy", state === "connecting" ? "true" : "false");
      this.onState(state, { key: this.key, ...details });
    }

    fail(error) {
      this.clearMediaWatchdog();
      this.setState("error", { message: error.message || String(error) });
      this.onError(error, this.key);
      const socket = this.socket;
      this.socket = null;
      this.epoch += 1;
      if (socket && socket.readyState !== this.WebSocketCtor.CLOSED) {
        socket.close(4000, String(error.message || "session failed").slice(0, 120));
      }
    }

    clearMediaWatchdog() {
      if (this.mediaWatchdogTimer === null) return;
      this.clearTimer(this.mediaWatchdogTimer);
      this.mediaWatchdogTimer = null;
    }

    armMediaWatchdog(epoch, phase) {
      this.clearMediaWatchdog();
      const timeoutMs = phase === "startup" ? this.startupTimeoutMs : this.stallTimeoutMs;
      if (!timeoutMs) return;
      this.mediaWatchdogTimer = this.setTimer(() => {
        this.mediaWatchdogTimer = null;
        if (epoch !== this.epoch || !this.socket) return;
        const error = new Error(
          phase === "startup"
            ? `${this.key} produced no frames within ${timeoutMs}ms`
            : `${this.key} media stream stalled for ${timeoutMs}ms`,
        );
        error.code = phase === "startup" ? "MEDIA_START_TIMEOUT" : "MEDIA_STALL_TIMEOUT";
        this.fail(error);
      }, timeoutMs);
    }

    snapshot() {
      const playback = this.playback.snapshot?.() || {};
      return {
        key: this.key,
        ...this.stats,
        ...playback,
        queueFrames: Number(playback.queueFrames ?? playback.queueLength ?? 0),
        decodeQueueLength: this.decodeQueue.length,
        renderFps: this.renderSamples.length,
      };
    }

    emitStats() {
      this.onStats(this.snapshot(), this.key);
    }
  }

  global.RealtimeModelSession = RealtimeModelSession;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { RealtimeModelSession };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
