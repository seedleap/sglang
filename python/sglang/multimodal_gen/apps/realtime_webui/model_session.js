(function (global) {
  const DEFAULT_WORKER_URL = "./decoder_worker.js?v=realtime-production-gateway-v17";

  function closeFrame(frame) {
    const image = frame?.image;
    if (image && typeof ImageData !== "undefined" && image instanceof ImageData) return;
    image?.close?.();
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
        minTargetLeadMs: 360,
        maxTargetLeadMs: 900,
      });
      this.socket = null;
      this.pendingHeader = null;
      this.traceId = "";
      this.epoch = 0;
      this.renderScheduled = false;
      this.decodeQueue = [];
      this.decodeInProgress = false;
      this.worker = null;
      this.decodeRequests = new Map();
      this.decodeRequestId = 1;
      this.stats = {
        frames: 0,
        bytes: 0,
        renderedFrames: 0,
        lastChunk: null,
        lastEventId: 0,
        lastDecodeMs: 0,
        lastDisplayLagMs: 0,
      };
      this.setState("idle");
    }

    configure({ mode, targetFps } = {}) {
      if (mode) this.playback.setMode?.(mode);
      if (targetFps) this.playback.setTargetFps?.(targetFps);
    }

    connect(init, url) {
      this.close("replace session", { notify: false });
      this.epoch += 1;
      const epoch = this.epoch;
      this.traceId = init.trace_id || "";
      this.pendingHeader = null;
      this.decodeQueue = [];
      this.decodeInProgress = false;
      this.stats = {
        frames: 0,
        bytes: 0,
        renderedFrames: 0,
        lastChunk: null,
        lastEventId: 0,
        lastDecodeMs: 0,
        lastDisplayLagMs: 0,
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
          socket.send(this.pack(init));
          this.setState("live");
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
          if (this.socket === socket) this.socket = null;
          this.setState(event.code === 1000 ? "closed" : "error", {
            code: event.code,
            reason: event.reason || "",
          });
          if (!opened) {
            reject(new Error(`${this.key} closed before startup (${event.code})`));
          } else if (event.code !== 1000 && event.code !== 1001) {
            this.onError(
              new Error(`${this.key} websocket closed (${event.code}): ${event.reason || "unknown"}`),
              this.key,
            );
          }
        };
      });
    }

    sendEvent(envelope) {
      if (!this.socket || this.socket.readyState !== this.WebSocketCtor.OPEN) return false;
      this.socket.send(this.pack({ ...envelope, trace_id: this.traceId }));
      this.stats.lastEventId = Number(envelope.event_id || this.stats.lastEventId);
      this.playback.noteInputEvent?.(envelope.event_id, this.now(), {
        cutoverMode: envelope.kind === "prompt" ? "motion" : "settle",
      });
      return true;
    }

    close(reason = "session closed", { notify = true } = {}) {
      this.epoch += 1;
      const socket = this.socket;
      this.socket = null;
      this.pendingHeader = null;
      this.decodeQueue = [];
      for (const frame of this.playback.clear?.() || []) closeFrame(frame);
      this.worker?.postMessage?.({ type: "reset" });
      if (socket && socket.readyState !== this.WebSocketCtor.CLOSED) {
        socket.close(1000, reason.slice(0, 120));
      }
      if (notify) this.setState("closed");
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
        throw new Error(message.content || `${this.key} server error`);
      }
      if (message.type === "frame_batch") {
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
      this.decodeQueue.push({ header, payload, epoch });
      this.pumpDecode();
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
          receivedAt: frame.receivedAt || item.header.__received_at,
          decodedAt,
          decodeMs: frame.decodeMs ?? decodeMs,
        }));
        const result = this.playback.enqueueDecodedFrames(item.header, prepared, decodedAt);
        for (const dropped of result.droppedFrames || []) closeFrame(dropped);
        const bytes = Number(item.payload?.byteLength || item.payload?.size || item.payload?.length || 0);
        this.stats.frames += Number(item.header.num_frames || prepared.length);
        this.stats.bytes += bytes;
        this.stats.lastChunk = Number(item.header.chunk_index || 0);
        this.stats.lastEventId = Number(item.header.event_id || this.stats.lastEventId);
        this.stats.lastDecodeMs = decodeMs;
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
        this.drawFrame(frame.image);
        this.stats.renderedFrames += 1;
        this.stats.lastChunk = Number(frame.chunk ?? this.stats.lastChunk ?? 0);
        this.stats.lastDisplayLagMs = now - Number(frame.receivedAt || now);
        this.onFrame({
          key: this.key,
          chunk: this.stats.lastChunk,
          eventId: frame.eventId,
          decodeMs: frame.decodeMs,
          displayLagMs: this.stats.lastDisplayLagMs,
        });
        this.emitStats();
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
      this.setState("live");
    }

    setState(state, details = {}) {
      if (this.root) this.root.dataset.sessionState = state;
      if (this.overlay?.style) {
        this.overlay.style.display = state === "connecting" ? "grid" : "none";
      }
      this.canvas.setAttribute?.("aria-busy", state === "connecting" ? "true" : "false");
      this.onState(state, { key: this.key, ...details });
    }

    fail(error) {
      this.setState("error", { message: error.message || String(error) });
      this.onError(error, this.key);
      const socket = this.socket;
      this.socket = null;
      this.epoch += 1;
      if (socket && socket.readyState !== this.WebSocketCtor.CLOSED) {
        socket.close(4000, String(error.message || "session failed").slice(0, 120));
      }
    }

    snapshot() {
      const playback = this.playback.snapshot?.() || {};
      return {
        key: this.key,
        ...this.stats,
        ...playback,
        queueFrames: Number(playback.queueFrames ?? playback.queueLength ?? 0),
        decodeQueueLength: this.decodeQueue.length,
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
