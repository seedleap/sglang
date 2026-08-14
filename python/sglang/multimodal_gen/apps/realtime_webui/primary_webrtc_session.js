(function (global) {
  const CONTROL_OPEN = 1;

  function delay(ms) {
    return new Promise((resolve) => global.setTimeout(resolve, ms));
  }

  function responseError(response, fallback) {
    return response.text()
      .catch(() => "")
      .then((text) => new Error(text || `${fallback} (${response.status})`));
  }

  function waitForControlOpen(socket, timeoutMs) {
    return new Promise((resolve, reject) => {
      if (socket.readyState === CONTROL_OPEN) {
        resolve();
        return;
      }
      const timeout = global.setTimeout(() => {
        cleanup();
        reject(new Error("WebRTC control socket timed out"));
      }, timeoutMs);
      const cleanup = () => {
        global.clearTimeout(timeout);
        socket.removeEventListener?.("open", handleOpen);
        socket.removeEventListener?.("error", handleError);
        socket.removeEventListener?.("close", handleClose);
      };
      const handleOpen = () => {
        cleanup();
        resolve();
      };
      const handleError = () => {
        cleanup();
        reject(new Error("WebRTC control socket failed"));
      };
      const handleClose = () => {
        cleanup();
        reject(new Error("WebRTC control socket closed before startup"));
      };
      socket.addEventListener?.("open", handleOpen, { once: true });
      socket.addEventListener?.("error", handleError, { once: true });
      socket.addEventListener?.("close", handleClose, { once: true });
      if (!("addEventListener" in socket)) {
        socket.onopen = handleOpen;
        socket.onerror = handleError;
        socket.onclose = handleClose;
      }
    });
  }

  function waitForIceGathering(peer, timeoutMs) {
    if (peer.iceGatheringState === "complete") return Promise.resolve();
    return new Promise((resolve) => {
      const timeout = global.setTimeout(finish, timeoutMs);
      function finish() {
        global.clearTimeout(timeout);
        peer.removeEventListener?.("icegatheringstatechange", handleChange);
        resolve();
      }
      function handleChange() {
        if (peer.iceGatheringState === "complete") finish();
      }
      peer.addEventListener?.("icegatheringstatechange", handleChange);
    });
  }

  function preferH264(transceiver) {
    const receiver = global.RTCRtpReceiver;
    if (!transceiver?.setCodecPreferences || !receiver?.getCapabilities) return;
    const codecs = receiver.getCapabilities("video")?.codecs || [];
    const h264 = codecs.filter((codec) => codec.mimeType?.toLowerCase() === "video/h264");
    const auxiliaries = codecs.filter((codec) => codec.mimeType?.toLowerCase() !== "video/h264");
    if (h264.length) transceiver.setCodecPreferences([...h264, ...auxiliaries]);
  }

  function configureReceiverPlayoutDelay(transceiver, playoutDelayMs) {
    const receiver = transceiver?.receiver;
    if (!receiver || !Number.isFinite(playoutDelayMs) || playoutDelayMs <= 0) return;
    const boundedDelayMs = Math.min(4000, Math.max(0, playoutDelayMs));
    try {
      if ("jitterBufferTarget" in receiver) {
        receiver.jitterBufferTarget = boundedDelayMs;
      } else if ("playoutDelayHint" in receiver) {
        receiver.playoutDelayHint = boundedDelayMs / 1000;
      }
    } catch {}
  }

  function frameDimensions(frame) {
    return {
      width: Number(frame?.displayWidth || frame?.codedWidth || frame?.width || 0),
      height: Number(frame?.displayHeight || frame?.codedHeight || frame?.height || 0),
    };
  }

  class PrimaryWebRTCSession {
    constructor({
      video,
      canvas,
      endpoint = "./api/webrtc/sessions",
      codec = "h264",
      bitrateKbps = 3500,
      playoutDelayMs = 0,
      managedPlayback = false,
      fetchImpl = global.fetch?.bind(global),
      WebSocketImpl = global.WebSocket,
      RTCPeerConnectionImpl = global.RTCPeerConnection,
      MediaStreamTrackProcessorImpl = global.MediaStreamTrackProcessor,
      mediaPollIntervalMs = 200,
      startupTimeoutMs = 30000,
      controlKeepaliveMs = 10000,
      controlReconnectBaseMs = 250,
      mediaDisconnectGraceMs = 6000,
      mediaReconnectBaseMs = 500,
      mediaReconnectMaxAttempts = 8,
      onState = () => {},
      onPlayable = () => {},
      onFrame = null,
      onStats = () => {},
      onError = () => {},
    }) {
      if (!video) throw new Error("PrimaryWebRTCSession requires a video element");
      this.video = video;
      this.canvas = canvas || null;
      this.endpoint = endpoint.replace(/\/$/, "");
      this.codec = codec;
      this.bitrateKbps = bitrateKbps;
      this.playoutDelayMs = Math.min(4000, Math.max(0, Number(playoutDelayMs) || 0));
      this.managedPlayback = Boolean(managedPlayback && onFrame);
      this.fetchImpl = fetchImpl;
      this.WebSocketImpl = WebSocketImpl;
      this.RTCPeerConnectionImpl = RTCPeerConnectionImpl;
      this.MediaStreamTrackProcessorImpl = MediaStreamTrackProcessorImpl;
      this.mediaPollIntervalMs = mediaPollIntervalMs;
      this.startupTimeoutMs = startupTimeoutMs;
      this.controlKeepaliveMs = controlKeepaliveMs;
      this.controlReconnectBaseMs = controlReconnectBaseMs;
      this.mediaDisconnectGraceMs = Math.max(0, Number(mediaDisconnectGraceMs) || 0);
      this.mediaReconnectBaseMs = Math.max(50, Number(mediaReconnectBaseMs) || 500);
      this.mediaReconnectMaxAttempts = Math.max(
        1,
        Math.trunc(Number(mediaReconnectMaxAttempts) || 8),
      );
      this.onState = onState;
      this.onPlayable = onPlayable;
      this.onFrame = onFrame;
      this.onStats = onStats;
      this.onError = onError;
      this.sessionId = "";
      this.control = null;
      this.peer = null;
      this.whepUrl = "";
      this.whepResourceUrl = "";
      this.generation = 0;
      this.state = "idle";
      this.expectedClose = false;
      this.playable = false;
      this.statsTimer = 0;
      this.controlKeepaliveTimer = 0;
      this.controlReconnectTimer = 0;
      this.controlReconnectAttempt = 0;
      this.mediaReconnectTimer = 0;
      this.mediaReconnectAttempt = 0;
      this.mediaReconnectInFlight = false;
      this.lastRtcSample = null;
      this.receiver = null;
      this.trackReader = null;
      this.trackPump = null;
      this.mediaBatches = [];
      this.mediaFps = 24;
      this.firstFrameTimestampUs = null;
      this.timestampFrameBase = 0;
      this.processedFrames = 0;
      this.playableResolve = null;
      this.playableReject = null;
      this.handleVideoPlayable = () => this._markPlayable();
      for (const eventName of ["playing", "loadeddata", "resize", "timeupdate"]) {
        this.video.addEventListener?.(eventName, this.handleVideoPlayable);
      }
    }

    get active() {
      return Boolean(this.sessionId || this.control || this.peer);
    }

    get connected() {
      return Boolean(
        this.playable
        && this.control
        && this.control.readyState === CONTROL_OPEN
        && this.peer
        && !["failed", "closed"].includes(this.peer.connectionState),
      );
    }

    get bufferedAmount() {
      return Number(this.control?.bufferedAmount || 0);
    }

    async connect(init) {
      await this.close("replacing WebRTC session", { emitState: false });
      if (!this.fetchImpl || !this.WebSocketImpl || !this.RTCPeerConnectionImpl) {
        throw new Error("This browser does not support the required WebRTC APIs");
      }
      const generation = ++this.generation;
      this.expectedClose = false;
      this.playable = false;
      this._setState("connecting", { protocol: "webrtc", codec: "h264" });
      try {
        const response = await this.fetchImpl(this.endpoint, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            codec: this.codec,
            bitrate_kbps: this.bitrateKbps,
            managed_playback: this._canManagePlayback(),
            init,
          }),
        });
        if (!response.ok) throw await responseError(response, "WebRTC session creation failed");
        const info = await response.json();
        if (generation !== this.generation) throw new Error("stale WebRTC session");
        this.sessionId = String(info.id || info.session_id || "");
        if (!this.sessionId) throw new Error("WebRTC bridge returned no session id");
        this.mediaFps = Math.max(1, Number(init?.fps || 24));
        await this._openControl(generation);
        const status = await this._waitForPublisher(generation);
        this.whepUrl = String(status.whep_url || info.whep_url || "");
        await this._openWhep(this.whepUrl, generation);
        await this._waitForPlayable(generation);
        this._startStats();
        return info;
      } catch (error) {
        if (generation === this.generation) {
          this._setState("error", { message: error.message, protocol: "webrtc", codec: "h264" });
          this.onError(error);
          await this.close(error.message || "WebRTC startup failed", { emitState: false });
        }
        throw error;
      }
    }

    sendEvent(envelope) {
      if (!this.control || this.control.readyState !== CONTROL_OPEN) return false;
      this.control.send(JSON.stringify(envelope));
      return true;
    }

    async close(reason = "WebRTC session closed", { emitState = true } = {}) {
      const sessionId = this.sessionId;
      const control = this.control;
      const peer = this.peer;
      const hadSession = Boolean(sessionId || control || peer);
      this.expectedClose = true;
      this.generation += 1;
      this.sessionId = "";
      this.control = null;
      this.peer = null;
      this.receiver = null;
      this.whepUrl = "";
      this.whepResourceUrl = "";
      this.playable = false;
      const trackReader = this.trackReader;
      this.trackReader = null;
      this.trackPump = null;
      this.mediaBatches = [];
      this.firstFrameTimestampUs = null;
      this.timestampFrameBase = 0;
      this.processedFrames = 0;
      this._stopStats();
      this._stopControlKeepalive();
      if (this.controlReconnectTimer) global.clearTimeout(this.controlReconnectTimer);
      this.controlReconnectTimer = 0;
      this.controlReconnectAttempt = 0;
      if (this.mediaReconnectTimer) global.clearTimeout(this.mediaReconnectTimer);
      this.mediaReconnectTimer = 0;
      this.mediaReconnectAttempt = 0;
      this.mediaReconnectInFlight = false;
      this.playableReject?.(new Error(reason));
      this.playableResolve = null;
      this.playableReject = null;
      try {
        control?.close?.(1000, String(reason).slice(0, 120));
      } catch {}
      try {
        peer?.close?.();
      } catch {}
      try {
        await trackReader?.cancel?.(reason);
      } catch {}
      try {
        this.video.pause?.();
      } catch {}
      this.video.srcObject = null;
      this.video.hidden = true;
      if (this.canvas) this.canvas.hidden = false;
      if (sessionId && this.fetchImpl) {
        await this.fetchImpl(`${this.endpoint}/${encodeURIComponent(sessionId)}`, {
          method: "DELETE",
          keepalive: true,
        }).catch(() => {});
      }
      if (emitState && hadSession) this._setState("closed", { reason });
    }

    async _openControl(generation) {
      const protocol = global.location?.protocol === "https:" ? "wss:" : "ws:";
      const host = global.location?.host || "localhost";
      const control = new this.WebSocketImpl(
        `${protocol}//${host}/api/webrtc/sessions/${encodeURIComponent(this.sessionId)}/control`,
      );
      this.control = control;
      control.onmessage = (message) => {
        if (generation !== this.generation) return;
        try {
          const event = JSON.parse(message.data);
          if (event.type === "error") {
            const error = new Error(event.message || "WebRTC upstream control error");
            this.onError(error);
          } else if (event.type === "control_ack") {
            const clientSentEpochMs = Number(event.client_sent_epoch_ms || 0);
            this.onStats({
              lastControlEventId: Number(event.event_id || 0),
              lastControlKind: String(event.kind || ""),
              controlBridgeRoundTripMs: clientSentEpochMs
                ? Math.max(0, Date.now() - clientSentEpochMs)
                : 0,
              controlBridgeForwardMs: Number(event.bridge_forward_ms || 0),
              controlBridgeReceivedEpochMs: Number(event.bridge_received_epoch_ms || 0),
              controlMinimumEventId: Number(event.minimum_event_id || 0),
            });
          } else if (event.type === "media_batch") {
            this._queueMediaBatch(event);
          } else if (event.type === "media_chunk_complete") {
            this._finalizeMediaChunk(event);
          }
        } catch {}
      };
      control.onclose = (event) => {
        if (generation !== this.generation || this.expectedClose || this.control !== control) return;
        this.control = null;
        this._stopControlKeepalive();
        this._scheduleControlReconnect(generation, event.code || 0);
      };
      await waitForControlOpen(control, this.startupTimeoutMs);
      if (generation !== this.generation) throw new Error("WebRTC control startup canceled");
      this.controlReconnectAttempt = 0;
      this._startControlKeepalive();
    }

    _startControlKeepalive() {
      this._stopControlKeepalive();
      if (this.controlKeepaliveMs <= 0) return;
      this.controlKeepaliveTimer = global.setInterval(() => {
        if (!this.control || this.control.readyState !== CONTROL_OPEN) return;
        try {
          this.control.send(JSON.stringify({
            type: "event",
            kind: "heartbeat",
            payload: { transport: "webrtc-control" },
          }));
        } catch {}
      }, this.controlKeepaliveMs);
    }

    _stopControlKeepalive() {
      if (this.controlKeepaliveTimer) global.clearInterval(this.controlKeepaliveTimer);
      this.controlKeepaliveTimer = 0;
    }

    _scheduleControlReconnect(generation, closeCode) {
      if (this.controlReconnectTimer || generation !== this.generation || !this.sessionId) return;
      const attempt = this.controlReconnectAttempt++;
      const delayMs = Math.min(4000, this.controlReconnectBaseMs * (2 ** Math.min(attempt, 4)));
      if (!this.playable) {
        this._setState("connecting", {
          protocol: "webrtc",
          codec: "h264",
          reason: `control socket closed (${closeCode})`,
        });
      }
      this.controlReconnectTimer = global.setTimeout(async () => {
        this.controlReconnectTimer = 0;
        if (generation !== this.generation || !this.sessionId) return;
        try {
          await this._openControl(generation);
          if (this.playable) {
            this._setState("live", {
              protocol: "webrtc",
              codec: "h264",
              reconnected: true,
              width: this.video.videoWidth,
              height: this.video.videoHeight,
            });
          }
        } catch (error) {
          if (generation !== this.generation || !this.sessionId) return;
          if (this.controlReconnectAttempt >= 8) this.onError(error);
          this._scheduleControlReconnect(generation, closeCode);
        }
      }, delayMs);
    }

    async _waitForPublisher(generation) {
      const deadline = Date.now() + this.startupTimeoutMs;
      let lastStatus = null;
      while (Date.now() < deadline) {
        if (generation !== this.generation) throw new Error("WebRTC startup canceled");
        const response = await this.fetchImpl(
          `${this.endpoint}/${encodeURIComponent(this.sessionId)}`,
          { cache: "no-store" },
        );
        if (!response.ok) throw await responseError(response, "WebRTC status failed");
        lastStatus = await response.json();
        this.onStats({
          sourceFrames: Number(lastStatus.frames || 0),
          sourceMbps: Number(lastStatus.average_source_mbps || 0),
          width: Number(lastStatus.width || 0),
          height: Number(lastStatus.height || 0),
          bitrateKbps: Number(lastStatus.bitrate_kbps || this.bitrateKbps),
          codec: String(lastStatus.codec || this.codec),
        });
        if (lastStatus.error || lastStatus.state === "error") {
          throw new Error(lastStatus.error || "WebRTC bridge failed");
        }
        if (Number(lastStatus.frames || 0) > 0 && lastStatus.whep_url) return lastStatus;
        await delay(this.mediaPollIntervalMs);
      }
      throw new Error(
        `WebRTC publisher timed out${lastStatus?.state ? ` (${lastStatus.state})` : ""}`,
      );
    }

    async _openWhep(whepUrl, generation) {
      if (!whepUrl) throw new Error("WebRTC bridge returned no WHEP URL");
      let lastError = null;
      const deadline = Date.now() + Math.min(10000, this.startupTimeoutMs);
      while (Date.now() < deadline) {
        if (generation !== this.generation) throw new Error("WebRTC startup canceled");
        const peer = new this.RTCPeerConnectionImpl({ bundlePolicy: "max-bundle" });
        this.peer = peer;
        const transceiver = peer.addTransceiver("video", { direction: "recvonly" });
        preferH264(transceiver);
        configureReceiverPlayoutDelay(transceiver, this.playoutDelayMs);
        this.receiver = transceiver.receiver || null;
        peer.ontrack = (event) => {
          if (generation !== this.generation) return;
          if (this._canManagePlayback() && event.track) {
            this.video.srcObject = null;
            this.video.hidden = true;
            if (this.canvas) this.canvas.hidden = false;
            this._startManagedTrack(event.track, generation);
            return;
          }
          const stream = event.streams?.[0]
            || (global.MediaStream ? new global.MediaStream([event.track]) : null);
          if (!stream) return;
          this.video.srcObject = stream;
          this.video.hidden = false;
          void this.video.play?.().catch(() => {});
        };
        peer.onconnectionstatechange = () => {
          this._handlePeerConnectionState(peer, generation);
        };
        try {
          const offer = await peer.createOffer();
          await peer.setLocalDescription(offer);
          await waitForIceGathering(peer, 2500);
          const response = await this.fetchImpl(whepUrl, {
            method: "POST",
            headers: { "content-type": "application/sdp" },
            body: peer.localDescription?.sdp || offer.sdp,
          });
          if (!response.ok) throw await responseError(response, "WHEP negotiation failed");
          const answerSdp = await response.text();
          if (!/H264\/90000/i.test(answerSdp)) {
            throw new Error("WHEP answer did not negotiate H.264");
          }
          const resourceLocation = response.headers?.get?.("location") || "";
          this.whepResourceUrl = resourceLocation
            ? new URL(resourceLocation, whepUrl).toString()
            : whepUrl;
          await peer.setRemoteDescription({ type: "answer", sdp: answerSdp });
          this._handlePeerConnectionState(peer, generation);
          return;
        } catch (error) {
          lastError = error;
          try { peer.close(); } catch {}
          if (this.peer === peer) this.peer = null;
          await delay(this.mediaPollIntervalMs);
        }
      }
      throw lastError || new Error("WHEP negotiation timed out");
    }

    _handlePeerConnectionState(peer, generation) {
      if (
        generation !== this.generation
        || this.expectedClose
        || this.peer !== peer
      ) return;
      const state = String(peer.connectionState || "");
      if (state === "connected") {
        const recovered = Boolean(
          this.mediaReconnectTimer
          || this.mediaReconnectInFlight
          || this.mediaReconnectAttempt,
        );
        const attempt = this.mediaReconnectAttempt;
        if (this.mediaReconnectTimer) global.clearTimeout(this.mediaReconnectTimer);
        this.mediaReconnectTimer = 0;
        this.mediaReconnectAttempt = 0;
        this.mediaReconnectInFlight = false;
        if (recovered && this.playable) {
          this._setState("live", {
            protocol: "webrtc",
            codec: "h264",
            reconnected: true,
            mediaReconnected: true,
            attempt,
            width: Number(this.canvas?.width || this.video.videoWidth || 0),
            height: Number(this.canvas?.height || this.video.videoHeight || 0),
          });
        }
        return;
      }
      if (state === "disconnected") {
        this._scheduleMediaReconnect(
          generation,
          "media disconnected",
          this.mediaDisconnectGraceMs,
        );
      } else if (state === "failed") {
        if (this.mediaReconnectTimer) global.clearTimeout(this.mediaReconnectTimer);
        this.mediaReconnectTimer = 0;
        this._scheduleMediaReconnect(generation, "media failed", 0);
      }
    }

    _scheduleMediaReconnect(generation, reason, delayMs = null) {
      if (
        this.mediaReconnectTimer
        || this.mediaReconnectInFlight
        || generation !== this.generation
        || this.expectedClose
        || !this.sessionId
        || !this.whepUrl
      ) return;
      const attempt = this.mediaReconnectAttempt + 1;
      const retryDelayMs = delayMs === null
        ? Math.min(8000, this.mediaReconnectBaseMs * (2 ** Math.min(attempt - 1, 4)))
        : Math.max(0, Number(delayMs) || 0);
      this._setState("connecting", {
        protocol: "webrtc",
        codec: "h264",
        reconnecting: true,
        reason,
        attempt,
      });
      this.mediaReconnectTimer = global.setTimeout(() => {
        this.mediaReconnectTimer = 0;
        void this._reconnectMedia(generation, reason);
      }, retryDelayMs);
    }

    async _reconnectMedia(generation, reason) {
      if (
        this.mediaReconnectInFlight
        || generation !== this.generation
        || this.expectedClose
        || !this.sessionId
      ) return;
      this.mediaReconnectInFlight = true;
      this.mediaReconnectAttempt += 1;
      const attempt = this.mediaReconnectAttempt;
      const previousPeer = this.peer;
      const previousReader = this.trackReader;
      this.peer = null;
      this.receiver = null;
      this.trackReader = null;
      this.trackPump = null;
      this.whepResourceUrl = "";
      this.firstFrameTimestampUs = null;
      this.timestampFrameBase = Math.max(
        this.processedFrames,
        Number(this.mediaBatches[0]?.firstFrameIndex || 0),
      );
      this.video.srcObject = null;
      try {
        void Promise.resolve(previousReader?.cancel?.("WebRTC media reconnect")).catch(() => {});
      } catch {}
      try {
        previousPeer?.close?.();
      } catch {}
      try {
        await this._openWhep(this.whepUrl, generation);
        if (generation !== this.generation || this.expectedClose) return;
        this.mediaReconnectInFlight = false;
        this._startStats();
        this._handlePeerConnectionState(this.peer, generation);
      } catch (error) {
        this.mediaReconnectInFlight = false;
        if (generation !== this.generation || this.expectedClose) return;
        if (attempt >= this.mediaReconnectMaxAttempts) {
          const terminalError = new Error(
            `WebRTC media reconnect failed after ${attempt} attempts: ${error.message || reason}`,
          );
          this._setState("error", {
            message: terminalError.message,
            protocol: "webrtc",
            codec: "h264",
          });
          this.onError(terminalError);
          return;
        }
        this._scheduleMediaReconnect(
          generation,
          error.message || reason || "media reconnect failed",
        );
      }
    }

    _canManagePlayback() {
      return Boolean(
        this.managedPlayback
        && this.onFrame
        && this.MediaStreamTrackProcessorImpl,
      );
    }

    _queueMediaBatch(event) {
      const firstFrameIndex = Number(event.first_frame_index);
      const numFrames = Math.max(0, Number(event.num_frames || 0));
      if (!Number.isFinite(firstFrameIndex) || !numFrames) return;
      const batch = {
        chunkIndex: Number(event.chunk_index || 0),
        eventId: Number(event.event_id || 0),
        firstFrameIndex,
        lastFrameIndex: firstFrameIndex + numFrames - 1,
        numFrames,
        frameBatchIndex: Number(event.frame_batch_index || 0),
        numFrameBatches: Number(event.num_frame_batches || 0),
        isFinalFrameBatch: Boolean(event.is_final_frame_batch),
      };
      if (this.mediaBatches.some((item) => (
        item.firstFrameIndex === batch.firstFrameIndex
        && item.lastFrameIndex === batch.lastFrameIndex
      ))) return;
      this.mediaBatches.push(batch);
      this.mediaBatches.sort((left, right) => left.firstFrameIndex - right.firstFrameIndex);
      if (this.mediaBatches.length > 1024) {
        this.mediaBatches.splice(0, this.mediaBatches.length - 1024);
      }
    }

    _finalizeMediaChunk(event) {
      const chunkIndex = Number(event.chunk_index || 0);
      const batches = this.mediaBatches.filter((item) => item.chunkIndex === chunkIndex);
      const finalBatch = batches.at(-1);
      if (!finalBatch) return;
      finalBatch.isFinalFrameBatch = true;
      finalBatch.numFrameBatches = Math.max(
        finalBatch.frameBatchIndex + 1,
        Number(finalBatch.numFrameBatches || 0),
      );
    }

    _metadataForFrame(frame, fallbackIndex) {
      const timestampUs = Number(frame?.timestamp);
      if (this.firstFrameTimestampUs === null && Number.isFinite(timestampUs)) {
        this.firstFrameTimestampUs = timestampUs;
      }
      const sourceFrameIndex = Number.isFinite(timestampUs) && this.firstFrameTimestampUs !== null
        ? this.timestampFrameBase + Math.max(0, Math.round(
            (timestampUs - this.firstFrameTimestampUs) * this.mediaFps / 1_000_000,
          ))
        : fallbackIndex;
      while (
        this.mediaBatches.length
        && this.mediaBatches[0].lastFrameIndex < sourceFrameIndex
      ) {
        this.mediaBatches.shift();
      }
      const batch = this.mediaBatches.find((item) => (
        item.firstFrameIndex <= sourceFrameIndex
        && item.lastFrameIndex >= sourceFrameIndex
      ));
      return {
        sourceFrameIndex,
        chunkIndex: batch?.chunkIndex ?? Math.floor(sourceFrameIndex / 16),
        eventId: batch?.eventId ?? 0,
        frameBatchIndex: batch?.frameBatchIndex ?? 0,
        numFrameBatches: batch
          ? batch.numFrameBatches > 0
            ? batch.numFrameBatches
            : batch.isFinalFrameBatch
              ? batch.frameBatchIndex + 1
              : 1_000_000
          : 1,
        isFinalFrameBatch: batch
          ? batch.isFinalFrameBatch && sourceFrameIndex >= batch.lastFrameIndex
          : true,
      };
    }

    _startManagedTrack(track, generation) {
      let processor;
      try {
        processor = new this.MediaStreamTrackProcessorImpl({ track });
      } catch (error) {
        this.onError(error);
        return;
      }
      const reader = processor.readable.getReader();
      this.trackReader = reader;
      this.trackPump = (async () => {
        while (generation === this.generation && this.trackReader === reader) {
          const result = await reader.read();
          if (result.done) break;
          const frame = result.value;
          if (!frame) continue;
          if (generation !== this.generation || this.trackReader !== reader) {
            frame.close?.();
            break;
          }
          const receivedAt = global.performance?.now?.() ?? Date.now();
          const sequence = this.processedFrames++;
          const metadata = this._metadataForFrame(frame, sequence);
          const dimensions = frameDimensions(frame);
          try {
            this.onFrame({
              frame,
              receivedAt,
              sequence,
              ...dimensions,
              ...metadata,
            });
          } catch (error) {
            frame.close?.();
            throw error;
          }
          this._markManagedPlayable(dimensions.width, dimensions.height);
        }
      })().catch((error) => {
        if (generation !== this.generation || this.expectedClose) return;
        this._setState("error", { message: error.message, protocol: "webrtc", codec: "h264" });
        this.onError(error);
      });
    }

    _waitForPlayable(generation) {
      if (this.playable) return Promise.resolve();
      return new Promise((resolve, reject) => {
        const timeout = global.setTimeout(() => {
          if (generation !== this.generation || this.playable) return;
          this.playableResolve = null;
          this.playableReject = null;
          reject(new Error("WebRTC video did not become playable"));
        }, this.startupTimeoutMs);
        this.playableResolve = () => {
          global.clearTimeout(timeout);
          resolve();
        };
        this.playableReject = (error) => {
          global.clearTimeout(timeout);
          reject(error);
        };
        this._markPlayable();
      });
    }

    _markPlayable() {
      if (this.playable || !this.sessionId) return;
      if (Number(this.video.readyState || 0) < 2) return;
      if (!Number(this.video.videoWidth || 0) || !Number(this.video.videoHeight || 0)) return;
      this.playable = true;
      this.video.hidden = false;
      if (this.canvas) this.canvas.hidden = true;
      this._setState("live", {
        protocol: "webrtc",
        codec: "h264",
        width: this.video.videoWidth,
        height: this.video.videoHeight,
      });
      this.onPlayable({
        width: this.video.videoWidth,
        height: this.video.videoHeight,
        codec: "h264",
        protocol: "webrtc",
      });
      this.playableResolve?.();
      this.playableResolve = null;
      this.playableReject = null;
    }

    _markManagedPlayable(width, height) {
      if (this.playable || !this.sessionId || width <= 0 || height <= 0) return;
      this.playable = true;
      this.video.hidden = true;
      if (this.canvas) this.canvas.hidden = false;
      this._setState("live", {
        protocol: "webrtc",
        codec: "h264",
        width,
        height,
        managedPlayback: true,
      });
      this.onPlayable({
        width,
        height,
        codec: "h264",
        protocol: "webrtc",
        managedPlayback: true,
      });
      this.playableResolve?.();
      this.playableResolve = null;
      this.playableReject = null;
    }

    _startStats() {
      this._stopStats();
      const sample = async () => {
        const peer = this.peer;
        if (!peer?.getStats || !this.sessionId) return;
        try {
          const reports = await peer.getStats();
          for (const report of reports.values()) {
            if (report.type !== "inbound-rtp" || (report.kind || report.mediaType) !== "video") {
              continue;
            }
            const now = performance.now();
            const previous = this.lastRtcSample;
            const framesDecoded = Number(report.framesDecoded || 0);
            const bytesReceived = Number(report.bytesReceived || 0);
            let receiveFps = 0;
            let receiveMbps = 0;
            if (previous && now > previous.at) {
              const seconds = (now - previous.at) / 1000;
              receiveFps = Math.max(0, (framesDecoded - previous.framesDecoded) / seconds);
              receiveMbps = Math.max(0, (bytesReceived - previous.bytesReceived) * 8 / seconds / 1_000_000);
            }
            this.lastRtcSample = { at: now, framesDecoded, bytesReceived };
            const jitterBufferCount = Number(report.jitterBufferEmittedCount || 0);
            const jitterBufferMs = jitterBufferCount
              ? Number(report.jitterBufferDelay || 0) * 1000 / jitterBufferCount
              : 0;
            const jitterBufferTargetMs = jitterBufferCount
              ? Number(report.jitterBufferTargetDelay || 0) * 1000 / jitterBufferCount
              : 0;
            let roundTripTimeMs = 0;
            let availableIncomingBitrate = 0;
            for (const candidate of reports.values()) {
              if (candidate.type !== "candidate-pair") continue;
              if (candidate.state !== "succeeded" && !candidate.nominated && !candidate.selected) continue;
              roundTripTimeMs = Math.max(
                roundTripTimeMs,
                Number(candidate.currentRoundTripTime || 0) * 1000,
              );
              availableIncomingBitrate = Math.max(
                availableIncomingBitrate,
                Number(candidate.availableIncomingBitrate || 0),
              );
            }
            this.onStats({
              framesDecoded,
              framesDropped: Number(report.framesDropped || 0),
              bytesReceived,
              receiveFps,
              receiveMbps,
              jitterMs: Number(report.jitter || 0) * 1000,
              jitterBufferMs,
              jitterBufferTargetMs,
              configuredPlayoutDelayMs: this.playoutDelayMs,
              packetsLost: Number(report.packetsLost || 0),
              packetsReceived: Number(report.packetsReceived || 0),
              nackCount: Number(report.nackCount || 0),
              pliCount: Number(report.pliCount || 0),
              freezeCount: Number(report.freezeCount || 0),
              totalFreezesDurationMs: Number(report.totalFreezesDuration || 0) * 1000,
              roundTripTimeMs,
              availableIncomingMbps: availableIncomingBitrate / 1_000_000,
              connectionState: String(peer.connectionState || ""),
              iceConnectionState: String(peer.iceConnectionState || ""),
              managedPlayback: this._canManagePlayback(),
              codec: "h264",
              protocol: "webrtc",
            });
            break;
          }
        } catch {}
      };
      void sample();
      this.statsTimer = global.setInterval(() => void sample(), 1000);
    }

    _stopStats() {
      if (this.statsTimer) global.clearInterval(this.statsTimer);
      this.statsTimer = 0;
      this.lastRtcSample = null;
    }

    _setState(state, details = {}) {
      this.state = state;
      this.onState(state, details);
    }
  }

  global.PrimaryWebRTCSession = PrimaryWebRTCSession;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { PrimaryWebRTCSession };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
