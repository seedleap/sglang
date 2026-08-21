#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");

function percentile(sorted, ratio) {
  if (!sorted.length) return null;
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * ratio;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;
  return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

function rounded(value) {
  return Math.round(value * 1000) / 1000;
}

function numeric(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function summarize(values) {
  const sorted = values.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return { count: 0, min: null, p50: null, p95: null, max: null, mean: null };
  return {
    count: sorted.length,
    min: rounded(sorted[0]),
    p50: rounded(percentile(sorted, 0.5)),
    p95: rounded(percentile(sorted, 0.95)),
    max: rounded(sorted.at(-1)),
    mean: rounded(sorted.reduce((sum, value) => sum + value, 0) / sorted.length),
  };
}

function traceHttpUrl(pageUrl, traceId) {
  const url = new URL(pageUrl);
  return `${url.origin}/v1/realtime_video/traces/${encodeURIComponent(traceId)}`;
}

function parseFpsText(text) {
  const values = [...String(text || "").matchAll(/(\d+(?:\.\d+)?)/g)].map((match) => Number(match[1]));
  return {
    source: Number.isFinite(values[0]) ? values[0] : 0,
    received: Number.isFinite(values[1]) ? values[1] : 0,
    rendered: Number.isFinite(values[2]) ? values[2] : 0,
  };
}

function traceEventDuration(event, fields) {
  for (const field of fields) {
    if (event[field] != null) {
      const value = Number(event[field]);
      if (Number.isFinite(value)) return value;
    }
  }
  return null;
}

function summarizeTrace(events, warmupChunks, fallbackFramesPerChunk = 16) {
  const chunkEvents = events.filter((event) => (
    event.event === "server.chunk_complete" &&
    Number(event.chunk_index ?? -1) >= warmupChunks
  ));
  const chunkTotalMs = chunkEvents
    .map((event) => traceEventDuration(event, ["chunk_total_ms", "duration_ms"]))
    .filter(Number.isFinite);
  const schedulerMs = chunkEvents
    .map((event) => traceEventDuration(event, ["scheduler_forward_ms"]))
    .filter(Number.isFinite);
  const denoiseMs = events
    .filter((event) => (
      event.event === "server.model_denoise_complete" &&
      Number(event.chunk_index ?? -1) >= warmupChunks
    ))
    .map((event) => traceEventDuration(event, ["cuda_ms", "duration_ms"]))
    .filter(Number.isFinite);
  const vaeDecodeMs = events
    .filter((event) => (
      (event.event === "server.remote_vae_complete" || event.event === "server.vae_decode_complete") &&
      Number(event.chunk_index ?? -1) >= warmupChunks
    ))
    .map((event) => traceEventDuration(event, ["vae_decode_ms", "cuda_ms", "duration_ms"]))
    .filter(Number.isFinite);
  const frameCountsByChunk = new Map();
  for (const event of events) {
    const chunk = Number(event.chunk_index ?? -1);
    if (chunk < warmupChunks) continue;
    const frames = Number(event.num_frames ?? event.frame_count ?? 0);
    if (!Number.isFinite(frames) || frames <= 0) continue;
    if (
      event.event === "server.vae_frame_batch_sent" ||
      event.event === "server.output_frame_batch_sent" ||
      event.event === "server.frame_batch_sent"
    ) {
      frameCountsByChunk.set(chunk, (frameCountsByChunk.get(chunk) || 0) + frames);
    }
  }
  const traceFrames = [...frameCountsByChunk.values()].reduce((sum, value) => sum + value, 0);
  const inferredFrames = chunkEvents.length * fallbackFramesPerChunk;
  const frames = traceFrames || inferredFrames;
  const computeSeconds = chunkTotalMs.reduce((sum, value) => sum + value, 0) / 1000;
  return {
    warmup_chunks: warmupChunks,
    steady_chunks: chunkEvents.length,
    steady_frames: frames,
    steady_frames_source: traceFrames ? "trace_frame_batches" : "chunk_count_x_fallback",
    steady_compute_seconds: rounded(computeSeconds),
    steady_compute_fps: computeSeconds > 0 ? rounded(frames / computeSeconds) : 0,
    chunk_total_ms: summarize(chunkTotalMs),
    scheduler_forward_ms: summarize(schedulerMs),
    model_denoise_ms: summarize(denoiseMs),
    vae_decode_ms: summarize(vaeDecodeMs),
  };
}

function parseArgs(argv) {
  const values = {};
  for (let index = 2; index < argv.length; index += 1) {
    const name = argv[index];
    if (!name.startsWith("--")) throw new Error(`unexpected argument ${name}`);
    values[name.slice(2)] = argv[++index];
  }
  if (!values.url || !values.output) throw new Error("--url and --output are required");
  return {
    url: values.url,
    output: values.output,
    screenshot: values.screenshot || "",
    minFrames: Number(values["min-frames"] || 64),
    warmupChunks: Number(values["warmup-chunks"] || 2),
    timeoutMs: Number(values["timeout-ms"] || 300000),
    traceTimeoutMs: Number(values["trace-timeout-ms"] || 90000),
    mode: values.mode || "t2v",
    fps: Number(values.fps || 24),
    size: values.size || "832x480",
    sink: values.sink == null ? null : Number(values.sink),
    windowFrames: values.window == null ? null : Number(values.window),
    guidance: Number(values.guidance || 0),
    numFrames: Number(values.frames || 121),
    prompt: values.prompt || "",
    preset: values.preset || "",
    referenceImage: values["reference-image"] ? path.resolve(values["reference-image"]) : "",
    measureMs: Number(values["measure-ms"] || 15000),
    continuous: values.continuous === "true",
    sendAction: values["send-action"] !== "false",
  };
}

async function collectTrace(request, endpoint, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let cursor = 0;
  let stable = 0;
  const events = new Map();
  while (Date.now() < deadline) {
    const response = await request.get(endpoint, {
      params: { after: String(cursor), limit: "100" },
      timeout: 30000,
    });
    if (response.ok()) {
      const payload = await response.json();
      let added = 0;
      for (const event of payload.events || []) {
        const sequence = Number(event.trace_seq || 0);
        if (!sequence || events.has(sequence)) continue;
        events.set(sequence, event);
        added += 1;
      }
      cursor = Math.max(cursor, Number(payload.next_cursor || 0), ...events.keys());
      stable = added ? 0 : stable + 1;
      const hasDisplay = [...events.values()].some(
        (event) => event.event === "client.chunk_first_rendered" && Number.isFinite(Number(event.display_lag_ms)),
      );
      const hasServerChunk = [...events.values()].some((event) => event.event === "server.chunk_complete");
      if ((hasDisplay || hasServerChunk) && stable >= 2) break;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  return [...events.values()].sort((left, right) => Number(left.trace_seq) - Number(right.trace_seq));
}

async function readBrowserStats(page) {
  return page.evaluate(() => {
    const text = (selector) => document.querySelector(selector)?.textContent?.trim() || "";
    const number = (value) => {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : 0;
    };
    const video = document.querySelector("#minwmH264Viewport");
    const quality = video?.getVideoPlaybackQuality?.();
    const root = document.querySelector('[data-model-key="minwm"]');
    return {
      debug: window.__sglangRealtimeDebug?.() || null,
      h264: {
        hidden: Boolean(video?.hidden),
        currentTime: number(video?.currentTime),
        readyState: number(video?.readyState),
        paused: Boolean(video?.paused),
        videoWidth: number(video?.videoWidth),
        videoHeight: number(video?.videoHeight),
        totalVideoFrames: number(quality?.totalVideoFrames),
        droppedVideoFrames: number(quality?.droppedVideoFrames),
      },
      ui: {
        modelState: root?.dataset?.sessionState || "",
        frames: number(root?.dataset?.frames),
        chunk: root?.dataset?.chunk || "",
        rateText: text("#minwmRateText"),
        perfFpsText: text("#minwmPerfFps"),
        status: text("#statusText"),
        firstFrameState: text("#firstFrameState"),
        referenceName: text("#referenceName"),
        worldDraftStatus: text("#worldDraftStatus"),
        historyTail: text("#historyList").slice(-3000),
      },
    };
  });
}

async function setControlValue(page, selector, value) {
  await page.evaluate(
    ({ selector: targetSelector, value: targetValue }) => {
      const control = document.querySelector(targetSelector);
      if (!control) throw new Error(`missing control ${targetSelector}`);
      control.value = String(targetValue);
      control.dispatchEvent(new Event("input", { bubbles: true }));
      control.dispatchEvent(new Event("change", { bubbles: true }));
    },
    { selector, value },
  );
}

async function setControlChecked(page, selector, checked) {
  await page.evaluate(
    ({ selector: targetSelector, checked: targetChecked }) => {
      const control = document.querySelector(targetSelector);
      if (!control) throw new Error(`missing control ${targetSelector}`);
      control.checked = Boolean(targetChecked);
      control.dispatchEvent(new Event("input", { bubbles: true }));
      control.dispatchEvent(new Event("change", { bubbles: true }));
    },
    { selector, checked },
  );
}

function fpsFromDelta(after, before, seconds) {
  const frameDelta = Math.max(
    numeric(after?.h264?.totalVideoFrames) - numeric(before?.h264?.totalVideoFrames),
    numeric(after?.ui?.frames) - numeric(before?.ui?.frames),
    0,
  );
  return seconds > 0 ? rounded(frameDelta / seconds) : 0;
}

async function run(args) {
  const { chromium } = require("playwright");
  const launch = { headless: true };
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE) {
    launch.executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
  }
  const browser = await chromium.launch(launch);
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const page = await context.newPage();
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") pageErrors.push(message.text());
  });
  try {
    const url = new URL(args.url);
    url.searchParams.set("mode", args.mode);
    await page.goto(url.toString(), { waitUntil: "networkidle", timeout: 60000 });
    await setControlValue(page, "#generationMode", args.mode);
    await setControlValue(page, "#size", args.size);
    await setControlValue(page, "#fps", args.fps);
    await setControlChecked(page, "#continuous", args.continuous);
    if (!args.continuous) await setControlValue(page, "#numFrames", args.numFrames);
    await setControlValue(page, "#guidance", args.guidance);
    if (args.sink != null) await setControlValue(page, "#sinkSize", args.sink);
    if (args.windowFrames != null) await setControlValue(page, "#windowFrames", args.windowFrames);
    if (args.mode === "i2v") {
      if (args.preset) {
        await page.locator(".preset", { hasText: args.preset }).first().click({ timeout: 60000 });
      } else if (args.referenceImage) {
        await page.setInputFiles("#firstFrame", args.referenceImage);
      }
      await page.waitForFunction(
        () => {
          const firstFrame = document.querySelector("#firstFrameState")?.textContent || "";
          const prompt = document.querySelector("#prompt")?.value || "";
          const button = document.querySelector("#connectBtn");
          return /已填写/.test(firstFrame) && prompt.trim().length > 0 && !button?.disabled;
        },
        null,
        { timeout: 60000, polling: 250 },
      );
    }
    await setControlValue(
      page,
      "#prompt",
      args.prompt || (
        args.mode === "i2v"
          ? "A smooth forward flight preserving the reference composition and daylight"
          : "A smooth forward camera move through a mountain valley in daylight"
      ),
    );
    await page.click("#connectBtn");
    try {
      await page.waitForFunction(
        (minFrames) => {
          const root = document.querySelector('[data-model-key="minwm"]');
          const video = document.querySelector("#minwmH264Viewport");
          const quality = video?.getVideoPlaybackQuality?.();
          const debug = window.__sglangRealtimeDebug?.();
          const h264Frames = Number(quality?.totalVideoFrames || root?.dataset?.frames || 0);
          const fallbackFrames = Number(debug?.frames || debug?.renderedPreviewFrames || 0);
          return (
            h264Frames >= minFrames ||
            fallbackFrames >= minFrames
          );
        },
        args.minFrames,
        { timeout: args.timeoutMs, polling: 250 },
      );
    } catch (error) {
      const diagnostics = await readBrowserStats(page);
      throw new Error(
        `browser did not reach requested frames: ${JSON.stringify({ ...diagnostics, pageErrors })}`,
        { cause: error },
      );
    }
    const preActionDebug = await page.evaluate(() => window.__sglangRealtimeDebug());
    if (args.sendAction) {
      await page.keyboard.down("w");
      await page.waitForTimeout(350);
      await page.keyboard.up("w");
    }
    const measurementStart = await readBrowserStats(page);
    const measurementStartMs = Date.now();
    await page.waitForTimeout(Math.max(0, args.measureMs));
    const measurementEndMs = Date.now();
    const debug = await page.evaluate(() => window.__sglangRealtimeDebug());
    const measurementEnd = await readBrowserStats(page);
    const connected = (
      measurementEnd.ui.modelState === "live" ||
      measurementEnd.ui.modelState === "ready" ||
      debug.socketReadyState === 1
    );
    if (args.continuous && !connected) {
      throw new Error(
        `continuous browser session closed after ${measurementEnd.ui.frames || debug.frames || 0} frames: ${JSON.stringify(measurementEnd)}`,
      );
    }
    const traceId = debug.currentSessionArtifact?.traceId;
    if (!traceId) throw new Error("browser session did not expose a trace id");
    if (args.screenshot) {
      await page.screenshot({ path: args.screenshot, fullPage: true });
    }
    if (args.continuous) {
      await page.click("#stopBtn");
      await page.waitForFunction(
        () => window.__sglangRealtimeDebug?.().socketReadyState !== WebSocket.OPEN,
        null,
        { timeout: 10000, polling: 100 },
      );
    }
    const events = await collectTrace(
      context.request,
      traceHttpUrl(url.toString(), traceId),
      args.traceTimeoutMs,
    );
    const perfFps = parseFpsText(measurementEnd.ui.perfFpsText || measurementEnd.ui.rateText);
    const measuredSeconds = Math.max(0, (measurementEndMs - measurementStartMs) / 1000);
    const displayEvents = events.filter(
      (event) =>
        event.event === "client.chunk_first_rendered" &&
        Number(event.chunk_index || 0) >= args.warmupChunks &&
        Number.isFinite(Number(event.display_lag_ms)),
    );
    const traceSummary = summarizeTrace(events, args.warmupChunks);
    return {
      trace_id: traceId,
      mode: args.mode,
      request: {
        size: args.size,
        fps: args.fps,
        sink: args.sink,
        window: args.windowFrames,
        continuous: args.continuous,
      },
      continuous: args.continuous,
      socket_open: connected,
      status: debug.status,
      display_lag_ms: summarize(displayEvents.map((event) => Number(event.display_lag_ms))),
      browser_measurement: {
        seconds: rounded(measuredSeconds),
        start: measurementStart,
        end: measurementEnd,
        decoded_or_presented_fps: fpsFromDelta(measurementEnd, measurementStart, measuredSeconds),
        ui_source_fps: perfFps.source,
        ui_received_fps: perfFps.received,
        ui_rendered_fps: perfFps.rendered,
      },
      trace_summary: traceSummary,
      rendered_frames: Number(measurementEnd.ui.frames || debug.frames || 0),
      render_fps: Number(perfFps.rendered || debug.renderedFps || 0),
      playback: debug.playback,
      pre_action: {
        playback: preActionDebug.playback,
        received_frames: Number(preActionDebug.frames || 0),
        rendered_frames: Number(preActionDebug.renderedPreviewFrames || 0),
        render_fps: Number(preActionDebug.renderedFps || 0),
      },
      rendered_preview_frames: Number(debug.renderedPreviewFrames || 0),
      evidence_events: displayEvents.length,
      trace_event_names: [...new Set(events.map((event) => event.event).filter(Boolean))].sort(),
    };
  } finally {
    await browser.close();
  }
}

if (require.main === module) {
  const args = parseArgs(process.argv);
  run(args)
    .then((result) => {
      fs.mkdirSync(path.dirname(path.resolve(args.output)), { recursive: true });
      fs.writeFileSync(args.output, `${JSON.stringify(result, null, 2)}\n`);
      console.log(`browser probe passed: ${args.output}`);
    })
    .catch((error) => {
      console.error(error.stack || error);
      process.exitCode = 1;
    });
}

module.exports = { collectTrace, parseArgs, run, summarize, traceHttpUrl };
