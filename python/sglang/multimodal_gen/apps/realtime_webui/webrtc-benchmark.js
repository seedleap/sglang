const presets = [
  {
    name: "Spring Valley",
    image: "./assets/presets/v1/reactor-spring-valley.png",
    prompt: "A stable third-person flight through a luminous spring valley, preserving the same dragon, rider, mountains, waterfalls, flowers, sunlight and cinematic composition; natural forward motion and coherent parallax, no morphing, no scene replacement.",
  },
  {
    name: "Misted Kingdom",
    image: "./assets/presets/v1/reactor-misted-kingdom.png",
    prompt: "A continuous cinematic journey through the same misted fantasy kingdom, keeping the rider, castle, cliffs, fog layers and lighting consistent while the camera advances smoothly; realistic parallax, no cuts, no object morphing.",
  },
  {
    name: "Dragon Ride",
    image: "./assets/presets/v1/reactor-dragon-ride.png",
    prompt: "A stable dragon ride through the same dramatic mountain world, preserving the dragon anatomy, rider, terrain and atmosphere while moving forward with smooth controllable camera motion; coherent geometry and natural parallax.",
  },
];

const keyActions = new Map([
  ["w", "w"], ["a", "a"], ["s", "s"], ["d", "d"],
  ["arrowup", "i"], ["arrowleft", "j"], ["arrowdown", "k"], ["arrowright", "l"],
]);
const sessions = new Map();
const activeActions = new Set();
let eventId = 1;
let pollTimer = 0;

const $ = (id) => document.getElementById(id);
for (const preset of presets) {
  const option = document.createElement("option");
  option.textContent = preset.name;
  option.value = preset.name;
  $("preset").append(option);
}

async function imageDataUrl(url) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) throw new Error(`首帧读取失败 (${response.status})`);
  const blob = await response.blob();
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function initPayload(preset, firstFrame, index) {
  return {
    type: "init",
    model: "zing-gs3200-20260810",
    generation_mode: "i2v",
    prompt: preset.prompt,
    size: "832x480",
    fps: 16,
    seed: 42 + index,
    num_inference_steps: 4,
    guidance_scale: 1,
    realtime_causal_sink_size: 8,
    realtime_causal_kv_cache_num_frames: 32,
    realtime_interactive_event_grace_ms: 1800,
    first_frame: firstFrame,
    trace_id: `zing-webrtc-browser-${Date.now()}-${index}`,
  };
}

function tileFor(index) {
  const tile = document.createElement("article");
  tile.className = "tile";
  tile.innerHTML = `<header><strong>用户 ${index + 1}</strong><span id="status-${index}">申请中</span></header><div class="waiting" id="media-${index}">等待 Zing 首批帧…</div><div class="metrics" id="metrics-${index}">-</div>`;
  $("tiles").append(tile);
}

async function createOne(index, preset, firstFrame) {
  tileFor(index);
  const response = await fetch("./api/webrtc/sessions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      codec: $("codec").value,
      bitrate_kbps: Number($("bitrate").value),
      init: initPayload(preset, firstFrame, index),
    }),
  });
  const text = await response.text();
  if (!response.ok) throw new Error(text || `会话申请失败 (${response.status})`);
  const info = JSON.parse(text);
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const control = new WebSocket(`${protocol}//${location.host}/api/webrtc/sessions/${info.id}/control`);
  control.onmessage = (message) => {
    try {
      const event = JSON.parse(message.data);
      if (event.type === "error") $("status-${index}").textContent = event.message;
    } catch {}
  };
  sessions.set(info.id, { ...info, index, control, iframeMounted: false });
}

async function start() {
  await stop();
  $("tiles").innerHTML = "";
  const preset = presets.find((item) => item.name === $("preset").value) || presets[0];
  const users = Number($("users").value);
  $("state").textContent = `正在并发申请 ${users} 个会话…`;
  const firstFrame = await imageDataUrl(preset.image);
  const results = await Promise.allSettled(
    Array.from({ length: users }, (_, index) => createOne(index, preset, firstFrame)),
  );
  let failures = 0;
  results.forEach((result, index) => {
    if (result.status === "rejected") {
      failures += 1;
      const status = $("status-${index}");
      const metrics = $("metrics-${index}");
      if (status) { status.textContent = "失败"; status.className = "error"; }
      if (metrics) metrics.textContent = result.reason?.message || String(result.reason);
    }
  });
  $("state").textContent = `已连接 ${sessions.size}/${users}，失败 ${failures}`;
  pollTimer = window.setInterval(poll, 1000);
  await poll();
}

async function poll() {
  await Promise.all([...sessions.entries()].map(async ([id, session]) => {
    try {
      const response = await fetch(`./api/webrtc/sessions/${id}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`status ${response.status}`);
      const status = await response.json();
      $("status-${session.index}").textContent = status.state;
      $("metrics-${session.index}").textContent = `${status.width || "-"}x${status.height || "-"} · ${status.codec.toUpperCase()} ${status.bitrate_kbps}kbps · frames=${status.frames} · source=${status.average_source_mbps}Mbps`;
      if (status.frames > 0 && !session.iframeMounted) {
        const media = $("media-${session.index}");
        const iframe = document.createElement("iframe");
        iframe.allow = "autoplay; fullscreen";
        iframe.src = status.stream_page_url;
        media.replaceWith(iframe);
        session.iframeMounted = true;
      }
    } catch (error) {
      $("status-${session.index}").textContent = "状态丢失";
      $("metrics-${session.index}").textContent = error.message;
    }
  }));
}

async function stop() {
  if (pollTimer) window.clearInterval(pollTimer);
  pollTimer = 0;
  const active = [...sessions.entries()];
  sessions.clear();
  await Promise.allSettled(active.map(async ([id, session]) => {
    session.control.close(1000, "test stopped");
    await fetch(`./api/webrtc/sessions/${id}`, { method: "DELETE" });
  }));
  if (active.length) $("state").textContent = "全部会话已停止";
}

function sendActions() {
  const envelope = {
    type: "event",
    kind: "camera_actions",
    payload: {
      mode: "state",
      transitions: [{ actions: [...activeActions].sort(), client_ts_ms: performance.now() }],
    },
    event_id: eventId++,
    client_sent_perf_ms: performance.now(),
    client_sent_epoch_ms: Date.now(),
  };
  for (const session of sessions.values()) {
    if (session.control.readyState === WebSocket.OPEN) {
      session.control.send(JSON.stringify(envelope));
    }
  }
}

window.addEventListener("keydown", (event) => {
  const action = keyActions.get(event.key.toLowerCase());
  if (!action || event.repeat) return;
  event.preventDefault();
  activeActions.add(action);
  sendActions();
});
window.addEventListener("keyup", (event) => {
  const action = keyActions.get(event.key.toLowerCase());
  if (!action) return;
  event.preventDefault();
  activeActions.delete(action);
  sendActions();
});
window.addEventListener("beforeunload", () => { void stop(); });
$("start").addEventListener("click", () => void start().catch((error) => {
  $("state").textContent = error.message || String(error);
  $("state").className = "pill error";
}));
$("stop").addEventListener("click", () => void stop());
