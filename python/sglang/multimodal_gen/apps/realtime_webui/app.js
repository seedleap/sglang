const $ = (id) => document.getElementById(id);
const RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb";
const RAW_RGB_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgb-delta-gzip";
const RAW_RGBA_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgba-delta-gzip";
const WEBP_FRAME_CONTENT_TYPE = "image/webp";
const JPEG_FRAME_CONTENT_TYPE = "image/jpeg";
const DECODER_WORKER_URL = "./decoder_worker.js?v=rgb-worker-v10";
const UI_CONFIG = Object.freeze(globalThis.SGLANG_REALTIME_UI_CONFIG || {});
const DUAL_MODEL_CONFIG = Object.freeze(UI_CONFIG.dualModels || {});
const H264_MSE_MIME_TYPE = 'video/mp4; codecs="avc1.4D401F"';
const H264_WEBSOCKET_REQUESTED = UI_CONFIG.h264WebSocketEnabled === true;
const H264_WEBSOCKET_ENABLED = H264_WEBSOCKET_REQUESTED
  && Boolean(globalThis.MediaSource?.isTypeSupported?.(H264_MSE_MIME_TYPE));
const REALTIME_PROTOCOL_VERSION = 2;
const DEFAULT_LINGBOT2_MODEL = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers";
const SESSION_ARTIFACT_SCHEMA_VERSION = 1;
const SESSION_ARTIFACT_EVENT_LIMIT = 20000;
const MAX_EMBEDDED_REFERENCE_IMAGE_BYTES = 2 * 1024 * 1024;

function configuredNumber(name, fallback) {
  const value = Number(UI_CONFIG[name]);
  return Number.isFinite(value) ? value : fallback;
}

function configuredModelNumber(key, name, fallback) {
  const value = Number(DUAL_MODEL_CONFIG[key]?.[name]);
  return Number.isFinite(value) ? value : fallback;
}

function h264CompressionInit(init, key) {
  const bitrateKbps = Math.max(
    250,
    Math.min(
      20000,
      Math.trunc(configuredModelNumber(
        key,
        "h264BitrateKbps",
        configuredNumber("h264CompressedBitrateKbps", 3000),
      )),
    ),
  );
  return {
    ...init,
    h264_bitrate_kbps: bitrateKbps,
    h264_crf: configuredModelNumber(
      key,
      "h264Crf",
      configuredNumber("h264CompressedCrf", 20),
    ),
    h264_preset: String(
      DUAL_MODEL_CONFIG[key]?.h264Preset
      || UI_CONFIG.h264CompressedPreset
      || "fast",
    ),
    h264_gop_seconds: configuredModelNumber(
      key,
      "h264GopSeconds",
      configuredNumber("h264CompressedGopSeconds", 2),
    ),
    h264_vbv_buffer_ms: configuredModelNumber(
      key,
      "h264VbvBufferMs",
      configuredNumber("h264CompressedVbvBufferMs", 250),
    ),
    h264_startup_drop_frames: Math.max(
      0,
      Math.min(
        120,
        Math.trunc(configuredModelNumber(
          key,
          "h264StartupDropFrames",
          key === "lingbot2" ? 8 : 0,
        )),
      ),
    ),
  };
}

function h264WebSocketEndpoint(key) {
  const defaultEndpoint = `/api/h264ws/${key}`;
  const configuredEndpoint = String(
    DUAL_MODEL_CONFIG[key]?.h264WsUrl || UI_CONFIG.h264WebSocketBaseUrl || "",
  ).trim();
  if (!configuredEndpoint) return defaultEndpoint;
  try {
    const endpoint = new URL(defaultEndpoint, configuredEndpoint);
    if (endpoint.protocol === "https:") endpoint.protocol = "wss:";
    if (endpoint.protocol === "http:") endpoint.protocol = "ws:";
    if (endpoint.protocol !== "ws:" && endpoint.protocol !== "wss:") {
      return defaultEndpoint;
    }
    return endpoint.toString();
  } catch {
    return defaultEndpoint;
  }
}

function configuredGenerationModes() {
  const requestedModes = Array.isArray(UI_CONFIG.generationModes)
    ? UI_CONFIG.generationModes
    : UI_CONFIG.generationMode || UI_CONFIG.defaultGenerationMode
    ? ["i2v", UI_CONFIG.generationMode || UI_CONFIG.defaultGenerationMode]
    : ["i2v"];
  const modes = requestedModes
    .map((mode) => String(mode).toLowerCase())
    .filter((mode, index, values) => (
      (mode === "i2v" || mode === "t2v") && values.indexOf(mode) === index
    ));
  return modes.length ? modes : ["i2v"];
}

const DEFAULT_PREVIEW_OUTPUT_FORMAT = "webp";
const DEFAULT_PREVIEW_OUTPUT_QUALITY = 55;
const MAX_WEBP_PREVIEW_OUTPUT_QUALITY = 80;
const SMOOTH_PREVIEW_OUTPUT_QUALITY = 70;
const SR_PREVIEW_OUTPUT_QUALITY = 70;
const HEAVY_PREVIEW_OUTPUT_QUALITY = 60;
const DEFAULT_TARGET_FPS = configuredNumber("targetFps", 24);
const DEFAULT_LINGBOT2_TARGET_FPS = configuredModelNumber("lingbot2", "targetFps", 16);
const DEFAULT_LINGBOT2_SINK_SIZE = configuredModelNumber("lingbot2", "sinkSize", 9);
const DEFAULT_LINGBOT2_WINDOW_FRAMES = configuredModelNumber("lingbot2", "windowFrames", 18);
const DEFAULT_PREVIEW_MAX_WIDTH = configuredNumber("previewMaxWidth", 832);
const MAX_AUTO_PREVIEW_WIDTH = configuredNumber("maxAutoPreviewWidth", 1280);
const DEFAULT_FRAME_INTERPOLATION_EXP = 1;
const DEFAULT_FRAME_INTERPOLATION_SCALE = 1.0;
const DEFAULT_SMOOTH_CATCHUP_RATE = Math.min(
  2.5,
  Math.max(1, configuredNumber("smoothCatchupRateMax", 1.1)),
);
const DEFAULT_UPSCALING_SCALE = 2;
const DEFAULT_UPSCALING_MODEL =
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth";
const DEFAULT_PREVIEW_SCALE = 100;
const ENABLED_GENERATION_MODES = configuredGenerationModes();
const CONFIGURED_DEFAULT_GENERATION_MODE = String(
  UI_CONFIG.defaultGenerationMode || UI_CONFIG.generationMode || "",
).toLowerCase();
const DEFAULT_GENERATION_MODE = ENABLED_GENERATION_MODES.includes(
  CONFIGURED_DEFAULT_GENERATION_MODE,
)
  ? CONFIGURED_DEFAULT_GENERATION_MODE
  : ENABLED_GENERATION_MODES[0];
const T2V_FRAME_STEP = Math.max(
  1,
  Math.trunc(configuredNumber("t2vFrameStep", 4)),
);
const DEFAULT_T2V_NUM_FRAMES = 9;
const RECONNECT_CLOSE_TIMEOUT_MS = 15000;
const DECODE_QUEUE_SECONDS = configuredNumber("decodeQueueSeconds", 5);
const STARTUP_DECODE_QUEUE_SECONDS = configuredNumber("startupDecodeQueueSeconds", 5);
const ONLINE_MAX_BUFFER_MS = configuredNumber("onlineMaxBufferMs", 1100);
const ONLINE_MAX_BUFFER_CHUNKS = Math.max(
  1,
  Math.trunc(configuredNumber("onlineMaxBufferChunks", 2)),
);
const ONLINE_MAX_FRAME_AGE_MS = configuredNumber("onlineMaxFrameAgeMs", 1800);
const ONLINE_DECODE_QUEUE_SLACK_FRAMES = configuredNumber("onlineDecodeQueueSlackFrames", 8);
const MAX_DECODE_QUEUE_BYTES = configuredNumber(
  "maxDecodeQueueBytes",
  192 * 1024 * 1024,
);
const RECENT_DROP_DISPLAY_MS = 1800;
const CONTROL_TRANSITION_FLUSH_DELAY_MS = 50;
const CONTROL_HELD_STATE_HEARTBEAT_MS = 100;
const SESSION_HEARTBEAT_MS = 15000;
const PLAYBACK_ACK_INTERVAL_MS = 50;
// Keep ACK flow-control opt-in until every deployed realtime worker supports
// the playback_ack protocol extension. Older workers return `invalid event`.
const PLAYBACK_ACK_ENABLED = UI_CONFIG.playbackAckEnabled === true;
const SESSION_MAX_LIFETIME_SECONDS = Math.max(
  1,
  Math.trunc(configuredNumber("sessionMaxLifetimeSeconds", 60)),
);
const SESSION_MAX_LIFETIME_MS = SESSION_MAX_LIFETIME_SECONDS * 1000;
const MAX_GOAL_MIN_PLAY_SECONDS = Math.max(0, SESSION_MAX_LIFETIME_SECONDS - 1);
const EXPERIENCE_BUSY_MESSAGE = `当前正有人体验，请等待${SESSION_MAX_LIFETIME_SECONDS}s`;
const GAMEPLAY_RECORDING_FPS = Math.max(
  8,
  Math.min(30, Math.trunc(configuredNumber("gameplayRecordingFps", DEFAULT_TARGET_FPS))),
);
const BROWSER_USER_ID_STORAGE_KEY = "sglang-realtime-user-id";
const MIN_RENDER_TIMER_FPS = 30;
const MAX_RENDER_TIMER_FPS = 60;
const CONTROL_KEY_ACTIONS = new Map([
  ["w", "w"],
  ["a", "a"],
  ["s", "s"],
  ["d", "d"],
  ["arrowup", "i"],
  ["arrowleft", "j"],
  ["arrowdown", "k"],
  ["arrowright", "l"],
]);
const CONTROL_ACTION_META = {
  w: {
    label: "Forward",
    type: "translation",
    axis: "+forward",
    amount: "0.05/frame",
  },
  a: { label: "Left", type: "translation", axis: "-right", amount: "0.05/frame" },
  s: {
    label: "Back",
    type: "translation",
    axis: "-forward",
    amount: "0.05/frame",
  },
  d: { label: "Right", type: "translation", axis: "+right", amount: "0.05/frame" },
  i: { label: "Pitch +", type: "rotation", axis: "+pitch", amount: "4deg/frame" },
  j: { label: "Yaw -", type: "rotation", axis: "-yaw", amount: "6deg/frame" },
  k: { label: "Pitch -", type: "rotation", axis: "-pitch", amount: "4deg/frame" },
  l: { label: "Yaw +", type: "rotation", axis: "+yaw", amount: "6deg/frame" },
};
const RECORDING_STAGE_WIDTH = Math.max(
  1280,
  Math.min(1920, Math.trunc(configuredNumber("gameplayRecordingWidth", 1920))),
) & ~1;
const RECORDING_STAGE_HEIGHT = Math.max(
  720,
  Math.min(
    1080,
    Math.trunc(configuredNumber(
      "gameplayRecordingHeight",
      Math.round(RECORDING_STAGE_WIDTH * 9 / 16),
    )),
  ),
) & ~1;
const RECORDING_STAGE_TOPBAR_HEIGHT = 48;
const RECORDING_STAGE_PREVIEW_HEIGHT = RECORDING_STAGE_HEIGHT - RECORDING_STAGE_TOPBAR_HEIGHT;
const RECORDING_STAGE_PADDING = 18;
const RECORDING_PROMPT_STATUS_HOLD_MS = 1600;
const RECORDING_READY_TOAST_MS = 5000;
const RECORDING_HEARTBEAT_MS = 250;
const RECORDING_IDLE_CAPTURE_TIMEOUT_MS = 80;
const RECORDING_MAX_ENCODER_QUEUE_SIZE = 2;
const RECORDING_KEYFRAME_INTERVAL_FRAMES = 120;

function applyRuntimeUiConfig() {
  for (const key of ["minwm", "lingbot2"]) {
    const isLingBot2 = key === "lingbot2";
    modelControl(key, "fps").value = String(
      isLingBot2 ? DEFAULT_LINGBOT2_TARGET_FPS : DEFAULT_TARGET_FPS,
    );
    modelControl(key, "guidance").value = String(
      configuredNumber("guidanceScale", Number(modelControl(key, "guidance").value)),
    );
    modelControl(key, "sinkSize").value = String(
      isLingBot2
        ? DEFAULT_LINGBOT2_SINK_SIZE
        : configuredNumber("sinkSize", Number(modelControl(key, "sinkSize").value)),
    );
    modelControl(key, "windowFrames").value = String(
      isLingBot2
        ? DEFAULT_LINGBOT2_WINDOW_FRAMES
        : configuredNumber("windowFrames", Number(modelControl(key, "windowFrames").value)),
    );
  }
  if (UI_CONFIG.titleSuffix) {
    const suffix = String(UI_CONFIG.titleSuffix);
    $("studioTitle").textContent = `Realtime Studio · ${suffix}`;
    document.title = `Realtime Studio · ${suffix}`;
  }
  if (UI_CONFIG.actionAmountLabel) {
    Object.values(CONTROL_ACTION_META).forEach((meta) => {
      meta.amount = String(UI_CONFIG.actionAmountLabel);
    });
  }
  if (H264_WEBSOCKET_ENABLED) {
    const bitrateKbps = configuredNumber("h264CompressedBitrateKbps", 3000);
    for (const key of ["minwm", "lingbot2"]) {
      const chip = document.querySelector(`[data-model-key="${key}"] .stream-chip`);
      if (chip) chip.textContent = `H.264 · WS · ${(bitrateKbps / 1000).toFixed(1)} Mbps`;
    }
  } else if (H264_WEBSOCKET_REQUESTED) {
    addHistory("当前浏览器不支持 H.264 MSE，已自动回退 WebP WebSocket");
  }
  configureGenerationModeSelect();
}

function modelControlId(key, id) {
  if (key !== "lingbot2") return id;
  return `lingbot2${id[0].toUpperCase()}${id.slice(1)}`;
}

function modelControl(key, id) {
  return $(modelControlId(key, id));
}

function configureGenerationModeSelect() {
  const select = $("generationMode");
  Array.from(select.options).forEach((option) => {
    const enabled = ENABLED_GENERATION_MODES.includes(option.value);
    option.disabled = !enabled;
    option.hidden = !enabled;
  });
  select.value = DEFAULT_GENERATION_MODE;
  $("generationModeField").hidden = ENABLED_GENERATION_MODES.length < 2;
  updateGenerationModeUi();
}

function selectedGenerationMode() {
  return $("generationMode").value === "t2v" ? "t2v" : "i2v";
}

function updateT2VFrameHint() {
  if (selectedGenerationMode() === "t2v" && $("continuous").checked) {
    $("t2vFrameHint").textContent = "Continuous T2V runs until Stop is pressed.";
    return;
  }
  const frames = Number($("numFrames").value);
  const fps = Number($("fps").value || DEFAULT_TARGET_FPS);
  const duration = Number.isFinite(frames) && Number.isFinite(fps) && fps > 0
    ? Math.max(0, frames) / fps
    : 0;
  $("t2vFrameHint").textContent = (
    `Zing requires 1 + N × ${T2V_FRAME_STEP}; `
    + `${frames || 0} frames ≈ ${duration.toFixed(2)}s at ${fps || 0}fps.`
  );
}

function updateGenerationModeUi() {
  const mode = selectedGenerationMode();
  const isT2V = mode === "t2v";
  if (lastGenerationMode !== mode) {
    if (isT2V) {
      savedI2VNumFrames = $("numFrames").value;
      savedI2VContinuous = $("continuous").checked;
      $("numFrames").value = savedT2VNumFrames;
      $("continuous").checked = savedT2VContinuous;
    } else if (lastGenerationMode === "t2v") {
      savedT2VNumFrames = $("numFrames").value;
      savedT2VContinuous = $("continuous").checked;
      $("numFrames").value = savedI2VNumFrames;
      $("continuous").checked = savedI2VContinuous;
    }
  }
  $("referenceSection").hidden = isT2V;
  $("t2vFrameHint").hidden = !isT2V;
  $("numFrames").min = isT2V ? "1" : "5";
  $("numFrames").step = isT2V ? String(T2V_FRAME_STEP) : "4";
  $("continuous").disabled = false;
  $("numFrames").disabled = isT2V && $("continuous").checked;
  $("continuousLabelText").textContent = isT2V
    ? "Continuous T2V session"
    : "Continuous session";
  lastGenerationMode = mode;
  updateT2VFrameHint();
}

function readT2VNumFrames() {
  const numFrames = Number($("numFrames").value);
  if (
    !Number.isInteger(numFrames)
    || numFrames < 1
    || (numFrames - 1) % T2V_FRAME_STEP !== 0
  ) {
    throw new Error(
      `Zing T2V Frames must equal 1 + N × ${T2V_FRAME_STEP}`,
    );
  }
  return numFrames;
}

const PRESET_ASSET_BASE_URL = "./assets/presets/v1";

const reactorPresets = [
  {
    name: "Dragon Ride",
    tone: "green",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A locked first-person dragon-rider view matching the reference image: both tan forearms in brown leather gloves stay visible at the bottom, gripping leather reins around the green-brown scaled dragon neck; the dragon head, horns, and both wide wings frame the jungle valley, waterfalls, mist, and tall castle on the right. Smooth forward flight only, keep the same rider hands, dragon body, wing silhouette, castle placement, and humid daylight colors in every frame.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-dragon-ride.png`,
    mime: "image/png",
    source: "Reactor LingBot preset",
  },
  {
    name: "Misted Kingdom",
    tone: "green",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person over-the-shoulder fantasy view following a sword-slung rider on a brown horse through curling valley mist, wildflower meadows, ruined stone arches, cottages, and a many-spired castle under a ringed gas giant and crescent moon.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-misted-kingdom.png`,
    mime: "image/png",
    source: "Reactor LingBot preset",
  },
  {
    name: "Storm Crossing",
    tone: "blue",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person stern view of a battered grey aluminum work boat pushing through slate-black storm swells, wet wooden deck, warm cabin lamp, orange life rings, salt mist, churning wake, and a pale silver break in the dark horizon.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-storm-crossing.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Citadel Approach",
    tone: "accent",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person rear view of a mud-streaked vintage Defender 4x4 driving along a cobblestone-and-sand track through a coral-lit desert canyon toward a cliff-built sandstone citadel, with cacti, red poppies, ochre dunes, and peach sunset haze.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-citadel-approach.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Spring Valley",
    tone: "green",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person over-the-shoulder view following a golden retriever through a sunlit meadow with a patterned floral rug, stone bench, open book, potted seedling, cherry blossoms, rounded green oaks, soft hills, and a tender watercolor storybook atmosphere.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-spring-valley.png`,
    mime: "image/png",
    source: "Reactor LingBot preset",
  },
  {
    name: "Reef Patrol",
    tone: "blue",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person follow view trailing a large grey reef shark through clear tropical water above a sunlit coral reef, with drifting sediment, shifting sun-ray lattices, clouds of reef fish, a sardine bait ball, and deep blue open-water haze.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-reef-patrol.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Alpine Run",
    tone: "blue",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person rear view of a yellow four-person whitewater raft plunging through churning rapids in an alpine canyon, red lifejackets, yellow helmets, wet paddles, dark boulders, conifer slopes, and a snow-capped mountain at the vanishing point.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-alpine-run.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Ice Kayak",
    tone: "blue",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A centered elevated third-person game camera behind a lone kayaker in a bright red kayak crossing a calm deep blue alpine lake, scattered ice blocks, mirror reflections, huge snow-covered mountain ranges, vivid sky, and crisp cold wilderness scale.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-ice-kayak.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Penguin Colony",
    tone: "green",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person follow view of a single black-and-white penguin waddling across a windswept Antarctic ice shelf toward a distant colony, crystalline snow, small flippers, scattered dark boulders, rocky shoreline, and pale polar sky.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-penguin-colony.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Mars Mountain",
    tone: "accent",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A centered third-person rear view of a six-wheeled Martian rover marked XR-7A P-3317 crossing cracked basalt toward a vast volcanic mountain, dusty rose twilight, ochre wheel plumes, weathered grey panels, and a cold alien horizon.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-mars-mountain.jpg`,
    source: "Reactor LingBot preset",
  },
  {
    name: "Seaside Adventurer",
    tone: "green",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A centered third-person anime view behind a young girl on a flower-covered coastal hillside overlooking a sparkling blue bay, rolling green hills, sailboats, dramatic cliffs, a small lighthouse, huge fluffy clouds, and warm hand-painted adventure atmosphere.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-seaside-adventurer.png`,
    source: "Reactor LingBot preset",
    mime: "image/png",
  },
  {
    name: "Roman Chariot",
    tone: "accent",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A centered elevated third-person game camera behind a Roman warrior riding an ancient chariot pulled by two white horses across an open grassy field, worn stone path, Roman ruins, broken columns, bright midday sky, and epic historical scale.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-roman-chariot.png`,
    source: "Reactor LingBot preset",
    mime: "image/png",
  },
  {
    name: "Asylum Corridor",
    tone: "accent",
    size: "832x480",
    fps: DEFAULT_TARGET_FPS,
    prompt: "A third-person over-the-shoulder traversal behind a man in a wet leather jacket holding a flashlight down a derelict asylum corridor, standing water, torn vinyl strips, rusted ceiling debris, bloodstains, a toppled wheelchair, and a distant cyan-grey doorway glow.",
    referenceUrl: `${PRESET_ASSET_BASE_URL}/reactor-asylum-corridor.jpg`,
    source: "Reactor LingBot preset",
  },
];

const examplePresets = [
  { name: "Dragon Dolly", tone: "green", size: "832x480", fps: DEFAULT_TARGET_FPS, prompt: "A stable first-person dolly from the same dragon-rider viewpoint, keeping the black dragon head, horns, wings, jungle canopy, and distant castle consistent; slow forward camera motion, natural parallax, no creature morphing, no scene replacement.", referenceUrl: `${PRESET_ASSET_BASE_URL}/lingbot-example-00-dragon-dolly.jpg`, source: "LingBot example 00" },
];

function presetKey(preset) {
  if (!preset) return "";
  if (preset.isCustom && preset.fingerprint) {
    return `custom-${String(preset.fingerprint).replace(/[^a-zA-Z0-9]/g, "").slice(0, 64).toLowerCase()}`;
  }
  if (!presets.includes(preset)) return "";
  return String(preset.name || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

const FEATURED_PRESET_NAMES = [
  "Misted Kingdom",
  "Penguin Colony",
  "Seaside Adventurer",
  "Dragon Ride",
  "Spring Valley",
];
const featuredPresetNames = new Set(FEATURED_PRESET_NAMES);
const presets = [
  ...FEATURED_PRESET_NAMES.map((name) =>
    reactorPresets.find((preset) => preset.name === name)
  ),
  ...reactorPresets.filter((preset) => !featuredPresetNames.has(preset.name)),
  ...examplePresets,
];
const CUSTOM_WORLD_DB_NAME = "world-studio-custom-worlds";
const CUSTOM_WORLD_DB_VERSION = 1;
const CUSTOM_WORLD_STORE_NAME = "worlds";
let customWorldPresets = [];
let customWorldDbPromise = null;
let customWorldLoadPromise = null;
const MODEL_SLOT_DEFAULTS = ["minwm", "lingbot2", "happyoyster"];
let activeModelSlotCount = 2;

function selectedModelKeys() {
  const keys = [];
  for (let index = 0; index < activeModelSlotCount; index += 1) {
    const key = $(`modelSlot${index}`)?.value || MODEL_SLOT_DEFAULTS[index];
    if (!keys.includes(key)) keys.push(key);
  }
  return keys;
}

function modelSelected(key) {
  return selectedModelKeys().includes(key);
}

function syncModelSlotUi() {
  const selected = selectedModelKeys();
  const grid = document.querySelector(".model-player-grid");
  for (const key of MODEL_SLOT_DEFAULTS) {
    const player = document.querySelector(`[data-model-key="${key}"]`);
    if (player) player.hidden = !selected.includes(key);
  }
  for (const key of selected) {
    const player = document.querySelector(`[data-model-key="${key}"]`);
    if (player && grid) grid.appendChild(player);
  }
  grid?.classList.toggle("is-three-up", selected.length === 3);
  $("modelSlot2Wrap").hidden = activeModelSlotCount < 3;
  $("addModelSlotBtn").hidden = activeModelSlotCount >= 3;
  $("removeModelSlotBtn").hidden = activeModelSlotCount < 3;
}

function ensureUniqueModelSlot(changedIndex) {
  const changed = $(`modelSlot${changedIndex}`);
  if (!changed) return;
  for (let index = 0; index < activeModelSlotCount; index += 1) {
    if (index === changedIndex) continue;
    const other = $(`modelSlot${index}`);
    if (other?.value !== changed.value) continue;
    const replacement = MODEL_SLOT_DEFAULTS.find((key) => (
      key !== changed.value
      && !Array.from({ length: activeModelSlotCount }, (_, slotIndex) => (
        slotIndex === index ? null : $(`modelSlot${slotIndex}`)?.value
      )).includes(key)
    ));
    if (replacement) other.value = replacement;
  }
  syncModelSlotUi();
}

function allWorldPresets() {
  return [...presets, ...customWorldPresets];
}

let ws = null;
const h264ModelStats = { minwm: {}, lingbot2: {} };
const activeH264Models = new Set();
let selectedPreset = null;
let selectedReferenceBytes = null;
let selectedReferenceUrl = "";
let selectedReferenceLabel = "";
let selectedReferenceMimeType = "";
let selectedReferencePreviewReady = false;
let worldCompletionPending = false;
let skillRuleNextId = 1;
let goalRuleNextId = 1;
let preparedWorldRulesCache = null;
let worldRulesDraftGeneration = 0;
let sessionLifetimeExpired = false;
let sessionPlayable = false;
let worldExperiencePending = false;
let worldExperienceReady = false;
let lastGenerationMode = null;
let savedI2VNumFrames = "9";
let savedT2VNumFrames = String(DEFAULT_T2V_NUM_FRAMES);
let savedI2VContinuous = true;
let savedT2VContinuous = true;
let pendingHeader = null;
let frames = 0;
let bytes = 0;
let clearQueueOnClose = false;
let fpsSamples = [];
let renderLoopSamples = [];
let decodeQueue = [];
let queuedDecodeFrames = 0;
let queuedDecodeBytes = 0;
let decodeInProgress = false;
let pendingDecodeBatches = 0;
let droppedDecodeFrames = 0;
let lastDecodeDropAt = 0;
let lastDecodeDropCount = 0;
let lastRawRgbFrame = null;
let decoderWorker = null;
let decodeWorkerUnavailable = false;
let decodeRequestId = 1;
let streamEpoch = 0;
let lastDecodeMs = 0;
let lastDisplayLagMs = 0;
let lastRenderedChunk = null;
let lastReceivedChunk = null;
let lastReceivedFrameBatchIndex = null;
let frameBatchGapCount = 0;
let primaryProtocolStats = {};
let primaryNetworkSample = null;
const primaryControlSentEpochByEvent = new Map();
let encodedDecodeErrors = 0;
let socketHadError = false;
let socketCloseExpected = false;
let socketServerError = "";
let renderedPreviewFrames = 0;
let previewScaleFrame = 0;
let recordingActive = false;
let recordingTracks = [];
let recordingFrameIndex = 0;
let recordingFps = DEFAULT_TARGET_FPS;
let recordingTimer = 0;
let recordingFrameTimer = 0;
let recordingCaptureHandle = 0;
let recordingCaptureUsesIdle = false;
let recordingCapturePending = false;
let recordingStartedPerfMs = 0;
let recordingElapsedMs = 0;
let recordingLastCaptureMs = 0;
let recordingLastPresentedMs = 0;
let recordingDroppedFrames = 0;
let recordingSaving = false;
let recordingMode = "";
let recordingDirectoryHandle = null;
let recordingBaseFileName = "";
let recordingDownloads = [];
let recordingReadyToastTimer = 0;
let recordingReadyToastHideTimer = 0;
let goalAchievementToastHideTimer = 0;
let goalAchievementToastFinalizeTimer = 0;
let recordingPromptDraft = "";
let recordingPromptSubmitted = "";
let recordingPromptStatus = "idle";
let recordingPromptStatusPerfMs = 0;
let recordingPromptChangeType = "";
const recordingActionPulseUntil = new Map();
let currentSessionArtifact = null;
let recordingArtifact = null;
let currentTrace = null;
let renderedTraceChunks = new Set();
const decodeRequests = new Map();
let controlStateController = null;
let worldRulesController = null;
let runtimeSkillCooldownUiTimer = 0;
let lastSentEventId = 0;
let lastSampledEventId = 0;
const pendingModelEvents = new Map();
const traceTopologyApi = window.SGLangRealtimeTraceTopology || {};
const traceTopology = traceTopologyApi.createRealtimeTraceTopology
  ? traceTopologyApi.createRealtimeTraceTopology({ maxEvents: 220 })
  : null;
const traceTransportApi = window.SGLangRealtimeTraceTransport || {};
const traceHttpClient = traceTransportApi.RealtimeTraceHttpClient
  ? new traceTransportApi.RealtimeTraceHttpClient({
      onServerEvents: (events) => {
        events.forEach((event) => recordTraceTopologyEvent(event));
      },
      onAggregate: (aggregate) => {
        const metricsChanged = traceTopology?.setAggregate?.(aggregate);
        if (metricsChanged) renderTraceTopology();
        else if (traceTopology) updateTraceSummary(traceTopology.summary());
      },
    })
  : null;
const formatTraceDuration = traceTopologyApi.formatTraceDuration || formatMs;
let activeWorkspaceView = "preview";
let traceRenderFrame = 0;

const stage = document.querySelector(".stage");
const previewFrame = document.querySelector(".stage");
const fullscreenController = window.SGLangFullscreen?.createFullscreenController?.({
  documentRef: document,
  target: stage,
  button: $("fullscreenBtn"),
  onError: (error) => {
    addHistory(`fullscreen unavailable: ${error?.message || error}`);
  },
});
const canvas = $("minwmViewport");
const ctx = canvas.getContext("2d", { alpha: false });
const minwmH264Video = $("minwmH264Viewport");
const lingbot2Canvas = $("lingbot2Viewport");
const lingbot2H264Video = $("lingbot2H264Viewport");
const scratchCanvas = document.createElement("canvas");
const scratchCtx = scratchCanvas.getContext("2d", { alpha: false });
const recordingCanvas = document.createElement("canvas");
let recordingCtx = recordingCanvas.getContext("2d", { alpha: false });
const zingRecordingCanvas = document.createElement("canvas");
const zingRecordingCtx = zingRecordingCanvas.getContext("2d", { alpha: false });
const playbackController = new RealtimePlaybackController({
  mode: "adaptive",
  targetFps: DEFAULT_TARGET_FPS,
  lowLatencyPlayback: true,
  holdForTargetLead: true,
  targetLeadChunkRatio: 0.7,
  minTargetLeadMs: 260,
  maxTargetLeadMs: 900,
  lowLatencyMaxLeadFrames: 12,
  smoothTimelinePlaybackRateMin: 0.85,
  smoothTimelinePlaybackRateMax: DEFAULT_SMOOTH_CATCHUP_RATE,
  realtimeMaxBufferMs: ONLINE_MAX_BUFFER_MS,
  realtimeMaxBufferChunks: ONLINE_MAX_BUFFER_CHUNKS,
  realtimeMaxFrameAgeMs: ONLINE_MAX_FRAME_AGE_MS,
  startLeadChunkRatio: 0.45,
  minStartLeadMs: 220,
  resumeLeadChunkRatio: 0.45,
  minResumeLeadMs: 180,
  maxResumeLeadMs: 650,
  maxDeliveryLeadBoostMs: 0,
  deliveryStallExpectedMultiplier: 1.8,
});
let lingbot2ReconnectTimer = 0;
let lingbot2ReconnectAttempt = 0;
let lingbot2ReconnectInFlight = false;

function cancelLingbot2Reconnect() {
  if (lingbot2ReconnectTimer) window.clearTimeout(lingbot2ReconnectTimer);
  lingbot2ReconnectTimer = 0;
  lingbot2ReconnectAttempt = 0;
  lingbot2ReconnectInFlight = false;
}

function canReconnectLingbot2() {
  return (
    !sessionLifetimeExpired &&
    selectedGenerationMode() === "i2v" &&
    primarySessionConnected()
  );
}

function scheduleLingbot2Reconnect(reason = "media stream unavailable") {
  if (!canReconnectLingbot2() || lingbot2ReconnectTimer || lingbot2ReconnectInFlight) return;
  const delaysMs = [250, 1000, 2500, 4000];
  const delayMs = delaysMs[Math.min(lingbot2ReconnectAttempt, delaysMs.length - 1)];
  lingbot2ReconnectAttempt += 1;
  addHistory(`LingBot2 recovering in ${delayMs}ms · ${reason}`);
  lingbot2ReconnectTimer = window.setTimeout(async () => {
    lingbot2ReconnectTimer = 0;
    if (!canReconnectLingbot2()) return;
    lingbot2ReconnectInFlight = true;
    try {
      const restored = await dualModelController.reconnect("lingbot2");
      if (!restored || !canReconnectLingbot2()) return;
      lingbot2ReconnectAttempt = 0;
      addHistory("LingBot2 connection restored");
    } catch (error) {
      addHistory(`LingBot2 recovery failed · ${error.message || error}`);
      lingbot2ReconnectInFlight = false;
      scheduleLingbot2Reconnect(error.message || "retry failed");
      return;
    } finally {
      lingbot2ReconnectInFlight = false;
    }
  }, delayMs);
}

const lingbot2FallbackSession = new RealtimeModelSession({
  key: "lingbot2",
  canvas: lingbot2Canvas,
  overlay: $("lingbot2PreviewOverlay"),
  root: document.querySelector('[data-model-key="lingbot2"]'),
  pack,
  unpack,
  workerUrl: DECODER_WORKER_URL,
  startupMinChunk: 1,
  startupTimeoutMs: configuredModelNumber("lingbot2", "startupTimeoutMs", 60000),
  onState: (state, details) => {
    setModelConnectionState("lingbot2", state);
    if (state === "error") {
      addHistory(`LingBot2 error · ${details.message || details.reason || "unknown"}`);
    }
    if (state === "closed" && isSessionLifetimeReason(details.reason)) {
      expireSessionLifetime({ closeSessions: true });
    }
  },
  onStats: (stats) => {
    const root = document.querySelector('[data-model-key="lingbot2"]');
    if (!root) return;
    root.dataset.chunk = stats.lastChunk ?? "";
    root.dataset.frames = String(stats.frames || 0);
    renderModelTelemetry("lingbot2", stats);
    renderProtocolPerformance("lingbot2", { ...stats, transport: "webp" });
    markModelEventApplied("lingbot2", stats.lastAppliedEventId);
  },
  onFrame: () => {
    markSessionPlayable("lingbot2");
    notifyRecordingPresentedFrame("lingbot2");
  },
  onError: (error) => {
    if (isExperienceBusyError(error)) {
      handleExperienceBusy();
      return;
    }
    addHistory(`LingBot2 session failed · ${error.message || "unknown"}`);
    scheduleLingbot2Reconnect(error.message || "stream failed");
  },
});

function mirrorH264Video(key) {
  const video = key === "lingbot2" ? lingbot2H264Video : minwmH264Video;
  const target = key === "lingbot2" ? lingbot2Canvas : canvas;
  const width = Number(video?.videoWidth || 0);
  const height = Number(video?.videoHeight || 0);
  if (!video || !width || !height) return;
  if (target.width !== width || target.height !== height) {
    target.width = width;
    target.height = height;
  }
  const targetContext = target.getContext("2d", { alpha: false });
  targetContext.drawImage(video, 0, 0, width, height);
}

function createH264ModelSession(key) {
  const isLingBot2 = key === "lingbot2";
  const video = isLingBot2 ? lingbot2H264Video : minwmH264Video;
  const fallbackCanvas = isLingBot2 ? lingbot2Canvas : canvas;
  return new H264WebSocketSession({
    video,
    overlay: $(`${key}PreviewOverlay`),
    root: document.querySelector(`[data-model-key="${key}"]`),
    endpoint: h264WebSocketEndpoint(key),
    liveEdgeTargetMs: configuredNumber("h264WebSocketLiveEdgeTargetMs", 80),
    liveEdgeSeekThresholdMs: configuredNumber("h264WebSocketSeekThresholdMs", 420),
    onState: (state, details = {}) => {
      setModelConnectionState(key, state);
      if (state === "connecting") {
        video.hidden = true;
        fallbackCanvas.hidden = false;
      }
      if (["closed", "error", "unavailable"].includes(state)) {
        activeH264Models.delete(key);
        video.hidden = true;
        fallbackCanvas.hidden = false;
      }
      if (state === "error") {
        addHistory(`${modelLabel(key)} H.264 error · ${details.message || "unknown"}`);
      }
    },
    onPlayable: ({ width, height }) => {
      activeH264Models.add(key);
      video.hidden = false;
      fallbackCanvas.hidden = true;
      addHistory(`${modelLabel(key)} H.264 WebSocket live · ${width}x${height}`);
      markSessionPlayable(key);
    },
    onPresentedFrame: ({ eventId, presentedAt }) => {
      mirrorH264Video(key);
      markModelEventApplied(key, eventId);
      notifyRecordingPresentedFrame(key, presentedAt);
    },
    onStats: (stats) => {
      h264ModelStats[key] = { ...h264ModelStats[key], ...stats };
      const root = document.querySelector(`[data-model-key="${key}"]`);
      if (root) {
        root.dataset.chunk = h264ModelStats[key].lastChunk ?? "";
        root.dataset.frames = String(h264ModelStats[key].frames || 0);
      }
      renderModelTelemetry(key, h264ModelStats[key]);
      renderProtocolPerformance(key, { ...h264ModelStats[key], transport: "h264" });
    },
    onError: (error) => {
      addHistory(`${modelLabel(key)} H.264 session failed · ${error.message || error}`);
    },
  });
}

const minwmH264Session = H264_WEBSOCKET_ENABLED
  ? createH264ModelSession("minwm")
  : null;
const lingbot2H264Session = H264_WEBSOCKET_ENABLED
  ? createH264ModelSession("lingbot2")
  : null;

function preferredRealtimeSession(key, h264Session, fallbackSession) {
  let selected = fallbackSession;
  return {
    async connect(init, url) {
      if (h264Session) {
        try {
          await h264Session.connect(h264CompressionInit(init, key));
          selected = h264Session;
          return;
        } catch (error) {
          await h264Session.close("H.264 startup failed", { emitState: false });
          addHistory(`${modelLabel(key)} H.264 启动失败，自动回退 WebP · ${error.message || error}`);
          const video = key === "lingbot2" ? lingbot2H264Video : minwmH264Video;
          const fallbackCanvas = key === "lingbot2" ? lingbot2Canvas : canvas;
          video.hidden = true;
          fallbackCanvas.hidden = false;
        }
      }
      selected = fallbackSession;
      return fallbackSession.connect(init, url);
    },
    sendEvent(envelope) { return selected?.sendEvent(envelope) || false; },
    close(reason) {
      void h264Session?.close(reason);
      fallbackSession?.close?.(reason);
    },
    setUnavailable(reason) {
      h264Session?.setUnavailable?.(reason);
      fallbackSession?.setUnavailable?.(reason);
    },
    configure(options) { fallbackSession?.configure?.(options); },
    snapshot() { return selected?.snapshot?.() || h264ModelStats[key] || {}; },
    get active() { return Boolean(selected?.active); },
    get connected() { return Boolean(selected?.connected); },
    get bufferedAmount() { return Number(selected?.bufferedAmount || 0); },
  };
}

const lingbot2Session = preferredRealtimeSession(
  "lingbot2",
  lingbot2H264Session,
  lingbot2FallbackSession,
);
const happyOysterSession = new HappyOysterSession({
  video: $("happyoysterViewport"),
  overlay: $("happyoysterPreviewOverlay"),
  root: document.querySelector('[data-model-key="happyoyster"]'),
  onState: (state, details) => {
    setModelConnectionState("happyoyster", state);
    const fallback = {
      preparing: "正在准备快乐生蚝 World…",
      ready: "World 已就绪，正在连接视频流…",
      connecting: "正在连接快乐生蚝视频流…",
      live: "快乐生蚝已连接",
      unavailable: details.reason || "当前模式不可用",
      error: details.message || details.reason || "快乐生蚝连接失败",
      closed: "快乐生蚝连接已关闭",
      idle: "等待进入世界",
    }[state];
    setHappyOysterStageText(details.message || fallback, state);
    setHappyOysterProgress(details.progress, state);
    if (state === "live") markSessionPlayable("happyoyster");
    if (state === "error") {
      addHistory(`快乐生蚝 error · ${details.message || details.reason || "unknown"}`);
    }
  },
  onError: (error) => {
    addHistory(`快乐生蚝 session failed · ${error.message || "unknown"}`);
  },
});

function buildHappyOysterInit(init) {
  const unchangedPresetKey = selectedWorldContentIsUnchanged(init.prompt)
    ? presetKey(selectedPreset)
    : "";
  const customKey = `custom-${fallbackBytesFingerprint(init.first_frame).split("-").at(-1)}-${fallbackBytesFingerprint(new TextEncoder().encode(init.prompt || "")).split("-").at(-1)}`;
  return {
    prompt: init.prompt,
    firstFrame: init.first_frame,
    firstFrameMimeType: selectedReferenceMimeType || selectedPreset?.mime || "image/png",
    perspective: /first[-_ ]person/i.test(init.prompt || "") ? "first_person" : "third_person",
    presetKey: unchangedPresetKey || customKey,
  };
}

let primaryUsesH264 = false;
const primarySessionAdapter = {
  async connect(init, url) {
    if (minwmH264Session) {
      try {
        await minwmH264Session.connect(h264CompressionInit(init, "minwm"));
        primaryUsesH264 = true;
        return;
      } catch (error) {
        await minwmH264Session.close("H.264 startup failed", { emitState: false });
        addHistory(`Zing H.264 启动失败，自动回退 WebP · ${error.message || error}`);
        minwmH264Video.hidden = true;
        canvas.hidden = false;
      }
    }
    primaryUsesH264 = false;
    return openPrimarySession(init, url);
  },
  sendEvent(envelope) {
    return primaryUsesH264
      ? minwmH264Session?.sendEvent(envelope) || false
      : sendPrimaryEventEnvelope(envelope);
  },
  close(reason) {
    if (primaryUsesH264) {
      streamEpoch += 1;
      primaryUsesH264 = false;
      void minwmH264Session?.close(reason);
      return;
    }
    abortCurrentSession(reason, { expectedClose: true });
  },
};

function primarySessionConnected() {
  return primaryUsesH264
    ? Boolean(minwmH264Session?.connected)
    : Boolean(ws && ws.readyState === WebSocket.OPEN);
}

function primaryTransportBufferedAmount() {
  return primaryUsesH264
    ? Number(minwmH264Session?.bufferedAmount || 0)
    : Number(ws?.bufferedAmount || 0);
}
const dualModelController = new DualModelController({
  sessions: {
    minwm: primarySessionAdapter,
    lingbot2: lingbot2Session,
    happyoyster: happyOysterSession,
  },
  backends: {
    minwm: {
      enabled: () => modelSelected("minwm"),
      model: () => String(
        DUAL_MODEL_CONFIG.minwm?.model
        || UI_CONFIG.minwmModel
        || $("model").value
        || "minwm"
      ),
      wsUrl: (init) => backendWebSocketUrl("minwm", init.trace_id),
    },
    lingbot2: {
      model: String(
        DUAL_MODEL_CONFIG.lingbot2?.model
        || UI_CONFIG.lingbot2Model
        || DEFAULT_LINGBOT2_MODEL
      ),
      transformInit: (init) => {
        const modelParams = readModelRequestParams("lingbot2", {
          generationMode: init.generation_mode,
          firstFrame: init.first_frame,
        });
        const interactiveInit = {
          ...init,
          ...modelParams,
          realtime_interactive_event_grace_ms: 1800,
        };
        const is720p = interactiveInit.size === "1280x704" || interactiveInit.size === "1280x720";
        if (!is720p) return interactiveInit;
        return {
          ...interactiveInit,
          size: "1280x720",
        };
      },
      enabled: (init) => modelSelected("lingbot2") && init.generation_mode !== "t2v",
      unavailableReason: "T2V unavailable",
      wsUrl: (init) => backendWebSocketUrl("lingbot2", init.trace_id),
    },
    happyoyster: {
      model: "happyoyster-adventure",
      nonBlocking: true,
      enabled: (init) => modelSelected("happyoyster") && init.generation_mode === "i2v",
      unavailableReason: "仅支持 I2V Adventure",
      transformInit: buildHappyOysterInit,
      wsUrl: "",
    },
  },
  onBackgroundState: ({ key, state, error }) => {
    if (key !== "happyoyster") return;
    if (state === "connected") {
      addHistory("快乐生蚝 RTC 已连接并加入同步控制");
      return;
    }
    if (state === "failed") {
      const message = error?.message || String(error || "连接失败");
      setHappyOysterStageText(message, "error");
      addHistory(`快乐生蚝后台接入失败 · ${message}`);
    }
  },
});
const PROMPT_LOG_LIMIT = 100;
let promptLogEntries = [];
let promptLogNextId = 1;

function promptLogTypeLabel(changeType) {
  return changeType === "one_time" ? "一次性" : "持久";
}

function promptLogSourceLabel(trigger) {
  if (trigger === "user") return "用户输入";
  if (trigger === "skill") return "技能按键";
  return "规则触发";
}

function promptLogRuleLabel(rule, afterMs, entry = {}) {
  if (rule === "session_start") return "进入世界，建立初始持久状态";
  if (rule === "preset_runtime_update") return "切换世界预设，重建持久状态";
  if (rule === "rewrite_failure_restore") return "新指令改写失败，恢复持久状态";
  if (rule === "goal_time_probability") {
    return `游玩 ${Number(entry.minPlaySeconds || 0)} 秒后目标概率命中 ${Number(entry.probability || 0)} · ${entry.goalName || "隐藏目标"}`;
  }
  if (rule === "one_time_timeout_restore") {
    return `一次性指令持续 ${Math.round(Number(afterMs || 10000) / 1000)} 秒后恢复`;
  }
  return "系统规则发送";
}

function clearPromptLog() {
  promptLogEntries = [];
  promptLogNextId = 1;
  renderPromptLog();
}

function appendPromptLog(prompt, metadata = {}) {
  const normalizedPrompt = String(prompt || "").trim();
  if (!normalizedPrompt) return;
  const trigger = metadata.trigger === "skill"
    ? "skill"
    : metadata.trigger === "rule" || metadata.phase === "restore"
      ? "rule"
      : "user";
  const changeType = metadata.changeType === "one_time" ? "one_time" : "persistent";
  promptLogEntries.push({
    id: promptLogNextId++,
    timestamp: new Date(),
    prompt: normalizedPrompt,
    trigger,
    changeType,
    instruction: trigger === "user" ? String(metadata.instruction || "").trim() : "",
    skillName: trigger === "skill" ? String(metadata.skillName || "").trim() : "",
    skillInstruction: trigger === "skill" ? String(metadata.instruction || "").trim() : "",
    rule: trigger === "rule" ? String(metadata.rule || "") : "",
    afterMs: Number(metadata.afterMs || 0),
    goalName: trigger === "rule" ? String(metadata.goalName || "") : "",
    probability: trigger === "rule" ? Number(metadata.probability || 0) : 0,
    minPlaySeconds: trigger === "rule" ? Number(metadata.minPlaySeconds || 0) : 0,
  });
  if (promptLogEntries.length > PROMPT_LOG_LIMIT) {
    promptLogEntries.splice(0, promptLogEntries.length - PROMPT_LOG_LIMIT);
  }
  renderPromptLog();
}

function renderPromptLog() {
  const list = $("promptLogList");
  const empty = $("promptLogEmpty");
  const count = $("promptLogCount");
  if (!list || !empty || !count) return;
  count.textContent = `${promptLogEntries.length} 条`;
  empty.hidden = promptLogEntries.length > 0;
  list.innerHTML = "";
  for (const entry of [...promptLogEntries].reverse()) {
    const item = document.createElement("li");
    item.className = "prompt-log-entry";
    item.dataset.trigger = entry.trigger;
    item.dataset.changeType = entry.changeType;
    const header = document.createElement("div");
    header.className = "prompt-log-entry-header";
    const sequence = document.createElement("b");
    sequence.textContent = `#${entry.id}`;
    const source = document.createElement("span");
    source.className = "prompt-log-badge prompt-log-source";
    source.textContent = promptLogSourceLabel(entry.trigger);
    const type = document.createElement("span");
    type.className = "prompt-log-badge prompt-log-type";
    type.textContent = promptLogTypeLabel(entry.changeType);
    const time = document.createElement("time");
    time.dateTime = entry.timestamp.toISOString();
    time.textContent = entry.timestamp.toLocaleTimeString("zh-CN", { hour12: false });
    header.append(sequence, source, type, time);
    const context = document.createElement("p");
    context.className = "prompt-log-context";
    context.textContent = entry.trigger === "user"
      ? `用户输入：${entry.instruction || "（未记录）"}`
      : entry.trigger === "skill"
        ? `技能：${entry.skillName || "未命名技能"} · ${entry.skillInstruction}`
        : `触发规则：${promptLogRuleLabel(entry.rule, entry.afterMs, entry)}`;
    const fullPrompt = document.createElement("pre");
    fullPrompt.className = "prompt-log-full";
    fullPrompt.textContent = entry.prompt;
    item.append(header, context, fullPrompt);
    list.appendChild(item);
  }
}

function beginPromptLogSession(prompt, rule = "session_start") {
  clearPromptLog();
  appendPromptLog(prompt, {
    trigger: "rule",
    changeType: "persistent",
    rule,
  });
}

const promptRewriteController = new PromptRewriteController({
  rewrite: rewriteRuntimePrompt,
  sendPrompt: (prompt, metadata) => {
    const eventId = sendEvent(
      "prompt",
      prompt,
      metadata.phase === "restore"
        ? "persistent prompt restored"
        : `${metadata.changeType} prompt update`,
    );
    if (metadata.phase === "restore" && eventId) {
      setPromptRewriteStatus("已恢复上一条持久指令", "persistent");
    }
    if (eventId) {
      appendPromptLog(prompt, metadata);
      markRecordingPromptSent(prompt, metadata, eventId);
    }
    return eventId;
  },
  restoreDelayMs: 10000,
});
worldRulesController = new WorldRulesController({
  completeRule: completeWorldRule,
  dispatchPrepared: (prepared, metadata) => {
    markRecordingPromptSubmitted(metadata.instruction || metadata.skillName || metadata.goalName || "");
    return promptRewriteController.submitPrepared(
      prepared,
      metadata.instruction || "",
      metadata,
    );
  },
  skillCooldownMs: 10000,
  achievementDelayMs: 5000,
  onAchievement: (goal) => showGoalAchievement(goal.name),
  onGoalResult: (result, goal) => {
    if (result?.triggered) {
      const changeType = result.result?.change_type === "persistent"
        ? "persistent"
        : "one_time";
      setPromptRewriteStatus("目标规则已自动触发 · 5 秒后显示达成提示", changeType);
      addHistory(
        `goal auto-triggered · ${goal.name} · ${goal.min_play_seconds}s · probability ${goal.probability}`,
      );
      return;
    }
    if (!result?.canceled) {
      addHistory(
        `goal probability missed · ${goal.name} · roll ${Number(result?.roll || 0).toFixed(4)} / ${goal.probability}`,
      );
    }
  },
  onGoalError: (error, goal) => {
    addHistory(`goal auto-trigger failed · ${goal.name} · ${error.message || error}`);
    setPromptRewriteStatus("目标规则自动发送失败", "error");
  },
  onStateChange: (snapshot) => renderRuntimeSkillBar(snapshot),
});
let playbackAckTimer = 0;
let lastRenderedEventId = 0;
let primaryHasVisibleFrame = false;
let sessionCountdownTimer = 0;
let sessionCountdownDeadlineMs = 0;
const sessionLifetimeGuard = new SessionLifetimeGuard({
  durationMs: SESSION_MAX_LIFETIME_MS,
  onExpire: () => expireSessionLifetime({ closeSessions: true }),
});

function isSessionLifetimeReason(reason) {
  return String(reason || "").toLowerCase().includes("maximum session lifetime reached");
}

function resetSessionLifetimeUi() {
  sessionLifetimeGuard.cancel();
  stopSessionCountdown();
  hideRecordingReadyToast({ immediate: true });
  hideGoalAchievement({ immediate: true });
  sessionLifetimeExpired = false;
  sessionPlayable = false;
  worldExperiencePending = false;
  worldExperienceReady = false;
  $("sessionNotice").hidden = true;
  updateRecordButton();
}

function markWorldExperienceReady(modelKey) {
  if (!worldExperiencePending || worldExperienceReady || sessionLifetimeExpired) return false;
  worldExperiencePending = false;
  worldExperienceReady = true;
  sessionLifetimeGuard.start();
  startSessionCountdown();
  startRecording({ source: "first_visible_frame" });
  worldRulesController?.startSession();
  setStatus("Live", "live");
  $("sessionNotice").hidden = true;
  renderRuntimeSkillBar();
  addHistory(`world ready · ${modelLabel(modelKey)} visible · timer and dual recordings started`);
  return true;
}

function stopWorldExperienceTiming({ recordingReason = "session_closed" } = {}) {
  worldExperiencePending = false;
  worldExperienceReady = false;
  sessionPlayable = false;
  sessionLifetimeGuard.cancel();
  stopSessionCountdown();
  worldRulesController?.endSession();
  hideGoalAchievement({ immediate: true });
  if (recordingActive) void stopRecording({ reason: recordingReason });
  updateRecordButton();
}

function markSessionPlayable(modelKey) {
  if (sessionPlayable || sessionLifetimeExpired || !modelSelected(modelKey)) return false;
  sessionPlayable = true;
  markWorldExperienceReady(modelKey);
  recordTrajectoryEvent("session_playable", { model: modelKey });
  return true;
}

function schedulePrimaryPlaybackAck() {
  if (!PLAYBACK_ACK_ENABLED || playbackAckTimer || !ws || ws.readyState !== WebSocket.OPEN) return;
  playbackAckTimer = window.setTimeout(() => {
    playbackAckTimer = 0;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(pack({
      type: "event",
      kind: "playback_ack",
      trace_id: currentTrace?.traceId,
      payload: {
        last_received_chunk: lastReceivedChunk,
        last_rendered_chunk: lastRenderedChunk,
        last_rendered_event_id: lastRenderedEventId,
        playable: primaryHasVisibleFrame,
      },
    }));
  }, PLAYBACK_ACK_INTERVAL_MS);
}

function formatSessionCountdown(seconds) {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function updateSessionCountdown() {
  const remainingSeconds = Math.max(
    0,
    Math.ceil((sessionCountdownDeadlineMs - Date.now()) / 1000),
  );
  $("sessionCountdownText").textContent = formatSessionCountdown(remainingSeconds);
  $("sessionCountdown").classList.toggle("is-ending", remainingSeconds <= 10);
  if (remainingSeconds === 0 && sessionCountdownTimer) {
    window.clearInterval(sessionCountdownTimer);
    sessionCountdownTimer = 0;
  }
}

function startSessionCountdown() {
  stopSessionCountdown();
  sessionCountdownDeadlineMs = Date.now() + SESSION_MAX_LIFETIME_MS;
  $("sessionCountdown").hidden = false;
  updateSessionCountdown();
  sessionCountdownTimer = window.setInterval(updateSessionCountdown, 1000);
}

function stopSessionCountdown() {
  if (sessionCountdownTimer) window.clearInterval(sessionCountdownTimer);
  sessionCountdownTimer = 0;
  sessionCountdownDeadlineMs = 0;
  $("sessionCountdown").hidden = true;
  $("sessionCountdown").classList.remove("is-ending");
  $("sessionCountdownText").textContent = formatSessionCountdown(
    Math.ceil(SESSION_MAX_LIFETIME_MS / 1000),
  );
}

function showSessionNotice(message) {
  $("sessionNotice").textContent = message;
  $("sessionNotice").hidden = false;
}

function hideRecordingReadyToast({ immediate = false } = {}) {
  if (recordingReadyToastTimer) window.clearTimeout(recordingReadyToastTimer);
  if (recordingReadyToastHideTimer) window.clearTimeout(recordingReadyToastHideTimer);
  recordingReadyToastTimer = 0;
  recordingReadyToastHideTimer = 0;
  const toast = $("recordingReadyToast");
  if (!toast) return;
  toast.classList.remove("is-visible");
  if (immediate) {
    toast.hidden = true;
    return;
  }
  recordingReadyToastHideTimer = window.setTimeout(() => {
    if (!toast.classList.contains("is-visible")) toast.hidden = true;
    recordingReadyToastHideTimer = 0;
  }, 180);
}

function showRecordingReadyToast() {
  const toast = $("recordingReadyToast");
  if (!toast) return;
  hideRecordingReadyToast({ immediate: true });
  toast.hidden = false;
  window.requestAnimationFrame(() => toast.classList.add("is-visible"));
  recordingReadyToastTimer = window.setTimeout(
    () => hideRecordingReadyToast(),
    RECORDING_READY_TOAST_MS,
  );
}

function hideGoalAchievement({ immediate = false } = {}) {
  if (goalAchievementToastHideTimer) window.clearTimeout(goalAchievementToastHideTimer);
  if (goalAchievementToastFinalizeTimer) window.clearTimeout(goalAchievementToastFinalizeTimer);
  goalAchievementToastHideTimer = 0;
  goalAchievementToastFinalizeTimer = 0;
  const toast = $("goalAchievementToast");
  if (!toast) return;
  toast.classList.remove("is-visible");
  if (immediate) {
    toast.hidden = true;
    return;
  }
  goalAchievementToastFinalizeTimer = window.setTimeout(() => {
    if (!toast.classList.contains("is-visible")) toast.hidden = true;
    goalAchievementToastFinalizeTimer = 0;
  }, 300);
}

function showGoalAchievement(goalName) {
  const toast = $("goalAchievementToast");
  if (!toast) return;
  hideGoalAchievement({ immediate: true });
  $("goalAchievementText").textContent = `你成功获得「${String(goalName || "隐藏目标").trim()}」`;
  toast.hidden = false;
  window.requestAnimationFrame(() => toast.classList.add("is-visible"));
  goalAchievementToastHideTimer = window.setTimeout(
    () => hideGoalAchievement(),
    4500,
  );
}

function isExperienceBusyError(error) {
  const reason = String(error?.reason || "");
  const message = String(error?.message || error || "");
  return reason === "USER_SESSION_LIMIT" || message.includes("USER_SESSION_LIMIT");
}

function handleExperienceBusy() {
  promptRewriteController.endSession();
  stopWorldExperienceTiming({ recordingReason: "admission_rejected" });
  dualModelController.close("showcase session is occupied");
  $("connectBtn").disabled = false;
  setStatus("Busy", "error");
  setPreviewState("idle");
  showSessionNotice(EXPERIENCE_BUSY_MESSAGE);
  addHistory(EXPERIENCE_BUSY_MESSAGE);
}

function expireSessionLifetime({ closeSessions = false } = {}) {
  if (sessionLifetimeExpired) return;
  sessionLifetimeExpired = true;
  promptRewriteController.endSession();
  const finalizingRecording = recordingActive;
  stopWorldExperienceTiming({ recordingReason: "session_timeout" });
  if (closeSessions) dualModelController.close("maximum session lifetime reached");
  showSessionNotice(finalizingRecording
    ? "本轮体验已结束，正在生成游玩录像…"
    : "连接已断开，请重新连接");
  $("connectBtn").disabled = false;
  setStatus("Disconnected", "error");
  addHistory("连接已断开，请重新连接");
}

function setStatus(text, kind = "") {
  $("statusText").textContent = text;
  $("statusDot").className = "dot" + (kind ? ` ${kind}` : "");
}

function setModelConnectionState(key, state) {
  const root = document.querySelector(`[data-model-key="${key}"]`);
  if (root) root.dataset.sessionState = state;
  const label = document.getElementById(`${key}ConnectionText`);
  if (!label) return;
  label.textContent = {
    preparing: "构建中",
    ready: "准备完成",
    connecting: "连接中",
    live: "已连接",
    unavailable: "不可用",
    error: "连接异常",
    closed: "已断开",
    idle: "待连接",
  }[state] || "待连接";
  renderRuntimeSkillBar();
}

function setHappyOysterStageText(message, state = "") {
  const text = $("happyoysterStageText");
  if (!text) return;
  text.textContent = message || "正在准备快乐生蚝…";
  if (state) text.dataset.state = state;
  else delete text.dataset.state;
}

function setHappyOysterProgress(progress, state = "") {
  const root = $("happyoysterProgress");
  const bar = $("happyoysterProgressBar");
  if (!root || !bar) return;
  const value = Math.max(0, Math.min(100, Number(progress) || 0));
  root.setAttribute("aria-valuenow", String(Math.round(value)));
  root.hidden = state === "live" || state === "idle" || state === "closed";
  bar.style.setProperty("--progress", `${value}%`);
}

function setHappyOysterReferencePreview(dataUrl = "") {
  const image = $("happyoysterReferenceImage");
  if (!image) return;
  image.src = dataUrl;
  image.hidden = !dataUrl;
}

function setPreviewState(state) {
  if (!stage) return;
  stage.dataset.previewState = state;
  canvas.setAttribute("aria-busy", state === "waiting" ? "true" : "false");
}

function addHistory(text) {
  const item = document.createElement("span");
  const now = new Date();
  const ms = String(now.getMilliseconds()).padStart(3, "0");
  item.textContent = `${now.toLocaleTimeString("zh-CN", { hour12: false })}.${ms} ${text}`;
  $("historyList").prepend(item);
  while ($("historyList").children.length > 8) $("historyList").lastChild.remove();
}

function createClientTrace() {
  return {
    traceId: createTraceId(),
    seq: 0,
    createdPerfMs: performance.now(),
    createdEpochMs: Date.now(),
    events: [],
  };
}

function createTraceId() {
  if (crypto.randomUUID) return crypto.randomUUID().replaceAll("-", "");
  const random = crypto.getRandomValues(new Uint32Array(4));
  return Array.from(random, (part) => part.toString(16).padStart(8, "0")).join("");
}

function stableBrowserUserId() {
  try {
    let value = localStorage.getItem(BROWSER_USER_ID_STORAGE_KEY);
    if (!value) {
      value = createTraceId();
      localStorage.setItem(BROWSER_USER_ID_STORAGE_KEY, value);
    }
    return value;
  } catch {
    return createTraceId();
  }
}

const browserUserId = stableBrowserUserId();

function traceWebSocketUrl(baseUrl) {
  try {
    const url = new URL(baseUrl, window.location.href);
    if (currentTrace) url.searchParams.set("trace_id", currentTrace.traceId);
    url.searchParams.set("user_id", browserUserId);
    return url.toString();
  } catch {
    const separator = baseUrl.includes("?") ? "&" : "?";
    const trace = currentTrace
      ? `trace_id=${encodeURIComponent(currentTrace.traceId)}&`
      : "";
    return `${baseUrl}${separator}${trace}user_id=${encodeURIComponent(browserUserId)}`;
  }
}

function backendWebSocketUrl(key, traceId) {
  const configuredUrl = DUAL_MODEL_CONFIG[key]?.wsUrl;
  const baseUrl = configuredUrl || $("serverUrl").value;
  const configuredUserId = UI_CONFIG.singleExperienceUserIds?.[key];
  const backendUserId = UI_CONFIG.singleExperience && configuredUserId
    ? String(configuredUserId)
    : `${browserUserId}:${key}`;
  try {
    const url = new URL(baseUrl, window.location.href);
    if (!configuredUrl) {
      url.pathname = `/backends/${key}/v1/realtime_video/generate`;
      url.search = "";
    }
    if (traceId) url.searchParams.set("trace_id", traceId);
    url.searchParams.set("user_id", backendUserId);
    return url.toString();
  } catch {
    return baseUrl;
  }
}

function markClientTrace(name, fields = {}, options = {}) {
  if (!currentTrace) return null;
  const event = {
    name,
    seq: ++currentTrace.seq,
    trace_id: currentTrace.traceId,
    client_perf_ms: roundTraceNumber(performance.now()),
    client_epoch_ms: Date.now(),
    ...fields,
  };
  currentTrace.events.push(event);
  if (currentTrace.events.length > 64) currentTrace.events.shift();
  recordTraceTopologyEvent(event);
  if (options.send !== false) traceHttpClient?.enqueueClientEvent(event);
  return event;
}

function roundTraceNumber(value) {
  return Math.round(Number(value || 0) * 1000) / 1000;
}

function recordTraceTopologyEvent(event, receivedPerfMs = performance.now()) {
  if (!traceTopology || !event) return;
  const traceEvent = event.trace ? event.trace : event;
  traceTopology.addEvent(traceEvent, receivedPerfMs);
  recordTrajectoryEvent("trace_event", { trace: traceEvent });
  renderTraceTopology();
}

function resetTraceTopology(traceId = "") {
  traceTopology?.reset(traceId);
  renderTraceTopology();
}

function renderTraceTopology() {
  if (traceRenderFrame) return;
  traceRenderFrame = requestAnimationFrame(() => {
    traceRenderFrame = 0;
    renderTraceTopologyNow();
  });
}

function renderTraceTopologyNow() {
  if (!traceTopology) return;
  const summary = traceTopology.summary();
  updateTraceSummary(summary);
  if (activeWorkspaceView !== "trace") return;
  renderTraceSvg(summary);
  renderTraceEventList(summary.recentEvents);
}

function updateTraceSummary(summary) {
  $("traceIdText").textContent = shortTraceId(summary.traceId);
  $("traceEventCountText").textContent = String(summary.eventCount);
  const aggregate = summary.aggregate;
  const observedLabel = aggregate
    ? `${aggregate.window?.seconds || 300}s · ${aggregate.stale ? "stale" : "fresh"} · ${aggregate.observed_at || "-"}`
    : "-";
  $("traceObservedText").textContent = observedLabel;
  const chunk = summary.latestChunk;
  $("traceChunkText").textContent = chunk ? `#${chunk.chunkIndex}` : "-";
  $("traceChunkTotalText").textContent = chunk ? formatTraceDuration(chunk.chunkTotalMs) : "-";
  $("traceSchedulerText").textContent = traceStageMetric(summary, "scheduler", chunk?.schedulerForwardMs);
  $("traceVaeEncodeText").textContent = traceStageMetric(summary, "vae_encode", chunk?.vaeEncodeMs);
  $("traceDenoiseText").textContent = traceStageMetric(summary, "denoise", chunk?.denoiseMs);
  const vaeDecodeMs = chunk
    ? sumTraceNumbers(chunk.vaeDecodeMs, chunk.postDecodeMs)
    : null;
  $("traceVaeDecodeText").textContent = traceStageMetric(summary, "vae_decode", vaeDecodeMs);
  $("traceAsyncEstimateText").textContent = formatAsyncEstimate(summary.asyncEstimate);
}

function traceStageMetric(summary, stageId, fallbackMs) {
  const stage = summary.aggregate?.stages?.find((candidate) => candidate.id === stageId);
  if (stage && Number(stage.count || 0) > 0) {
    return `p50 ${formatTraceDuration(stage.p50_ms)} · p95 ${formatTraceDuration(stage.p95_ms)}`;
  }
  return formatTraceDuration(fallbackMs);
}

function renderTraceSvg(summary) {
  const container = $("traceTopology");
  const nodes = summary.nodes || [];
  if (!nodes.length || summary.eventCount === 0) {
    container.innerHTML = `<svg viewBox="0 0 1180 240" role="img" aria-label="Trace topology"><text class="trace-empty" x="36" y="122">Trace events will appear after Generate starts.</text></svg>`;
    return;
  }

  const width = 1180;
  const height = 240;
  const marginX = 28;
  const nodeW = nodes.length > 8 ? 112 : 124;
  const nodeH = 74;
  const gap = (width - marginX * 2 - nodeW * nodes.length) / Math.max(1, nodes.length - 1);
  const nodeY = 72;
  const positions = new Map();
  nodes.forEach((node, index) => {
    positions.set(node.id, {
      x: marginX + index * (nodeW + gap),
      y: nodeY,
    });
  });

  const edges = (summary.edges || []).map((edge) => {
    const from = positions.get(edge.from);
    const to = positions.get(edge.to);
    if (!from || !to) return "";
    const x1 = from.x + nodeW;
    const x2 = to.x;
    const y = nodeY + nodeH / 2;
    return `
      <line class="trace-edge-line" x1="${x1}" y1="${y}" x2="${x2 - 8}" y2="${y}" />
      <text class="trace-edge-label" x="${(x1 + x2) / 2}" y="${y - 10}" text-anchor="middle">${escapeHtml(nodeLabel(edge.label || "-"))}</text>
    `;
  }).join("");

  const nodeMarkup = nodes.map((node) => {
    const pos = positions.get(node.id);
    return `
      <g class="trace-node ${node.status === "active" ? "is-active" : ""}" transform="translate(${pos.x} ${pos.y})">
        <rect width="${nodeW}" height="${nodeH}" rx="8"></rect>
        <text class="trace-node-title" x="12" y="24">${escapeHtml(node.title)}</text>
        <text class="trace-node-subtitle" x="12" y="43">${escapeHtml(node.subtitle || "")}</text>
        <text class="trace-node-metric" x="12" y="62">${escapeHtml(node.metric || "-")}</text>
      </g>
    `;
  }).join("");

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Realtime trace topology">
      <defs>
        <marker id="traceArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#8c9288"></path>
        </marker>
      </defs>
      ${edges}
      ${nodeMarkup}
    </svg>
  `;
}

function renderTraceEventList(events) {
  const list = $("traceEventList");
  list.replaceChildren();
  for (const event of [...events].reverse()) {
    const item = document.createElement("div");
    item.className = "trace-event-item";
    const name = document.createElement("b");
    name.textContent = event.event || event.name || "-";
    const time = document.createElement("span");
    time.textContent = traceEventTimeLabel(event);
    const details = document.createElement("code");
    details.textContent = traceEventDetails(event);
    item.append(name, time, details);
    list.appendChild(item);
  }
}

function traceEventTimeLabel(event) {
  if (Number.isFinite(Number(event.server_elapsed_ms))) {
    return `server +${formatTraceDuration(event.server_elapsed_ms)}`;
  }
  if (Number.isFinite(Number(event.client_perf_ms)) && currentTrace) {
    return `client +${formatTraceDuration(Number(event.client_perf_ms) - currentTrace.createdPerfMs)}`;
  }
  return "-";
}

function traceEventDetails(event) {
  const parts = [];
  if (event.chunk_index !== null && event.chunk_index !== undefined) parts.push(`chunk=${event.chunk_index}`);
  if (event.event_id !== null && event.event_id !== undefined) parts.push(`event=${event.event_id}`);
  if (Number.isFinite(Number(event.duration_ms))) parts.push(`duration=${formatTraceDuration(event.duration_ms)}`);
  if (Number.isFinite(Number(event.cuda_ms))) parts.push(`cuda=${formatTraceDuration(event.cuda_ms)}`);
  if (Number.isFinite(Number(event.chunk_total_ms))) parts.push(`chunk_total=${formatTraceDuration(event.chunk_total_ms)}`);
  if (Number.isFinite(Number(event.display_lag_ms))) parts.push(`display_lag=${formatTraceDuration(event.display_lag_ms)}`);
  if (event.content_type) parts.push(shortPayloadMode(event.content_type));
  return parts.join(" · ") || "-";
}

function formatAsyncEstimate(estimate) {
  if (!estimate) return "-";
  return `${formatTraceDuration(estimate.savedMs)} saved · ${estimate.speedup.toFixed(2)}x`;
}

function sumTraceNumbers(...values) {
  let total = 0;
  let seen = false;
  for (const value of values) {
    if (!Number.isFinite(Number(value))) continue;
    total += Number(value);
    seen = true;
  }
  return seen ? total : null;
}

function shortTraceId(traceId) {
  const value = String(traceId || "");
  if (!value) return "-";
  if (value.length <= 12) return value;
  return `${value.slice(0, 8)}...${value.slice(-4)}`;
}

function nodeLabel(value) {
  return value;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setWorkspaceView(view) {
  activeWorkspaceView = view === "trace" ? "trace" : "preview";
  document.querySelectorAll("[data-workspace-view]").forEach((button) => {
    const active = button.dataset.workspaceView === activeWorkspaceView;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".workspace-pane").forEach((pane) => {
    const active = pane.id === `${activeWorkspaceView}Pane`;
    pane.classList.toggle("is-active", active);
    pane.hidden = !active;
  });
  if (activeWorkspaceView === "trace") {
    renderTraceTopologyNow();
    traceHttpClient?.setActive(true, 5000);
  } else {
    traceHttpClient?.setActive(false);
  }
}

function updateControlDebugText() {
  const activeActions = controlStateController
    ? Array.from(controlStateController.activeActions).sort().join("+")
    : "";
  const activeText = activeActions || "idle";
  const sentText = lastSentEventId ? `sent #${lastSentEventId}` : "sent -";
  const sampledText = lastSampledEventId ? `sampled #${lastSampledEventId}` : "sampled -";
  $("actionStateText").textContent = `${activeText} · ${sentText} · ${sampledText}`;
}

function drawIdle() {
  const w = 1280, h = 720;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  setPreviewState("idle");
  renderedPreviewFrames = 0;
  ctx.fillStyle = "#11140f";
  ctx.fillRect(0, 0, w, h);
  if (lingbot2Canvas.width !== w || lingbot2Canvas.height !== h) {
    lingbot2Canvas.width = w;
    lingbot2Canvas.height = h;
  }
  const lingbot2Context = lingbot2Canvas.getContext("2d", { alpha: false });
  lingbot2Context.fillStyle = "#11140f";
  lingbot2Context.fillRect(0, 0, w, h);
}

function resetStreamStats() {
  pendingHeader = null;
  h264ModelStats.minwm = {};
  h264ModelStats.lingbot2 = {};
  activeH264Models.clear();
  minwmH264Video.hidden = true;
  lingbot2H264Video.hidden = true;
  canvas.hidden = false;
  lingbot2Canvas.hidden = false;
  clearFrameQueue();
  playbackController.reset({
    mode: selectedPlaybackMode(),
    targetFps: previewPlaybackTargetFps(),
  });
  frames = 0;
  bytes = 0;
  fpsSamples = [];
  clearQueueOnClose = false;
  decodeQueue = [];
  queuedDecodeFrames = 0;
  queuedDecodeBytes = 0;
  decodeInProgress = false;
  pendingDecodeBatches = 0;
  droppedDecodeFrames = 0;
  lastDecodeDropAt = 0;
  lastDecodeDropCount = 0;
  lastReceivedChunk = null;
  lastReceivedFrameBatchIndex = null;
  frameBatchGapCount = 0;
  primaryProtocolStats = {};
  primaryNetworkSample = null;
  primaryControlSentEpochByEvent.clear();
  encodedDecodeErrors = 0;
  renderedPreviewFrames = 0;
  lastSentEventId = 0;
  lastRenderedEventId = 0;
  primaryHasVisibleFrame = false;
  if (playbackAckTimer) window.clearTimeout(playbackAckTimer);
  playbackAckTimer = 0;
  lastSampledEventId = 0;
  pendingModelEvents.clear();
  lastRenderedChunk = null;
  renderedTraceChunks = new Set();
  controlStateController?.reset({ sendRelease: false });
  resetDecoderState();
  resetModelTelemetry("minwm");
  resetModelTelemetry("lingbot2");
  updateStats();
  $("actionStateText").textContent = "-";
  updateOutputSizeText();
}

function rejectPendingDecodes(message) {
  for (const request of decodeRequests.values()) {
    request.reject(new Error(message));
  }
  decodeRequests.clear();
}

function ensureDecoderWorker() {
  if (decoderWorker || decodeWorkerUnavailable) return;
  if (typeof Worker === "undefined") {
    decodeWorkerUnavailable = true;
    return;
  }

  decoderWorker = new Worker(DECODER_WORKER_URL);
  decoderWorker.onmessage = (event) => {
    const message = event.data;
    const request = decodeRequests.get(message.id);
    if (!request) return;
    decodeRequests.delete(message.id);
    if (message.type === "error") {
      request.reject(new Error(message.message || "decode failed"));
      return;
    }
    request.resolve(message);
  };
  decoderWorker.onerror = (event) => {
    decodeWorkerUnavailable = true;
    decoderWorker?.terminate();
    decoderWorker = null;
    rejectPendingDecodes(event.message || "decode worker failed");
  };
}

function resetDecoderState() {
  lastRawRgbFrame = null;
  if (decoderWorker) decoderWorker.postMessage({ type: "reset" });
}

async function decodeFrameBatch(header, data) {
  const decodeStartedAt = performance.now();
  if (!isWorkerDecodableContentType(header.content_type)) {
    const items = await framePayloadToImageData(header, data);
    const decodedAt = performance.now();
    lastDecodeMs = decodedAt - decodeStartedAt;
    return items.map((item) => ({
      ...item,
      receivedAt: header.__received_at,
      decodedAt,
      decodeMs: lastDecodeMs,
    }));
  }

  ensureDecoderWorker();
  if (!decoderWorker || decodeWorkerUnavailable) {
    const items = await framePayloadToImageData(header, data);
    const decodedAt = performance.now();
    lastDecodeMs = decodedAt - decodeStartedAt;
    return items.map((item) => ({
      ...item,
      receivedAt: header.__received_at,
      decodedAt,
      decodeMs: lastDecodeMs,
    }));
  }

  const payload = await payloadToArrayBuffer(data);
  const id = decodeRequestId++;
  const decodeHeader = { ...header, __decode_id: id };
  const useTransfer =
    isWorkerDecodableRawContentType(header.content_type) ||
    isEncodedPreviewContentType(header.content_type);
  try {
    return await new Promise((resolve, reject) => {
      decodeRequests.set(id, {
        resolve: (message) => {
          const decodedAt = performance.now();
          lastDecodeMs = decodedAt - decodeStartedAt;
          resolve(message.frames.map((frame) => ({
            image: message.frame_type === "bitmap"
              ? frame
              : new ImageData(new Uint8ClampedArray(frame), message.width, message.height),
            chunk: message.chunk,
            receivedAt: header.__received_at,
            decodedAt,
            decodeMs: lastDecodeMs,
          })));
        },
        reject,
      });
      try {
        decoderWorker.postMessage(
          { type: "decode", header: decodeHeader, payload },
          useTransfer ? [payload] : [],
        );
      } catch (error) {
        decodeRequests.delete(id);
        reject(error);
      }
    });
  } catch (error) {
    if (isEncodedPreviewContentType(header.content_type) && !useTransfer) {
      const items = await framePayloadToImageData(header, data);
      const decodedAt = performance.now();
      lastDecodeMs = decodedAt - decodeStartedAt;
      return items.map((item) => ({
        ...item,
        receivedAt: header.__received_at,
        decodedAt,
        decodeMs: lastDecodeMs,
      }));
    }
    throw error;
  }
}

function isWorkerDecodableContentType(contentType) {
  return isWorkerDecodableRawContentType(contentType);
}

function isWorkerDecodableRawContentType(contentType) {
  return (
    contentType === RAW_RGB_CONTENT_TYPE ||
    contentType === RAW_RGB_DELTA_GZIP_CONTENT_TYPE ||
    contentType === RAW_RGBA_DELTA_GZIP_CONTENT_TYPE
  );
}

function updateStats() {
  if (activeH264Models.has("minwm")) {
    renderModelTelemetry("minwm", h264ModelStats.minwm);
    renderProtocolPerformance("minwm", { ...h264ModelStats.minwm, transport: "h264" });
    return;
  }
  const playback = playbackController.snapshot();
  const totalDroppedFrames = playback.droppedFrames + droppedDecodeFrames;
  renderModelTelemetry("minwm", {
    ...playback,
    ...primaryProtocolStats,
    frames,
    bytes,
    lastChunk: lastRenderedChunk,
    frameBatchGapCount,
    lastDecodeMs,
    lastDisplayLagMs,
    renderFps: fpsSamples.length,
    droppedFrames: totalDroppedFrames,
    decodeQueueLength: pendingDecodeBatches,
  });
  renderProtocolPerformance("minwm", {
    ...playback,
    ...primaryProtocolStats,
    bytes,
    transport: "webp",
  });
}

function resetModelTelemetry(key) {
  renderModelTelemetry(key, {
    frames: 0,
    bytes: 0,
    lastChunk: null,
    lastDecodeMs: 0,
    lastDisplayLagMs: 0,
    renderFps: 0,
    sourceFps: 0,
    bufferMs: 0,
    queueFrames: 0,
  });
  renderProtocolPerformance(key, {});
}

function performanceMs(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0
    ? `${number.toFixed(number < 10 ? 1 : 0)} ms`
    : "-";
}

function protocolMetric(...values) {
  for (const value of values) {
    if (value === null || value === undefined || value === "") continue;
    const number = Number(value);
    if (Number.isFinite(number) && number >= 0) return number;
  }
  return null;
}

function protocolMetricText(value) {
  return value === null ? "-" : performanceMs(value);
}

function renderProtocolPerformance(key, stats = {}) {
  const telemetry = stats.chunkTelemetry || {};
  const isH264 = stats.transport === "h264" || stats.codec === "h264";
  const receiveMbps = protocolMetric(stats.receiveMbps);
  const bytesReceived = Number(stats.bytesReceived ?? stats.bytes ?? 0);
  const sourceFps = Number(stats.serverFps || stats.sourceFps || 0);
  const deliveryFps = Number(stats.deliveryFps || 0);
  const renderFps = Number(stats.renderFps || 0);
  const vaeQueueMs = protocolMetric(telemetry.vae_queue_wait_ms);
  const vaeDecodeMs = protocolMetric(
    telemetry.vae_decode_ms,
    telemetry.model_vae_decode_ms,
  );
  const h264FeedMs = protocolMetric(stats.lastBridgeEncoderFeedMs);
  const bridgeQueueMs = protocolMetric(stats.lastBridgeQueueMs);
  const webSocketDownlinkMs = protocolMetric(
    stats.lastWebSocketDownlinkMs,
    stats.lastDownlinkMs,
  );
  const mseQueueMs = protocolMetric(stats.lastMseQueueMs);
  const mseAppendMs = protocolMetric(stats.lastMseAppendMs);
  const playbackBufferMs = protocolMetric(
    stats.playbackBufferMs,
    stats.mseBufferMs,
    stats.bufferMs,
  );
  const inputUplinkMs = protocolMetric(
    stats.lastInputUplinkMs,
    telemetry.input_uplink_ms,
  );
  const e2eMs = protocolMetric(
    stats.lastPresentedControlToVideoMs,
    stats.lastControlToVideoMs,
  );
  $(`${key}PerfData`).textContent = bytesReceived > 0
    ? `${Number(receiveMbps || 0).toFixed(1)} Mb/s`
    : "-";
  $(`${key}PerfFps`).textContent = sourceFps > 0 || deliveryFps > 0 || renderFps > 0
    ? `源 ${sourceFps.toFixed(1)} · 收 ${deliveryFps.toFixed(1)} · 显 ${renderFps.toFixed(1)}`
    : "-";
  $(`${key}PerfUplink`).textContent = protocolMetricText(inputUplinkMs);
  $(`${key}PerfScheduler`).textContent = telemetry.scheduler_forward_ms != null
    ? performanceMs(telemetry.scheduler_forward_ms)
    : "-";
  $(`${key}PerfDenoise`).textContent = telemetry.model_denoise_ms != null
    ? performanceMs(telemetry.model_denoise_ms)
    : "-";
  $(`${key}PerfVae`).textContent = vaeQueueMs !== null || vaeDecodeMs !== null
    ? `q ${protocolMetricText(vaeQueueMs)} · dec ${protocolMetricText(vaeDecodeMs)}`
    : "-";
  $(`${key}PerfH264Queue`).textContent = isH264
    ? protocolMetricText(bridgeQueueMs)
    : "不适用";
  $(`${key}PerfH264Feed`).textContent = isH264
    ? protocolMetricText(h264FeedMs)
    : "不适用";
  $(`${key}PerfDownlink`).textContent = protocolMetricText(webSocketDownlinkMs);
  $(`${key}PerfMseQueue`).textContent = isH264
    ? protocolMetricText(mseQueueMs)
    : "不适用";
  $(`${key}PerfMseAppend`).textContent = isH264
    ? protocolMetricText(mseAppendMs)
    : "不适用";
  $(`${key}PerfPlaybackBuffer`).textContent = protocolMetricText(playbackBufferMs);
  $(`${key}PerfE2E`).textContent = protocolMetricText(e2eMs);
}

function renderModelTelemetry(key, stats = {}) {
  const prefix = key === "lingbot2" ? "lingbot2" : "minwm";
  const renderFps = Number(stats.renderFps || 0);
  const sourceFps = Number(stats.sourceFps || 0);
  const serverFps = Number(stats.serverFps || sourceFps);
  const deliveryFps = Number(stats.deliveryFps || sourceFps);
  const bufferMs = Number(stats.bufferMs || 0);
  const queueFrames = Number(stats.queueFrames ?? stats.queueLength ?? 0);
  const droppedFrames = Number(stats.droppedFrames || 0);
  const decodeQueueLength = Number(stats.decodeQueueLength || 0);
  const frameBatchGapCount = Number(stats.frameBatchGapCount || 0);
  const totalFrames = Number(stats.frames || 0);
  const bufferParts = [formatMs(bufferMs), `q ${queueFrames}`];
  if (decodeQueueLength) bufferParts.push(`decode ${decodeQueueLength}`);
  if (droppedFrames) bufferParts.push(`drop ${droppedFrames}`);
  if (frameBatchGapCount) bufferParts.push(`gap ${frameBatchGapCount}`);
  $(`${prefix}ChunkText`).textContent = stats.lastChunk == null ? "-" : `#${stats.lastChunk}`;
  $(`${prefix}RateText`).textContent = serverFps > 0 || deliveryFps > 0 || renderFps > 0
    ? `${serverFps.toFixed(1)} source · ${deliveryFps.toFixed(1)} recv · ${renderFps} render`
    : "-";
  $(`${prefix}BufferText`).textContent = bufferParts.join(" · ");
  $(`${prefix}FramesText`).textContent = `${totalFrames} · ${(Number(stats.bytes || 0) / 1048576).toFixed(1)} MB`;
  $(`${prefix}DecodeText`).textContent = stats.lastDecodeMs > 0
    ? `${Math.round(stats.lastDecodeMs)} ms`
    : "-";
  $(`${prefix}DisplayLagText`).textContent = stats.lastDisplayLagMs > 0
    ? `${(stats.lastDisplayLagMs / 1000).toFixed(1)} s`
    : "-";
}

function requestedInputFps(key = "minwm") {
  return Number(modelControl(key, "fps").value || DEFAULT_TARGET_FPS);
}

function frameInterpolationMultiplier(key = "minwm") {
  return modelControl(key, "frameInterpolation").checked
    ? 2 ** DEFAULT_FRAME_INTERPOLATION_EXP
    : 1;
}

function previewPlaybackTargetFps(key = "minwm") {
  return requestedInputFps(key) * frameInterpolationMultiplier(key);
}

function syncPlaybackTargetFps() {
  playbackController.setTargetFps(previewPlaybackTargetFps("minwm"));
  lingbot2Session.configure({ targetFps: previewPlaybackTargetFps("lingbot2") });
  updateStats();
}

function syncSmoothCatchupRate() {
  const rate = Math.min(2.5, Math.max(1, Number($("smoothCatchupRate").value) || 1.1));
  $("smoothCatchupRate").value = String(rate);
  $("smoothCatchupRateText").textContent = `${rate.toFixed(2)}x`;
  playbackController.setSmoothTimelinePlaybackRateMax(rate);
  lingbot2Session.configure({ smoothTimelinePlaybackRateMax: rate });
}

function syncZingFrameInterpolation({ fromTopbar = true } = {}) {
  const topbar = $("zingFrameInterpolation");
  const requestControl = modelControl("minwm", "frameInterpolation");
  if (fromTopbar) requestControl.checked = topbar.checked;
  else topbar.checked = requestControl.checked;
  tunePreviewQualityForPostprocess("minwm");
  syncPlaybackTargetFps();
}

function selectedPlaybackMode(key = "minwm") {
  const value = modelControl(key, "playbackMode")?.value;
  if (value === "timeline" || value === "adaptive" || value === "smooth_timeline") return value;
  return "live";
}

function syncPlaybackMode({ addToHistory = true } = {}) {
  const mode = selectedPlaybackMode("minwm");
  const lingbot2Mode = selectedPlaybackMode("lingbot2");
  playbackController.setMode(mode);
  lingbot2Session.configure({ mode: lingbot2Mode });
  if (addToHistory) {
    const historyText =
      mode === "timeline"
        ? "playback · full timeline (no frame skipping)"
        : mode === "smooth_timeline"
        ? "playback · smooth timeline (bounded lag)"
        : mode === "adaptive"
        ? "playback · adaptive (buffered, fast input)"
        : "playback · low latency (may skip old frames)";
    addHistory(`${historyText} · LingBot2 ${lingbot2Mode}`);
  }
  trimDecodeQueue();
  updateStats();
}

function clearFrameQueue() {
  closeFrames(playbackController.clear());
}

function closeFrames(items) {
  for (const item of items || []) item.image?.close?.();
}

function recordingFileName(extension = "mp4") {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `world-studio-gameplay-${stamp}.${extension}`;
}

function createRecordingTrack({ key, label, variant, canvas: targetCanvas, ctx: targetCtx }) {
  return {
    key,
    label,
    variant,
    canvas: targetCanvas,
    ctx: targetCtx,
    samples: [],
    encoder: null,
    encoderReady: null,
    encoderConfig: null,
    encodeChain: Promise.resolve(),
    encodePending: false,
    mediaRecorder: null,
    mediaChunks: [],
    captureStream: null,
    mimeType: recordingMode === "mediarecorder-webm"
      ? supportedWebmMimeType()
      : "video/mp4",
    frameIndex: 0,
    lastTimestampUs: -1,
    droppedFrames: 0,
  };
}

function desiredRecordingFps() {
  const selected = selectedModelKeys()
    .filter((key) => key === "minwm" || key === "lingbot2")
    .map((key) => previewPlaybackTargetFps(key))
    .filter((fps) => Number.isFinite(fps) && fps > 0);
  const targetFps = selected.length
    ? Math.max(...selected)
    : previewPlaybackTargetFps("minwm");
  return Math.max(1, Math.min(GAMEPLAY_RECORDING_FPS, targetFps));
}

function recordingVideoBitrate(width = RECORDING_STAGE_WIDTH, height = RECORDING_STAGE_HEIGHT, fps = recordingFps) {
  return Math.round(Math.min(
    20_000_000,
    Math.max(8_000_000, width * height * Math.max(1, fps) * 0.55),
  ));
}

function updateRecordButton() {
  const button = $("recordBtn");
  button.classList.toggle("is-recording", recordingActive);
  button.classList.toggle("is-saving", recordingSaving);
  const sessionLive = worldExperienceReady
    && sessionCountdownDeadlineMs > Date.now()
    && !sessionLifetimeExpired;
  button.disabled = recordingSaving || (!recordingActive && !sessionLive);
  button.setAttribute("aria-pressed", recordingActive ? "true" : "false");
  $("recordLabel").textContent = recordingSaving
    ? "生成录像"
    : recordingActive ? "录制中" : "游玩录像";
  const elapsedMs = recordingActive
    ? Math.max(0, performance.now() - recordingStartedPerfMs)
    : recordingElapsedMs;
  $("recordDuration").textContent = formatRecordingDuration(elapsedMs);
  button.title = recordingActive
    ? "点击提前结束录像"
    : sessionLive ? "开始录制当前游玩" : "进入世界后自动开始录制";
  updateRecordingDownloadButton();
}

function updateRecordingDownloadButton() {
  const button = $("recordDownloadBtn");
  if (!button) return;
  const ready = recordingDownloads.length === 2;
  button.hidden = !ready;
  button.disabled = !ready || recordingSaving;
  button.setAttribute("aria-disabled", !ready || recordingSaving ? "true" : "false");
  button.title = ready
    ? `同步下载 ${recordingDownloads.map((item) => item.fileName).join("、")}`
    : "两份录像生成后可下载";
}

function setRecordingDownloads(outputs = []) {
  for (const item of recordingDownloads) {
    if (item.url) URL.revokeObjectURL(item.url);
  }
  recordingDownloads = outputs
    .filter((item) => item?.videoBlob && item?.fileName)
    .map((item) => ({
      ...item,
      url: URL.createObjectURL(item.videoBlob),
    }));
  updateRecordingDownloadButton();
}

function downloadGameplayRecordings(event) {
  if (recordingDownloads.length !== 2 || recordingSaving) {
    event?.preventDefault?.();
    return;
  }
  event?.preventDefault?.();
  for (const item of recordingDownloads) {
    const link = document.createElement("a");
    link.href = item.url;
    link.download = item.fileName;
    document.body.appendChild(link);
    link.click();
    link.remove();
  }
  addHistory(`downloaded both gameplay recordings · ${recordingDownloads.map((item) => item.fileName).join(" · ")}`);
}

function resetRecordingPromptOverlay() {
  recordingPromptDraft = $("runtimePrompt")?.value || "";
  recordingPromptSubmitted = "";
  recordingPromptStatus = recordingPromptDraft ? "typing" : "idle";
  recordingPromptStatusPerfMs = performance.now();
  recordingPromptChangeType = "";
}

function updateRecordingPromptDraft(value) {
  const next = String(value || "");
  if (next === recordingPromptDraft) return;
  recordingPromptDraft = next;
  if (!recordingPromptSubmitted) {
    recordingPromptStatus = next ? "typing" : "idle";
    recordingPromptStatusPerfMs = performance.now();
  }
  if (recordingActive) recordTrajectoryEvent("runtime_prompt_input", { value: next });
}

function markRecordingPromptSubmitted(prompt) {
  recordingPromptSubmitted = String(prompt || "").trim();
  recordingPromptStatus = "rewriting";
  recordingPromptStatusPerfMs = performance.now();
  recordingPromptChangeType = "";
  if (recordingActive) {
    recordTrajectoryEvent("runtime_prompt_submitted", {
      prompt: recordingPromptSubmitted,
    });
  }
}

function markRecordingPromptSent(prompt, metadata = {}, eventId = null) {
  if (metadata.trigger === "rule" || metadata.phase === "restore") return;
  recordingPromptSubmitted = String(
    metadata.instruction || recordingPromptSubmitted || prompt || "",
  ).trim();
  recordingPromptStatus = "sent";
  recordingPromptStatusPerfMs = performance.now();
  recordingPromptChangeType = metadata.changeType || "persistent";
  if (recordingActive) {
    recordTrajectoryEvent("runtime_prompt_sent", {
      event_id: eventId,
      user_prompt: recordingPromptSubmitted,
      rewritten_prompt: prompt,
      change_type: recordingPromptChangeType,
    });
  }
}

function markRecordingPromptFailed(message = "") {
  recordingPromptStatus = "error";
  recordingPromptStatusPerfMs = performance.now();
  if (recordingActive) {
    recordTrajectoryEvent("runtime_prompt_failed", {
      user_prompt: recordingPromptSubmitted,
      error: String(message || "prompt rewrite failed"),
    });
  }
}

function recordingPromptOverlaySnapshot(now = performance.now()) {
  const heldStatus = ["sent", "error"].includes(recordingPromptStatus);
  if (heldStatus && now - recordingPromptStatusPerfMs > RECORDING_PROMPT_STATUS_HOLD_MS) {
    recordingPromptSubmitted = "";
    recordingPromptStatus = recordingPromptDraft ? "typing" : "idle";
    recordingPromptChangeType = "";
  }
  return {
    text: recordingPromptSubmitted || recordingPromptDraft,
    status: recordingPromptStatus,
    changeType: recordingPromptChangeType,
  };
}

function formatRecordingDuration(elapsedMs) {
  const seconds = Math.max(0, Math.floor(elapsedMs / 1000));
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function selectRecordingMode() {
  if (window.VideoEncoder && window.VideoFrame) return "webcodecs-mp4";
  if (window.MediaRecorder && recordingCanvas.captureStream && supportedWebmMimeType()) {
    return "mediarecorder-webm";
  }
  return "";
}

function supportedWebmMimeType() {
  if (!window.MediaRecorder) return "";
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  return candidates.find((mimeType) => (
    typeof MediaRecorder.isTypeSupported !== "function" ||
    MediaRecorder.isTypeSupported(mimeType)
  )) || "";
}

function updateRecordFolderButton() {
  const button = $("recordFolderBtn");
  if (!button) return;
  const supported = typeof window.showDirectoryPicker === "function";
  button.disabled = recordingSaving || !supported;
  button.classList.toggle("is-selected", Boolean(recordingDirectoryHandle));
  $("recordFolderLabel").textContent = recordingDirectoryHandle ? "Set" : "Folder";
  button.title = supported
    ? recordingDirectoryHandle
      ? "Recording artifacts will be saved to the selected folder"
      : "Choose a folder for MP4, JSON, and HTML recording artifacts"
    : "Folder save is unavailable in this browser; artifacts will download";
}

async function chooseRecordingDirectory() {
  if (typeof window.showDirectoryPicker !== "function") {
    addHistory("record folder unavailable · using downloads");
    updateRecordFolderButton();
    return;
  }
  try {
    recordingDirectoryHandle = await window.showDirectoryPicker({ mode: "readwrite" });
    addHistory("record folder selected");
  } catch (error) {
    if (error?.name !== "AbortError") {
      addHistory(error.message || "record folder selection failed");
    }
  } finally {
    updateRecordFolderButton();
  }
}

function recordingAssetBaseUrl() {
  return String(UI_CONFIG.recordingAssetBaseUrl || "").trim().replace(/\/+$/, "");
}

function recordingAssetUrl(fileName) {
  const baseUrl = recordingAssetBaseUrl();
  return baseUrl ? `${baseUrl}/${encodeURIComponent(fileName)}` : fileName;
}

function generateTraceId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `trace-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function artifactClientMs(artifact = currentSessionArtifact) {
  if (!artifact?.client_started_at_ms) return 0;
  return Math.round(performance.now() - artifact.client_started_at_ms);
}

function currentRequestSnapshot() {
  const generationMode = selectedGenerationMode();
  const continuousT2V = generationMode === "t2v" && $("continuous").checked;
  const numFrames = generationMode === "t2v"
    ? (continuousT2V ? undefined : readT2VNumFrames())
    : Number($("numFrames").value);
  return compact({
    type: "init_snapshot",
    generation_mode: generationMode,
    model: $("model").value,
    prompt: $("prompt").value,
    size: $("size").value,
    fps: Number($("fps").value || DEFAULT_TARGET_FPS),
    num_frames: continuousT2V ? undefined : numFrames,
    seed: Number($("seed").value),
    num_inference_steps: Number($("steps").value),
    guidance_scale: Number($("guidance").value),
    realtime_causal_sink_size: readOptionalInteger("sinkSize"),
    realtime_causal_kv_cache_num_frames: readOptionalInteger("windowFrames"),
    max_chunks: generationMode === "t2v" || $("continuous").checked ? undefined : 1,
    ...readPreviewTransportParams(),
    ...readFrameInterpolationParams(),
    ...readSuperResolutionParams(),
  });
}

function createSessionArtifact(init = currentRequestSnapshot(), referenceImage = null) {
  const now = new Date();
  const artifact = {
    schema_version: SESSION_ARTIFACT_SCHEMA_VERSION,
    trace_id: generateTraceId(),
    created_at: now.toISOString(),
    page_url: window.location.href,
    user_agent: navigator.userAgent,
    client_started_at_ms: performance.now(),
    server_url: $("serverUrl").value,
    request: {},
    prompt_history: [],
    events: [],
    chunks: [],
    first_rendered_chunks: [],
    recording: null,
  };
  updateSessionArtifactRequest(artifact, init, referenceImage);
  recordPromptHistory(init.prompt, "init", null, artifact);
  return artifact;
}

function updateSessionArtifactRequest(artifact, init, referenceImage = null) {
  artifact.server_url = $("serverUrl").value;
  artifact.request = {
    ...stripBinaryFields(init),
    reference_image: referenceImage,
  };
  artifact.reference_image = referenceImage || null;
  artifact.generation_mode = init.generation_mode || artifact.request.generation_mode || null;
  artifact.model = init.model || artifact.model || "";
}

function beginSessionArtifact(init, referenceImage = null) {
  const artifact = recordingActive && recordingArtifact
    ? recordingArtifact
    : createSessionArtifact(init, referenceImage);
  updateSessionArtifactRequest(artifact, init, referenceImage);
  currentSessionArtifact = artifact;
  if (recordingActive) recordingArtifact = artifact;
  recordPromptHistory(init.prompt, "init", null, artifact);
  recordTrajectoryEvent("session_init", {
    generation_mode: init.generation_mode,
    has_reference_image: Boolean(referenceImage),
    num_frames: init.num_frames,
    max_chunks: init.max_chunks ?? null,
  });
  return artifact;
}

function ensureSessionArtifact() {
  if (!currentSessionArtifact) {
    currentSessionArtifact = createSessionArtifact(currentRequestSnapshot(), null);
  }
  return currentSessionArtifact;
}

function recordTrajectoryEvent(kind, details = {}) {
  if (!currentSessionArtifact && !recordingActive) return null;
  const artifact = ensureSessionArtifact();
  const event = {
    kind,
    client_ms: artifactClientMs(artifact),
    ...jsonSafe(details),
  };
  artifact.events.push(event);
  if (artifact.events.length > SESSION_ARTIFACT_EVENT_LIMIT) {
    artifact.events.splice(0, artifact.events.length - SESSION_ARTIFACT_EVENT_LIMIT);
  }
  return event;
}

function recordPromptHistory(prompt, kind = "prompt_update", eventId = null, artifact = null) {
  const target = artifact || currentSessionArtifact;
  if (!target || typeof prompt !== "string") return;
  const lastPrompt = target.prompt_history[target.prompt_history.length - 1];
  if (lastPrompt && lastPrompt.prompt === prompt && lastPrompt.kind === kind) return;
  target.prompt_history.push(compact({
    kind,
    event_id: eventId,
    client_ms: artifactClientMs(target),
    prompt,
  }));
}

async function createReferenceImageMeta(firstFrame) {
  if (!firstFrame) return null;
  const file = $("firstFrame").files[0];
  const mime = file?.type || selectedReferenceMimeType || selectedPreset?.mime || mimeFromReferenceUrl(selectedReferenceUrl);
  const bytes = firstFrame.byteLength || firstFrame.length || 0;
  const meta = {
    source: file ? "upload" : selectedReferenceUrl ? "preset_url" : "bytes",
    label: file?.name || selectedReferenceLabel || selectedPreset?.name || "",
    url: selectedReferenceUrl || undefined,
    mime,
    bytes,
    first_frame_sha256: await sha256Bytes(firstFrame),
  };
  if (bytes > 0 && bytes <= MAX_EMBEDDED_REFERENCE_IMAGE_BYTES) {
    meta.data_url = await bytesToDataUrl(firstFrame, mime);
  }
  return compact(meta);
}

function mimeFromReferenceUrl(url) {
  const path = String(url || "").split("?")[0].toLowerCase();
  if (path.endsWith(".png")) return "image/png";
  if (path.endsWith(".webp")) return "image/webp";
  return "image/jpeg";
}

async function sha256Bytes(bytes) {
  if (!bytes || !globalThis.crypto?.subtle) return null;
  const buffer = bytes instanceof Uint8Array
    ? bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    : bytes;
  const digest = await globalThis.crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function normalizeWorldDescription(value) {
  return String(value || "").trim().replace(/\s+/g, " ");
}

function fallbackBytesFingerprint(bytes) {
  const source = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes || 0);
  let hash = 0x811c9dc5;
  for (let index = 0; index < source.length; index += 1) {
    hash ^= source[index];
    hash = Math.imul(hash, 0x01000193);
  }
  return `fnv1a-${source.length}-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

async function customWorldFingerprint(firstFrame, description, knownImageHash = "") {
  const normalizedDescription = normalizeWorldDescription(description);
  const descriptionBytes = new TextEncoder().encode(normalizedDescription);
  const imageHash = knownImageHash
    || await sha256Bytes(firstFrame)
    || fallbackBytesFingerprint(firstFrame);
  const descriptionHash = await sha256Bytes(descriptionBytes)
    || fallbackBytesFingerprint(descriptionBytes);
  return `${imageHash}:${descriptionHash}`;
}

function requestResult(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("browser storage request failed"));
  });
}

function openCustomWorldDatabase() {
  if (customWorldDbPromise) return customWorldDbPromise;
  if (!globalThis.indexedDB) {
    return Promise.reject(new Error("this browser does not support persistent world storage"));
  }
  customWorldDbPromise = new Promise((resolve, reject) => {
    const request = globalThis.indexedDB.open(CUSTOM_WORLD_DB_NAME, CUSTOM_WORLD_DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(CUSTOM_WORLD_STORE_NAME)) {
        const store = database.createObjectStore(CUSTOM_WORLD_STORE_NAME, {
          keyPath: "fingerprint",
        });
        store.createIndex("createdAt", "createdAt", { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => {
      customWorldDbPromise = null;
      reject(request.error || new Error("could not open custom world storage"));
    };
    request.onblocked = () => {
      customWorldDbPromise = null;
      reject(new Error("custom world storage upgrade is blocked"));
    };
  });
  return customWorldDbPromise;
}

async function readStoredCustomWorlds() {
  const database = await openCustomWorldDatabase();
  const transaction = database.transaction(CUSTOM_WORLD_STORE_NAME, "readonly");
  const records = await requestResult(
    transaction.objectStore(CUSTOM_WORLD_STORE_NAME).getAll(),
  );
  return records.sort((left, right) => Number(left.createdAt) - Number(right.createdAt));
}

async function writeStoredCustomWorld(record) {
  const database = await openCustomWorldDatabase();
  await new Promise((resolve, reject) => {
    const transaction = database.transaction(CUSTOM_WORLD_STORE_NAME, "readwrite");
    transaction.objectStore(CUSTOM_WORLD_STORE_NAME).put(record);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(
      transaction.error || new Error("could not save custom world"),
    );
    transaction.onabort = () => reject(
      transaction.error || new Error("custom world save was aborted"),
    );
  });
}

function customWorldPresetFromRecord(record) {
  const imageBlob = record.imageBlob instanceof Blob
    ? record.imageBlob
    : new Blob([record.imageBlob], { type: record.mime || "image/png" });
  return {
    name: record.name,
    tone: "green",
    size: record.size || "832x480",
    fps: Number(record.fps || DEFAULT_TARGET_FPS),
    prompt: record.prompt,
    referenceUrl: URL.createObjectURL(imageBlob),
    source: "自定义世界",
    mime: record.mime || imageBlob.type || "image/png",
    imageBlob,
    rules: record.rules || null,
    fingerprint: record.fingerprint,
    createdAt: Number(record.createdAt || Date.now()),
    isCustom: true,
  };
}

async function loadCustomWorldPresets() {
  try {
    const records = await readStoredCustomWorlds();
    customWorldPresets = records.map(customWorldPresetFromRecord);
    renderPresets();
    if (records.length) addHistory(`loaded ${records.length} custom worlds`);
  } catch (error) {
    addHistory(`custom world library unavailable · ${error.message || error}`);
  }
}

function ensureCustomWorldPresetsLoaded() {
  if (!customWorldLoadPromise) {
    customWorldLoadPromise = loadCustomWorldPresets();
  }
  return customWorldLoadPromise;
}

function selectedWorldContentIsUnchanged(description, preset = selectedPreset) {
  return Boolean(
    preset
    && normalizeWorldDescription(preset.prompt) === normalizeWorldDescription(description)
  );
}

function selectedWorldIsUnchanged(
  description,
  preset = selectedPreset,
  rules = readWorldRulesDraft(),
) {
  return Boolean(
    selectedWorldContentIsUnchanged(description, preset)
    && worldRulesStorageSignature(preset.rules || {}) === worldRulesStorageSignature(rules)
  );
}

async function matchesBuiltInWorld(firstFrame, description, imageHash) {
  const candidates = presets.filter((preset) => (
    normalizeWorldDescription(preset.prompt) === description
  ));
  for (const preset of candidates) {
    try {
      const presetBytes = await fetchReferenceBytes(preset.referenceUrl);
      const presetHash = await sha256Bytes(presetBytes)
        || fallbackBytesFingerprint(presetBytes);
      if (presetHash === imageHash) return true;
    } catch (error) {
      addHistory(`preset duplicate check skipped · ${preset.name} · ${error.message || error}`);
    }
  }
  return false;
}

async function rememberEnteredWorld(firstFrame, referenceImage, entrySnapshot = {}) {
  const description = normalizeWorldDescription(
    entrySnapshot.description ?? $("prompt").value,
  );
  const entryPreset = entrySnapshot.preset ?? selectedPreset;
  const entryRules = normalizedWorldRulesForStorage(
    entrySnapshot.rules ?? readWorldRulesDraft(),
  );
  const shouldKeepSelection = () => (
    selectedPreset === entryPreset
    && normalizeWorldDescription($("prompt").value) === description
    && worldRulesStorageSignature(readWorldRulesDraft()) === worldRulesStorageSignature(entryRules)
  );
  if (!firstFrame?.byteLength || !description || selectedWorldIsUnchanged(
    description,
    entryPreset,
    entryRules,
  )) {
    return false;
  }
  try {
    await ensureCustomWorldPresetsLoaded();
    const imageHash = referenceImage?.first_frame_sha256
      || await sha256Bytes(firstFrame)
      || fallbackBytesFingerprint(firstFrame);
    if (!hasConfiguredWorldRules(entryRules)
      && await matchesBuiltInWorld(firstFrame, description, imageHash)) return false;
    const fingerprint = await customWorldFingerprint(
      firstFrame,
      description,
      imageHash,
    );
    const existing = customWorldPresets.find((preset) => preset.fingerprint === fingerprint);
    if (existing) {
      if (worldRulesStorageSignature(existing.rules || {}) !== worldRulesStorageSignature(entryRules)) {
        const updatedRecord = {
          fingerprint: existing.fingerprint,
          name: existing.name,
          prompt: existing.prompt,
          mime: existing.mime,
          size: existing.size,
          fps: existing.fps,
          createdAt: existing.createdAt,
          imageBlob: existing.imageBlob,
          rules: entryRules,
        };
        await writeStoredCustomWorld(updatedRecord);
        existing.rules = entryRules;
        addHistory(`updated ${existing.name} world rules`);
      }
      if (shouldKeepSelection()) selectedPreset = existing;
      renderPresets();
      return false;
    }
    const mime = referenceImage?.mime
      || $("firstFrame").files[0]?.type
      || selectedReferenceMimeType
      || "image/png";
    const createdAt = Date.now();
    const record = {
      fingerprint,
      name: `自定义世界 ${customWorldPresets.length + 1}`,
      prompt: description,
      mime,
      size: modelControl("minwm", "size").value || "832x480",
      fps: Number(modelControl("minwm", "fps").value || DEFAULT_TARGET_FPS),
      createdAt,
      imageBlob: new Blob([firstFrame], { type: mime }),
      rules: entryRules,
    };
    await writeStoredCustomWorld(record);
    const preset = customWorldPresetFromRecord(record);
    customWorldPresets.push(preset);
    if (shouldKeepSelection()) selectedPreset = preset;
    renderPresets();
    addHistory(`saved ${preset.name} to world library`);
    return true;
  } catch (error) {
    addHistory(`custom world save failed · ${error.message || error}`);
    return false;
  }
}

function bytesToDataUrl(bytes, mime = "application/octet-stream") {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("reference image encode failed"));
    reader.readAsDataURL(new Blob([bytes], { type: mime }));
  });
}

function stripBinaryFields(value) {
  const safe = jsonSafe(value);
  if (value?.first_frame instanceof Uint8Array) {
    safe.first_frame = {
      byte_length: value.first_frame.byteLength,
      note: "binary bytes summarized; see request.reference_image",
    };
  }
  return safe;
}

function jsonSafe(value, depth = 0) {
  if (depth > 8) return "[MaxDepth]";
  if (value == null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (value instanceof Uint8Array) {
    return { binary_type: "Uint8Array", byte_length: value.byteLength };
  }
  if (value instanceof ArrayBuffer) {
    return { binary_type: "ArrayBuffer", byte_length: value.byteLength };
  }
  if (value instanceof Blob) {
    return { binary_type: "Blob", byte_length: value.size, type: value.type };
  }
  if (Array.isArray(value)) return value.map((item) => jsonSafe(item, depth + 1));
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, item]) => typeof item !== "function" && item !== undefined)
        .map(([key, item]) => [key, jsonSafe(item, depth + 1)]),
    );
  }
  return String(value);
}

function createStageRecordingTracks() {
  return [
    createRecordingTrack({
      key: "comparison",
      label: "Zing × LingBot2",
      variant: "comparison",
      canvas: recordingCanvas,
      ctx: recordingCtx,
    }),
    createRecordingTrack({
      key: "zing",
      label: "Zing",
      variant: "zing",
      canvas: zingRecordingCanvas,
      ctx: zingRecordingCtx,
    }),
  ];
}

function startRecording({ source = "manual" } = {}) {
  if (recordingActive || recordingSaving) return false;
  recordingMode = selectRecordingMode();
  if (!recordingMode) {
    setStatus("Recording unsupported", "error");
    addHistory("recording requires WebCodecs MP4 or MediaRecorder WebM support");
    return false;
  }
  recordingActive = true;
  setRecordingDownloads([]);
  recordingTracks = createStageRecordingTracks();
  recordingFrameIndex = 0;
  recordingFps = desiredRecordingFps();
  recordingStartedPerfMs = performance.now();
  recordingElapsedMs = 0;
  recordingLastCaptureMs = 0;
  recordingLastPresentedMs = 0;
  recordingDroppedFrames = 0;
  recordingBaseFileName = recordingFileName().replace(/\.[^.]*$/, "");
  recordingActionPulseUntil.clear();
  resetRecordingPromptOverlay();
  recordingArtifact = ensureSessionArtifact();
  recordingArtifact.recording = {
    base_file_name: recordingBaseFileName,
    started_at: new Date().toISOString(),
    started_client_ms: artifactClientMs(recordingArtifact),
    mode: recordingMode,
    mime_type: recordingTracks[0].mimeType,
    capture_scope: "stage",
    capture_width: RECORDING_STAGE_WIDTH,
    capture_height: RECORDING_STAGE_HEIGHT,
    target_fps: recordingFps,
    timing: "wall_clock",
    capture_driver: "presented_frame_idle",
    video_bits_per_second: recordingVideoBitrate(
      RECORDING_STAGE_WIDTH,
      RECORDING_STAGE_HEIGHT,
      recordingFps,
    ),
    source,
    variants: recordingTracks.map((track) => track.key),
  };
  if (recordingMode === "mediarecorder-webm") {
    for (const track of recordingTracks) startWebmRecording(track);
  }
  startRecordingFramePump();
  recordTrajectoryEvent("record_start", {
    target_fps: recordingFps,
    capture_driver: "presented_frame_idle",
    capture_scope: "stage",
    source,
  });
  recordingTimer = window.setInterval(updateRecordButton, 250);
  updateRecordButton();
  updateRecordFolderButton();
  addHistory("dual recording started · comparison + Zing");
  return true;
}

async function stopRecording({ reason = "manual" } = {}) {
  if (!recordingActive || recordingSaving) return;
  recordingElapsedMs = Math.max(0, performance.now() - recordingStartedPerfMs);
  recordingActive = false;
  stopRecordingFramePump();
  if (recordingTimer) {
    window.clearInterval(recordingTimer);
    recordingTimer = 0;
  }
  recordingSaving = true;
  updateRecordButton();
  updateRecordFolderButton();

  const extension = recordingMode === "mediarecorder-webm" ? "webm" : "mp4";
  try {
    recordTrajectoryEvent("record_stop", {
      encoded_frames: Object.fromEntries(
        recordingTracks.map((track) => [track.key, track.samples.length]),
      ),
      captured_frames: recordingFrameIndex,
      mode: recordingMode,
      reason,
      elapsed_ms: Math.round(recordingElapsedMs),
      dropped_frames: Object.fromEntries(
        recordingTracks.map((track) => [track.key, track.droppedFrames]),
      ),
    });
    const baseFileName = recordingBaseFileName
      || recordingFileName(extension).replace(/\.[^.]*$/, "");
    const outputs = await Promise.all(recordingTracks.map(async (track) => ({
      key: track.key,
      label: track.label,
      fileName: `${baseFileName}-${track.key}.${extension}`,
      videoBlob: recordingMode === "mediarecorder-webm"
        ? await stopWebmRecording(track)
        : await buildMp4RecordingBlob(track),
    })));
    await saveRecordingArtifactFiles(outputs, { deferDownload: true });
    addHistory(`both gameplay recordings ready · ${recordingFrameIndex} synchronized frames · ${extension}`);
    if (["session_timeout", "session_closed", "primary_disconnected"].includes(reason)) {
      showSessionNotice("两份游玩录像已生成，可点击右上角同步下载");
      showRecordingReadyToast();
    }
  } catch (error) {
    if (error?.name === "AbortError") {
      addHistory("recording save canceled");
    } else {
      addHistory(error.message || "recording save failed");
      setStatus("Save failed", "error");
      showSessionNotice("游玩录像生成失败，请重新体验后再试");
    }
  } finally {
    for (const track of recordingTracks) {
      track.encoder?.close?.();
      track.encoder = null;
      track.encoderReady = null;
      stopRecordingCaptureStream(track);
      track.mediaRecorder = null;
      track.mediaChunks = [];
      track.captureStream = null;
    }
    recordingMode = "";
    recordingSaving = false;
    recordingTracks = [];
    updateRecordButton();
    updateRecordFolderButton();
  }
}

async function buildMp4RecordingBlob(track) {
  await track.encodeChain;
  if (!track.encoder) throw new Error(`No ${track.label} frames were recorded`);
  await track.encoder.flush();
  if (!track.samples.length) throw new Error(`No ${track.label} frames were recorded`);
  return buildRecordingMp4(track);
}

function startWebmRecording(track) {
  drawRecordingStageFrame(canvas, track);
  track.mediaChunks = [];
  track.captureStream = track.canvas.captureStream(recordingFps);
  track.mediaRecorder = new MediaRecorder(
    track.captureStream,
    track.mimeType
      ? {
        mimeType: track.mimeType,
        videoBitsPerSecond: recordingVideoBitrate(
          RECORDING_STAGE_WIDTH,
          RECORDING_STAGE_HEIGHT,
          recordingFps,
        ),
      }
      : {
        videoBitsPerSecond: recordingVideoBitrate(
          RECORDING_STAGE_WIDTH,
          RECORDING_STAGE_HEIGHT,
          recordingFps,
        ),
      },
  );
  track.mimeType = track.mediaRecorder.mimeType || track.mimeType || "video/webm";
  track.mediaRecorder.ondataavailable = (event) => {
    if (event.data?.size) track.mediaChunks.push(event.data);
  };
  track.mediaRecorder.onerror = (event) => {
    recordingActive = false;
    stopRecordingFramePump();
    addHistory(event.error?.message || `${track.label} recorder failed`);
    updateRecordButton();
  };
  track.mediaRecorder.start(250);
}

function stopWebmRecording(track) {
  return new Promise((resolve, reject) => {
    const recorder = track.mediaRecorder;
    if (!recorder) {
      reject(new Error(`No ${track.label} WebM recorder was started`));
      return;
    }
    recorder.onstop = () => {
      stopRecordingCaptureStream(track);
      if (!track.mediaChunks.length) {
        reject(new Error(`No ${track.label} frames were recorded`));
        return;
      }
      resolve(new Blob(track.mediaChunks, { type: track.mimeType || "video/webm" }));
    };
    recorder.onerror = (event) => {
      stopRecordingCaptureStream(track);
      reject(event.error || new Error(`${track.label} recorder failed`));
    };
    if (recorder.state === "inactive") {
      recorder.onstop();
      return;
    }
    try {
      recorder.requestData?.();
      recorder.stop();
    } catch (error) {
      reject(error);
    }
  });
}

function stopRecordingCaptureStream(track) {
  for (const streamTrack of track?.captureStream?.getTracks?.() || []) streamTrack.stop();
  if (track) track.captureStream = null;
}

function startRecordingFramePump() {
  stopRecordingFramePump();
  captureRecordingFrame({ reason: "start", force: true });
  recordingFrameTimer = window.setInterval(recordingHeartbeatCapture, RECORDING_HEARTBEAT_MS);
}

function stopRecordingFramePump() {
  if (recordingFrameTimer) window.clearInterval(recordingFrameTimer);
  recordingFrameTimer = 0;
  cancelPendingRecordingCapture();
}

function cancelPendingRecordingCapture() {
  if (!recordingCapturePending) return;
  if (recordingCaptureUsesIdle && typeof window.cancelIdleCallback === "function") {
    window.cancelIdleCallback(recordingCaptureHandle);
  } else {
    window.clearTimeout(recordingCaptureHandle);
  }
  recordingCaptureHandle = 0;
  recordingCaptureUsesIdle = false;
  recordingCapturePending = false;
}

function recordingHeartbeatCapture() {
  if (!recordingActive || recordingSaving) return;
  const now = performance.now();
  const maxGapMs = Math.max(RECORDING_HEARTBEAT_MS, Math.round(2000 / Math.max(1, recordingFps)));
  if (!recordingLastCaptureMs || now - recordingLastCaptureMs >= maxGapMs) {
    scheduleRecordingCapture({ reason: "heartbeat", force: true, at: now });
  }
}

function notifyRecordingPresentedFrame(modelKey, presentedAt = performance.now()) {
  if (!recordingActive || recordingSaving) return;
  const at = Number.isFinite(Number(presentedAt)) ? Number(presentedAt) : performance.now();
  recordingLastPresentedMs = at;
  scheduleRecordingCapture({ reason: `${modelKey}_presented`, at });
}

function scheduleRecordingCapture({ reason = "presented_frame", force = false, at = performance.now() } = {}) {
  if (!recordingActive || recordingSaving) return;
  const minIntervalMs = 1000 / Math.max(1, recordingFps);
  if (!force && recordingLastCaptureMs && at - recordingLastCaptureMs < minIntervalMs * 0.75) return;
  if (recordingCapturePending) return;
  recordingCapturePending = true;
  const run = () => {
    recordingCaptureHandle = 0;
    recordingCaptureUsesIdle = false;
    recordingCapturePending = false;
    captureRecordingFrame({ reason, captureTimeMs: performance.now(), force });
  };
  if (typeof window.requestIdleCallback === "function") {
    recordingCaptureUsesIdle = true;
    recordingCaptureHandle = window.requestIdleCallback(run, {
      timeout: RECORDING_IDLE_CAPTURE_TIMEOUT_MS,
    });
  } else {
    recordingCaptureUsesIdle = false;
    recordingCaptureHandle = window.setTimeout(run, 0);
  }
}

function captureRecordingFrame({ reason = "capture", captureTimeMs = performance.now(), force = false } = {}) {
  if (!recordingActive || recordingSaving) return;
  const elapsedMs = Math.max(0, captureTimeMs - recordingStartedPerfMs);
  recordingElapsedMs = elapsedMs;
  const minIntervalMs = 1000 / Math.max(1, recordingFps);
  if (!force && recordingLastCaptureMs && captureTimeMs - recordingLastCaptureMs < minIntervalMs * 0.75) return;
  if (recordingMode === "webcodecs-mp4" && recordingTracks.some(recordingTrackBackpressured)) {
    for (const track of recordingTracks) {
      if (recordingTrackBackpressured(track)) track.droppedFrames += 1;
    }
    recordingDroppedFrames = recordingTracks.reduce(
      (sum, track) => sum + track.droppedFrames,
      0,
    );
    return;
  }
  for (const track of recordingTracks) drawRecordingStageFrame(canvas, track);
  let captured = false;
  if (recordingMode === "mediarecorder-webm") {
    for (const track of recordingTracks) track.frameIndex += 1;
    captured = true;
  } else {
    for (const track of recordingTracks) {
      captured = captureRecordingTrack(track, elapsedMs) || captured;
    }
  }
  if (captured) {
    recordingFrameIndex += 1;
    recordingLastCaptureMs = captureTimeMs;
  }
  recordingDroppedFrames = recordingTracks.reduce(
    (sum, track) => sum + track.droppedFrames,
    0,
  );
  void reason;
}

function recordingTrackBackpressured(track) {
  return Boolean(track.encodePending)
    || Number(track.encoder?.encodeQueueSize || 0) >= RECORDING_MAX_ENCODER_QUEUE_SIZE;
}

function captureRecordingTrack(track, elapsedMs) {
  if (recordingTrackBackpressured(track)) {
    track.droppedFrames += 1;
    return false;
  }
  const frameIndex = track.frameIndex;
  const duration = Math.round(1_000_000 / Math.max(1, recordingFps));
  const timestamp = Math.max(track.lastTimestampUs + 1, Math.round(elapsedMs * 1000));
  track.lastTimestampUs = timestamp;
  let frame;
  try {
    frame = new VideoFrame(track.canvas, { timestamp, duration });
  } catch (error) {
    recordingActive = false;
    stopRecordingFramePump();
    addHistory(error.message || `${track.label} frame capture failed`);
    updateRecordButton();
    return false;
  }
  track.encodePending = true;
  track.encodeChain = track.encodeChain
    .then(async () => {
      try {
        await ensureRecordingEncoder(track, frame.displayWidth, frame.displayHeight);
        if (Number(track.encoder?.encodeQueueSize || 0) >= RECORDING_MAX_ENCODER_QUEUE_SIZE) {
          track.droppedFrames += 1;
          return;
        }
        track.encoder.encode(frame, {
          keyFrame: frameIndex === 0 || frameIndex % RECORDING_KEYFRAME_INTERVAL_FRAMES === 0,
        });
        track.frameIndex += 1;
      } finally {
        frame.close();
        track.encodePending = false;
      }
    })
    .catch((error) => {
      track.encodePending = false;
      recordingActive = false;
      stopRecordingFramePump();
      addHistory(error.message || `${track.label} encode failed`);
      updateRecordButton();
    });
  return true;
}

function drawRecordingStageFrame(image, track = recordingTracks[0] || {
  variant: "comparison",
  canvas: recordingCanvas,
  ctx: recordingCtx,
}) {
  const previousCtx = recordingCtx;
  recordingCtx = track.ctx;
  ensureRecordingStageCanvas(track.canvas);
  const minwmSource = recordingDrawableSource(image || canvas);
  recordingCtx.save();
  try {
    recordingCtx.imageSmoothingEnabled = true;
    recordingCtx.imageSmoothingQuality = "high";
    recordingCtx.fillStyle = "#11140f";
    recordingCtx.fillRect(0, 0, RECORDING_STAGE_WIDTH, RECORDING_STAGE_HEIGHT);
    drawRecordingTopbar(track.variant);
    if (track.variant === "zing") drawRecordingZingPreview(minwmSource);
    else drawRecordingComparisonPreview(minwmSource, lingbot2Canvas);
    drawRecordingBottomGradient();
    drawRecordingControls();
    drawRecordingPromptComposer();
  } finally {
    recordingCtx.restore();
    recordingCtx = previousCtx;
  }
}

function ensureRecordingStageCanvas(targetCanvas = recordingCanvas) {
  if (
    targetCanvas.width !== RECORDING_STAGE_WIDTH ||
    targetCanvas.height !== RECORDING_STAGE_HEIGHT
  ) {
    targetCanvas.width = RECORDING_STAGE_WIDTH;
    targetCanvas.height = RECORDING_STAGE_HEIGHT;
  }
}

function recordingDrawableSource(image) {
  if (image instanceof ImageData) {
    if (scratchCanvas.width !== image.width || scratchCanvas.height !== image.height) {
      scratchCanvas.width = image.width;
      scratchCanvas.height = image.height;
    }
    scratchCtx.putImageData(image, 0, 0);
    return scratchCanvas;
  }
  return image || canvas;
}

function drawRecordingTopbar(variant = "comparison") {
  const y = 0;
  fillRecordingRect(0, y, RECORDING_STAGE_WIDTH, RECORDING_STAGE_TOPBAR_HEIGHT, "#0b1110");
  recordingCtx.fillStyle = "rgba(232, 234, 223, 0.12)";
  recordingCtx.fillRect(0, RECORDING_STAGE_TOPBAR_HEIGHT - 1, RECORDING_STAGE_WIDTH, 1);

  let x = RECORDING_STAGE_PADDING;
  recordingCtx.beginPath();
  recordingCtx.moveTo(x, 33);
  recordingCtx.lineTo(x + 12, 13);
  recordingCtx.lineTo(x + 24, 33);
  recordingCtx.closePath();
  recordingCtx.strokeStyle = "#79dfbd";
  recordingCtx.lineWidth = 4;
  recordingCtx.stroke();
  x += 38;
  drawRecordingLabel("World Studio", x, y + 31, {
    color: "#f7faf8",
    font: "700 17px ui-sans-serif, system-ui, sans-serif",
    maxWidth: 150,
  });

  x += 174;
  recordingCtx.beginPath();
  recordingCtx.arc(x + 5, y + RECORDING_STAGE_TOPBAR_HEIGHT / 2, 5, 0, Math.PI * 2);
  recordingCtx.fillStyle = recordingActive ? "#e48674" : "#687164";
  recordingCtx.fill();
  drawRecordingLabel(recordingSaving ? "正在生成录像" : "游玩录制", x + 17, y + 30, {
    color: "rgba(247, 250, 248, 0.82)",
    font: "600 14px ui-sans-serif, system-ui, sans-serif",
    maxWidth: 110,
  });
  drawRecordingLabel(formatRecordingDuration(recordingElapsedMs), x + 128, y + 30, {
    color: "#f7faf8",
    font: "700 14px ui-monospace, SFMono-Regular, monospace",
    maxWidth: 60,
  });

  drawRecordingLabel(variant === "zing" ? "Zing" : "Zing  ×  LingBot2", RECORDING_STAGE_WIDTH - RECORDING_STAGE_PADDING, y + 30, {
    align: "right",
    color: "rgba(247, 250, 248, 0.72)",
    font: "600 14px ui-sans-serif, system-ui, sans-serif",
    maxWidth: 180,
  });
}

function drawRecordingZingPreview(minwmSource) {
  const y = RECORDING_STAGE_TOPBAR_HEIGHT;
  fillRecordingRect(0, y, RECORDING_STAGE_WIDTH, RECORDING_STAGE_PREVIEW_HEIGHT, "#11140f");
  drawRecordingFittedSource(minwmSource, {
    x: 0,
    y,
    width: RECORDING_STAGE_WIDTH,
    height: RECORDING_STAGE_PREVIEW_HEIGHT,
  });
}

function drawRecordingComparisonPreview(minwmSource, lingbot2Source) {
  const y = RECORDING_STAGE_TOPBAR_HEIGHT;
  fillRecordingRect(0, y, RECORDING_STAGE_WIDTH, RECORDING_STAGE_PREVIEW_HEIGHT, "#11140f");
  const gap = 12;
  const inset = 24;
  const titleHeight = 34;
  const width = (RECORDING_STAGE_WIDTH - inset * 2 - gap) / 2;
  const height = RECORDING_STAGE_PREVIEW_HEIGHT - titleHeight;
  const players = [
    { label: "Zing", source: minwmSource, x: inset },
    { label: "LingBot2", source: lingbot2Source, x: inset + width + gap },
  ];
  for (const player of players) {
    drawRecordingLabel(player.label, player.x + 4, y + 24, {
      color: "#fffdf7",
      font: "600 16px ui-sans-serif, system-ui, sans-serif",
      maxWidth: width - 8,
    });
    drawRecordingFittedSource(player.source, {
      x: player.x,
      y: y + titleHeight,
      width,
      height,
    });
  }
}

function drawRecordingFittedSource(source, previewRect) {
  const sourceWidth = source?.width || 1280;
  const sourceHeight = source?.height || 720;
  const scale = Math.min(
    previewRect.width / Math.max(1, sourceWidth),
    previewRect.height / Math.max(1, sourceHeight),
  );
  const drawWidth = Math.round(sourceWidth * scale);
  const drawHeight = Math.round(sourceHeight * scale);
  const drawX = Math.round(previewRect.x + (previewRect.width - drawWidth) / 2);
  const drawY = Math.round(previewRect.y + (previewRect.height - drawHeight) / 2);
  fillRecordingRect(previewRect.x, previewRect.y, previewRect.width, previewRect.height, "#151912");
  if (sourceWidth > 0 && sourceHeight > 0) {
    recordingCtx.drawImage(source, drawX, drawY, drawWidth, drawHeight);
  }
}

function drawRecordingControls() {
  const y = RECORDING_STAGE_HEIGHT - 112;
  drawRecordingControlCluster("移动", 24, y, [
    [null, "w", null],
    ["a", "s", "d"],
  ]);
  drawRecordingControlCluster("视角", 176, y, [
    [null, "i", null],
    ["j", "k", "l"],
  ]);
}

function drawRecordingControlCluster(title, x, y, rows) {
  drawRecordingLabel(title, x, y - 8, {
    color: "rgba(255, 255, 255, 0.7)",
    font: "600 11px ui-sans-serif, system-ui, sans-serif",
    maxWidth: 72,
  });
  const cellGap = 5;
  const buttonSize = 38;
  rows.forEach((row, rowIndex) => {
    row.forEach((action, columnIndex) => {
      if (!action) return;
      drawRecordingControlButton(
        action,
        x + columnIndex * (buttonSize + cellGap),
        y + rowIndex * (buttonSize + cellGap),
        buttonSize,
        buttonSize,
      );
    });
  });
}

function drawRecordingControlButton(action, x, y, width, height) {
  const active = controlStateController?.activeActions?.has(action)
    || Number(recordingActionPulseUntil.get(action) || 0) > performance.now();
  const radius = 10;
  fillRecordingRoundedRect(
    x,
    y,
    width,
    height,
    radius,
    active ? "rgba(121, 223, 189, 0.92)" : "rgba(15, 19, 18, 0.72)",
  );
  strokeRecordingRoundedRect(
    x,
    y,
    width,
    height,
    radius,
    active ? "rgba(227, 255, 246, 0.94)" : "rgba(255, 255, 255, 0.35)",
  );
  const keyLabel = action === "i" ? "↑" : action === "j" ? "←" : action === "k" ? "↓" : action === "l" ? "→" : action.toUpperCase();
  drawRecordingLabel(keyLabel, x + width / 2, y + 25, {
    align: "center",
    color: active ? "#0b1411" : "rgba(255, 255, 255, 0.9)",
    font: "700 16px ui-sans-serif, system-ui, sans-serif",
    maxWidth: width - 10,
  });
}

function drawRecordingBottomGradient() {
  const height = 210;
  const y = RECORDING_STAGE_HEIGHT - height;
  const gradient = recordingCtx.createLinearGradient(0, y, 0, RECORDING_STAGE_HEIGHT);
  gradient.addColorStop(0, "rgba(5, 10, 9, 0)");
  gradient.addColorStop(0.46, "rgba(5, 10, 9, 0.46)");
  gradient.addColorStop(1, "rgba(5, 10, 9, 0.9)");
  fillRecordingRect(0, y, RECORDING_STAGE_WIDTH, height, gradient);
}

function drawRecordingPromptComposer() {
  const snapshot = recordingPromptOverlaySnapshot();
  const x = 450;
  const y = RECORDING_STAGE_HEIGHT - 88;
  const width = 700;
  const height = 56;
  const sendSize = 42;
  const sendX = x + width - sendSize - 7;
  const sent = snapshot.status === "sent";
  const failed = snapshot.status === "error";

  fillRecordingRoundedRect(x, y, width, height, 18, "rgba(244, 247, 245, 0.82)");
  strokeRecordingRoundedRect(x, y, width, height, 18, "rgba(255, 255, 255, 0.82)");

  const displayText = snapshot.text || "输入世界指令…";
  const textColor = snapshot.text ? "#17201d" : "rgba(23, 32, 29, 0.48)";
  drawRecordingLabel(displayText, x + 20, y + 35, {
    color: textColor,
    font: "500 18px ui-sans-serif, system-ui, sans-serif",
    maxWidth: width - sendSize - 54,
  });

  if (snapshot.text && ["typing", "rewriting"].includes(snapshot.status)) {
    recordingCtx.font = "500 18px ui-sans-serif, system-ui, sans-serif";
    const cursorX = Math.min(
      sendX - 16,
      x + 21 + recordingCtx.measureText(snapshot.text).width,
    );
    if (Math.floor(performance.now() / 450) % 2 === 0) {
      fillRecordingRect(cursorX, y + 17, 1.5, 22, "rgba(23, 32, 29, 0.78)");
    }
  }

  fillRecordingRoundedRect(
    sendX,
    y + 7,
    sendSize,
    sendSize,
    14,
    failed ? "#e48674" : sent ? "#79dfbd" : "rgba(24, 34, 31, 0.9)",
  );
  drawRecordingLabel(sent ? "✓" : failed ? "!" : "→", sendX + sendSize / 2, y + 34, {
    align: "center",
    color: sent ? "#0b1411" : "#f7faf8",
    font: "700 20px ui-sans-serif, system-ui, sans-serif",
    maxWidth: 26,
  });

  if (snapshot.status === "rewriting") {
    drawRecordingLabel("AI 改写中", sendX - 14, y - 8, {
      align: "right",
      color: "rgba(255, 255, 255, 0.82)",
      font: "600 11px ui-sans-serif, system-ui, sans-serif",
      maxWidth: 90,
    });
  } else if (sent) {
    const typeLabel = snapshot.changeType === "one_time" ? "一次性指令已发送" : "持久指令已发送";
    drawRecordingLabel(typeLabel, sendX - 14, y - 8, {
      align: "right",
      color: "#b7f4df",
      font: "600 11px ui-sans-serif, system-ui, sans-serif",
      maxWidth: 120,
    });
  }
}

function drawRecordingLabel(text, x, y, {
  color = "#fffdf7",
  font = "14px ui-sans-serif, system-ui, sans-serif",
  align = "left",
  maxWidth = undefined,
} = {}) {
  recordingCtx.save();
  recordingCtx.fillStyle = color;
  recordingCtx.font = font;
  recordingCtx.textAlign = align;
  recordingCtx.textBaseline = "alphabetic";
  if (maxWidth === undefined) {
    recordingCtx.fillText(String(text), x, y);
  } else {
    recordingCtx.fillText(String(text), x, y, maxWidth);
  }
  recordingCtx.restore();
}

function fillRecordingRect(x, y, width, height, fillStyle) {
  recordingCtx.fillStyle = fillStyle;
  recordingCtx.fillRect(x, y, width, height);
}

function fillRecordingRoundedRect(x, y, width, height, radius, fillStyle) {
  recordingCtx.beginPath();
  recordingRoundedRectPath(x, y, width, height, radius);
  recordingCtx.fillStyle = fillStyle;
  recordingCtx.fill();
}

function strokeRecordingRoundedRect(x, y, width, height, radius, strokeStyle) {
  recordingCtx.beginPath();
  recordingRoundedRectPath(x, y, width, height, radius);
  recordingCtx.strokeStyle = strokeStyle;
  recordingCtx.lineWidth = 1;
  recordingCtx.stroke();
}

function recordingRoundedRectPath(x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  if (recordingCtx.roundRect) {
    recordingCtx.roundRect(x, y, width, height, r);
    return;
  }
  recordingCtx.moveTo(x + r, y);
  recordingCtx.lineTo(x + width - r, y);
  recordingCtx.quadraticCurveTo(x + width, y, x + width, y + r);
  recordingCtx.lineTo(x + width, y + height - r);
  recordingCtx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  recordingCtx.lineTo(x + r, y + height);
  recordingCtx.quadraticCurveTo(x, y + height, x, y + height - r);
  recordingCtx.lineTo(x, y + r);
  recordingCtx.quadraticCurveTo(x, y, x + r, y);
}

async function ensureRecordingEncoder(track, width, height) {
  if (track.encoderReady) return track.encoderReady;
  track.encoderReady = createRecordingEncoder(track, width, height);
  return track.encoderReady;
}

async function createRecordingEncoder(track, width, height) {
  const fps = Math.max(1, recordingFps);
  const bitrate = recordingVideoBitrate(width, height, fps);
  const configs = [
    { codec: "avc1.640028", width, height, bitrate, framerate: fps },
    { codec: "avc1.4d4028", width, height, bitrate, framerate: fps },
    { codec: "avc1.42e028", width, height, bitrate, framerate: fps },
  ];
  let supported = null;
  for (const config of configs) {
    const candidate = {
      ...config,
      avc: { format: "avc" },
      bitrateMode: "variable",
      hardwareAcceleration: "prefer-hardware",
      latencyMode: "realtime",
    };
    const result = await VideoEncoder.isConfigSupported(candidate);
    if (result.supported) {
      supported = result.config;
      break;
    }
  }
  if (!supported) throw new Error("This browser cannot encode H.264 MP4");
  track.encoderConfig = supported;
  track.encoder = new VideoEncoder({
    output: (chunk, metadata) => recordEncodedChunk(track, chunk, metadata),
    error: (error) => {
      recordingActive = false;
      addHistory(error.message || `${track.label} encoder failed`);
      updateRecordButton();
    },
  });
  track.encoder.configure(supported);
}

function recordEncodedChunk(track, chunk, metadata) {
  if (metadata?.decoderConfig?.description) {
    track.encoderConfig.description = metadata.decoderConfig.description;
  }
  const data = new Uint8Array(chunk.byteLength);
  chunk.copyTo(data);
  track.samples.push({
    data,
    timestamp: chunk.timestamp,
    duration: chunk.duration || 0,
    key: chunk.type === "key",
  });
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function sidecarFileName(fileName, extension) {
  return `${String(fileName).replace(/\.[^.]*$/, "")}.${extension}`;
}

async function saveRecordingArtifactFiles(outputs, { deferDownload = false } = {}) {
  const artifact = finalizeRecordingArtifact(outputs);
  const jsonFileName = artifact.recording.json_file;
  const htmlFileName = artifact.recording.html_file;
  const jsonBlob = new Blob(
    [JSON.stringify(artifact, null, 2)],
    { type: "application/json" },
  );
  const htmlBlob = new Blob(
    [buildReplayHtml(artifact)],
    { type: "text/html" },
  );
  const files = [
    ...outputs.map((output) => ({ name: output.fileName, blob: output.videoBlob })),
    { name: jsonFileName, blob: jsonBlob },
    { name: htmlFileName, blob: htmlBlob },
  ];
  if (recordingDirectoryHandle) {
    await saveRecordingFiles(files);
  } else if (!deferDownload) {
    await saveRecordingFiles(files);
  }
  setRecordingDownloads(outputs);
}

function finalizeRecordingArtifact(outputs) {
  const artifact = recordingArtifact || ensureSessionArtifact();
  const primary = outputs.find((output) => output.key === "comparison") || outputs[0];
  if (!primary) throw new Error("No recording outputs were generated");
  const sidecarBaseName = primary.fileName.replace(/-comparison\.[^.]*$/, "");
  const jsonFileName = `${sidecarBaseName}.json`;
  const htmlFileName = `${sidecarBaseName}.html`;
  const tracksByKey = Object.fromEntries(recordingTracks.map((track) => [track.key, track]));
  const videos = Object.fromEntries(outputs.map((output) => {
    const track = tracksByKey[output.key];
    return [output.key, {
      label: output.label,
      mime_type: output.videoBlob.type || track?.mimeType || "video/mp4",
      frames: track?.frameIndex || 0,
      dropped_frames: track?.droppedFrames || 0,
      encoded_chunks: recordingMode === "mediarecorder-webm"
        ? track?.mediaChunks.length || 0
        : track?.samples.length || 0,
      video_file: output.fileName,
      video_url: recordingAssetUrl(output.fileName),
      video_bytes: output.videoBlob.size,
    }];
  }));
  artifact.recording = {
    ...(artifact.recording || {}),
    stopped_at: new Date().toISOString(),
    stopped_client_ms: artifactClientMs(artifact),
    mode: recordingMode,
    mime_type: primary.videoBlob.type || tracksByKey.comparison?.mimeType || "video/mp4",
    fps: recordingFps,
    frames: recordingFrameIndex,
    dropped_frames: recordingDroppedFrames,
    duration_ms: Math.round(recordingElapsedMs),
    encoded_chunks: videos.comparison?.encoded_chunks || 0,
    video_file: primary.fileName,
    video_url: recordingAssetUrl(primary.fileName),
    video_bytes: primary.videoBlob.size,
    videos,
    json_file: jsonFileName,
    json_url: recordingAssetUrl(jsonFileName),
    html_file: htmlFileName,
    html_url: recordingAssetUrl(htmlFileName),
    asset_base_url: recordingAssetBaseUrl() || null,
  };
  return artifact;
}

async function saveRecordingFiles(files) {
  if (recordingDirectoryHandle) {
    await ensureRecordingDirectoryWritable(recordingDirectoryHandle);
    for (const file of files) {
      const handle = await recordingDirectoryHandle.getFileHandle(file.name, { create: true });
      const writable = await handle.createWritable();
      await writable.write(file.blob);
      await writable.close();
    }
    return;
  }
  for (const file of files) downloadBlob(file.blob, file.name);
}

async function ensureRecordingDirectoryWritable(directoryHandle) {
  const options = { mode: "readwrite" };
  if (directoryHandle.queryPermission) {
    const existing = await directoryHandle.queryPermission(options);
    if (existing === "granted") return;
  }
  if (directoryHandle.requestPermission) {
    const requested = await directoryHandle.requestPermission(options);
    if (requested === "granted") return;
  }
  throw new Error("recording folder permission denied");
}

function buildReplayHtml(artifact) {
  const recording = artifact.recording || {};
  const request = artifact.request || {};
  const referenceImage = request.reference_image || null;
  const prompts = artifact.prompt_history || [];
  const events = artifact.events || [];
  const eventRows = events.slice(-600).map((event) => (
    `<tr><td>${escapeHtmlText(event.kind)}</td><td>${formatReplayMs(event.client_ms)}</td><td><code>${escapeHtmlText(JSON.stringify(event))}</code></td></tr>`
  )).join("");
  const promptRows = prompts.map((item) => (
    `<li><b>${escapeHtmlText(item.kind)}</b> ${formatReplayMs(item.client_ms)}<pre>${escapeHtmlText(item.prompt || "")}</pre></li>`
  )).join("");
  function replayReferenceImageSrc(referenceImage) {
    return referenceImage?.data_url || referenceImage?.url || referenceImage?.source_url || referenceImage?.preset_url || "";
  }
  const referenceSrc = replayReferenceImageSrc(referenceImage);
  const referenceBlock = referenceSrc
    ? `<img class="reference" src="${escapeHtmlAttribute(referenceSrc)}" alt="reference image" />`
    : `<div class="reference empty">${escapeHtmlText(referenceImage ? referenceImage.label || "reference image" : "T2V session: no reference image")}</div>`;
  const artifactJson = JSON.stringify(artifact)
    .replace(/</g, "\\u003c")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SGLang realtime replay ${escapeHtmlText(artifact.trace_id || "")}</title>
  <style>
    body { margin: 0; background: #eef1ec; color: #171a16; font: 14px/1.45 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width: min(1480px, calc(100vw - 32px)); margin: 24px auto 56px; }
    h1 { margin: 0 0 6px; font-size: 24px; }
    .meta, .grid, .events, .prompt-list { margin-top: 16px; }
    .grid { display: grid; grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr); gap: 16px; align-items: start; }
    .replay-stage { overflow: hidden; border: 1px solid #11140f; border-radius: 8px; background: #11140f; box-shadow: 0 18px 60px rgba(23, 26, 22, 0.12); }
    .replay-topbar, .replay-timeline { display: flex; align-items: center; gap: 10px; min-width: 0; height: 44px; padding: 0 14px; color: #e8eadf; background: rgba(17, 20, 15, 0.9); font-size: 12px; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .replay-topbar-spacer { flex: 1; }
    .replay-dot { width: 8px; height: 8px; border-radius: 50%; background: #8ecf9d; box-shadow: 0 0 0 4px rgba(142, 207, 157, 0.14); }
    .replay-pill { display: inline-flex; align-items: center; gap: 6px; height: 28px; padding: 0 10px; border: 1px solid rgba(232, 234, 223, 0.22); border-radius: 6px; background: rgba(238, 241, 236, 0.08); color: #e8eadf; }
    .replay-video-shell { position: relative; display: grid; place-items: center; min-height: 320px; background: #11140f; }
    .replay-video { display: block; width: 100%; max-height: 72vh; border: 0; border-radius: 0; background: #11140f; }
    .replay-cursor { position: absolute; inset: 0 auto 0 0; width: 2px; transform: translateX(var(--replay-cursor-x, -200%)); background: rgba(142, 207, 157, 0.86); box-shadow: 0 0 0 1px rgba(17, 20, 15, 0.62); pointer-events: none; opacity: 0; }
    .replay-video-shell.is-inspecting .replay-cursor { opacity: 1; }
    .replay-inspector { position: fixed; left: 0; top: 0; z-index: 40; width: min(430px, calc(100vw - 28px)); max-height: min(520px, calc(100vh - 28px)); overflow: auto; border: 1px solid rgba(232, 234, 223, 0.34); border-radius: 8px; background: rgba(251, 250, 245, 0.95); color: #171a16; box-shadow: 0 18px 50px rgba(17, 20, 15, 0.34); pointer-events: none; transform: translate(14px, 14px); }
    .replay-inspector[hidden] { display: none; }
    .replay-inspector-header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding: 10px 12px 8px; border-bottom: 1px solid #cbd2c4; }
    .replay-inspector-header b { font-size: 13px; }
    .replay-inspector-header span { color: #687164; font-size: 12px; font-variant-numeric: tabular-nums; }
    .replay-inspector-grid { display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 6px 10px; padding: 10px 12px; }
    .replay-inspector-grid span { color: #687164; font-size: 12px; }
    .replay-inspector-grid b { min-width: 0; font-size: 12px; word-break: break-word; }
    .replay-inspector-block { padding: 0 12px 10px; }
    .replay-inspector-block span { display: block; margin-bottom: 4px; color: #687164; font-size: 12px; }
    .replay-inspector-block pre { max-height: 110px; margin: 0; padding: 8px; border-radius: 6px; font-size: 11px; }
    .replay-inspector-image { display: none; width: 86px; height: 48px; object-fit: cover; margin: 0 0 8px; border: 1px solid #cbd2c4; border-radius: 5px; }
    .replay-inspector-image.has-image { display: block; }
    .replay-timeline { justify-content: flex-end; border-top: 1px solid rgba(232, 234, 223, 0.12); }
    video, .reference { width: 100%; border: 1px solid #cbd2c4; border-radius: 8px; background: #11140f; }
    .replay-stage video { border: 0; border-radius: 0; }
    .reference.empty { min-height: 160px; display: grid; place-items: center; border: 1px dashed #cbd2c4; border-radius: 8px; color: #687164; }
    pre, code { white-space: pre-wrap; word-break: break-word; }
    pre { margin: 8px 0 0; padding: 12px; background: #fbfaf5; border: 1px solid #cbd2c4; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; background: #fbfaf5; border: 1px solid #cbd2c4; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #d8ddd2; vertical-align: top; text-align: left; }
    th { color: #687164; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
    .card { padding: 10px; border: 1px solid #cbd2c4; border-radius: 8px; background: #fbfaf5; }
    .card b { display: block; font-size: 16px; }
    @media (max-width: 860px) { .grid, .cards { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>SGLang realtime replay</h1>
    <div class="meta">Trace ${escapeHtmlText(artifact.trace_id || "-")} · ${escapeHtmlText(request.generation_mode || "-")} · ${escapeHtmlText(recording.video_file || "-")}</div>
    <div class="cards">
      <div class="card"><span>Frames</span><b>${escapeHtmlText(recording.frames ?? "-")}</b></div>
      <div class="card"><span>FPS</span><b>${escapeHtmlText(recording.fps ?? request.fps ?? "-")}</b></div>
      <div class="card"><span>Events</span><b>${escapeHtmlText(events.length)}</b></div>
      <div class="card"><span>Mode</span><b>${escapeHtmlText(request.generation_mode || "-")}</b></div>
    </div>
    <section class="grid">
      <div>
        <section class="replay-stage" aria-label="Recorded realtime stage">
          <div class="replay-topbar">
            <span class="replay-dot" aria-hidden="true"></span>
            <span>Replay</span>
            <span>frames ${escapeHtmlText(recording.frames ?? "-")}</span>
            <span class="replay-pill">Record ${escapeHtmlText(recording.fps ?? request.fps ?? "-")} fps</span>
            <span class="replay-topbar-spacer"></span>
            <span>mode ${escapeHtmlText(request.generation_mode || "-")}</span>
            <span>scope ${escapeHtmlText(recording.capture_scope || "viewport")}</span>
          </div>
          <div class="replay-video-shell">
            <video id="replayVideo" class="replay-video" controls preload="metadata" src="${escapeHtmlAttribute(recording.video_url || recording.video_file || "")}"></video>
            <div id="replayCursor" class="replay-cursor" aria-hidden="true"></div>
            <aside id="replayInspector" class="replay-inspector" hidden aria-live="polite">
              <div class="replay-inspector-header">
                <b>Cursor trace</b>
                <span id="replayInspectorTime">-</span>
              </div>
              <div class="replay-inspector-grid">
                <span>User keys</span><b id="replayInspectorUserKeys">-</b>
                <span>SGLang keys</span><b id="replayInspectorSglangKeys">-</b>
                <span>Chunk / event</span><b id="replayInspectorChunk">-</b>
                <span>Reference image</span><b id="replayInspectorImageMeta">-</b>
              </div>
              <div class="replay-inspector-block">
                <img id="replayInspectorImage" class="replay-inspector-image" alt="reference image at cursor" />
                <span>Prompt at cursor</span>
                <pre id="replayInspectorPrompt">-</pre>
              </div>
              <div class="replay-inspector-block">
                <span>Nearby events</span>
                <pre id="replayInspectorEvents">-</pre>
              </div>
            </aside>
          </div>
          <div class="replay-timeline">
            <span id="replayActiveText">input idle</span>
            <span>${escapeHtmlText(recording.video_file || "-")}</span>
          </div>
        </section>
        <h2>Prompt History</h2>
        <ol class="prompt-list">${promptRows}</ol>
      </div>
      <aside>
        <h2>Reference</h2>
        ${referenceBlock}
        <h2>Request</h2>
        <pre>${escapeHtmlText(JSON.stringify(request, null, 2))}</pre>
      </aside>
    </section>
    <section class="events">
      <h2>Recent Events</h2>
      <table>
        <thead><tr><th>Kind</th><th>Time</th><th>Payload</th></tr></thead>
        <tbody>${eventRows}</tbody>
      </table>
    </section>
  </main>
  <script id="recording-artifact" type="application/json">${artifactJson}</script>
  <script>
    (() => {
      const artifactNode = document.getElementById("recording-artifact");
      const video = document.getElementById("replayVideo");
      const activeText = document.getElementById("replayActiveText");
      const videoShell = video && video.closest(".replay-video-shell");
      const inspector = document.getElementById("replayInspector");
      const inspectorTime = document.getElementById("replayInspectorTime");
      const inspectorUserKeys = document.getElementById("replayInspectorUserKeys");
      const inspectorSglangKeys = document.getElementById("replayInspectorSglangKeys");
      const inspectorChunk = document.getElementById("replayInspectorChunk");
      const inspectorPrompt = document.getElementById("replayInspectorPrompt");
      const inspectorEvents = document.getElementById("replayInspectorEvents");
      const inspectorImage = document.getElementById("replayInspectorImage");
      const inspectorImageMeta = document.getElementById("replayInspectorImageMeta");
      if (!artifactNode || !video) return;
      const artifact = JSON.parse(artifactNode.textContent || "{}");
      const events = Array.isArray(artifact.events)
        ? artifact.events.slice().sort((left, right) => Number(left.client_ms || 0) - Number(right.client_ms || 0))
        : [];
      const recordingStartMs = Number(artifact.recording && artifact.recording.started_client_ms) || 0;
      const recording = artifact.recording || {};
      const request = artifact.request || {};
      const prompts = Array.isArray(artifact.prompt_history)
        ? artifact.prompt_history.slice().sort((left, right) => Number(left.client_ms || 0) - Number(right.client_ms || 0))
        : [];
      const tracedChunks = events
        .filter((event) => event.kind === "trace_event" && event.trace?.event === "server.chunk_complete")
        .map((event) => ({
          ...event.trace,
          client_ms: event.client_ms,
          received_client_ms: event.client_ms,
        }));
      const legacyChunks = events.filter((event) => event.kind === "server_chunk_stats");
      const chunks = Array.isArray(artifact.chunks) && artifact.chunks.length
        ? artifact.chunks.slice().sort((left, right) => replayEventTime(left) - replayEventTime(right))
        : (tracedChunks.length ? tracedChunks : legacyChunks)
          .sort((left, right) => replayEventTime(left) - replayEventTime(right));
      const referenceImage = request.reference_image || artifact.reference_image || null;
      const referenceSrc = replayReferenceImageSrc(referenceImage);
      const cameraEventsById = new Map();
      events.forEach((event) => {
        if (event.kind === "camera_actions_sent" && event.event_id !== undefined && event.event_id !== null) {
          cameraEventsById.set(Number(event.event_id), event);
        }
      });
      const replayActionLabels = {
        w: "W Forward",
        a: "A Left",
        s: "S Back",
        d: "D Right",
        i: "↑ Pitch +",
        j: "← Yaw -",
        k: "↓ Pitch -",
        l: "→ Yaw +",
      };
      const REPLAY_INSPECTOR_OFFSET_PX = 16;

      function applyReplayEvent(active, event) {
        if (!event || typeof event.kind !== "string") return active;
        if (event.kind === "camera_actions_sent" && Array.isArray(event.active_actions)) {
          return new Set(event.active_actions.map(String));
        }
        const action = typeof event.action === "string" ? event.action : "";
        if (!action) return active;
        if (event.kind === "key_down" || event.kind === "control_button_down") active.add(action);
        if (event.kind === "key_up" || event.kind === "control_button_up") active.delete(action);
        return active;
      }

      function replayActionsAt(clientMs) {
        let active = new Set();
        for (const event of events) {
          if (Number(event.client_ms || 0) > clientMs) break;
          active = applyReplayEvent(active, event);
        }
        return active;
      }

      function userActionsAt(clientMs) {
        const active = new Set();
        for (const event of events) {
          if (Number(event.client_ms || 0) > clientMs) break;
          const action = typeof event.action === "string" ? event.action : "";
          if (!action) continue;
          if (event.kind === "key_down" || event.kind === "control_button_down") active.add(action);
          if (event.kind === "key_up" || event.kind === "control_button_up") active.delete(action);
        }
        return active;
      }

      function replayEventTime(event) {
        return Number(event.received_client_ms ?? event.client_ms ?? 0);
      }

      function promptAt(clientMs) {
        let prompt = { prompt: request.prompt || "-", kind: "request", client_ms: 0 };
        for (const item of prompts) {
          if (Number(item.client_ms || 0) > clientMs) break;
          prompt = item;
        }
        return prompt;
      }

      function chunkAt(clientMs) {
        let selected = null;
        for (const chunk of chunks) {
          if (replayEventTime(chunk) > clientMs) break;
          selected = chunk;
        }
        return selected || chunks[0] || null;
      }

      function sglangActionsForEventId(eventId) {
        if (eventId === undefined || eventId === null || eventId === "") return new Set();
        const event = cameraEventsById.get(Number(eventId));
        return new Set(Array.isArray(event?.active_actions) ? event.active_actions.map(String) : []);
      }

      function actionText(actions) {
        const labels = Array.from(actions)
          .sort()
          .map((action) => replayActionLabels[action] || action.toUpperCase());
        return labels.length ? labels.join(" + ") : "idle";
      }

      function eventSummaryAt(clientMs) {
        const interesting = events.filter((event) => (
          [
            "key_down",
            "key_up",
            "control_button_down",
            "control_button_up",
            "camera_actions_sent",
            "prompt_update",
            "server_chunk_stats",
          ].includes(event.kind)
        ));
        let nearby = interesting.filter((event) => Math.abs(Number(event.client_ms || 0) - clientMs) <= 750);
        if (!nearby.length) {
          nearby = interesting.filter((event) => Number(event.client_ms || 0) <= clientMs).slice(-6);
        } else {
          nearby = nearby.slice(-8);
        }
        return nearby.map(formatReplayEventSummary).join("\\n") || "-";
      }

      function formatReplayEventSummary(event) {
        const parts = [
          formatReplayClientMs(Number(event.client_ms || 0)),
          event.kind,
        ];
        if (event.action) parts.push("action=" + event.action.toUpperCase());
        if (event.event_id !== undefined) parts.push("event#" + event.event_id);
        if (event.chunk_index !== undefined) parts.push("chunk#" + event.chunk_index);
        if (Array.isArray(event.active_actions)) {
          parts.push("active=" + actionText(new Set(event.active_actions.map(String))));
        }
        return parts.join(" · ");
      }

      function referenceImageText() {
        if (!referenceImage) return "T2V / no reference image";
        const parts = [
          referenceImage.label || "reference image",
          referenceImage.source || "",
          referenceImage.mime || "",
          referenceImage.bytes ? String(referenceImage.bytes) + " bytes" : "",
        ].filter(Boolean);
        return parts.join(" · ");
      }

      function replayReferenceImageSrc(referenceImage) {
        return referenceImage?.data_url || referenceImage?.url || referenceImage?.source_url || referenceImage?.preset_url || "";
      }

      function replayDurationSeconds() {
        if (Number.isFinite(video.duration) && video.duration > 0) return video.duration;
        const frames = Number(recording.frames || 0);
        const fps = Number(recording.fps || request.fps || 0);
        return frames > 0 && fps > 0 ? frames / fps : 0;
      }

      function clampReplayRatio(value) {
        return Math.min(1, Math.max(0, value));
      }

      function replayClientMsFromPointer(event) {
        const rect = video.getBoundingClientRect();
        const ratio = clampReplayRatio((event.clientX - rect.left) / Math.max(1, rect.width));
        const durationSeconds = replayDurationSeconds();
        if (videoShell) videoShell.style.setProperty("--replay-cursor-x", (ratio * 100).toFixed(2) + "%");
        return recordingStartMs + durationSeconds * ratio * 1000;
      }

      function formatReplayClientMs(ms) {
        const relative = Math.max(0, ms - recordingStartMs);
        return (relative / 1000).toFixed(2) + "s";
      }

      function positionReplayInspector(event) {
        if (!inspector || !event) return;
        inspector.hidden = false;
        const left = event.clientX + REPLAY_INSPECTOR_OFFSET_PX;
        const top = event.clientY + REPLAY_INSPECTOR_OFFSET_PX;
        inspector.style.transform = "translate(" + Math.round(left) + "px, " + Math.round(top) + "px)";
      }

      function inspectReplayAt(clientMs) {
        if (!inspector) return;
        const userActions = userActionsAt(clientMs);
        const chunk = chunkAt(clientMs);
        const sglangActions = sglangActionsForEventId(chunk?.event_id);
        const prompt = promptAt(clientMs);
        inspector.hidden = false;
        videoShell?.classList.add("is-inspecting");
        if (inspectorTime) inspectorTime.textContent = formatReplayClientMs(clientMs);
        if (inspectorUserKeys) inspectorUserKeys.textContent = actionText(userActions);
        if (inspectorSglangKeys) inspectorSglangKeys.textContent = actionText(sglangActions);
        if (inspectorChunk) {
          inspectorChunk.textContent = chunk
            ? "chunk #" + (chunk.chunk_index ?? "-") + " · event #" + (chunk.event_id ?? "-")
            : "-";
        }
        if (inspectorPrompt) {
          inspectorPrompt.textContent = (prompt.kind || "prompt") + " " + formatReplayClientMs(Number(prompt.client_ms || 0)) + "\\n" + (prompt.prompt || "-");
        }
        if (inspectorEvents) inspectorEvents.textContent = eventSummaryAt(clientMs);
        if (inspectorImageMeta) inspectorImageMeta.textContent = referenceImageText();
        if (inspectorImage) {
          if (referenceSrc) {
            inspectorImage.src = referenceSrc;
            inspectorImage.classList.add("has-image");
          } else {
            inspectorImage.removeAttribute("src");
            inspectorImage.classList.remove("has-image");
          }
        }
      }

      function syncReplayControls() {
        const clientMs = recordingStartMs + video.currentTime * 1000;
        const active = replayActionsAt(clientMs);
        if (activeText) {
          const labels = Array.from(active).sort().map((action) => action.toUpperCase());
          activeText.textContent = labels.length ? "input " + labels.join(" + ") : "input idle";
        }
        if (!video.paused && !video.ended) requestAnimationFrame(syncReplayControls);
      }

      ["loadedmetadata", "timeupdate", "seeked", "play", "pause"].forEach((eventName) => {
        video.addEventListener(eventName, syncReplayControls);
      });
      video.addEventListener("mousemove", (event) => {
        positionReplayInspector(event);
        inspectReplayAt(replayClientMsFromPointer(event));
      });
      video.addEventListener("mouseleave", () => {
        if (inspector) inspector.hidden = true;
        videoShell?.classList.remove("is-inspecting");
        syncReplayControls();
      });
      syncReplayControls();
    })();
  </script>
</body>
</html>`;
}

function formatReplayMs(value) {
  const ms = Number(value || 0);
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${Math.round(ms)}ms`;
}

function escapeHtmlText(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeHtmlAttribute(value) {
  return escapeHtmlText(value)
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function buildRecordingMp4(track) {
  if (!track.encoderConfig?.description) {
    throw new Error("H.264 encoder did not return MP4 decoder config");
  }
  const width = track.encoderConfig.width;
  const height = track.encoderConfig.height;
  const samples = normalizeRecordingSamples(track.samples);
  const mdatPayload = concatBytes(samples.map((sample) => sample.data));
  const ftyp = mp4Box("ftyp", ascii("isom"), u32(0x200), ascii("isom"), ascii("iso2"), ascii("avc1"), ascii("mp41"));
  const mdat = mp4Box("mdat", mdatPayload);
  const firstSampleOffset = ftyp.byteLength + 8;
  const moov = buildMoovBox({
    width,
    height,
    samples,
    firstSampleOffset,
    avcConfig: new Uint8Array(track.encoderConfig.description),
  });
  return new Blob([ftyp, mdat, moov], { type: "video/mp4" });
}

function normalizeRecordingSamples(samples) {
  const ordered = [...samples].sort((left, right) => left.timestamp - right.timestamp);
  const timescale = 90_000;
  const fallbackDuration = Math.round(timescale / Math.max(1, recordingFps));
  const normalized = ordered.map((sample) => ({
    ...sample,
    time: Math.round(sample.timestamp * timescale / 1_000_000),
  }));
  for (let i = 0; i < normalized.length; i++) {
    const next = normalized[i + 1];
    normalized[i].duration = next
      ? Math.max(1, next.time - normalized[i].time)
      : Math.max(1, Math.round((ordered[i].duration || 0) * timescale / 1_000_000) || fallbackDuration);
  }
  return normalized;
}

function buildMoovBox({ width, height, samples, firstSampleOffset, avcConfig }) {
  const timescale = 90_000;
  const duration = samples.reduce((sum, sample) => sum + sample.duration, 0);
  const movieTimescale = 1000;
  const movieDuration = Math.ceil(duration * movieTimescale / timescale);
  return mp4Box(
    "moov",
    buildMvhdBox(movieTimescale, movieDuration),
    mp4Box(
      "trak",
      buildTkhdBox(width, height, movieDuration),
      mp4Box(
        "mdia",
        buildMdhdBox(timescale, duration),
        buildHdlrBox(),
        mp4Box(
          "minf",
          buildVmhdBox(),
          buildDinfBox(),
          buildStblBox({ width, height, samples, firstSampleOffset, avcConfig }),
        ),
      ),
    ),
  );
}

function buildMvhdBox(timescale, duration) {
  return mp4Box(
    "mvhd",
    u32(0),
    u32(0),
    u32(0),
    u32(timescale),
    u32(duration),
    u32(0x00010000),
    u16(0x0100),
    u16(0),
    zeros(8),
    u32(0x00010000), u32(0), u32(0),
    u32(0), u32(0x00010000), u32(0),
    u32(0), u32(0), u32(0x40000000),
    zeros(24),
    u32(2),
  );
}

function buildTkhdBox(width, height, duration) {
  return mp4Box(
    "tkhd",
    u32(0x00000007),
    u32(0),
    u32(0),
    u32(1),
    u32(0),
    u32(duration),
    zeros(8),
    u16(0),
    u16(0),
    u16(0),
    u16(0),
    u32(0x00010000), u32(0), u32(0),
    u32(0), u32(0x00010000), u32(0),
    u32(0), u32(0), u32(0x40000000),
    u32(width << 16),
    u32(height << 16),
  );
}

function buildMdhdBox(timescale, duration) {
  return mp4Box(
    "mdhd",
    u32(0),
    u32(0),
    u32(0),
    u32(timescale),
    u32(duration),
    u16(0x55c4),
    u16(0),
  );
}

function buildHdlrBox() {
  return mp4Box("hdlr", u32(0), u32(0), ascii("vide"), zeros(12), ascii("VideoHandler\0"));
}

function buildVmhdBox() {
  return mp4Box("vmhd", u32(0x00000001), u16(0), u16(0), u16(0), u16(0));
}

function buildDinfBox() {
  return mp4Box(
    "dinf",
    mp4Box(
      "dref",
      u32(0),
      u32(1),
      mp4Box("url ", u32(0x00000001)),
    ),
  );
}

function buildStblBox({ width, height, samples, firstSampleOffset, avcConfig }) {
  return mp4Box(
    "stbl",
    buildStsdBox(width, height, avcConfig),
    buildSttsBox(samples),
    buildStssBox(samples),
    buildStscBox(samples.length),
    buildStszBox(samples),
    buildStcoBox(firstSampleOffset),
  );
}

function buildStsdBox(width, height, avcConfig) {
  const compressor = new Uint8Array(32);
  return mp4Box(
    "stsd",
    u32(0),
    u32(1),
    mp4Box(
      "avc1",
      zeros(6),
      u16(1),
      zeros(16),
      u16(width),
      u16(height),
      u32(0x00480000),
      u32(0x00480000),
      u32(0),
      u16(1),
      compressor,
      u16(24),
      u16(0xffff),
      mp4Box("avcC", avcConfig),
    ),
  );
}

function buildSttsBox(samples) {
  const entries = [];
  for (const sample of samples) {
    const last = entries[entries.length - 1];
    if (last && last.duration === sample.duration) {
      last.count += 1;
    } else {
      entries.push({ count: 1, duration: sample.duration });
    }
  }
  return mp4Box("stts", u32(0), u32(entries.length), ...entries.flatMap((entry) => [u32(entry.count), u32(entry.duration)]));
}

function buildStssBox(samples) {
  const keySamples = samples
    .map((sample, index) => sample.key ? index + 1 : 0)
    .filter(Boolean);
  if (!keySamples.length && samples.length) keySamples.push(1);
  return mp4Box("stss", u32(0), u32(keySamples.length), ...keySamples.map(u32));
}

function buildStscBox(sampleCount) {
  return mp4Box("stsc", u32(0), u32(1), u32(1), u32(sampleCount), u32(1));
}

function buildStszBox(samples) {
  return mp4Box("stsz", u32(0), u32(0), u32(samples.length), ...samples.map((sample) => u32(sample.data.byteLength)));
}

function buildStcoBox(firstSampleOffset) {
  return mp4Box("stco", u32(0), u32(1), u32(firstSampleOffset));
}

function mp4Box(type, ...payloads) {
  const size = 8 + payloads.reduce((sum, payload) => sum + payload.byteLength, 0);
  const output = new Uint8Array(size);
  const view = new DataView(output.buffer);
  view.setUint32(0, size, false);
  output.set(ascii(type), 4);
  let offset = 8;
  for (const payload of payloads) {
    output.set(payload, offset);
    offset += payload.byteLength;
  }
  return output;
}

function concatBytes(parts) {
  const output = new Uint8Array(parts.reduce((sum, part) => sum + part.byteLength, 0));
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.byteLength;
  }
  return output;
}

function ascii(text) {
  const output = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) output[i] = text.charCodeAt(i);
  return output;
}

function zeros(length) {
  return new Uint8Array(length);
}

function u16(value) {
  const output = new Uint8Array(2);
  new DataView(output.buffer).setUint16(0, value, false);
  return output;
}

function u32(value) {
  const output = new Uint8Array(4);
  new DataView(output.buffer).setUint32(0, value >>> 0, false);
  return output;
}

function hasPendingPlaybackInput() {
  return (
    pendingDecodeBatches > 0 ||
    decodeInProgress ||
    decodeQueue.length > 0 ||
    Boolean(ws && ws.readyState === WebSocket.OPEN)
  );
}

function enqueueDecodeBatch(header, data, epoch) {
  const frameCount = Number(header.num_frames || 1);
  const payloadBytes = payloadByteLength(data);
  const eventId = Number(header.event_id || 0);
  if (lastSentEventId > 0 && eventId >= lastSentEventId) {
    dropQueuedDecodeBatchesBeforeEvent(eventId);
  }
  decodeQueue.push({ header, data, epoch, frameCount, payloadBytes });
  queuedDecodeFrames += frameCount;
  queuedDecodeBytes += payloadBytes;
  pendingDecodeBatches += 1;
  trimDecodeQueue();
  pumpDecodeQueue();
  updateStats();
}

function dropQueuedDecodeBatchesBeforeEvent(eventId) {
  const kept = [];
  for (const item of decodeQueue) {
    if (Number(item.header?.event_id || 0) >= eventId) {
      kept.push(item);
      continue;
    }
    queuedDecodeFrames = Math.max(0, queuedDecodeFrames - item.frameCount);
    queuedDecodeBytes = Math.max(0, queuedDecodeBytes - item.payloadBytes);
    pendingDecodeBatches = Math.max(0, pendingDecodeBatches - 1);
    droppedDecodeFrames += item.frameCount;
    lastDecodeDropAt = performance.now();
    lastDecodeDropCount = item.frameCount;
  }
  decodeQueue = kept;
}

function payloadByteLength(data) {
  if (!data) return 0;
  return Number(data.byteLength || data.size || data.length || 0);
}

function trimDecodeQueue() {
  if (recordingActive) return;
  if (!decodeQueue.length) return;
  const playbackMode = selectedPlaybackMode();
  const preservesTimeline = playbackMode === "timeline";
  const boundedRealtime = playbackMode === "smooth_timeline";
  const playback = playbackController.snapshot();
  let maxQueuedFrames;
  if (preservesTimeline) {
    maxQueuedFrames = Number.POSITIVE_INFINITY;
  } else if (boundedRealtime && renderedPreviewFrames) {
    const fallbackFrames = Math.max(
      1,
      Math.floor(previewPlaybackTargetFps() * ONLINE_MAX_BUFFER_MS / 1000),
    );
    maxQueuedFrames = Math.max(
      2,
      Number(playback.maxRealtimeBufferFrames || fallbackFrames) +
        ONLINE_DECODE_QUEUE_SLACK_FRAMES,
    );
  } else {
    const decodeWindowSeconds = renderedPreviewFrames
      ? Math.max(DECODE_QUEUE_SECONDS, (playback.maxLeadMs || 0) / 1000)
      : STARTUP_DECODE_QUEUE_SECONDS;
    maxQueuedFrames = Math.max(
      2,
      Math.round(previewPlaybackTargetFps() * decodeWindowSeconds),
    );
  }
  while (
    (queuedDecodeFrames > maxQueuedFrames || queuedDecodeBytes > MAX_DECODE_QUEUE_BYTES) &&
    decodeQueue.length > 1
  ) {
    const dropIndex = decodeQueue.findIndex((item, index) => (
      index < decodeQueue.length - 1 && canDropQueuedDecodeItem(item)
    ));
    if (dropIndex < 0) break;
    const [item] = decodeQueue.splice(dropIndex, 1);
    queuedDecodeFrames = Math.max(0, queuedDecodeFrames - item.frameCount);
    queuedDecodeBytes = Math.max(0, queuedDecodeBytes - item.payloadBytes);
    pendingDecodeBatches = Math.max(0, pendingDecodeBatches - 1);
    droppedDecodeFrames += item.frameCount;
    lastDecodeDropAt = performance.now();
    lastDecodeDropCount = item.frameCount;
  }
}

function canDropQueuedDecodeItem(item) {
  return (
    item?.header?.content_type === RAW_RGB_CONTENT_TYPE ||
    isEncodedPreviewContentType(item?.header?.content_type)
  );
}

async function pumpDecodeQueue() {
  if (decodeInProgress) return;
  const item = decodeQueue.shift();
  if (!item) return;
  queuedDecodeFrames = Math.max(0, queuedDecodeFrames - item.frameCount);
  queuedDecodeBytes = Math.max(0, queuedDecodeBytes - item.payloadBytes);
  decodeInProgress = true;
  try {
    await decodeAndEnqueueFrameBatch(item.header, item.data, item.epoch);
  } catch (error) {
    handleReceiveError(error, item.epoch);
  } finally {
    pendingDecodeBatches = Math.max(0, pendingDecodeBatches - 1);
    decodeInProgress = false;
    updateStats();
    if (decodeQueue.length) pumpDecodeQueue();
  }
}

function rgbToImageData(header, payload) {
  const width = Number(header.width), height = Number(header.height);
  const channels = Number(header.channels), count = Number(header.num_frames);
  const frameBytes = Number(header.bytes_per_frame);
  const src = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const items = [];
  for (let f = 0; f < count; f++) {
    const img = ctx.createImageData(width, height);
    let s = f * frameBytes, d = 0;
    for (let p = 0; p < width * height; p++) {
      img.data[d++] = src[s++];
      img.data[d++] = src[s++];
      img.data[d++] = src[s++];
      if (channels > 3) s += channels - 3;
      img.data[d++] = 255;
    }
    items.push({ image: img, chunk: header.chunk_index });
  }
  return items;
}

function rgbaToImageData(header, payload) {
  const width = Number(header.width), height = Number(header.height);
  const count = Number(header.num_frames);
  const frameBytes = Number(header.bytes_per_frame);
  const src = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const items = [];
  for (let f = 0; f < count; f++) {
    const offset = f * frameBytes;
    const imageBytes = new Uint8ClampedArray(
      src.buffer,
      src.byteOffset + offset,
      frameBytes,
    );
    items.push({ image: new ImageData(imageBytes, width, height), chunk: header.chunk_index });
  }
  return items;
}

async function gunzipBytes(payload) {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser does not support gzip stream decoding");
  }
  const stream = new Blob([payload]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function restoreDeltaGzipRawRgb(header, payload) {
  const frameBytes = Number(header.bytes_per_frame);
  const count = Number(header.num_frames);
  const expectedSize = frameBytes * count;
  const restored = await gunzipBytes(payload);
  if (restored.length !== expectedSize) {
    throw new Error(`delta payload size mismatch: expected ${expectedSize}, got ${restored.length}`);
  }
  let previous = header.delta_reference === "previous-frame" ? lastRawRgbFrame : null;
  if (header.delta_reference === "previous-frame" && !previous) {
    throw new Error("Missing previous frame for delta payload");
  }
  for (let f = 0; f < count; f++) {
    const current = f * frameBytes;
    if (previous) {
      for (let i = 0; i < frameBytes; i++) {
        restored[current + i] ^= previous[i];
      }
    }
    previous = restored.slice(current, current + frameBytes);
  }
  return restored;
}

async function framePayloadToImageData(header, payload) {
  let rawPayload;
  const isRgba = header.content_type === RAW_RGBA_DELTA_GZIP_CONTENT_TYPE;
  if (
    header.content_type === WEBP_FRAME_CONTENT_TYPE ||
    header.content_type === JPEG_FRAME_CONTENT_TYPE
  ) {
    return encodedImageToImageData(header, payload);
  } else if (header.content_type === RAW_RGB_CONTENT_TYPE) {
    rawPayload = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  } else if (header.content_type === RAW_RGB_DELTA_GZIP_CONTENT_TYPE) {
    rawPayload = await restoreDeltaGzipRawRgb(header, payload);
  } else if (isRgba) {
    rawPayload = await restoreDeltaGzipRawRgb(header, payload);
  } else {
    throw new Error(`Unsupported content type ${header.content_type}`);
  }
  const frameBytes = Number(header.bytes_per_frame);
  const frameCount = Number(header.num_frames);
  if (frameCount > 0) {
    const offset = (frameCount - 1) * frameBytes;
    lastRawRgbFrame = rawPayload.slice(offset, offset + frameBytes);
  }
  if (isRgba) {
    return rgbaToImageData(header, rawPayload);
  }
  return rgbToImageData(header, rawPayload);
}

function isEncodedPreviewContentType(contentType) {
  return (
    contentType === WEBP_FRAME_CONTENT_TYPE ||
    contentType === JPEG_FRAME_CONTENT_TYPE
  );
}

async function encodedImageToImageData(header, payload) {
  const framePayloads = splitEncodedPayload(header, payload);
  if (typeof createImageBitmap === "function") {
    try {
      return await Promise.all(framePayloads.map(async (framePayload) => ({
        image: await createImageBitmap(new Blob([framePayload], { type: header.content_type })),
        chunk: header.chunk_index,
      })));
    } catch (error) {
      return Promise.all(framePayloads.map((framePayload) => (
        encodedImageElementFallback(
          new Blob([framePayload], { type: header.content_type }),
          header,
          error,
        )
      )));
    }
  }
  return Promise.all(framePayloads.map((framePayload) => (
    encodedImageElementFallback(
      new Blob([framePayload], { type: header.content_type }),
      header,
      new Error("createImageBitmap unavailable"),
    )
  )));
}

function splitEncodedPayload(header, payload) {
  const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const lengths = Array.isArray(header.payload_lengths) && header.payload_lengths.length
    ? header.payload_lengths.map(Number)
    : [bytes.byteLength];
  const payloads = [];
  let offset = 0;
  for (const length of lengths) {
    payloads.push(bytes.buffer.slice(
      bytes.byteOffset + offset,
      bytes.byteOffset + offset + length,
    ));
    offset += length;
  }
  return payloads;
}

async function encodedImageElementFallback(blob, header, createBitmapError) {
  const url = URL.createObjectURL(blob);
  try {
    const image = await loadImageElement(url, createBitmapError);
    if (
      scratchCanvas.width !== image.naturalWidth ||
      scratchCanvas.height !== image.naturalHeight
    ) {
      scratchCanvas.width = image.naturalWidth;
      scratchCanvas.height = image.naturalHeight;
    }
    scratchCtx.drawImage(image, 0, 0);
    return {
      image: scratchCtx.getImageData(0, 0, image.naturalWidth, image.naturalHeight),
      chunk: header.chunk_index,
    };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function loadImageElement(url, createBitmapError) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(createBitmapError);
    image.src = url;
  });
}

function handleEncodedPreviewDecodeError(error, header, data, payloadBytes) {
  encodedDecodeErrors += 1;
  const signature = payloadSignature(data);
  const mode = shortPayloadMode(header.content_type);
  const message = error?.message || "encoded preview decode failed";
  $("minwmDecodeText").textContent = `drop ${encodedDecodeErrors}`;
  setStatus("Decode dropped", "error");
  addHistory(
    `decode drop c${header.chunk_index} ${mode} ${formatBytes(payloadBytes)} ${signature} · ${message}`,
  );
}

function payloadSignature(data) {
  let bytes;
  if (data instanceof Uint8Array) {
    bytes = data.subarray(0, Math.min(12, data.byteLength));
  } else if (data instanceof ArrayBuffer) {
    bytes = new Uint8Array(data, 0, Math.min(12, data.byteLength));
  } else {
    return "";
  }
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function payloadToArrayBuffer(data) {
  if (data instanceof ArrayBuffer) return data;
  if (data instanceof Uint8Array) {
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  }
  return data.arrayBuffer();
}

function drawFrame(image, { close = true, markRendered = true } = {}) {
  const sourceWidth = image.width;
  const sourceHeight = image.height;
  let drawSource = image;
  if (image instanceof ImageData) {
    if (scratchCanvas.width !== sourceWidth || scratchCanvas.height !== sourceHeight) {
      scratchCanvas.width = sourceWidth;
      scratchCanvas.height = sourceHeight;
    }
    scratchCtx.putImageData(image, 0, 0);
    drawSource = scratchCanvas;
  }

  if (canvas.width !== sourceWidth || canvas.height !== sourceHeight) {
    canvas.width = sourceWidth;
    canvas.height = sourceHeight;
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "medium";
  ctx.drawImage(drawSource, 0, 0, sourceWidth, sourceHeight);
  primaryHasVisibleFrame = true;
  if (markRendered) renderedPreviewFrames += 1;
  setPreviewState("live");
  markSessionPlayable("minwm");
  if (close && !(image instanceof ImageData)) image.close?.();
}

function renderLoop(now) {
  renderLoopSamples.push(now);
  renderLoopSamples = renderLoopSamples.filter((t) => now - t < 1000);
  const decision = playbackController.render(now, {
    hasPendingInput: hasPendingPlaybackInput(),
  });
  closeFrames(decision.droppedFrames);
  if (decision.action === "draw") {
    const item = decision.frame;
    drawFrame(item.image);
    notifyRecordingPresentedFrame("minwm", now);
    fpsSamples.push(now);
    fpsSamples = fpsSamples.filter((t) => now - t < 1000);
    lastRenderedChunk = item.chunk;
    lastRenderedEventId = Number(item.eventId || lastRenderedEventId || 0);
    const appliedEventIds = Array.from(primaryControlSentEpochByEvent.keys())
      .filter((eventId) => eventId <= lastRenderedEventId)
      .sort((left, right) => left - right);
    if (appliedEventIds.length) {
      const sentEpochMs = primaryControlSentEpochByEvent.get(appliedEventIds[0]);
      primaryProtocolStats = {
        ...primaryProtocolStats,
        lastControlToVideoMs: Math.max(0, Date.now() - sentEpochMs),
      };
      for (const eventId of appliedEventIds) primaryControlSentEpochByEvent.delete(eventId);
    }
    lastDisplayLagMs = now - (item.receivedAt || now);
    recordChunkFirstRendered(item.chunk, {
      render_loop: true,
      display_lag_ms: lastDisplayLagMs,
      decode_ms: item.decodeMs || lastDecodeMs,
    });
    updateStats();
    schedulePrimaryPlaybackAck();
  } else if (decision.action === "hold") {
    updateStats();
  }
  scheduleRenderLoop();
}

function scheduleRenderLoop() {
  if (
    document.visibilityState !== "hidden" &&
    typeof window.requestAnimationFrame === "function"
  ) {
    window.requestAnimationFrame(renderLoop);
    return;
  }
  const timerFps = Math.min(
    MAX_RENDER_TIMER_FPS,
    Math.max(MIN_RENDER_TIMER_FPS, previewPlaybackTargetFps() * 2),
  );
  window.setTimeout(() => renderLoop(performance.now()), 1000 / timerFps);
}

async function readFirstFrame() {
  const file = $("firstFrame").files[0];
  if (file) return new Uint8Array(await file.arrayBuffer());
  if (selectedReferenceBytes) return selectedReferenceBytes;
  if (selectedReferenceUrl) {
    selectedReferenceBytes = await fetchReferenceBytes(selectedReferenceUrl);
    return selectedReferenceBytes;
  }
  return undefined;
}

function drawPlaceholderImage(targetCanvas, image, sizeText) {
  const requestedSize = parseSizeValue(sizeText);
  const width = requestedSize?.width || image.width || image.naturalWidth || 1280;
  const height = requestedSize?.height || image.height || image.naturalHeight || 720;
  if (targetCanvas.width !== width || targetCanvas.height !== height) {
    targetCanvas.width = width;
    targetCanvas.height = height;
  }
  const targetContext = targetCanvas.getContext("2d", { alpha: false });
  const sourceWidth = image.width || image.naturalWidth || width;
  const sourceHeight = image.height || image.naturalHeight || height;
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  targetContext.fillStyle = "#11140f";
  targetContext.fillRect(0, 0, width, height);
  targetContext.imageSmoothingEnabled = true;
  targetContext.imageSmoothingQuality = "high";
  targetContext.drawImage(
    image,
    (width - drawWidth) / 2,
    (height - drawHeight) / 2,
    drawWidth,
    drawHeight,
  );
}

function drawVisibleReferencePlaceholders() {
  if (!selectedReferencePreviewReady) return;
  const referencePreview = $("referencePreview");
  drawPlaceholderImage(canvas, referencePreview, modelControl("minwm", "size").value);
  drawPlaceholderImage(
    lingbot2Canvas,
    referencePreview,
    modelControl("lingbot2", "size").value,
  );
}

async function drawInitialReferencePlaceholders(firstFrame) {
  if (!firstFrame?.byteLength || typeof createImageBitmap !== "function") return;
  let image;
  try {
    image = await createImageBitmap(new Blob([firstFrame]));
    drawPlaceholderImage(canvas, image, modelControl("minwm", "size").value);
    drawPlaceholderImage(
      lingbot2Canvas,
      image,
      modelControl("lingbot2", "size").value,
    );
  } catch (error) {
    addHistory(`reference placeholder unavailable · ${error?.message || error}`);
  } finally {
    image?.close?.();
  }
}

async function fetchReferenceBytes(url) {
  try {
    const response = await fetch(url, { cache: "force-cache", mode: "cors" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (!bytes.byteLength) {
      throw new Error("empty image");
    }
    return bytes;
  } catch (error) {
    throw new Error(
      `reference image fetch failed: ${error.message || String(error)}`
    );
  }
}

function drawReferencePreviewFromImageSource(src, label) {
  const preview = $("referencePreview");
  const previewCtx = preview.getContext("2d", { alpha: false });
  selectedReferencePreviewReady = false;
  previewCtx.fillStyle = "#101515";
  previewCtx.fillRect(0, 0, preview.width, preview.height);
  $("referenceName").textContent = label;
  const img = new Image();
  return new Promise((resolve) => {
    img.onload = () => {
      const scale = Math.min(preview.width / img.width, preview.height / img.height);
      const w = img.width * scale, h = img.height * scale;
      previewCtx.fillRect(0, 0, preview.width, preview.height);
      previewCtx.drawImage(img, (preview.width - w) / 2, (preview.height - h) / 2, w, h);
      setHappyOysterReferencePreview(preview.toDataURL("image/jpeg", 0.86));
      selectedReferencePreviewReady = true;
      updateWorldDraftState();
      drawVisibleReferencePlaceholders();
      if (src.startsWith("blob:")) URL.revokeObjectURL(src);
      resolve(true);
    };
    img.onerror = () => {
      selectedReferencePreviewReady = false;
      updateWorldDraftState();
      previewCtx.fillStyle = "#11140f";
      previewCtx.fillRect(0, 0, preview.width, preview.height);
      previewCtx.fillStyle = "#8c9288";
      previewCtx.font = "14px ui-sans-serif, Avenir Next, Helvetica Neue, sans-serif";
      previewCtx.textAlign = "center";
      previewCtx.textBaseline = "middle";
      previewCtx.fillText("reference image unavailable", preview.width / 2, preview.height / 2);
      setHappyOysterReferencePreview("");
      if (src.startsWith("blob:")) URL.revokeObjectURL(src);
      resolve(false);
    };
    img.src = src;
  });
}

function clearReferencePreview() {
  const preview = $("referencePreview");
  const previewCtx = preview.getContext("2d", { alpha: false });
  selectedReferencePreviewReady = false;
  previewCtx.fillStyle = "#101515";
  previewCtx.fillRect(0, 0, preview.width, preview.height);
  $("referenceName").textContent = "尚未选择图片";
  setHappyOysterReferencePreview("");
}

function hasFirstFrame() {
  return Boolean(
    $("firstFrame").files[0]
    || selectedReferenceBytes?.byteLength
    || (selectedReferenceUrl && selectedReferencePreviewReady)
  );
}

function hasWorldDescription() {
  return Boolean($("prompt").value.trim());
}

function skillRuleElements() {
  return Array.from(document.querySelectorAll(".skill-rule-item"));
}

function goalRuleElements() {
  return Array.from(document.querySelectorAll(".goal-rule-item"));
}

function setWorldRulesStatus(message, state = "") {
  const status = $("worldRulesStatus");
  status.textContent = message;
  if (state) status.dataset.state = state;
  else delete status.dataset.state;
}

function readWorldRulesDraft() {
  const goals = goalRuleElements().map((item) => {
    const goalInput = item.querySelector("[data-rule-field='input']").value;
    const goalMinPlaySeconds = item.querySelector("[data-rule-field='min_play_seconds']").value;
    if (goalInput.trim() && goalMinPlaySeconds !== "") {
      const seconds = Number(goalMinPlaySeconds);
      if (!Number.isFinite(seconds) || seconds < 0 || seconds > MAX_GOAL_MIN_PLAY_SECONDS) {
        throw new Error(`目标至少游玩时间必须在 0–${MAX_GOAL_MIN_PLAY_SECONDS} 秒之间`);
      }
    }
    return {
      id: item.dataset.goalRuleId,
      min_play_seconds: goalMinPlaySeconds,
      probability: item.querySelector("[data-rule-field='probability']").value,
      input: goalInput,
    };
  });
  return {
    skills: skillRuleElements().map((item) => ({
      id: item.dataset.skillRuleId,
      input: item.querySelector("[data-rule-field='input']").value,
    })),
    goals,
  };
}

function normalizedWorldRulesForStorage(draft = readWorldRulesDraft()) {
  return normalizeWorldRulesDraft(draft);
}

function worldRulesGoalCount(normalized) {
  if (Array.isArray(normalized?.goals)) return normalized.goals.length;
  return normalized?.goal ? 1 : 0;
}

function worldRulesRuleCount(normalized) {
  return (Array.isArray(normalized?.skills) ? normalized.skills.length : 0)
    + worldRulesGoalCount(normalized);
}

function worldRulesStorageSignature(draft = readWorldRulesDraft()) {
  try {
    return JSON.stringify(normalizedWorldRulesForStorage(draft));
  } catch {
    return JSON.stringify(draft || {});
  }
}

function hasConfiguredWorldRules(draft = readWorldRulesDraft()) {
  try {
    const normalized = normalizedWorldRulesForStorage(draft);
    return Boolean(worldRulesRuleCount(normalized));
  } catch {
    return true;
  }
}

function worldRulesPreparationSignature(description, draft = readWorldRulesDraft()) {
  return JSON.stringify({
    description: normalizeWorldDescription(description),
    rules: normalizedWorldRulesForStorage(draft),
  });
}

function invalidatePreparedWorldRules() {
  preparedWorldRulesCache = null;
  worldRulesDraftGeneration += 1;
  skillRuleElements().forEach((item) => {
    const state = item.querySelector(".skill-rule-state");
    state.textContent = item.querySelector("[data-rule-field='input']").value.trim()
      ? "待进入世界时由 AI 补全"
      : "填写技能标签或动作描述后启用";
    delete state.dataset.state;
  });
  goalRuleElements().forEach((item) => {
    const state = item.querySelector(".goal-rule-state");
    state.textContent = item.querySelector("[data-rule-field='input']").value.trim()
      ? "待进入世界时由 AI 补全"
      : "填写奖励或触发效果后启用";
    delete state.dataset.state;
  });
}

function updateWorldRulesDraftUi() {
  const items = skillRuleElements();
  items.forEach((item, index) => {
    item.querySelector(".skill-rule-key").textContent = String(index + 1);
  });
  $("skillRuleEmpty").hidden = items.length > 0;
  $("addSkillRuleBtn").disabled = items.length >= 9;
  const goalItems = goalRuleElements();
  goalItems.forEach((item, index) => {
    item.querySelector(".goal-rule-key").textContent = String(index + 1);
    item.querySelector(".goal-rule-title").textContent = `目标 ${index + 1}`;
  });
  $("goalRuleEmpty").hidden = goalItems.length > 0;
  $("addGoalRuleBtn").disabled = goalItems.length >= 9;
  const skillCount = items.filter((item) => (
    item.querySelector("[data-rule-field='input']").value.trim()
  )).length;
  const configuredGoalItems = goalItems.filter((item) => (
    item.querySelector("[data-rule-field='input']").value.trim()
  ));
  const parts = [];
  if (skillCount) parts.push(`${skillCount} 个技能`);
  if (configuredGoalItems.length) {
    const minPlaySecondsText = configuredGoalItems
      .map((item) => item.querySelector("[data-rule-field='min_play_seconds']").value.trim() || "10")
      .join("/");
    parts.push(`${configuredGoalItems.length} 个目标 · ≥${minPlaySecondsText}s`);
  }
  $("worldRulesSummary").textContent = parts.length ? parts.join(" · ") : "未配置";

  if (!parts.length) {
    setWorldRulesStatus("规则非必填", "");
    return;
  }
  try {
    normalizedWorldRulesForStorage();
    setWorldRulesStatus("进入世界前 AI 会自动补全名称与完整 Prompt", "ready");
  } catch (error) {
    setWorldRulesStatus(error.message || "规则配置不完整", "error");
  }
}

function handleWorldRulesDraftInput() {
  invalidatePreparedWorldRules();
  updateWorldRulesDraftUi();
}

function addSkillRule(skill = {}, { focus = true } = {}) {
  const list = $("skillRuleList");
  if (skillRuleElements().length >= 9) {
    setWorldRulesStatus("最多可以配置 9 个技能", "error");
    return null;
  }
  const item = document.createElement("article");
  item.className = "skill-rule-item";
  const existingIds = new Set(skillRuleElements().map((entry) => entry.dataset.skillRuleId));
  let skillId = String(skill.id || "").trim();
  if (!skillId || existingIds.has(skillId)) {
    do {
      skillId = `skill-${skillRuleNextId++}`;
    } while (existingIds.has(skillId));
  } else {
    const numericId = /^skill-(\d+)$/.exec(skillId);
    if (numericId) skillRuleNextId = Math.max(skillRuleNextId, Number(numericId[1]) + 1);
  }
  item.dataset.skillRuleId = skillId;

  const head = document.createElement("div");
  head.className = "skill-rule-item-head";
  const key = document.createElement("span");
  key.className = "skill-rule-key";
  key.setAttribute("aria-hidden", "true");
  const input = document.createElement("textarea");
  input.rows = 2;
  input.maxLength = 2000;
  input.placeholder = "输入技能标签或动作描述，例如：召唤飞船；或：从云层中召唤一艘发光飞船……";
  input.value = String(skill.input || skill.instruction || skill.name || "");
  input.dataset.ruleField = "input";
  input.setAttribute("aria-label", "技能标签或动作描述");
  const remove = document.createElement("button");
  remove.className = "skill-rule-remove";
  remove.type = "button";
  remove.setAttribute("aria-label", "删除技能");
  remove.title = "删除技能";
  remove.textContent = "×";
  head.append(key, input, remove);
  const state = document.createElement("span");
  state.className = "skill-rule-state";
  state.textContent = input.value.trim()
    ? "待进入世界时由 AI 补全"
    : "填写技能标签或动作描述后启用";
  item.append(head, state);
  list.appendChild(item);

  input.addEventListener("input", handleWorldRulesDraftInput);
  remove.onclick = () => {
    item.remove();
    handleWorldRulesDraftInput();
  };
  updateWorldRulesDraftUi();
  if (focus) input.focus({ preventScroll: true });
  return item;
}

function addGoalRule(goal = {}, { focus = true } = {}) {
  const list = $("goalRuleList");
  if (goalRuleElements().length >= 9) {
    setWorldRulesStatus("最多可以配置 9 个目标", "error");
    return null;
  }
  const item = document.createElement("article");
  item.className = "goal-rule-item";
  const existingIds = new Set(goalRuleElements().map((entry) => entry.dataset.goalRuleId));
  let goalId = String(goal.id || "").trim();
  if (!goalId || existingIds.has(goalId)) {
    do {
      goalId = `goal-${goalRuleNextId++}`;
    } while (existingIds.has(goalId));
  } else {
    const numericId = /^goal-(\d+)$/.exec(goalId);
    if (numericId) goalRuleNextId = Math.max(goalRuleNextId, Number(numericId[1]) + 1);
  }
  item.dataset.goalRuleId = goalId;

  const head = document.createElement("div");
  head.className = "goal-rule-item-head";
  const key = document.createElement("span");
  key.className = "goal-rule-key";
  key.setAttribute("aria-hidden", "true");
  const title = document.createElement("strong");
  title.className = "goal-rule-title";
  const remove = document.createElement("button");
  remove.className = "goal-rule-remove";
  remove.type = "button";
  remove.setAttribute("aria-label", "删除目标");
  remove.title = "删除目标";
  remove.textContent = "×";
  head.append(key, title, remove);

  const fields = document.createElement("div");
  fields.className = "goal-rule-fields";
  const minLabel = document.createElement("label");
  minLabel.className = "goal-min-play-field";
  minLabel.innerHTML = "<span>至少游玩（秒）</span>";
  const minPlay = document.createElement("input");
  minPlay.type = "number";
  minPlay.min = "0";
  minPlay.max = String(MAX_GOAL_MIN_PLAY_SECONDS);
  minPlay.step = "1";
  minPlay.placeholder = "10";
  minPlay.inputMode = "decimal";
  minPlay.value = goal.min_play_seconds == null
    ? (goal.minPlaySeconds == null ? "" : String(goal.minPlaySeconds))
    : String(goal.min_play_seconds);
  minPlay.dataset.ruleField = "min_play_seconds";
  minPlay.setAttribute("aria-label", "目标至少游玩秒数");
  minLabel.appendChild(minPlay);

  const probabilityLabel = document.createElement("label");
  probabilityLabel.className = "goal-probability-field";
  probabilityLabel.innerHTML = "<span>触发概率</span>";
  const probability = document.createElement("input");
  probability.type = "number";
  probability.min = "0";
  probability.max = "1";
  probability.step = "0.01";
  probability.placeholder = "0.2";
  probability.inputMode = "decimal";
  probability.value = goal.probability == null ? "" : String(goal.probability);
  probability.dataset.ruleField = "probability";
  probability.setAttribute("aria-label", "目标触发概率");
  probabilityLabel.appendChild(probability);

  const inputLabel = document.createElement("label");
  inputLabel.className = "goal-rule-input-field";
  inputLabel.innerHTML = "<span>达成奖励或触发效果</span>";
  const input = document.createElement("textarea");
  input.rows = 3;
  input.maxLength = 2000;
  input.placeholder = "例如：星光徽章；或：天空落下一枚发光徽章，被主角稳稳接住……";
  input.value = String(goal.input || goal.instruction || goal.name || "");
  input.dataset.ruleField = "input";
  input.setAttribute("aria-label", "目标达成奖励或触发效果");
  inputLabel.appendChild(input);

  fields.append(minLabel, probabilityLabel, inputLabel);
  const state = document.createElement("span");
  state.className = "goal-rule-state";
  state.textContent = input.value.trim()
    ? "待进入世界时由 AI 补全"
    : "填写奖励或触发效果后启用";
  item.append(head, fields, state);
  list.appendChild(item);

  for (const control of [minPlay, probability, input]) {
    control.addEventListener("input", handleWorldRulesDraftInput);
  }
  remove.onclick = () => {
    item.remove();
    handleWorldRulesDraftInput();
  };
  updateWorldRulesDraftUi();
  if (focus) input.focus({ preventScroll: true });
  return item;
}

function applyWorldRulesDraft(draft = null) {
  $("skillRuleList").innerHTML = "";
  $("goalRuleList").innerHTML = "";
  const skills = Array.isArray(draft?.skills) ? draft.skills : [];
  skills.slice(0, 9).forEach((skill) => addSkillRule(skill, { focus: false }));
  const goals = Array.isArray(draft?.goals) ? draft.goals : (draft?.goal ? [draft.goal] : []);
  goals.slice(0, 9).forEach((goal) => addGoalRule(goal, { focus: false }));
  invalidatePreparedWorldRules();
  updateWorldRulesDraftUi();
}

function setWorldRulesPreparing(pending) {
  $("worldRulesPanel").setAttribute("aria-busy", pending ? "true" : "false");
  $("worldRulesPanel").querySelectorAll("input, textarea, button").forEach((control) => {
    control.disabled = pending;
  });
  $("addSkillRuleBtn").disabled = pending || skillRuleElements().length >= 9;
  $("addGoalRuleBtn").disabled = pending || goalRuleElements().length >= 9;
}

async function prepareWorldRulesForEntry(description) {
  const draft = readWorldRulesDraft();
  const signature = worldRulesPreparationSignature(description, draft);
  if (preparedWorldRulesCache?.signature === signature) {
    return preparedWorldRulesCache.prepared;
  }
  const normalized = normalizedWorldRulesForStorage(draft);
  const ruleCount = worldRulesRuleCount(normalized);
  if (!ruleCount) {
    setWorldRulesStatus("当前世界未配置规则", "");
    return { skills: [], goals: [] };
  }

  const generation = worldRulesDraftGeneration;
  setWorldRulesPreparing(true);
  setWorldRulesStatus(`正在并行补全 ${ruleCount} 条规则…`, "working");
  skillRuleElements().forEach((item) => {
    const state = item.querySelector(".skill-rule-state");
    if (item.querySelector("[data-rule-field='input']").value.trim()) {
      state.textContent = "AI 正在补全名称与 Prompt…";
      delete state.dataset.state;
    }
  });
  try {
    const prepared = await worldRulesController.prepare(normalized, description);
    if (generation !== worldRulesDraftGeneration) {
      throw new Error("规则已发生变化，请重新进入世界");
    }
    prepared.skills.forEach((skill) => {
      const item = skillRuleElements().find((candidate) => (
        candidate.dataset.skillRuleId === skill.id
      ));
      const state = item?.querySelector(".skill-rule-state");
      if (!state) return;
      state.textContent = skill.prepared.change_type === "one_time"
        ? `✓ ${skill.name} · 一次性`
        : `✓ ${skill.name} · 持久`;
      state.dataset.state = "ready";
    });
    const preparedGoals = Array.isArray(prepared.goals) ? prepared.goals : (
      prepared.goal ? [prepared.goal] : []
    );
    preparedGoals.forEach((goal) => {
      const item = goalRuleElements().find((candidate) => (
        candidate.dataset.goalRuleId === goal.id
      ));
      const state = item?.querySelector(".goal-rule-state");
      if (!state) return;
      state.textContent = `✓ ${goal.name} · ≥${goal.min_play_seconds}s · p=${goal.probability}`;
      state.dataset.state = "ready";
    });
    preparedWorldRulesCache = { signature, prepared };
    setWorldRulesStatus(`${ruleCount} 条规则已补全；技能可立即使用，目标将按时间自动触发`, "ready");
    return prepared;
  } catch (error) {
    $("worldRulesPanel").open = true;
    skillRuleElements().forEach((item) => {
      const state = item.querySelector(".skill-rule-state");
      if (!state.dataset.state) {
        state.textContent = "补全失败，请重试";
        state.dataset.state = "error";
      }
    });
    goalRuleElements().forEach((item) => {
      const state = item.querySelector(".goal-rule-state");
      if (!state.dataset.state) {
        state.textContent = "补全失败，请重试";
        state.dataset.state = "error";
      }
    });
    setWorldRulesStatus(error.message || "规则补全失败", "error");
    throw error;
  } finally {
    setWorldRulesPreparing(false);
  }
}

function hasLiveWorldRuleTarget() {
  return selectedModelKeys().some((key) => (
    document.querySelector(`[data-model-key="${key}"]`)?.dataset.sessionState === "live"
  ));
}

function renderRuntimeSkillBar(snapshot = null) {
  snapshot = snapshot || worldRulesController?.snapshot() || { skills: [] };
  const bar = $("runtimeSkillBar");
  const container = $("runtimeSkillButtons");
  const hint = $("runtimeSkillHint");
  if (!bar || !container) return;
  const cooldownRemainingMs = Math.max(0, Number(snapshot.skillCooldownRemainingMs || 0));
  const cooldownActive = cooldownRemainingMs > 0;
  const cooldownSeconds = Math.max(1, Math.ceil(cooldownRemainingMs / 1000));
  if (cooldownActive && !runtimeSkillCooldownUiTimer) {
    runtimeSkillCooldownUiTimer = window.setInterval(() => {
      renderRuntimeSkillBar(worldRulesController?.snapshot());
    }, 200);
  } else if (!cooldownActive && runtimeSkillCooldownUiTimer) {
    window.clearInterval(runtimeSkillCooldownUiTimer);
    runtimeSkillCooldownUiTimer = 0;
  }
  container.innerHTML = "";
  bar.hidden = snapshot.skills.length === 0;
  bar.classList.toggle("is-cooldown", cooldownActive);
  if (hint) {
    hint.textContent = cooldownActive
      ? `全部技能共享冷却 · ${cooldownSeconds}s`
      : "点击或按数字键触发 · 共享 10s CD";
  }
  const canTrigger = worldExperienceReady
    && sessionPlayable
    && !sessionLifetimeExpired
    && hasLiveWorldRuleTarget();
  snapshot.skills.forEach((skill, index) => {
    const button = document.createElement("button");
    button.className = "runtime-skill-button";
    button.type = "button";
    button.dataset.skillId = skill.id;
    button.disabled = !canTrigger || skill.pending || cooldownActive;
    button.classList.toggle("is-pending", Boolean(skill.pending));
    button.classList.toggle("is-cooldown", cooldownActive);
    button.title = cooldownActive
      ? `全部技能冷却中，还剩 ${cooldownSeconds} 秒`
      : skill.instruction;
    const shortcut = document.createElement("kbd");
    shortcut.textContent = String(index + 1);
    const label = document.createElement("span");
    label.textContent = skill.pending
      ? `${skill.name}…`
      : cooldownActive
        ? `${skill.name} · ${cooldownSeconds}s`
        : skill.name;
    button.append(shortcut, label);
    button.onclick = () => triggerWorldSkill(skill.id);
    container.appendChild(button);
  });
}

async function triggerWorldSkill(skillId) {
  const rulesSnapshot = worldRulesController.snapshot();
  const skill = rulesSnapshot.skills.find((item) => item.id === skillId);
  if (!skill || !worldExperienceReady || !sessionPlayable || !hasLiveWorldRuleTarget()) {
    setPromptRewriteStatus("模型连接已断开，请重新进入世界", "error");
    renderRuntimeSkillBar();
    return;
  }
  if (rulesSnapshot.skillCooldownRemainingMs > 0) {
    setPromptRewriteStatus(
      `全部技能冷却中，还剩 ${Math.ceil(rulesSnapshot.skillCooldownRemainingMs / 1000)} 秒`,
      "working",
    );
    renderRuntimeSkillBar(rulesSnapshot);
    return;
  }
  setPromptRewriteStatus(`正在触发技能「${skill.name}」…`, "working");
  try {
    const result = await worldRulesController.triggerSkill(skillId);
    if (result?.ignored) {
      if (result.reason === "shared_cooldown") {
        setPromptRewriteStatus(
          `全部技能冷却中，还剩 ${Math.ceil(Number(result.remaining_ms || 0) / 1000)} 秒`,
          "working",
        );
      }
      return;
    }
    setPromptRewriteStatus(
      result.change_type === "one_time"
        ? `已触发「${skill.name}」· 一次性，10 秒后恢复`
        : `已触发「${skill.name}」· 持久状态`,
      result.change_type,
    );
    canvas.focus({ preventScroll: true });
  } catch (error) {
    setPromptRewriteStatus(error.message || "技能触发失败，请重试", "error");
    addHistory(`skill trigger failed · ${skill.name} · ${error.message || error}`);
  }
}

function setWorldDraftStatus(message, state = "") {
  const status = $("worldDraftStatus");
  status.textContent = message;
  if (state) status.dataset.state = state;
  else delete status.dataset.state;
}

function setWorldCompletionBusy(pending, completingFromImage = false) {
  worldCompletionPending = pending;
  const button = $("enhanceBtn");
  button.disabled = pending;
  button.classList.toggle("is-loading", pending);
  button.setAttribute("aria-busy", pending ? "true" : "false");
  $("enhanceBtnLabel").textContent = pending ? "正在补全…" : "补全世界";
  $("enhanceBtnHint").textContent = pending
    ? (completingFromImage ? "正在理解首帧并生成完整描述" : "正在生成世界描述和首帧")
    : "补齐首帧与完整世界描述";
  $("clearWorldBtn").disabled = pending;
  $("prompt").readOnly = pending;
  $("firstFrame").disabled = pending;
  document.querySelectorAll(".preset").forEach((preset) => {
    preset.disabled = pending;
  });
  updateWorldDraftState();
}

function updateWorldDraftState() {
  const hasImage = hasFirstFrame();
  const hasDescription = hasWorldDescription();
  $("firstFrameState").textContent = hasImage ? "✓ 已填写" : "未填写";
  $("worldDescriptionState").textContent = hasDescription ? "✓ 已填写" : "未填写";
  $("firstFrameState").classList.toggle("is-complete", hasImage);
  $("worldDescriptionState").classList.toggle("is-complete", hasDescription);
  document.querySelector(".reference-upload").classList.toggle("has-image", hasImage);
  $("referenceUploadTitle").textContent = hasImage ? "更换首帧" : "上传首帧";
  const complete = hasImage && hasDescription;
  $("connectBtn").disabled = worldCompletionPending;
  $("connectBtn").title = complete
    ? "进入当前世界"
    : "需要首帧图片和世界描述；点击后会提示缺少项";
  if (!worldCompletionPending && !complete) {
    const missing = [
      !hasImage ? "首帧图片" : "",
      !hasDescription ? "世界描述" : "",
    ].filter(Boolean).join("和");
    setWorldDraftStatus(`还需要${missing}，可点击补全世界`, "incomplete");
  } else if (!worldCompletionPending && complete) {
    setWorldDraftStatus("世界已完整，可以进入", "ready");
  }
}

function clearWorldDraft() {
  if (ws && ws.readyState === WebSocket.OPEN) closeSession("world draft cleared");
  selectedPreset = null;
  selectedReferenceBytes = null;
  selectedReferenceUrl = "";
  selectedReferenceLabel = "";
  selectedReferenceMimeType = "";
  $("firstFrame").value = "";
  $("prompt").value = "";
  applyWorldRulesDraft(null);
  document.querySelectorAll(".preset").forEach((button) => {
    button.classList.remove("is-selected");
    button.setAttribute("aria-pressed", "false");
  });
  clearReferencePreview();
  updateWorldDraftState();
  $("prompt").focus({ preventScroll: true });
  addHistory("world draft cleared");
}

async function worldCompletionImage() {
  const file = $("firstFrame").files[0];
  if (file) return file;
  if (!hasFirstFrame()) return null;
  const bytes = await readFirstFrame();
  if (!bytes?.byteLength) return null;
  const mime = selectedReferenceMimeType
    || selectedPreset?.mime
    || mimeFromReferenceUrl(selectedReferenceUrl)
    || "image/png";
  return new File([bytes], selectedReferenceLabel || "first-frame", { type: mime });
}

async function completeWorldDraft() {
  if (worldCompletionPending) return;
  const seedText = $("prompt").value.trim();
  if (!seedText && !hasFirstFrame()) {
    setWorldDraftStatus("请先写一句世界描述或上传一张图片", "error");
    $("prompt").focus({ preventScroll: true });
    return;
  }
  let finalStatus = null;
  const completingFromImage = hasFirstFrame();
  setWorldCompletionBusy(true, completingFromImage);
  setWorldDraftStatus(
    completingFromImage ? "正在理解首帧并补全世界描述…" : "正在生成世界描述和首帧，请稍候…",
    "working",
  );
  await new Promise((resolve) => requestAnimationFrame(() => resolve()));
  try {
    const form = new FormData();
    if (seedText) form.append("world_description", seedText);
    const image = await worldCompletionImage();
    if (image) form.append("first_frame", image, image.name);
    const response = await fetch("./api/world/complete", { method: "POST", body: form });
    let result = null;
    try {
      result = await response.json();
    } catch {
      result = null;
    }
    if (!response.ok) {
      throw new Error(result?.error || `world completion failed (${response.status})`);
    }
    $("prompt").value = String(result.world_description || "").trim();
    invalidatePreparedWorldRules();
    updateWorldRulesDraftUi();
    if (result.image_url) {
      selectedPreset = null;
      selectedReferenceBytes = null;
      selectedReferenceUrl = result.image_url;
      selectedReferenceLabel = "AI 生成首帧";
      selectedReferenceMimeType = "image/png";
      $("firstFrame").value = "";
      await drawReferencePreviewFromImageSource(result.image_url, selectedReferenceLabel);
    }
    if (!hasFirstFrame() || !hasWorldDescription()) {
      throw new Error("world completion did not produce both required fields");
    }
    finalStatus = {
      message: result.image_generated
        ? "已生成首帧并补全世界描述"
        : "已根据首帧补全世界描述",
      state: "ready",
    };
    addHistory("world draft completed");
  } catch (error) {
    finalStatus = {
      message: error.message || "世界补全失败，请重试",
      state: "error",
    };
    addHistory(`world completion failed · ${error.message || error}`);
  } finally {
    setWorldCompletionBusy(false);
    if (finalStatus) setWorldDraftStatus(finalStatus.message, finalStatus.state);
  }
}

function drawReferencePreview(file) {
  selectedReferenceBytes = null;
  selectedReferenceUrl = "";
  selectedReferenceLabel = file ? file.name : "";
  selectedReferenceMimeType = file?.type || "";
  if (!file) {
    clearReferencePreview();
    updateWorldDraftState();
    return;
  }
  drawReferencePreviewFromImageSource(URL.createObjectURL(file), file.name);
  updateWorldDraftState();
}

function clearSelectedWorldPreset() {
  selectedPreset = null;
  document.querySelectorAll(".preset").forEach((button) => {
    button.classList.remove("is-selected");
    button.setAttribute("aria-pressed", "false");
  });
}

function isSupportedFirstFrameImage(file) {
  if (!file) return false;
  const supportedTypes = new Set(["image/png", "image/jpeg", "image/webp"]);
  if (file.type) return supportedTypes.has(file.type.toLowerCase());
  return /\.(?:png|jpe?g|webp)$/i.test(file.name || "");
}

async function useFirstFrameFile(file, { fromDrop = false } = {}) {
  if (!file) return false;
  if (!isSupportedFirstFrameImage(file)) {
    setWorldDraftStatus("请拖入 PNG、JPG 或 WebP 图片", "error");
    addHistory("unsupported first-frame file rejected");
    return false;
  }
  clearSelectedWorldPreset();
  selectedReferenceBytes = null;
  selectedReferenceUrl = "";
  selectedReferenceLabel = file.name || "拖入的首帧";
  selectedReferenceMimeType = file.type || mimeFromReferenceUrl(file.name);
  $("firstFrame").value = "";
  const previewPromise = drawReferencePreviewFromImageSource(
    URL.createObjectURL(file),
    selectedReferenceLabel,
  );
  selectedReferenceBytes = new Uint8Array(await file.arrayBuffer());
  await previewPromise;
  updateWorldDraftState();
  if (fromDrop) addHistory(`first frame dropped · ${selectedReferenceLabel}`);
  return true;
}

function setupFirstFrameDropZone() {
  const dropZone = $("referenceDropZone");
  if (!dropZone) return;
  let dragDepth = 0;
  const hasFiles = (event) => {
    const types = event.dataTransfer?.types;
    if (!types) return Boolean(event.dataTransfer?.files?.length);
    for (let index = 0; index < types.length; index += 1) {
      if (types[index] === "Files") return true;
    }
    return Boolean(event.dataTransfer?.files?.length);
  };
  const clearDragging = () => {
    dragDepth = 0;
    dropZone.classList.remove("is-dragging");
  };
  dropZone.addEventListener("dragenter", (event) => {
    if (!hasFiles(event) || worldCompletionPending) return;
    event.preventDefault();
    dragDepth += 1;
    dropZone.classList.add("is-dragging");
  });
  dropZone.addEventListener("dragover", (event) => {
    if (!hasFiles(event) || worldCompletionPending) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  dropZone.addEventListener("dragleave", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dropZone.classList.remove("is-dragging");
  });
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    if (worldCompletionPending) {
      clearDragging();
      return;
    }
    const files = Array.from(event.dataTransfer?.files || []);
    clearDragging();
    void useFirstFrameFile(files[0], { fromDrop: true }).catch((error) => {
      setWorldDraftStatus("首帧图片读取失败，请重试", "error");
      addHistory(`dropped first frame failed · ${error.message || error}`);
    });
  });
  window.addEventListener("dragend", clearDragging);
  window.addEventListener("drop", clearDragging);
}

async function setPresetReference(preset) {
  selectedReferenceBytes = preset.imageBlob
    ? new Uint8Array(await preset.imageBlob.arrayBuffer())
    : null;
  selectedReferenceUrl = preset.referenceUrl;
  selectedReferenceLabel = preset.source;
  selectedReferenceMimeType = preset.mime || mimeFromReferenceUrl(preset.referenceUrl);
  $("firstFrame").value = "";
  const previewUrl = preset.imageBlob
    ? URL.createObjectURL(preset.imageBlob)
    : preset.referenceUrl;
  await drawReferencePreviewFromImageSource(previewUrl, selectedReferenceLabel);
  updateWorldDraftState();
}

function showError(error) {
  setStatus("Reference load failed", "error");
  if (!renderedPreviewFrames) setPreviewState("idle");
  addHistory(error.message || "reference load failed");
}

function abortCurrentSession(reason = "session closed by client", {
  clearFrames = true,
  expectedClose = true,
  keepConnectDisabled = false,
  resetControls = true,
} = {}) {
  promptRewriteController.endSession();
  recordTrajectoryEvent(expectedClose ? "session_close_requested" : "session_abort_requested", {
    reason,
    clear_frames: clearFrames,
  });
  const socket = ws;
  ws = null;
  streamEpoch++;
  clearQueueOnClose = clearFrames;
  socketCloseExpected = expectedClose;
  if (resetControls) controlStateController?.reset({ sendRelease: false });
  pendingHeader = null;
  rejectPendingDecodes("session aborted");
  resetDecoderState();
  if (clearFrames) {
    clearFrameQueue();
    updateStats();
  }
  if (!socket) {
    clearQueueOnClose = false;
    if (!keepConnectDisabled) $("connectBtn").disabled = false;
    setStatus("Closed");
    if (!renderedPreviewFrames) setPreviewState("idle");
    return null;
  }
  if (!keepConnectDisabled) $("connectBtn").disabled = false;
  setStatus(expectedClose ? "Closing" : "Aborting");
  if (!renderedPreviewFrames) setPreviewState("idle");
  addHistory(reason);
  socket.close(expectedClose ? 1000 : 4000, reason.slice(0, 120));
  return socket;
}

function closeSession(reason = "session closed by client", clearFrames = true) {
  promptRewriteController.endSession();
  cancelLingbot2Reconnect();
  stopWorldExperienceTiming({ recordingReason: "session_closed" });
  clearQueueOnClose = clearFrames;
  dualModelController.close(reason);
}

function waitForSocketClose(socket, timeoutMs = RECONNECT_CLOSE_TIMEOUT_MS) {
  return new Promise((resolve) => {
    if (!socket || socket.readyState === WebSocket.CLOSED) {
      resolve();
      return;
    }
    const finish = () => {
      socket.removeEventListener("close", finish);
      window.clearTimeout(timer);
      resolve();
    };
    const timer = window.setTimeout(finish, timeoutMs);
    socket.addEventListener("close", finish, { once: true });
    socket.close(1000, "replace session");
  });
}

async function connect() {
  if (recordingActive) {
    await stopRecording({ reason: "session_replaced" });
  }
  promptRewriteController.endSession();
  worldRulesController.endSession();
  setPromptRewriteStatus("进入世界后可发送新指令", "");
  cancelLingbot2Reconnect();
  resetSessionLifetimeUi();
  $("connectBtn").disabled = true;
  setModelConnectionState("minwm", "connecting");
  setModelConnectionState("lingbot2", "connecting");
  setModelConnectionState("happyoyster", "connecting");
  setStatus("Preparing");
  setPreviewState("waiting");
  addHistory("preparing session");
  try {
    if (ws && ws.readyState !== WebSocket.CLOSED) {
      setStatus("Replacing");
      const oldSocket = abortCurrentSession("closing previous socket before reconnect", {
        keepConnectDisabled: true,
      });
      await waitForSocketClose(oldSocket);
    }
    resetStreamStats();
    const epoch = ++streamEpoch;
    currentTrace = createClientTrace();
    traceHttpClient?.reset(currentTrace.traceId, $("serverUrl").value);
    resetTraceTopology(currentTrace.traceId);
    markClientTrace("client.generate_clicked", {
      generation_mode: selectedGenerationMode(),
      transport: $("transportFormat").value || "raw",
      fps: Number($("fps").value || DEFAULT_TARGET_FPS),
    });
    const generationMode = selectedGenerationMode();
    if (!hasWorldDescription() || !hasFirstFrame()) {
      setStatus("Complete world first", "error");
      setWorldDraftStatus("请先补齐首帧图片和世界描述", "error");
      setModelConnectionState("minwm", "idle");
      setModelConnectionState("lingbot2", "idle");
      setModelConnectionState("happyoyster", "idle");
      setPreviewState("idle");
      addHistory("world draft incomplete");
      $("connectBtn").disabled = false;
      (hasWorldDescription() ? $("firstFrame") : $("prompt")).focus?.({ preventScroll: true });
      return;
    }
    const enteredWorldRules = normalizedWorldRulesForStorage();
    const preparedWorldRules = await prepareWorldRulesForEntry($("prompt").value.trim());
    const continuousT2V = generationMode === "t2v" && $("continuous").checked;
    const enteredWorldSnapshot = {
      description: $("prompt").value,
      preset: selectedPreset,
      rules: enteredWorldRules,
    };
    let enteredFirstFrame;
    let firstFrame;
    let numFrames = Number($("numFrames").value);
    if (generationMode === "i2v") {
      drawVisibleReferencePlaceholders();
      enteredFirstFrame = await readFirstFrame();
      firstFrame = enteredFirstFrame;
      if (!firstFrame) {
        setModelConnectionState("minwm", "idle");
        setModelConnectionState("lingbot2", "idle");
        setModelConnectionState("happyoyster", "idle");
        setStatus("Pick a reference", "error");
        setPreviewState("idle");
        addHistory("reference image required for I2V");
        $("connectBtn").disabled = false;
        return;
      }
    } else {
      enteredFirstFrame = await readFirstFrame();
      numFrames = continuousT2V ? undefined : readT2VNumFrames();
    }
    await drawInitialReferencePlaceholders(firstFrame);
    const init = compact({
      type: "init",
      model: $("model").value,
      trace_id: currentTrace.traceId,
      playback_ack_enabled: PLAYBACK_ACK_ENABLED,
      ...readModelRequestParams("minwm", {
        generationMode,
        firstFrame,
        numFrames: continuousT2V ? undefined : numFrames,
      }),
    });
    const referenceImage = await createReferenceImageMeta(enteredFirstFrame);
    beginSessionArtifact(init, referenceImage);
    if (currentSessionArtifact && currentTrace) {
      currentSessionArtifact.trace_id = currentTrace.traceId;
    }
    // Mount prepared skills before waiting for every comparison backend. A slow
    // or reconnecting secondary model must not leave an already-entered world
    // without its controls. The buttons stay disabled until a live target exists.
    promptRewriteController.beginSession(init.prompt);
    worldRulesController.activate(preparedWorldRules);
    beginPromptLogSession(init.prompt);
    document.activeElement?.blur?.();
    canvas.tabIndex = 0;
    canvas.focus();
    worldExperiencePending = true;
    worldExperienceReady = false;
    setStatus("Loading world", "live");
    addHistory("model connected · waiting for first visible Zing frame");
    const connectionReport = await dualModelController.connect(init);
    if (epoch !== streamEpoch) return;
    void rememberEnteredWorld(
      enteredFirstFrame,
      referenceImage,
      enteredWorldSnapshot,
    ).catch((error) => {
      addHistory(`custom world save failed · ${error.message || error}`);
    });
    if (connectionReport.failed.length) {
      addHistory(
        `partial session · ${connectionReport.failed
          .map(({ key, error }) => `${modelLabel(key)} unavailable: ${error?.message || error}`)
          .join(" · ")}`,
      );
      if (connectionReport.failed.some(({ key }) => key === "lingbot2")) {
        scheduleLingbot2Reconnect("initial connection failed");
      }
    }
    if (connectionReport.pending?.includes("happyoyster")) {
      setHappyOysterStageText("正在创建快乐生蚝 World…", "preparing");
      addHistory("快乐生蚝 World 正在独立构建 · Zing/LingBot2 已先行启动");
    }
    if (!worldExperienceReady) setStatus("Loading world", "live");
  } catch (error) {
    stopWorldExperienceTiming({ recordingReason: "startup_failed" });
    $("connectBtn").disabled = false;
    setModelConnectionState("minwm", "error");
    setStatus("Init failed", "error");
    if (!renderedPreviewFrames) setPreviewState("idle");
    addHistory(error.message || "init failed");
  }
}

function openPrimarySession(init, url) {
  const epoch = streamEpoch;
  const generationMode = init.generation_mode;
  const referenceImage = currentSessionArtifact?.reference_image || null;
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(url);
    ws = socket;
    socket.binaryType = "arraybuffer";
    socketHadError = false;
    socketCloseExpected = false;
    socketServerError = "";
    let opened = false;
    socket.onopen = () => {
      if (epoch !== streamEpoch) return;
      opened = true;
      setModelConnectionState("minwm", "live");
      markClientTrace("client.ws_open", { url });
      recordTrajectoryEvent("socket_open", { url });
      const initPayload = pack(init);
      socket.send(initPayload);
      markClientTrace("client.init_sent", {
        generation_mode: generationMode,
        num_frames: init.num_frames,
        has_reference_image: Boolean(referenceImage),
        payload_bytes: initPayload.byteLength,
      });
      recordTrajectoryEvent("init_sent", {
        generation_mode: generationMode,
        num_frames: init.num_frames,
        has_reference_image: Boolean(referenceImage),
      });
      setStatus("Starting", "live");
      const source = generationMode === "t2v"
        ? `${init.num_frames || "continuous"} frames from text`
        : selectedReferenceLabel || "uploaded reference";
      addHistory(`${generationMode.toUpperCase()} dual session started · ${source}`);
      resolve();
    };
    socket.onclose = (event) => {
      if (epoch !== streamEpoch) return;
      if (ws === socket) ws = null;
      markClientTrace("client.ws_close", {
        code: event.code,
        reason: event.reason || "",
      });
      $("connectBtn").disabled = false;
      if (clearQueueOnClose) {
        clearFrameQueue();
        updateStats();
      }
      clearQueueOnClose = false;
      const reason = event.reason ? ` · ${event.reason}` : "";
      const closeText = `Zing socket closed code=${event.code}${reason}`;
      const normalClose = event.code === 1000 || event.code === 1001;
      setModelConnectionState("minwm", normalClose ? "closed" : "error");
      if (socketServerError) {
        setStatus("Server closed", "error");
        addHistory(`${closeText} · ${socketServerError}`);
      } else if (socketHadError && !socketCloseExpected && !normalClose) {
        setStatus("Socket closed", "error");
        addHistory(`${closeText} · transport error`);
      } else {
        setStatus("Closed");
        addHistory(closeText);
      }
      recordTrajectoryEvent("socket_close", {
        backend: "minwm",
        code: event.code,
        reason: event.reason || "",
        normal_close: normalClose,
        expected_close: socketCloseExpected,
      });
      if (isSessionLifetimeReason(event.reason)) {
        expireSessionLifetime({ closeSessions: true });
      } else if (!socketCloseExpected) {
        const hadReadyWorld = worldExperienceReady;
        stopWorldExperienceTiming({ recordingReason: "primary_disconnected" });
        lingbot2Session.close("Zing primary session closed");
        if (hadReadyWorld) {
          showSessionNotice("Zing 连接已中断，已结束计时并生成当前录像");
        }
      }
      void traceHttpClient?.flushClientEvents().catch(() => {});
      if (!renderedPreviewFrames) setPreviewState("idle");
      if (!opened) reject(new Error(`Zing closed before startup (${event.code})`));
      socketCloseExpected = false;
    };
    socket.onerror = () => {
      if (epoch !== streamEpoch) return;
      markClientTrace("client.ws_error");
      recordTrajectoryEvent("socket_error", { backend: "minwm", ready_state: socket.readyState });
      if (!socketCloseExpected) {
        setModelConnectionState("minwm", "error");
        socketHadError = true;
        $("connectBtn").disabled = false;
      }
      if (!opened) reject(new Error("Zing websocket transport error"));
    };
    socket.onmessage = (event) => {
      if (epoch !== streamEpoch) return;
      try {
        receive(event.data, epoch);
      } catch (error) {
        handleReceiveError(error, epoch);
      }
    };
  });
}

function sendPrimaryEventEnvelope(envelope) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  ws.send(pack({ ...envelope, trace_id: currentTrace?.traceId }));
  const eventId = Number(envelope.event_id || 0);
  if (eventId > 0 && ["camera_actions", "prompt", "scene_cut"].includes(envelope.kind)) {
    primaryControlSentEpochByEvent.set(
      eventId,
      Number(envelope.client_sent_epoch_ms || Date.now()),
    );
    while (primaryControlSentEpochByEvent.size > 64) {
      primaryControlSentEpochByEvent.delete(
        primaryControlSentEpochByEvent.keys().next().value,
      );
    }
  }
  return true;
}

function handleReceiveError(error, epoch) {
  if (epoch !== streamEpoch) return;
  setStatus("Receive failed", "error");
  addHistory(error.message || "receive failed");
  abortCurrentSession(error.message || "receive failed", {
    clearFrames: false,
    expectedClose: false,
    resetControls: false,
  });
}

function receive(data, epoch) {
  if (!pendingHeader) {
    const receivedAt = performance.now();
    const message = unpack(new Uint8Array(data));
    message.__received_at = receivedAt;
    if (message.type === "error") {
      markClientTrace("client.server_error_received", {
        payload_bytes: data.byteLength || data.size || 0,
      });
      socketServerError = message.content || "unknown";
      // The protocol defines invalid events as non-fatal. Do not convert one
      // rejected control extension into a complete dual-model disconnect.
      if (socketServerError === "invalid event") {
        addHistory("server rejected one event · session kept alive");
        return;
      }
      if (isExperienceBusyError(message)) {
        handleExperienceBusy();
        return;
      }
      setStatus(socketServerError, "error");
      addHistory(`server error: ${socketServerError}`);
      recordTrajectoryEvent("server_error", { content: socketServerError });
      if (ws && ws.readyState === WebSocket.OPEN) {
        socketCloseExpected = true;
        ws.close(1000, socketServerError.slice(0, 120));
      }
      $("connectBtn").disabled = false;
      if (!renderedPreviewFrames) setPreviewState("idle");
      return;
    }
    if (message.type === "control_ack") {
      if (message.stage === "worker") {
        const clientSentEpochMs = Number(message.client_sent_epoch_ms || 0);
        const serverReceivedEpochMs = Number(message.server_received_epoch_ms || 0);
        const serverSentEpochMs = Number(message.server_sent_epoch_ms || 0);
        const clientReceivedEpochMs = Date.now();
        if (clientSentEpochMs && serverReceivedEpochMs && serverSentEpochMs) {
          const serverProcessingMs = Math.max(0, serverSentEpochMs - serverReceivedEpochMs);
          const controlRoundTripMs = Math.max(0, clientReceivedEpochMs - clientSentEpochMs);
          primaryProtocolStats = {
            ...primaryProtocolStats,
            controlRoundTripMs,
            lastInputUplinkMs: Math.max(0, (controlRoundTripMs - serverProcessingMs) / 2),
            serverClockOffsetMs: (
              (serverReceivedEpochMs - clientSentEpochMs)
              + (serverSentEpochMs - clientReceivedEpochMs)
            ) / 2,
          };
          updateStats();
        }
      }
      return;
    }
    if (message.type === "chunk_telemetry") {
      primaryProtocolStats = {
        ...primaryProtocolStats,
        chunkTelemetry: { ...message },
      };
      updateStats();
      return;
    }
    if (message.type === "session_ready" || message.type === "heartbeat") return;
    if (message.type === "frame_batch") {
      const payload = message.payload;
      delete message.payload;
      markClientTrace("client.frame_batch_received", {
        chunk_index: Number(message.chunk_index || 0),
        event_id: Number(message.event_id || 0),
        content_type: message.content_type || "",
        num_frames: Number(message.num_frames || 0),
        payload_bytes: payload?.byteLength || payload?.size || payload?.length || 0,
        frame_batch_gap_count: observeFrameBatchGap(message),
      });
      recordFrameBatchReceived(message, payload?.byteLength || payload?.size || payload?.length || 0);
      enqueueDecodeBatch(message, payload, epoch);
      schedulePrimaryPlaybackAck();
      if (!renderedPreviewFrames) setStatus("Receiving", "live");
      return;
    }
    if (message.type === "media_chunk_complete") {
      recordTrajectoryEvent("media_chunk_complete", {
        chunk_index: Number(message.chunk_index || 0),
        event_id: Number(message.event_id || 0),
        num_frames: Number(message.num_frames || 0),
      });
      return;
    }
    if (message.type === "frame_batch_header" || (!message.type && message.content_type)) {
      pendingHeader = message;
    } else {
      recordTrajectoryEvent("server_control_ignored", {
        type: String(message.type || "unknown"),
      });
      return;
    }
    if (pendingHeader && !renderedPreviewFrames) setStatus("Receiving", "live");
    return;
  }
  const header = pendingHeader;
  pendingHeader = null;
  header.__received_at = performance.now();
  markClientTrace("client.frame_batch_received", {
    chunk_index: Number(header.chunk_index || 0),
    event_id: Number(header.event_id || 0),
    content_type: header.content_type || "",
    num_frames: Number(header.num_frames || 0),
    payload_bytes: data.byteLength || data.size || data.length || 0,
    frame_batch_gap_count: observeFrameBatchGap(header),
  });
  recordFrameBatchReceived(header, data?.byteLength || data?.size || data?.length || 0);
  enqueueDecodeBatch(header, data, epoch);
  schedulePrimaryPlaybackAck();
}

async function decodeAndEnqueueFrameBatch(header, data, epoch) {
  const chunkFrameCount = Number(header.num_frames || 0);
  const payloadBytes = data.byteLength || data.size || 0;
  let decodedFrames;
  try {
    decodedFrames = await decodeFrameBatch(header, data);
    if (isEncodedPreviewContentType(header.content_type)) encodedDecodeErrors = 0;
  } catch (error) {
    if (!isEncodedPreviewContentType(header.content_type)) throw error;
    handleEncodedPreviewDecodeError(error, header, data, payloadBytes);
    return;
  }
  if (epoch !== streamEpoch) {
    for (const item of decodedFrames) item.image?.close?.();
    return;
  }
  markClientTrace("client.decode_batch_done", {
    chunk_index: Number(header.chunk_index || 0),
    event_id: Number(header.event_id || 0),
    content_type: header.content_type || "",
    num_frames: decodedFrames.length,
    payload_bytes: payloadBytes,
    decode_ms: roundTraceNumber(lastDecodeMs),
  });
  const now = performance.now();
  if (!renderedPreviewFrames && decodedFrames.length) {
    drawFrame(decodedFrames[0].image, { close: false, markRendered: false });
    recordChunkFirstRendered(decodedFrames[0].chunk, {
      initial_preview: true,
      display_lag_ms: now - (decodedFrames[0].receivedAt || now),
      decode_ms: decodedFrames[0].decodeMs || lastDecodeMs,
    });
  }
  // Gameplay recording is driven by presented-frame idle capture, not by
  // decoded network batches.
  const enqueueResult = playbackController.enqueueDecodedFrames(header, decodedFrames, now);
  closeFrames(enqueueResult.droppedFrames);
  lastSampledEventId = Number(header.event_id || lastSampledEventId);
  markModelEventApplied("minwm", lastSampledEventId);
  updateControlDebugText();
  frames += chunkFrameCount;
  bytes += payloadBytes;
  const networkNow = performance.now();
  if (primaryNetworkSample && networkNow > primaryNetworkSample.at) {
    const elapsedSeconds = (networkNow - primaryNetworkSample.at) / 1000;
    primaryProtocolStats = {
      ...primaryProtocolStats,
      receiveMbps: Math.max(
        0,
        (bytes - primaryNetworkSample.bytes) * 8 / elapsedSeconds / 1_000_000,
      ),
    };
  }
  primaryNetworkSample = { at: networkNow, bytes };
  updateOutputSizeFromHeader(header);
  setStatus("Live", "live");
  updateStats();
}

function recordFrameBatchReceived(header, payloadBytes) {
  const serverSentEpochMs = Number(header.server_sent_epoch_ms || 0);
  if (serverSentEpochMs > 0) {
    primaryProtocolStats = {
      ...primaryProtocolStats,
      lastDownlinkMs: Math.max(
        0,
        Date.now()
          - serverSentEpochMs
          + Number(primaryProtocolStats.serverClockOffsetMs || 0),
      ),
    };
  }
  recordTrajectoryEvent("frame_batch_received", {
    chunk_index: header.chunk_index,
    event_id: header.event_id,
    content_type: header.content_type,
    encoding: header.encoding,
    num_frames: header.num_frames,
    width: header.width,
    height: header.height,
    source_width: header.source_width,
    source_height: header.source_height,
    preview_width: header.preview_width,
    preview_height: header.preview_height,
    payload_bytes: payloadBytes,
    frame_batch_gap_count: frameBatchGapCount,
  });
}

function observeFrameBatchGap(header) {
  const chunkIndex = Number(header.chunk_index || 0);
  const frameBatchIndex = Number(header.frame_batch_index || 0);
  if (lastReceivedChunk === null) {
    frameBatchGapCount += Math.max(0, frameBatchIndex);
  } else if (chunkIndex === lastReceivedChunk) {
    const expected = Number(lastReceivedFrameBatchIndex || 0) + 1;
    if (frameBatchIndex > expected) frameBatchGapCount += frameBatchIndex - expected;
  } else if (chunkIndex > lastReceivedChunk) {
    frameBatchGapCount += Math.max(0, frameBatchIndex);
    if (chunkIndex > lastReceivedChunk + 1) {
      frameBatchGapCount += chunkIndex - lastReceivedChunk - 1;
    }
  }
  lastReceivedChunk = chunkIndex;
  lastReceivedFrameBatchIndex = frameBatchIndex;
  return frameBatchGapCount;
}

function recordChunkFirstRendered(chunkIndex, details = {}) {
  if (chunkIndex === undefined || chunkIndex === null) return;
  const key = String(chunkIndex);
  if (renderedTraceChunks.has(key)) return;
  renderedTraceChunks.add(key);
  const event = recordTrajectoryEvent("client.chunk_first_rendered", {
    chunk_index: chunkIndex,
    ...details,
  });
  markClientTrace("client.chunk_first_rendered", {
    chunk_index: Number(chunkIndex || 0),
    display_lag_ms: roundTraceNumber(details.display_lag_ms),
    decode_ms: roundTraceNumber(details.decode_ms),
  });
  if (event && currentSessionArtifact) {
    currentSessionArtifact.first_rendered_chunks.push(event);
    if (currentSessionArtifact.first_rendered_chunks.length > SESSION_ARTIFACT_EVENT_LIMIT) {
      currentSessionArtifact.first_rendered_chunks.splice(
        0,
        currentSessionArtifact.first_rendered_chunks.length - SESSION_ARTIFACT_EVENT_LIMIT,
      );
    }
  }
}

function sendEvent(kind, payload, historyText = null) {
  const delivery = dualModelController.sendEvent(kind, payload);
  const deliveredModels = Object.entries(delivery.sent)
    .filter(([, sent]) => sent)
    .map(([key]) => key);
  if (!deliveredModels.length) {
    addHistory(`${historyText || `${kind} event`} · no model socket open`);
    recordTrajectoryEvent(`${kind}_event_dropped`, {
      reason: "no model socket open",
      payload,
    });
    return null;
  }
  const eventId = delivery.eventId;
  const clientSentPerfMs = performance.now();
  markClientTrace("client.event_sent", {
    kind,
    event_id: eventId,
    delivered_models: deliveredModels,
    ws_buffered_amount: primaryTransportBufferedAmount(),
  });
  lastSentEventId = eventId;
  updateControlDebugText();
  if (kind === "prompt") {
    recordPromptHistory(payload, "prompt_update", eventId);
    recordTrajectoryEvent("prompt_update", { event_id: eventId, prompt: payload });
  } else if (kind === "camera_actions") {
    recordTrajectoryEvent("camera_actions_sent", {
      event_id: eventId,
      payload,
      delivered_models: deliveredModels,
      active_actions: controlStateController
        ? Array.from(controlStateController.activeActions).sort()
        : [],
    });
  } else {
    recordTrajectoryEvent(`${kind}_event_sent`, { event_id: eventId, payload });
  }
  if (kind === "camera_actions" || kind === "prompt") {
    if (delivery.sent.minwm) {
      playbackController.noteInputEvent(eventId, performance.now(), {
        cutoverMode: kind === "prompt"
          ? "prompt"
          : cameraActionHasActiveMotion(payload)
            ? "motion"
            : "settle",
      });
      updateStats();
    }
    setStatus("Updating", "live");
  }
  trackPendingModelEvent(delivery, kind);
  addHistory(
    `${historyText || `${kind} event sent`} · event#${eventId} · ${formatModelDelivery(delivery.sent)}`,
  );
  return eventId;
}

function modelLabel(key) {
  if (key === "lingbot2") return "LingBot2";
  if (key === "happyoyster") return "快乐生蚝";
  return "Zing";
}

function formatModelDelivery(sent = {}) {
  const entries = Object.entries(sent);
  if (!entries.length) return "no active model";
  return entries
    .map(([key, delivered]) => `${modelLabel(key)} ${delivered ? "sent" : "send failed"}`)
    .join(" · ");
}

function trackPendingModelEvent(delivery, kind) {
  if (kind !== "prompt" && kind !== "camera_actions") return;
  const pending = new Set(
    Object.entries(delivery.sent)
      .filter(([, sent]) => sent)
      .map(([key]) => key),
  );
  if (!pending.size) return;
  pendingModelEvents.set(delivery.eventId, { kind, pending });
}

function markModelEventApplied(key, eventId) {
  const appliedEventId = Number(eventId || 0);
  if (!appliedEventId) return;
  const root = document.querySelector(`[data-model-key="${key}"]`);
  if (root) root.dataset.lastAppliedEventId = String(appliedEventId);
  for (const [pendingEventId, event] of pendingModelEvents) {
    if (pendingEventId > appliedEventId || !event.pending.has(key)) continue;
    event.pending.delete(key);
    if (event.kind === "prompt") {
      const outcome = pendingEventId === appliedEventId ? "applied" : "superseded";
      addHistory(`${modelLabel(key)} prompt ${outcome} · event#${pendingEventId}`);
    }
    if (!event.pending.size) pendingModelEvents.delete(pendingEventId);
  }
}

function cameraActionHasActiveMotion(payload) {
  const transitions = payload?.transitions || [];
  const finalTransition = transitions[transitions.length - 1];
  return Array.isArray(finalTransition?.actions) && finalTransition.actions.length > 0;
}

function sendCameraControlTransitions(transitions) {
  if (!transitions.length) return null;
  const payload = {
    mode: "state",
    transitions: transitions.map((transition) => ({
      actions: transition.actions,
      client_ts_ms: transition.clientTsMs,
    })),
  };
  return sendEvent(
    "camera_actions",
    payload,
    describeCameraStateEvent(transitions),
  );
}

async function applyPreset(preset, options = {}) {
  const sendRuntimeEvents = options.sendRuntimeEvents
    ?? Boolean(ws && ws.readyState === WebSocket.OPEN);
  let preparedPresetRules = null;
  selectedPreset = preset;
  document.querySelectorAll(".preset").forEach((button) => {
    const selected = button.dataset.presetName === preset.name;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-pressed", selected ? "true" : "false");
  });
  $("prompt").value = preset.prompt;
  applyWorldRulesDraft(preset.rules || null);
  modelControl("minwm", "fps").value = UI_CONFIG.targetFps == null
    ? preset.fps
    : DEFAULT_TARGET_FPS;
  updateOutputSizeText();
  syncPlaybackTargetFps();
  await setPresetReference(preset);
  updateWorldDraftState();
  const presetRuleCount = worldRulesRuleCount(normalizedWorldRulesForStorage(preset.rules || {}));
  setWorldDraftStatus(
    `已填充「${preset.name}」的首帧、世界描述${presetRuleCount ? `和 ${presetRuleCount} 条规则` : ""}`,
    "ready",
  );
  if (sendRuntimeEvents) {
    preparedPresetRules = await prepareWorldRulesForEntry(preset.prompt);
    promptRewriteController.beginSession(preset.prompt);
    worldRulesController.activate(preparedPresetRules);
    const eventId = sendEvent("prompt", preset.prompt, `prompt update · ${preset.name}`);
    if (eventId) beginPromptLogSession(preset.prompt, "preset_runtime_update");
  }
  addHistory(`preset ${preset.name}`);
}

function describeCameraStateEvent(transitions) {
  const parts = transitions
    .map((transition) => describeControlActions(transition.actions))
    .join(" -> ");
  return `camera state · ${parts} · transitions=${transitions.length}`;
}

function describeControlActions(actions) {
  return actions.map((action) => describeControlAction(action)).join(" + ") || "No-op";
}

function describeControlAction(action, samples = 1) {
  const meta = CONTROL_ACTION_META[action];
  if (!meta) return `${action} (custom)`;
  const distance = describeControlDistance(meta.amount, samples);
  return `${meta.label} [${meta.type}, ${meta.axis}, ${distance}]`;
}

function describeControlDistance(amount, samples) {
  const match = /^([0-9.]+)(deg)?\/frame$/.exec(amount);
  if (!match) return amount;
  const perFrame = Number(match[1]);
  const unit = match[2] || "";
  const total = perFrame * Math.max(1, Number(samples || 1));
  return `${amount} x ${samples} frames = ${formatControlDistance(total, unit)}`;
}

function formatControlDistance(value, unit) {
  if (unit === "deg") return `${value.toFixed(0)}deg`;
  return value.toFixed(2);
}

function modelsUrlFromServerUrl(serverUrl) {
  const url = new URL(serverUrl, window.location.href);
  if (url.protocol === "ws:") url.protocol = "http:";
  if (url.protocol === "wss:") url.protocol = "https:";
  const backendPrefix = url.pathname.match(/^\/backends\/[^/]+/)?.[0] || "";
  url.pathname = `${backendPrefix}/v1/models`;
  url.search = "";
  url.hash = "";
  return url.toString();
}

function realtimeServerUrlFromLocation() {
  if (!window.location.host) return "";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/backends/minwm/v1/realtime_video/generate`;
}

function applyDefaultServerUrl() {
  const current = $("serverUrl").value.trim();
  const locationServerUrl = realtimeServerUrlFromLocation();
  if (!locationServerUrl) return;
  if (current.includes("127.0.0.1") || current.includes("localhost")) {
    $("serverUrl").value = locationServerUrl;
  }
}

function firstServedModelInfo(payload) {
  if (Array.isArray(payload?.data) && payload.data.length > 0) return payload.data[0];
  if (payload && typeof payload === "object") return payload;
  return null;
}

function servedModelId(info) {
  return String(info?.id || info?.model || info?.root || "");
}

function presetForModelInfo(info) {
  const id = servedModelId(info).toLowerCase();
  if (!id) return null;
  return presets.find((preset) => (
    preset.model && id.includes(preset.model.toLowerCase())
  )) || null;
}

async function queryServerModelInfo(options = {}) {
  const applyPresetForModel = options.applyPresetForModel ?? true;
  const preserveSize = options.preserveSize ?? false;
  let info;
  try {
    const response = await fetch(modelsUrlFromServerUrl($("serverUrl").value), {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`/v1/models ${response.status}`);
    info = firstServedModelInfo(await response.json());
  } catch (error) {
    addHistory(`model query failed · ${error.message || "unknown"}`);
    return null;
  }
  if (!info) return null;

  const modelId = servedModelId(info);
  const preset = presetForModelInfo(info);
  if (preset && applyPresetForModel && preset !== selectedPreset) {
    await applyPreset(preset, { sendRuntimeEvents: false, preserveSize });
  }
  if (modelId) $("model").value = modelId;
  addHistory(
    preset
      ? `server model · ${preset.name}`
      : `server model · ${modelId || "unknown"}`,
  );
  return info;
}

function compact(obj) {
  return Object.fromEntries(
    Object.entries(obj).filter(([, v]) => v !== undefined && v !== "" && v !== null)
  );
}

function readOptionalInteger(id) {
  const value = $(id).value;
  if (value === "") return undefined;
  return Number(value);
}

function readPreviewTransportParams(key = "minwm") {
  const outputFormat = modelControl(key, "transportFormat").value;
  const outputQuality = Number(
    modelControl(key, "transportQuality").value || DEFAULT_PREVIEW_OUTPUT_QUALITY,
  );
  if (!outputFormat) return {};
  const params = {
    realtime_output_format: outputFormat,
  };
  const baseSize = parseSizeValue(modelControl(key, "size").value);
  params.realtime_preview_max_width = previewMaxWidthForSize(baseSize);
  if (outputFormat === "webp" || outputFormat === "jpeg") {
    params.output_compression = outputQuality;
    if (
      modelControl(key, "superResolution").checked
      && modelControl(key, "frameInterpolation").checked
    ) {
      if (baseSize?.width) params.realtime_preview_max_width = baseSize.width;
    }
  }
  return params;
}

function tunePreviewQualityForPostprocess(key = "minwm") {
  if (modelControl(key, "transportFormat").value !== "webp") return;
  const currentQuality = Number(
    modelControl(key, "transportQuality").value || DEFAULT_PREVIEW_OUTPUT_QUALITY,
  );
  let qualityCap = MAX_WEBP_PREVIEW_OUTPUT_QUALITY;
  if (
    modelControl(key, "frameInterpolation").checked
    && modelControl(key, "superResolution").checked
  ) {
    qualityCap = HEAVY_PREVIEW_OUTPUT_QUALITY;
  } else if (modelControl(key, "frameInterpolation").checked) {
    qualityCap = SMOOTH_PREVIEW_OUTPUT_QUALITY;
  } else if (modelControl(key, "superResolution").checked) {
    qualityCap = SR_PREVIEW_OUTPUT_QUALITY;
  }
  if (currentQuality > qualityCap) {
    modelControl(key, "transportQuality").value = String(qualityCap);
  }
}

function readFrameInterpolationParams(key = "minwm") {
  if (!modelControl(key, "frameInterpolation").checked) return {};
  return {
    enable_frame_interpolation: true,
    frame_interpolation_exp: DEFAULT_FRAME_INTERPOLATION_EXP,
    frame_interpolation_scale: DEFAULT_FRAME_INTERPOLATION_SCALE,
  };
}

function readUpscalingScale(key = "minwm") {
  return Number(modelControl(key, "upscalingScale").value || DEFAULT_UPSCALING_SCALE);
}

function readSuperResolutionParams(key = "minwm") {
  if (!modelControl(key, "superResolution").checked) return {};
  const params = {
    enable_upscaling: true,
    upscaling_scale: readUpscalingScale(key),
  };
  const modelPath = modelControl(key, "upscalingModel").value;
  if (modelPath) params.upscaling_model_path = modelPath;
  return params;
}

function readModelRequestParams(key, { generationMode, firstFrame, numFrames } = {}) {
  const continuous = modelControl(key, "continuous").checked;
  const requestedFrames = numFrames ?? Number(modelControl(key, "numFrames").value);
  return compact({
    generation_mode: generationMode,
    prompt: $("prompt").value,
    size: modelControl(key, "size").value,
    fps: requestedInputFps(key),
    num_frames: generationMode === "t2v" && continuous ? undefined : requestedFrames,
    seed: Number(modelControl(key, "seed").value),
    num_inference_steps: Number(modelControl(key, "steps").value),
    guidance_scale: Number(modelControl(key, "guidance").value),
    realtime_causal_sink_size: readOptionalInteger(modelControlId(key, "sinkSize")),
    realtime_causal_kv_cache_num_frames: readOptionalInteger(modelControlId(key, "windowFrames")),
    max_chunks: generationMode === "t2v" || continuous ? undefined : 1,
    first_frame: firstFrame,
    ...readPreviewTransportParams(key),
    ...readFrameInterpolationParams(key),
    ...readSuperResolutionParams(key),
  });
}

function parseSizeValue(sizeText) {
  const match = /^(\d+)\s*x\s*(\d+)$/i.exec(String(sizeText || "").trim());
  if (!match) return null;
  return {
    width: Number(match[1]),
    height: Number(match[2]),
  };
}

function previewMaxWidthForSize(baseSize) {
  const baseWidth = Number(baseSize?.width || 0);
  if (!baseWidth) return DEFAULT_PREVIEW_MAX_WIDTH;
  return Math.max(
    DEFAULT_PREVIEW_MAX_WIDTH,
    Math.min(baseWidth, MAX_AUTO_PREVIEW_WIDTH),
  );
}

function updateOutputSizeText(width = null, height = null) {
  let outputWidth = Number(width || 0);
  let outputHeight = Number(height || 0);
  const srEnabled = $("superResolution").checked;
  const scale = srEnabled ? readUpscalingScale() : 1;
  if (!outputWidth || !outputHeight) {
    const base = parseSizeValue($("size").value);
    if (base) {
      outputWidth = base.width * scale;
      outputHeight = base.height * scale;
    }
  }
  $("outputSizeText").textContent = outputWidth && outputHeight
    ? `${outputWidth}x${outputHeight}${srEnabled ? ` · SR ${scale}x` : ""}`
    : "-";
}

function updateOutputSizeFromHeader(header) {
  const requestSize = parseSizeValue($("size").value);
  const frameWidth = Number(header.width || 0);
  const frameHeight = Number(header.height || 0);
  const sourceWidth = Number(header.source_width || requestSize?.width || frameWidth || 0);
  const sourceHeight = Number(header.source_height || requestSize?.height || frameHeight || 0);
  if (!sourceWidth || !sourceHeight) return;
  updateOutputSizeText(sourceWidth, sourceHeight);
  const previewWidth = Number(header.preview_width || 0) || (
    frameWidth && frameWidth !== sourceWidth ? frameWidth : 0
  );
  const previewHeight = Number(header.preview_height || 0) || (
    frameHeight && frameHeight !== sourceHeight ? frameHeight : 0
  );
  if (previewWidth && previewHeight) {
    $("outputSizeText").textContent += ` · preview ${previewWidth}x${previewHeight}`;
  }
}

function updateSuperResolutionControls(key = "minwm") {
  const disabled = !modelControl(key, "superResolution").checked;
  modelControl(key, "upscalingScale").disabled = disabled;
  modelControl(key, "upscalingModel").disabled = disabled;
  if (key === "minwm") updateOutputSizeText();
}

function setPreviewScale(value) {
  if (!previewFrame) return;
  const scale = Math.max(80, Math.min(170, Number(value || DEFAULT_PREVIEW_SCALE)));
  $("previewScale").value = String(scale);
  $("previewScaleText").textContent = `${scale}%`;
  if (previewScaleFrame) cancelAnimationFrame(previewScaleFrame);
  previewScaleFrame = requestAnimationFrame(() => {
    previewScaleFrame = 0;
    previewFrame.style.setProperty("--preview-scale", String(scale / 100));
  });
}

function selectedTransportLabel() {
  const select = $("transportFormat");
  return select.options[select.selectedIndex]?.textContent || "raw RGB";
}

function shortPayloadMode(contentType) {
  if (contentType === WEBP_FRAME_CONTENT_TYPE) return "webp";
  if (contentType === JPEG_FRAME_CONTENT_TYPE) return "jpeg";
  if (contentType === RAW_RGB_DELTA_GZIP_CONTENT_TYPE) return "delta-gzip";
  if (contentType === RAW_RGB_CONTENT_TYPE) return "raw RGB";
  return contentType;
}

function payloadModeLabelFromHeader(header) {
  if (header?.encoding) return header.encoding;
  const label = shortPayloadMode(header?.content_type || "");
  return label || selectedTransportLabel();
}

function formatBytes(value) {
  return `${(Number(value || 0) / 1048576).toFixed(1)} MB`;
}

function formatMs(value) {
  const ms = Number(value || 0);
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}

function renderPresets() {
  $("presetList").innerHTML = "";
  allWorldPresets().forEach((preset) => {
    const btn = document.createElement("button");
    btn.className = "preset";
    btn.type = "button";
    btn.dataset.presetName = preset.name;
    const selected = preset === selectedPreset
      || Boolean(preset.fingerprint && preset.fingerprint === selectedPreset?.fingerprint);
    btn.classList.toggle("is-selected", selected);
    btn.setAttribute("aria-pressed", selected ? "true" : "false");
    btn.dataset.tone = preset.tone;
    const thumb = document.createElement("img");
    thumb.className = "preset-thumb";
    thumb.src = preset.referenceUrl;
    thumb.alt = "";
    thumb.loading = "lazy";
    thumb.onerror = () => thumb.replaceWith(createPresetThumbFallback(preset));
    const title = document.createElement("b");
    title.textContent = ({
      "Dragon Ride": "山谷飞龙",
      "Misted Kingdom": "雾谷骑行",
      "Storm Crossing": "风暴航行",
      "Citadel Approach": "峡谷越野",
      "Spring Valley": "春日山谷",
      "Reef Patrol": "珊瑚巡游",
      "Alpine Run": "高山漂流",
      "Ice Kayak": "冰湖泛舟",
      "Penguin Colony": "企鹅冰原",
      "Mars Mountain": "火星远征",
      "Seaside Adventurer": "海岸冒险",
      "Roman Chariot": "罗马战车",
      "Asylum Corridor": "废墟走廊",
    })[preset.name] || preset.name;
    const meta = document.createElement("span");
    const presetRules = normalizedWorldRulesForStorage(preset.rules || {});
    const ruleCount = worldRulesRuleCount(presetRules);
    meta.textContent = preset.isCustom
      ? `已保存的自定义世界${ruleCount ? ` · ${ruleCount} 条规则` : ""}`
      : `填充首帧 + 世界描述${ruleCount ? ` + ${ruleCount} 条规则` : ""}`;
    btn.append(thumb, title, meta);
    btn.onclick = () => applyPreset(preset).catch(showError);
    $("presetList").appendChild(btn);
  });
}

function createPresetThumbFallback(preset) {
  const fallback = document.createElement("span");
  fallback.className = "preset-thumb preset-thumb-fallback";
  fallback.textContent = preset.name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0] || "")
    .join("")
    .toUpperCase();
  fallback.title = `${preset.name} reference image unavailable`;
  return fallback;
}

async function applyQueryParams() {
  const params = new URLSearchParams(window.location.search);
  const server = params.get("server");
  if (server) $("serverUrl").value = server;
  else applyDefaultServerUrl();
  const model = params.get("model");
  if (model) $("model").value = model;
  const generationMode = String(params.get("mode") || "").toLowerCase();
  if (ENABLED_GENERATION_MODES.includes(generationMode)) {
    $("generationMode").value = generationMode;
    updateGenerationModeUi();
  }
  const playbackParam = params.get("playback");
  const srParam = params.get("sr");
  const smoothParam = params.get("smooth");
  const catchupParam = Number(params.get("catchup"));
  for (const key of ["minwm", "lingbot2"]) {
    modelControl(key, "transportFormat").value = params.get("transport") || DEFAULT_PREVIEW_OUTPUT_FORMAT;
    modelControl(key, "transportQuality").value = params.get("quality") || String(DEFAULT_PREVIEW_OUTPUT_QUALITY);
    if (
      playbackParam === "live"
      || playbackParam === "timeline"
      || playbackParam === "adaptive"
      || playbackParam === "smooth_timeline"
    ) {
      modelControl(key, "playbackMode").value = playbackParam;
    }
    modelControl(key, "superResolution").checked = srParam === "1" || srParam === "true";
    modelControl(key, "frameInterpolation").checked = smoothParam === "1" || smoothParam === "true";
    modelControl(key, "upscalingScale").value = params.get("sr_scale") || String(DEFAULT_UPSCALING_SCALE);
    modelControl(key, "upscalingModel").value = params.get("sr_model") || DEFAULT_UPSCALING_MODEL;
    tunePreviewQualityForPostprocess(key);
    updateSuperResolutionControls(key);
  }
  $("smoothCatchupRate").value = String(
    Number.isFinite(catchupParam) && catchupParam > 0
      ? catchupParam
      : DEFAULT_SMOOTH_CATCHUP_RATE,
  );
  syncSmoothCatchupRate();
  syncZingFrameInterpolation({ fromTopbar: false });
  setPreviewScale(params.get("preview_scale") || params.get("zoom"));
  syncPlaybackTargetFps();
  syncPlaybackMode({ addToHistory: false });

  const presetKey = params.get("preset");
  let appliedPreset = false;
  if (presetKey) {
    const normalized = presetKey.toLowerCase();
    const preset = presets.find((item) => (
      item.name.toLowerCase() === normalized
      || item.name.toLowerCase().replaceAll(" ", "-") === normalized
    ));
    if (preset && preset !== selectedPreset) {
      await applyPreset(preset, { sendRuntimeEvents: false });
      appliedPreset = true;
    }
  }
  return {
    model: Boolean(model),
    preset: Boolean(presetKey && appliedPreset),
  };
}

function pack(value) {
  const rootValue = (
    value
    && typeof value === "object"
    && !Array.isArray(value)
    && !(value instanceof Uint8Array)
    && typeof value.type === "string"
    && value.version === undefined
  )
    ? { version: REALTIME_PROTOCOL_VERSION, ...value }
    : value;
  const out = [];
  const bytes = (arr) => {
    for (const item of arr) out.push(item);
  };
  const str = (s) => new TextEncoder().encode(s);
  const u16 = (n) => [(n >> 8) & 255, n & 255];
  const u32 = (n) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
  const write = (v) => {
    if (v === null) return out.push(0xc0);
    if (typeof v === "boolean") return out.push(v ? 0xc3 : 0xc2);
    if (typeof v === "number") {
      if (Number.isInteger(v) && v >= 0 && v < 128) return out.push(v);
      if (Number.isInteger(v) && v < 0 && v >= -32) return out.push(0xe0 | (v + 32));
      if (Number.isInteger(v) && v >= 0 && v < 256) return bytes([0xcc, v]);
      if (Number.isInteger(v) && v >= 0 && v < 65536) return bytes([0xcd, ...u16(v)]);
      const b = new ArrayBuffer(9), view = new DataView(b);
      view.setUint8(0, 0xcb); view.setFloat64(1, v);
      return bytes(new Uint8Array(b));
    }
    if (typeof v === "string") {
      const b = str(v), n = b.length;
      if (n < 32) bytes([0xa0 | n]); else if (n < 256) bytes([0xd9, n]); else bytes([0xda, ...u16(n)]);
      return bytes(b);
    }
    if (v instanceof Uint8Array) {
      if (v.length < 256) bytes([0xc4, v.length]); else if (v.length < 65536) bytes([0xc5, ...u16(v.length)]); else bytes([0xc6, ...u32(v.length)]);
      return bytes(v);
    }
    if (Array.isArray(v)) {
      v.length < 16 ? bytes([0x90 | v.length]) : bytes([0xdc, ...u16(v.length)]);
      return v.forEach(write);
    }
    const entries = Object.entries(v);
    entries.length < 16 ? bytes([0x80 | entries.length]) : bytes([0xde, ...u16(entries.length)]);
    entries.forEach(([k, val]) => { write(k); write(val); });
  };
  write(rootValue);
  return new Uint8Array(out);
}

function unpack(buf) {
  let i = 0;
  const text = new TextDecoder();
  const readU32 = () => (
    (buf[i++] * 16777216) + (buf[i++] << 16) + (buf[i++] << 8) + buf[i++]
  );
  const readI32 = () => {
    const value = readU32();
    return value > 0x7fffffff ? value - 0x100000000 : value;
  };
  const readU64 = () => {
    const hi = readU32();
    const lo = readU32();
    const value = BigInt(hi) * 4294967296n + BigInt(lo);
    return value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value.toString();
  };
  const read = () => {
    const b = buf[i++];
    if (b <= 0x7f) return b;
    if ((b & 0xe0) === 0xa0) return readStr(b & 0x1f);
    if ((b & 0xf0) === 0x80) return readMap(b & 0x0f);
    if ((b & 0xf0) === 0x90) return Array.from({ length: b & 0x0f }, read);
    if (b === 0xc0) return null;
    if (b === 0xc2 || b === 0xc3) return b === 0xc3;
    if (b === 0xcc) return buf[i++];
    if (b === 0xcd) return (buf[i++] << 8) | buf[i++];
    if (b === 0xce) return readU32();
    if (b === 0xcf) return readU64();
    if (b === 0xca) {
      const value = new DataView(buf.buffer, buf.byteOffset + i, 4).getFloat32(0);
      i += 4;
      return value;
    }
    if (b === 0xcb) {
      const value = new DataView(buf.buffer, buf.byteOffset + i, 8).getFloat64(0);
      i += 8;
      return value;
    }
    if (b === 0xc4) return readBin(buf[i++]);
    if (b === 0xc5) return readBin((buf[i++] << 8) | buf[i++]);
    if (b === 0xc6) return readBin(readU32());
    if (b === 0xd2) return readI32();
    if (b === 0xd3) {
      const hi = readI32();
      const lo = readU32();
      return hi * 4294967296 + lo;
    }
    if (b === 0xdc) return Array.from({ length: (buf[i++] << 8) | buf[i++] }, read);
    if (b === 0xdd) return Array.from({ length: readU32() }, read);
    if (b === 0xd9) return readStr(buf[i++]);
    if (b === 0xda) return readStr((buf[i++] << 8) | buf[i++]);
    if (b === 0xde) return readMap((buf[i++] << 8) | buf[i++]);
    throw new Error(`Unsupported msgpack byte ${b}`);
  };
  const readStr = (n) => text.decode(buf.slice(i, i += n));
  const readBin = (n) => buf.subarray(i, i += n);
  const readMap = (n) => {
    const obj = {};
    for (let j = 0; j < n; j++) obj[read()] = read();
    return obj;
  };
  return read();
}

applyRuntimeUiConfig();
syncModelSlotUi();
renderPresets();
void ensureCustomWorldPresetsLoaded();
drawIdle();
setPreviewScale(DEFAULT_PREVIEW_SCALE);
updateSuperResolutionControls("minwm");
updateSuperResolutionControls("lingbot2");
applyQueryParams()
  .then(async (query) => {
    if (!query.preset) clearWorldDraft();
    return query;
  })
  .then((query) => queryServerModelInfo({
    applyPresetForModel: false,
    preserveSize: true,
  }))
  .catch(showError);
scheduleRenderLoop();
renderTraceTopology();
updateRecordButton();
updateRecordFolderButton();
$("connectBtn").onclick = connect;
function closeForModelSlotChange() {
  if (!ws && dualModelController.activeKeys.size === 0 && !happyOysterSession.connected) return;
  closeSession("model comparison selection changed");
  setStatus("Selection changed");
}
for (let slotIndex = 0; slotIndex < MODEL_SLOT_DEFAULTS.length; slotIndex += 1) {
  $(`modelSlot${slotIndex}`).addEventListener("change", () => {
    closeForModelSlotChange();
    ensureUniqueModelSlot(slotIndex);
  });
}
$("addModelSlotBtn").onclick = () => {
  const sessionActive = Boolean(
    ws
    || dualModelController.activeKeys.size > 0
    || happyOysterSession.connected
  );
  activeModelSlotCount = 3;
  ensureUniqueModelSlot(2);
  if (!sessionActive || !modelSelected("happyoyster")) return;
  setModelConnectionState("happyoyster", "preparing");
  setHappyOysterStageText("正在创建快乐生蚝 World…", "preparing");
  addHistory("正在把快乐生蚝加入当前对比会话");
  void dualModelController.activate("happyoyster").catch((error) => {
    const message = error?.message || String(error || "连接失败");
    setModelConnectionState("happyoyster", "error");
    setHappyOysterStageText(message, "error");
  });
};
$("removeModelSlotBtn").onclick = () => {
  closeForModelSlotChange();
  activeModelSlotCount = 2;
  syncModelSlotUi();
};
$("clearWorldBtn").onclick = clearWorldDraft;
$("enhanceBtn").onclick = completeWorldDraft;
$("prompt").addEventListener("input", () => {
  updateWorldDraftState();
  handleWorldRulesDraftInput();
});
$("addSkillRuleBtn").onclick = () => {
  $("worldRulesPanel").open = true;
  addSkillRule();
  handleWorldRulesDraftInput();
};
$("addGoalRuleBtn").onclick = () => {
  $("worldRulesPanel").open = true;
  addGoalRule();
  handleWorldRulesDraftInput();
};
$("stopBtn").onclick = () => {
  closeSession();
  setModelConnectionState("minwm", "closed");
  setModelConnectionState("lingbot2", "closed");
  setModelConnectionState("happyoyster", "closed");
};

function setPromptRewriteStatus(message, state = "") {
  const status = $("promptRewriteStatus");
  status.textContent = message;
  if (state) status.dataset.state = state;
  else delete status.dataset.state;
}

async function rewriteRuntimePrompt(payload) {
  const response = await fetch("./api/prompt/rewrite", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let result = null;
  try {
    result = await response.json();
  } catch {
    result = null;
  }
  if (!response.ok) {
    throw new Error(result?.error || `prompt rewrite failed (${response.status})`);
  }
  return result;
}

async function completeWorldRule(payload) {
  const response = await fetch("./api/world-rule/complete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let result = null;
  try {
    result = await response.json();
  } catch {
    result = null;
  }
  if (!response.ok) {
    throw new Error(result?.error || `world rule completion failed (${response.status})`);
  }
  return result;
}

let runtimePromptRewritePending = false;

async function sendRuntimePromptUpdate() {
  const input = $("runtimePrompt");
  const prompt = input.value.trim();
  if (!prompt) {
    input.focus();
    return;
  }
  if (runtimePromptRewritePending) return;
  runtimePromptRewritePending = true;
  markRecordingPromptSubmitted(prompt);
  input.blur();
  canvas.focus({ preventScroll: true });
  $("sendPromptBtn").disabled = true;
  setPromptRewriteStatus("正在理解并改写指令…", "working");
  try {
    const result = await promptRewriteController.submit(prompt);
    if (result.ignored) return;
    input.value = "";
    updateRecordingPromptDraft("");
    if (result.change_type === "one_time") {
      setPromptRewriteStatus("已发送 · 一次性指令，10 秒后恢复持久状态", "one_time");
    } else {
      setPromptRewriteStatus("已发送 · 持久指令", "persistent");
    }
  } catch (error) {
    markRecordingPromptFailed(error.message || error);
    setPromptRewriteStatus(error.message || "指令改写失败，请重试", "error");
    addHistory(`prompt rewrite failed · ${error.message || error}`);
    input.focus({ preventScroll: true });
  } finally {
    runtimePromptRewritePending = false;
    $("sendPromptBtn").disabled = false;
  }
}

$("sendPromptBtn").onclick = sendRuntimePromptUpdate;
$("runtimePrompt").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
  event.preventDefault();
  sendRuntimePromptUpdate();
});
$("runtimePrompt").addEventListener("input", (event) => {
  updateRecordingPromptDraft(event.currentTarget.value);
});

function setupVoicePromptInput() {
  const button = $("voicePromptBtn");
  const status = $("voicePromptStatus");
  const input = $("runtimePrompt");
  const secureBaseUrl = String(UI_CONFIG.secureBaseUrl || "").trim();
  if (!window.isSecureContext) {
    if (secureBaseUrl) {
      status.textContent = "切换 HTTPS";
      button.title = "点击切换到 HTTPS 后使用语音输入";
      button.onclick = () => {
        try {
          const secureUrl = new URL(
            `${window.location.pathname}${window.location.search}${window.location.hash}`,
            secureBaseUrl,
          );
          if (secureUrl.protocol !== "https:") throw new Error("secureBaseUrl must use HTTPS");
          window.location.assign(secureUrl);
        } catch (error) {
          status.textContent = "需要 HTTPS";
          button.disabled = true;
        }
      };
    } else {
      button.disabled = true;
      button.title = "语音输入需要 HTTPS 安全连接";
      status.textContent = "需要 HTTPS";
    }
    return;
  }
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    button.disabled = true;
    button.title = "当前浏览器不支持语音输入";
    status.textContent = "暂不支持";
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "zh-CN";
  recognition.continuous = false;
  recognition.interimResults = true;
  let listening = false;
  let prefix = "";
  let idleStatus = "点击说话";

  const speechErrorLabels = {
    "not-allowed": "麦克风未授权",
    "service-not-allowed": "语音服务未授权",
    "audio-capture": "未检测到麦克风",
    network: "语音服务网络异常",
    "no-speech": "未听清，请重试",
  };

  function focusInputAtEnd() {
    input.focus({ preventScroll: true });
    const end = input.value.length;
    input.setSelectionRange(end, end);
  }

  function setListening(next) {
    listening = next;
    button.classList.toggle("is-listening", next);
    button.setAttribute("aria-pressed", next ? "true" : "false");
    status.textContent = next ? "正在聆听" : idleStatus;
  }

  recognition.onstart = () => {
    prefix = input.value.trim();
    idleStatus = "点击说话";
    setListening(true);
  };
  recognition.onresult = (event) => {
    let transcript = "";
    for (let index = 0; index < event.results.length; index += 1) {
      transcript += event.results[index][0]?.transcript || "";
    }
    const spacer = prefix && transcript ? " " : "";
    input.value = `${prefix}${spacer}${transcript}`.trimStart();
    if (document.activeElement === input) {
      const end = input.value.length;
      input.setSelectionRange(end, end);
    }
    input.dispatchEvent(new Event("input", { bubbles: true }));
    if (transcript) idleStatus = "已识别";
  };
  recognition.onerror = (event) => {
    idleStatus = event.error === "aborted"
      ? "点击说话"
      : speechErrorLabels[event.error] || "语音识别失败";
    setListening(false);
  };
  recognition.onend = () => {
    setListening(false);
  };
  button.addEventListener("pointerdown", (event) => {
    // Keep the textarea focused while pressing the microphone. Native buttons
    // otherwise take focus before the speech recognizer starts.
    event.preventDefault();
    focusInputAtEnd();
  });
  button.onclick = () => {
    focusInputAtEnd();
    try {
      if (listening) recognition.stop();
      else {
        idleStatus = "正在启动";
        status.textContent = idleStatus;
        recognition.start();
      }
    } catch (error) {
      idleStatus = "请重试";
      setListening(false);
    }
  };
}

setupVoicePromptInput();
setupFirstFrameDropZone();
$("recordBtn").onclick = async () => {
  if (recordingActive) {
    await stopRecording({ reason: "manual" });
  } else {
    const sessionLive = worldExperienceReady
      && sessionCountdownDeadlineMs > Date.now()
      && !sessionLifetimeExpired;
    if (!sessionLive) {
      showSessionNotice("请先进入世界，再开始游玩录像");
      return;
    }
    startRecording({ source: "manual" });
  }
};
$("recordDownloadBtn").onclick = downloadGameplayRecordings;
$("recordFolderBtn").onclick = () => {
  chooseRecordingDirectory().catch((error) => {
    addHistory(error.message || "record folder selection failed");
  });
};
$("firstFrame").onchange = () => {
  clearSelectedWorldPreset();
  drawReferencePreview($("firstFrame").files[0]);
};
$("generationMode").addEventListener("change", updateGenerationModeUi);
$("continuous").addEventListener("change", updateGenerationModeUi);
$("numFrames").addEventListener("input", updateT2VFrameHint);
for (const key of ["minwm", "lingbot2"]) {
  modelControl(key, "size").addEventListener("input", () => {
    if (key === "minwm") updateOutputSizeText();
  });
  modelControl(key, "fps").addEventListener("input", () => {
    syncPlaybackTargetFps();
    if (key === "minwm") updateT2VFrameHint();
  });
  modelControl(key, "playbackMode").addEventListener("change", () => syncPlaybackMode());
  modelControl(key, "superResolution").addEventListener("change", () => {
    updateSuperResolutionControls(key);
    tunePreviewQualityForPostprocess(key);
  });
  modelControl(key, "upscalingScale").addEventListener("change", () => {
    if (key === "minwm") updateOutputSizeText();
  });
  modelControl(key, "frameInterpolation").addEventListener("change", () => {
    if (key === "minwm") syncZingFrameInterpolation({ fromTopbar: false });
    tunePreviewQualityForPostprocess(key);
    syncPlaybackTargetFps();
  });
}
$("previewScale").addEventListener("input", () => setPreviewScale($("previewScale").value));
$("smoothCatchupRate").addEventListener("input", syncSmoothCatchupRate);
$("zingFrameInterpolation").addEventListener("change", () => {
  syncZingFrameInterpolation({ fromTopbar: true });
});
canvas.addEventListener("pointerdown", () => canvas.focus({ preventScroll: true }));
lingbot2Canvas.addEventListener("pointerdown", () => canvas.focus({ preventScroll: true }));
$("serverUrl").addEventListener("change", () => {
  queryServerModelInfo({ applyPresetForModel: true }).catch(showError);
});
document.querySelectorAll("[data-workspace-view]").forEach((button) => {
  button.addEventListener("click", () => setWorkspaceView(button.dataset.workspaceView));
});
document.querySelectorAll("button").forEach((btn) => {
  btn.addEventListener("pointerdown", () => btn.classList.add("is-pressed"));
  ["pointerup", "pointercancel", "pointerleave", "blur"].forEach((eventName) => {
    btn.addEventListener(eventName, () => btn.classList.remove("is-pressed"));
  });
});
document.querySelectorAll("[data-action]").forEach((btn) => {
  const action = btn.dataset.action;
  btn.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    if (controlStateController.setAction(action, true)) {
      recordTrajectoryEvent("control_button_down", { action });
    }
  });
  ["pointerup", "pointercancel", "pointerleave", "blur"].forEach((eventName) => {
    btn.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (controlStateController.setAction(action, false)) {
        recordTrajectoryEvent("control_button_up", { action, event: eventName });
      }
    });
  });
});

function isTypingTarget(target) {
  return target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function keyboardAction(event) {
  return CONTROL_KEY_ACTIONS.get(event.key.toLowerCase()) || null;
}

function keyboardSkill(event) {
  if (event.altKey || event.ctrlKey || event.metaKey) return null;
  if (!/^[1-9]$/.test(event.key)) return null;
  return worldRulesController.snapshot().skills[Number(event.key) - 1] || null;
}

function setControlButtonActive(action, active) {
  document.querySelectorAll(`[data-action="${action}"]`).forEach((btn) => {
    btn.classList.toggle("is-key-active", active);
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

class ControlStateController {
  constructor() {
    this.activeActions = new Set();
    this.pendingTransitions = [];
    this.flushTimer = 0;
    this.stateHeartbeatTimer = 0;
  }

  reset({ sendRelease = false } = {}) {
    const hadActions = this.activeActions.size > 0;
    this.activeActions.clear();
    this.pendingTransitions = [];
    this.clearFlushTimer();
    this.clearStateHeartbeatTimer();
    this.updateButtons();
    if (sendRelease && hadActions) {
      this.enqueueTransition();
    }
  }

  setAction(action, active) {
    const hadAction = this.activeActions.has(action);
    if (active === hadAction) return false;
    if (active) {
      this.activeActions.add(action);
      if (recordingActive) {
        recordingActionPulseUntil.set(action, performance.now() + 120);
      }
    } else {
      this.activeActions.delete(action);
    }
    this.updateButtons();
    // Send presses immediately so a running model has the largest possible
    // chance of sampling the held state at its next chunk boundary. Releases
    // keep the short batching window to compact rapid key/chord changes.
    this.enqueueTransition({ immediate: active });
    if (this.activeActions.size) this.scheduleStateHeartbeat();
    else this.clearStateHeartbeatTimer();
    return true;
  }

  releaseAll() {
    this.reset({ sendRelease: true });
  }

  enqueueTransition({ immediate = false } = {}) {
    const actions = Array.from(this.activeActions).sort();
    const last = this.pendingTransitions[this.pendingTransitions.length - 1];
    if (last && this.sameActions(last.actions, actions)) return;
    this.pendingTransitions.push({
      actions,
      clientTsMs: Math.round(performance.now()),
    });
    this.compactPendingIfNeeded();
    if (immediate) this.flush();
    else this.scheduleFlush();
  }

  scheduleFlush() {
    if (this.flushTimer) return;
    this.flushTimer = window.setTimeout(() => {
      this.flushTimer = 0;
      this.flush();
    }, CONTROL_TRANSITION_FLUSH_DELAY_MS);
  }

  scheduleStateHeartbeat() {
    if (this.stateHeartbeatTimer || !this.activeActions.size) return;
    this.stateHeartbeatTimer = window.setTimeout(() => {
      this.stateHeartbeatTimer = 0;
      if (!this.activeActions.size) return;
      sendCameraControlTransitions([{
        actions: Array.from(this.activeActions).sort(),
        clientTsMs: Math.round(performance.now()),
      }]);
      this.scheduleStateHeartbeat();
    }, CONTROL_HELD_STATE_HEARTBEAT_MS);
  }

  flush() {
    this.clearFlushTimer();
    if (!this.pendingTransitions.length) return;
    const transitions = this.pendingTransitions;
    this.pendingTransitions = [];
    sendCameraControlTransitions(transitions);
  }

  compactPendingIfNeeded() {
    if (this.pendingTransitions.length <= 8) return;
    this.compactPendingToLatestPulse();
  }

  compactPendingToLatestPulse() {
    const final = this.pendingTransitions[this.pendingTransitions.length - 1];
    const latestPulse = [...this.pendingTransitions]
      .reverse()
      .find((transition) => transition.actions.length > 0);
    if (latestPulse && !this.sameActions(latestPulse.actions, final.actions)) {
      this.pendingTransitions = [latestPulse, final];
    } else {
      this.pendingTransitions = [final];
    }
  }

  updateButtons() {
    CONTROL_ACTION_META_KEYS.forEach((action) => {
      setControlButtonActive(action, this.activeActions.has(action));
    });
    updateControlDebugText();
  }

  sameActions(left, right) {
    return left.length === right.length && left.every((item, idx) => item === right[idx]);
  }

  clearFlushTimer() {
    if (!this.flushTimer) return;
    window.clearTimeout(this.flushTimer);
    this.flushTimer = 0;
  }

  clearStateHeartbeatTimer() {
    if (!this.stateHeartbeatTimer) return;
    window.clearTimeout(this.stateHeartbeatTimer);
    this.stateHeartbeatTimer = 0;
  }
}

const CONTROL_ACTION_META_KEYS = Object.keys(CONTROL_ACTION_META);
controlStateController = new ControlStateController();
updateControlDebugText();
window.setInterval(() => {
  dualModelController.sendEvent("heartbeat", {
    active_actions: Array.from(controlStateController.activeActions).sort(),
  });
}, SESSION_HEARTBEAT_MS);

document.addEventListener("keydown", (event) => {
  if (isTypingTarget(event.target)) return;
  const skill = keyboardSkill(event);
  if (skill) {
    event.preventDefault();
    if (!event.repeat) void triggerWorldSkill(skill.id);
    return;
  }
  const action = keyboardAction(event);
  if (!action) return;
  event.preventDefault();
  if (event.repeat) {
    recordTrajectoryEvent("key_repeat_ignored", {
      key: event.key,
      code: event.code,
      action,
    });
    return;
  }
  if (controlStateController.setAction(action, true)) {
    recordTrajectoryEvent("key_down", {
      key: event.key,
      code: event.code,
      action,
    });
  }
});

document.addEventListener("keyup", (event) => {
  if (isTypingTarget(event.target)) return;
  const action = keyboardAction(event);
  if (!action) return;
  event.preventDefault();
  if (controlStateController.setAction(action, false)) {
    recordTrajectoryEvent("key_up", {
      key: event.key,
      code: event.code,
      action,
    });
  }
});

window.addEventListener("blur", () => {
  controlStateController.releaseAll();
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    controlStateController.releaseAll();
  }
});

window.__sglangRealtimeDebug = () => ({
  activeActions: controlStateController
    ? Array.from(controlStateController.activeActions).sort()
    : [],
  bytes,
  decodeInProgress,
  queuedDecodeBytes,
  queuedDecodeFrames,
  decodeQueueLength: decodeQueue.length,
  droppedDecodeFrames,
  frames,
  lastDecodeMs,
  lastDisplayLagMs,
  lastSampledEventId,
  lastSentEventId,
  pendingDecodeBatches,
  pendingHeader: Boolean(pendingHeader),
  playback: playbackController.snapshot(),
  renderedFps: fpsSamples.length,
  renderedPreviewFrames,
  renderLoopFps: renderLoopSamples.length,
  recordingArtifact: recordingArtifact ? {
    events: recordingArtifact.events.length,
    firstRenderedChunks: recordingArtifact.first_rendered_chunks.length,
    promptHistory: recordingArtifact.prompt_history.length,
    traceId: recordingArtifact.trace_id,
  } : null,
  currentSessionArtifact: currentSessionArtifact ? {
    events: currentSessionArtifact.events.length,
    firstRenderedChunks: currentSessionArtifact.first_rendered_chunks.length,
    promptHistory: currentSessionArtifact.prompt_history.length,
    traceId: currentSessionArtifact.trace_id,
  } : null,
  socketBufferedAmount: ws ? ws.bufferedAmount : 0,
  socketCloseExpected,
  socketHadError,
  socketReadyState: ws ? ws.readyState : null,
  socketServerError,
  status: $("statusText").textContent,
  streamEpoch,
  visibilityState: document.visibilityState,
});
