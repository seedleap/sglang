#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUAL_GATEWAY_HTTP="${REALTIME_DUAL_GATEWAY_HTTP:-http://k8s-minwmrea-zingling-081cdcf247-5c21747e962f7bd4.elb.us-east-2.amazonaws.com}"

if [[ "${DUAL_GATEWAY_HTTP}" == https://* ]]; then
  DEFAULT_GATEWAY_WS="wss://${DUAL_GATEWAY_HTTP#https://}"
else
  DEFAULT_GATEWAY_WS="ws://${DUAL_GATEWAY_HTTP#http://}"
fi
DUAL_GATEWAY_WS="${REALTIME_DUAL_GATEWAY_WS:-${DEFAULT_GATEWAY_WS}}"

export WEBUI_PORT="${WEBUI_PORT:-18083}"
export REALTIME_UPSTREAM_HTTP="${DUAL_GATEWAY_HTTP%/}"
export REALTIME_UPSTREAM_WS="${DUAL_GATEWAY_WS%/}"
export MINWM_UPSTREAM_HTTP="${DUAL_GATEWAY_HTTP%/}/backends/minwm"
export MINWM_UPSTREAM_WS="${DUAL_GATEWAY_WS%/}/backends/minwm"
export LINGBOT2_UPSTREAM_HTTP="${DUAL_GATEWAY_HTTP%/}/backends/lingbot2"
export LINGBOT2_UPSTREAM_WS="${DUAL_GATEWAY_WS%/}/backends/lingbot2"
if [[ -z "${REALTIME_UI_CONFIG_JSON:-}" ]]; then
  export REALTIME_UI_CONFIG_JSON="{\"generationModes\":[\"i2v\",\"t2v\"],\"defaultGenerationMode\":\"i2v\",\"t2vFrameStep\":4,\"t2vDefaultNumFrames\":121,\"singleExperience\":true,\"singleExperienceUserIds\":{\"minwm\":\"showcase:zing\",\"lingbot2\":\"showcase:lingbot2\"},\"smoothCatchupRateMax\":1.1,\"dualModels\":{\"minwm\":{\"label\":\"Zing\",\"wsUrl\":\"${DUAL_GATEWAY_WS%/}/backends/minwm/v1/realtime_video/generate\"},\"lingbot2\":{\"label\":\"LingBot2\",\"wsUrl\":\"${DUAL_GATEWAY_WS%/}/backends/lingbot2/v1/realtime_video/generate\",\"targetFps\":16,\"sinkSize\":9,\"windowFrames\":18}}}"
fi

exec python3 "${SCRIPT_DIR}/server.py"
