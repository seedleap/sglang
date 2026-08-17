# MinWM 与 LingBot2 双模型实时对比 WebUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留现有 Realtime Studio 全部输入和控制能力的前提下，将单播放器改成左 MinWM、右 LingBot2 的双模型实时对比界面，并在一台 8×H100 Spot 节点上以两个模型各 `2 replicas × SP2` 完成 720p 端到端验证与并发压测。

**Architecture:** 浏览器使用一个 `DualRealtimeController` 管理共享表单、事件编号和按键状态，并为 MinWM、LingBot2 各创建一个独立 `RealtimeModelSession`。两个模型通过同源 comparison gateway 的不同 WebSocket 前缀访问各自服务，媒体解码与播放完全独立。Kubernetes 使用四个 2-GPU Pod 填满当前 H100 节点，MinWM 继续连接现有 L4 异步 TAEHV。

**Tech Stack:** Vanilla JavaScript、Canvas、Web Workers、aiohttp、pytest、Node.js assertion tests、Kubernetes/Kustomize、AWS EKS/ECR/S3、Playwright、Python WebSocket load test。

---

## File Structure

### SGLang worktree

- Create `python/sglang/multimodal_gen/apps/realtime_webui/model_session.js`: one model's socket, decode queue, playback controller, canvas and stats.
- Create `python/sglang/multimodal_gen/apps/realtime_webui/dual_model_controller.js`: shared event IDs, fan-out, lifecycle aggregation and payload cloning.
- Modify `python/sglang/multimodal_gen/apps/realtime_webui/index.html`: two labeled player surfaces while retaining one controls/presets section.
- Modify `python/sglang/multimodal_gen/apps/realtime_webui/styles.css`: responsive two-column player grid and isolated status rows.
- Modify `python/sglang/multimodal_gen/apps/realtime_webui/app.js`: build one request snapshot, delegate media sessions, broadcast input, retain recording/trace/preset behavior.
- Modify `python/sglang/multimodal_gen/apps/realtime_webui/server.py`: route two prefixed HTTP/WebSocket backends and strip the prefix before proxying.
- Create `python/sglang/multimodal_gen/apps/realtime_webui/dual_model_contract_test.js`: DOM and shared-controller source contract tests.
- Create `python/sglang/multimodal_gen/apps/realtime_webui/dual_model_controller_test.js`: executable fake-WebSocket lifecycle and event fan-out tests.
- Modify `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_webui.py`: aiohttp proxy and runtime-config contracts.
- Create `benchmark/minwm_realtime_async_vae/dual_model_load_test.py`: single-model and comparison load profiles.
- Create `benchmark/minwm_realtime_async_vae/test_dual_model_load_test.py`: pairing, percentile and stop-condition tests.
- Create `benchmark/minwm_realtime_async_vae/dual_model_report.py`: deterministic JSON/Markdown benchmark report generation.

### Deployment worktree

- Create `deploy/k8s/overlays/minwm-lingbot2-dual/kustomization.yaml`: comparison profile entrypoint.
- Create `deploy/k8s/overlays/minwm-lingbot2-dual/workloads.yaml`: two MinWM SP2 Pods, two LingBot2 SP2 Pods, Services and comparison gateway.
- Create `deploy/k8s/overlays/minwm-lingbot2-dual/runtime-config.json`: browser model labels/endpoints without exposing hardware labels.
- Create `deploy/k8s/overlays/minwm-lingbot2-dual/test_profile.py`: immutable profile, GPU split and checkpoint identity assertions.
- Create `docs/minwm-lingbot2-dual-h100.md`: exact image digest, artifact identity, rollout/rollback, E2E and benchmark commands.

### Result artifacts

- Create `benchmark/minwm_realtime_async_vae/results/$RUN_ID-dual-h100-720p/raw/*.json`: raw runs, where `RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)` is created once before the benchmark.
- Create `benchmark/minwm_realtime_async_vae/results/$RUN_ID-dual-h100-720p/report.zh-CN.md`: final Chinese report.
- Create `benchmark/minwm_realtime_async_vae/results/$RUN_ID-dual-h100-720p/browser/*.png`: desktop/mobile and running comparison screenshots.
- Create `benchmark/minwm_realtime_async_vae/results/$RUN_ID-dual-h100-720p/browser/comparison-30s.webm`: comparison recording.

---

### Task 1: Establish Clean Baseline

**Files:**
- Verify: `python/sglang/multimodal_gen/apps/realtime_webui/*_test.js`
- Verify: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_webui.py`
- Verify: `benchmark/minwm_realtime_async_vae/test_load_test.py`

- [ ] **Step 1: Verify branch and clean worktree**

Run:

```bash
git status --short --branch
git rev-parse HEAD
```

Expected: branch `codex/minwm-lingbot2-dual-webui`, only committed design/plan changes, base descendant of `origin/main@2b801149ed`.

- [ ] **Step 2: Run existing browser source and playback tests**

Run:

```bash
for test in python/sglang/multimodal_gen/apps/realtime_webui/*_test.js; do node "$test"; done
```

Expected: every script exits 0.

- [ ] **Step 3: Run existing Python unit tests**

Run:

```bash
python3 -m pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_realtime_webui.py \
  benchmark/minwm_realtime_async_vae/test_load_test.py
```

Expected: all tests pass before feature edits.

---

### Task 2: Define Dual-Model DOM and Runtime Contract

**Files:**
- Create: `python/sglang/multimodal_gen/apps/realtime_webui/dual_model_contract_test.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/index.html`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/styles.css`

- [ ] **Step 1: Write the failing DOM contract test**

The test must assert:

```javascript
assert.equal((html.match(/class="model-player"/g) || []).length, 2);
assert.match(html, /id="minwmViewport"/);
assert.match(html, /id="lingbot2Viewport"/);
assert.match(html, /data-model-label="MinWM"/);
assert.match(html, /data-model-label="LingBot2"/);
assert.equal((html.match(/id="connectBtn"/g) || []).length, 1);
assert.equal((html.match(/class="stage-controls"/g) || []).length, 1);
assert.doesNotMatch(html, /SP2|CUDA Graph|4 GPU profile/);
assert.match(css, /\.model-player-grid/);
assert.match(css, /grid-template-columns:\s*repeat\(2,/);
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
node python/sglang/multimodal_gen/apps/realtime_webui/dual_model_contract_test.js
```

Expected: FAIL because dual player elements do not exist.

- [ ] **Step 3: Implement the minimal dual-player markup**

Use one stage topbar and one stage controls block. Inside the preview region create:

```html
<div class="model-player-grid">
  <article class="model-player" data-model-label="MinWM">
    <header><strong>MinWM</strong><span id="minwmStatusText">Idle</span></header>
    <div class="model-player-frame">
      <canvas id="minwmViewport" width="1280" height="704" tabindex="0"></canvas>
      <div id="minwmPreviewOverlay" class="preview-overlay" aria-hidden="true"></div>
    </div>
    <div id="minwmTelemetry" class="model-player-telemetry"></div>
  </article>
  <article class="model-player" data-model-label="LingBot2">
    <header><strong>LingBot2</strong><span id="lingbot2StatusText">Idle</span></header>
    <div class="model-player-frame">
      <canvas id="lingbot2Viewport" width="1280" height="704" tabindex="0"></canvas>
      <div id="lingbot2PreviewOverlay" class="preview-overlay" aria-hidden="true"></div>
    </div>
    <div id="lingbot2Telemetry" class="model-player-telemetry"></div>
  </article>
</div>
```

Keep all existing reference/preset/parameter controls exactly once. Add a mobile media query that changes the grid to one column.

- [ ] **Step 4: Run DOM and existing UI contract tests**

Run:

```bash
node python/sglang/multimodal_gen/apps/realtime_webui/dual_model_contract_test.js
python3 -m pytest -q python/sglang/multimodal_gen/test/unit/realtime/test_realtime_webui.py
```

Expected: PASS after updating intentional old single-canvas assertions.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/multimodal_gen/apps/realtime_webui/index.html \
  python/sglang/multimodal_gen/apps/realtime_webui/styles.css \
  python/sglang/multimodal_gen/apps/realtime_webui/dual_model_contract_test.js \
  python/sglang/multimodal_gen/test/unit/realtime/test_realtime_webui.py
git commit -m "feat: add dual realtime player layout"
```

---

### Task 3: Implement Independent Model Sessions and Shared Event Fan-Out

**Files:**
- Create: `python/sglang/multimodal_gen/apps/realtime_webui/model_session.js`
- Create: `python/sglang/multimodal_gen/apps/realtime_webui/dual_model_controller.js`
- Create: `python/sglang/multimodal_gen/apps/realtime_webui/dual_model_controller_test.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/index.html`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/app.js`

- [ ] **Step 1: Write failing executable controller tests**

Use fake sessions with `connect`, `sendEvent`, and `close` call logs. Verify this exact API:

```javascript
const controller = new DualRealtimeController({
  sessions: [minwm, lingbot2],
  eventIdStart: 1,
});
await controller.connectAll(baseInit);
const eventId = controller.broadcastEvent("camera_actions", payload);
assert.equal(eventId, 1);
assert.deepEqual(minwm.events[0], lingbot2.events[0]);
assert.equal(minwm.events[0].event_id, 1);
await controller.closeAll("test stop");
assert.equal(minwm.closeCalls, 1);
assert.equal(lingbot2.closeCalls, 1);
```

Add isolation tests where MinWM connect/send fails but LingBot2 remains live, and where `closeAll` is invoked twice.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
node python/sglang/multimodal_gen/apps/realtime_webui/dual_model_controller_test.js
```

Expected: FAIL because controllers are not defined.

- [ ] **Step 3: Implement `DualRealtimeController`**

Required behavior:

```javascript
class DualRealtimeController {
  constructor({ sessions, eventIdStart = 1 }) {
    this.sessions = sessions;
    this.nextEventId = eventIdStart;
  }
  async connectAll(baseInit) {
    return Promise.allSettled(this.sessions.map((session) => session.connect(baseInit)));
  }
  broadcastEvent(kind, payload, metadata = {}) {
    const event_id = this.nextEventId++;
    const event = { type: "event", kind, payload, event_id, ...metadata };
    for (const session of this.sessions) session.sendEvent(event);
    return event_id;
  }
  async closeAll(reason) {
    await Promise.allSettled(this.sessions.map((session) => session.close(reason)));
  }
}
```

Export to both `globalThis` and CommonJS so browser and Node tests use the same implementation.

- [ ] **Step 4: Implement `RealtimeModelSession`**

Constructor dependencies must be explicit: model key/label, endpoint resolver, canvas, status elements, decoder worker URL, playback-controller factory, codec pack/unpack and callbacks. Session-owned mutable state includes socket, epoch, pending header, decoder worker, decode queue, playback controller, frame/byte counts and model-specific sampled event ID. No media state may be shared between instances.

- [ ] **Step 5: Delegate `app.js` lifecycle**

Build the base init payload once, create one `comparison_id`, and connect sessions in parallel. Replace direct `ws.send` in Action/Prompt/Heartbeat paths with `dualController.broadcastEvent`. Keep trajectory recording once, but include per-model `last_sampled_event_id` snapshots. Stop must call `closeAll` and be idempotent.

- [ ] **Step 6: Run controller and regression tests**

Run:

```bash
node python/sglang/multimodal_gen/apps/realtime_webui/dual_model_controller_test.js
for test in python/sglang/multimodal_gen/apps/realtime_webui/*_test.js; do node "$test"; done
python3 -m pytest -q python/sglang/multimodal_gen/test/unit/realtime/test_realtime_webui.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/multimodal_gen/apps/realtime_webui
git commit -m "feat: fan out realtime controls to two model sessions"
```

---

### Task 4: Add Same-Origin Dual Backend Proxy

**Files:**
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/server.py`
- Modify: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_webui.py`

- [ ] **Step 1: Write failing proxy selection tests**

Load `server.py` with env vars and verify:

```python
assert resolve_backend("minwm").http == "http://minwm-gateway:8080"
assert resolve_backend("lingbot2").ws == "ws://lingbot2:30000"
with pytest.raises(web.HTTPNotFound):
    resolve_backend("unknown")
assert strip_backend_prefix("/backends/minwm/v1/models", "minwm") == "/v1/models"
```

Also assert the router exposes WebSocket and HTTP wildcard routes under `/backends/{backend}`.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python3 -m pytest -q python/sglang/multimodal_gen/test/unit/realtime/test_realtime_webui.py
```

Expected: new dual-backend tests fail.

- [ ] **Step 3: Implement backend routing**

Use an immutable mapping populated from:

```text
MINWM_UPSTREAM_HTTP / MINWM_UPSTREAM_WS
LINGBOT2_UPSTREAM_HTTP / LINGBOT2_UPSTREAM_WS
```

Register:

```python
app.router.add_get(
    "/backends/{backend}/v1/realtime_video/generate", _proxy_backend_websocket
)
app.router.add_route("*", "/backends/{backend}/v1/{path:.*}", _proxy_backend_http)
```

Preserve the legacy unprefixed route for single-model deployments.

- [ ] **Step 4: Verify tests and commit**

```bash
python3 -m pytest -q python/sglang/multimodal_gen/test/unit/realtime/test_realtime_webui.py
git add python/sglang/multimodal_gen/apps/realtime_webui/server.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_realtime_webui.py
git commit -m "feat: proxy two realtime model backends"
```

---

### Task 5: Add 720p Dual-Model Benchmark Harness

**Files:**
- Create: `benchmark/minwm_realtime_async_vae/dual_model_load_test.py`
- Create: `benchmark/minwm_realtime_async_vae/test_dual_model_load_test.py`
- Create: `benchmark/minwm_realtime_async_vae/dual_model_report.py`

- [ ] **Step 1: Write failing benchmark-unit tests**

Cover:

```python
assert concurrency_levels("1,2,4,6,8") == [1, 2, 4, 6, 8]
assert paired_event_ids(minwm_events, lingbot_events) == {1, 2, 3}
assert should_stop({"success_rate": 0.98}) is True
assert should_stop({"source_fps_p50": 15.9}) is True
assert should_stop({"pod_restarts": 1}) is True
```

Verify report rows preserve separate MinWM, LingBot2 and dual profiles.

- [ ] **Step 2: Run and verify failure**

```bash
python3 -m pytest -q benchmark/minwm_realtime_async_vae/test_dual_model_load_test.py
```

Expected: FAIL because harness does not exist.

- [ ] **Step 3: Implement the harness**

Reuse the existing MessagePack init/event protocol and trace HTTP collector. Add CLI args:

```text
--minwm-ws-url
--lingbot2-ws-url
--profile minwm|lingbot2|dual
--concurrency 1,2,4,6,8
--size 1280x704
--fps 24
--num-frames 9
--warmup-chunks 2
--duration-s 60
--output-dir
```

Every dual virtual user sends identical event IDs and Action payloads to both sockets. Persist per-session, per-model raw samples and a combined summary.

- [ ] **Step 4: Implement deterministic report rendering**

Write JSON and Chinese Markdown tables containing success rate, first-frame/chunk/stage percentiles, FPS, event response, dropped frames, buffer and hardware samples. Stop-condition reasons must be explicit.

- [ ] **Step 5: Verify and commit**

```bash
python3 -m pytest -q benchmark/minwm_realtime_async_vae/test_dual_model_load_test.py
git add benchmark/minwm_realtime_async_vae/dual_model_load_test.py \
  benchmark/minwm_realtime_async_vae/test_dual_model_load_test.py \
  benchmark/minwm_realtime_async_vae/dual_model_report.py
git commit -m "test: add dual model realtime load harness"
```

---

### Task 6: Build Immutable Runtime Image

**Files:**
- Modify as needed: `benchmark/minwm_realtime_async_vae/docker/Dockerfile.gpu-code-overlay`
- Modify as needed: `benchmark/minwm_realtime_async_vae/docker/Dockerfile.cpu-code-overlay`
- Verify: `benchmark/minwm_realtime_async_vae/docker/requirements-*.lock`

- [ ] **Step 1: Run image contract tests before build**

```bash
python3 -m pytest -q \
  benchmark/minwm_realtime_async_vae/test_deploy_production.py \
  benchmark/minwm_realtime_async_vae/test_aws_infrastructure.py
```

- [ ] **Step 2: Build and push commit-addressed GPU and CPU images**

Use the committed branch SHA as the tag, enable BuildKit cache, and push to the existing ECR repositories. Do not use `latest` in manifests.

- [ ] **Step 3: Resolve and record image digests**

```bash
GIT_SHA=$(git rev-parse HEAD)
aws ecr describe-images --region us-east-2 \
  --repository-name leap-world/minwm-realtime \
  --image-ids imageTag="$GIT_SHA" \
  --query 'imageDetails[0].imageDigest' --output text
```

Expected: non-empty `sha256:` digests for GPU and CPU images.

- [ ] **Step 4: Run container smoke checks**

Verify imports, TAEHV weight presence, WebUI static files and CLI help without network access.

---

### Task 7: Create and Validate the 4+4 Deployment Profile

**Files:**
- Create in deployment worktree: `deploy/k8s/overlays/minwm-lingbot2-dual/*`
- Create in deployment worktree: `docs/minwm-lingbot2-dual-h100.md`

- [ ] **Step 1: Create a deployment-repo worktree from `origin/main`**

Branch: `codex/minwm-lingbot2-dual-h100`.

- [ ] **Step 2: Write failing profile assertions**

The test must parse rendered YAML and assert:

```python
assert replicas("minwm-denoiser") == 2
assert replicas("lingbot2-server") == 2
assert gpu_request("minwm-denoiser") == 2
assert gpu_request("lingbot2-server") == 2
assert env("minwm-denoiser", "SP_DEGREE") == "2"
assert env("lingbot2-server", "SP_DEGREE") == "2"
assert "--enable-cuda-graph" in args("minwm-denoiser")
assert total_gpu_requests(rendered) == 8
assert checkpoint_sha(rendered) == "36de945826273583a8cfdfbfa1d0e6eff726c092a7e0b071e92d055028d941ca"
```

- [ ] **Step 3: Render and verify expected failure**

```bash
kubectl kustomize deploy/k8s/overlays/minwm-lingbot2-dual > /tmp/dual.yaml
python3 -m pytest -q deploy/k8s/overlays/minwm-lingbot2-dual/test_profile.py
```

- [ ] **Step 4: Implement workloads and runtime config**

Use image digests from Task 6. Runtime config must contain only user-facing model labels and endpoints:

```json
{
  "comparisonMode": true,
  "models": [
    {"key": "minwm", "label": "MinWM", "endpoint": "/backends/minwm/v1/realtime_video/generate"},
    {"key": "lingbot2", "label": "LingBot2", "endpoint": "/backends/lingbot2/v1/realtime_video/generate"}
  ]
}
```

- [ ] **Step 5: Validate immutable model identity read-only**

Run `aws s3api head-object` with the pinned VersionId and compare bytes/checksum to the design. Stage under a content-addressed key only if not already present.

- [ ] **Step 6: Verify rendered manifest and commit deployment profile**

```bash
kubectl kustomize deploy/k8s/overlays/minwm-lingbot2-dual > /tmp/dual.yaml
python3 -m pytest -q deploy/k8s/overlays/minwm-lingbot2-dual/test_profile.py
git add deploy/k8s/overlays/minwm-lingbot2-dual docs/minwm-lingbot2-dual-h100.md
git commit -m "deploy: add MinWM LingBot2 H100 split profile"
```

---

### Task 8: Deploy and Perform End-to-End Verification

**Files:**
- Capture: `/tmp/dual.yaml`
- Record: deployment doc and SGLang benchmark result directory.

- [ ] **Step 1: Snapshot existing resources and verify the target node**

Capture current StatefulSets/Deployments/Pods, H100 allocation, L4 VAE readiness, Service endpoints, image digests and old rollback manifests.

- [ ] **Step 2: Deploy CPU gateway and zero-replica GPU resources**

Apply the comparison gateway/Services first. Verify static UI and backend route configuration before allocating GPUs.

- [ ] **Step 3: Drain current MinWM sessions and release old GPU reservations**

Scale the old `8×SP1` workload to zero only after confirming the replacement manifests and rollback snapshot. Verify the H100 node reports eight allocatable GPUs.

- [ ] **Step 4: Start four 2-GPU Pods together**

Scale both MinWM and LingBot2 workloads to two replicas. Wait for all four Pods to be Ready and ensure each requests exactly two GPUs on the intended Spot node.

- [ ] **Step 5: Verify server contracts**

Check `/health`, `/v1/models`, model revision, MinWM checkpoint identity, SP degree, CUDA Graph log evidence, TAEHV remote path, no startup clone/install, and Service endpoint membership.

- [ ] **Step 6: Run browser E2E with Playwright**

Verify I2V, T2V support matrix, one reference picker, one parameter form, left MinWM/right LingBot2, shared W/A event IDs, Prompt update, independent FPS/buffer stats, Stop cleanup and one-side failure isolation. Capture screenshots and a 30-second recording.

- [ ] **Step 7: Verify resource cleanup after each session**

Coordinator reservations, worker sessions, decoder workers and WebSocket connections must return to baseline without restarting Pods.

---

### Task 9: Run 720p Capacity Benchmark and Publish Results

**Files:**
- Create: `benchmark/minwm_realtime_async_vae/results/$RUN_ID-dual-h100-720p/*`
- Modify: deployment doc with deployed SHA/digests and measured capacity.

- [ ] **Step 1: Warm both models**

Run two warmup chunks for each replica and verify no compilation is included in steady-state percentiles.

- [ ] **Step 2: Run MinWM-only levels**

Run concurrency `1,2,4,6,8`, 60 seconds per level, stopping at the first design-defined capacity condition.

- [ ] **Step 3: Run LingBot2-only levels**

Use the identical prompt, size, FPS, frames, steps and Action schedule.

- [ ] **Step 4: Run dual-model levels**

Each virtual user opens one socket per model and sends identical event IDs to both.

- [ ] **Step 5: Collect hardware and browser samples**

Capture GPU utilization/memory per Pod, CPU/memory, restarts, source/render FPS, display lag, queue lead, WebSocket bytes and frame drops.

- [ ] **Step 6: Generate and review the report**

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_DIR="benchmark/minwm_realtime_async_vae/results/$RUN_ID-dual-h100-720p"
python3 benchmark/minwm_realtime_async_vae/dual_model_report.py \
  --input-dir "$RESULT_DIR/raw" \
  --output-json "$RESULT_DIR/report.json" \
  --output-markdown "$RESULT_DIR/report.zh-CN.md"
```

Expected: no missing profile/metric columns; raw results and report agree.

- [ ] **Step 7: Commit reproducible result artifacts**

Commit reports and bounded-size JSON/screenshot artifacts. Do not commit large videos; record their S3 URI and checksum in the report.

---

### Task 10: Final Regression, Code Review and Handoff

**Files:**
- Verify all modified files in both worktrees.

- [ ] **Step 1: Run complete targeted regression suite**

```bash
for test in python/sglang/multimodal_gen/apps/realtime_webui/*_test.js; do node "$test"; done
python3 -m pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_realtime_webui.py \
  benchmark/minwm_realtime_async_vae/test_load_test.py \
  benchmark/minwm_realtime_async_vae/test_dual_model_load_test.py \
  benchmark/minwm_realtime_async_vae/test_deploy_production.py
```

Expected: all pass.

- [ ] **Step 2: Run diff and secret checks**

```bash
git diff --check origin/main...HEAD
git status --short
rg -n "AKIA|LTAI|BEGIN.*PRIVATE KEY|aws_secret_access_key" \
  python/sglang/multimodal_gen/apps/realtime_webui \
  benchmark/minwm_realtime_async_vae docs_new/docs/sglang-diffusion/minwm
```

Expected: clean diff check, no secrets.

- [ ] **Step 3: Request code review and address findings**

Review for lifecycle leaks, event-ID divergence, shared mutable playback state, unbounded queues, mutable deployment inputs and benchmark bias.

- [ ] **Step 4: Push both branches and report live URL**

Push the SGLang feature branch and deployment branch. Report exact commits, image digests, model identities, live WebUI URL, remaining Spot cost, E2E evidence and benchmark conclusions.
