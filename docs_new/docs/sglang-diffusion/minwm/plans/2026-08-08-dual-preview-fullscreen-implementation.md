# Dual Preview Fullscreen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an accessible native fullscreen comparison view for the existing MinWM and LingBot2 players, align the deployed MinWM UI defaults with its trained 8/32 temporal profile, and preserve the real model output without cosmetic fog filters.

**Architecture:** A focused browser-side `FullscreenController` owns the Fullscreen API and button state, while `app.js` only wires the controller to the existing stage and history UI. CSS makes the existing stage fill the fullscreen viewport without duplicating controls or players. The Kubernetes overlay supplies the same sink/window defaults shown in the UI and already used by the MinWM server.

**Tech Stack:** Browser Fullscreen API, vanilla JavaScript, Node.js `assert` contract tests, CSS Grid, Python `unittest`, Kubernetes YAML.

---

## File Map

- Create `python/sglang/multimodal_gen/apps/realtime_webui/fullscreen_controller.js`: isolated Fullscreen API state machine.
- Create `python/sglang/multimodal_gen/apps/realtime_webui/fullscreen_controller_test.js`: behavioral unit tests with a fake document and stage.
- Modify `python/sglang/multimodal_gen/apps/realtime_webui/dual_model_contract_test.js`: static DOM and CSS contract for the fullscreen affordance.
- Modify `python/sglang/multimodal_gen/apps/realtime_webui/index.html`: add the icon-only fullscreen button and controller script.
- Modify `python/sglang/multimodal_gen/apps/realtime_webui/app.js`: instantiate the controller and surface Fullscreen API failures in History.
- Modify `python/sglang/multimodal_gen/apps/realtime_webui/styles.css`: make the complete comparison stage fit desktop and narrow fullscreen viewports.
- Modify `deploy/k8s/overlays/minwm-lingbot2-dual/test_profile.py`: assert the deployed UI temporal defaults.
- Modify `deploy/k8s/overlays/minwm-lingbot2-dual/workloads.yaml`: publish `sinkSize=8` and `windowFrames=32` to the UI.

### Task 1: Fullscreen Controller

**Files:**
- Create: `python/sglang/multimodal_gen/apps/realtime_webui/fullscreen_controller.js`
- Create: `python/sglang/multimodal_gen/apps/realtime_webui/fullscreen_controller_test.js`

- [ ] **Step 1: Write the failing controller tests**

Create a fake document that records `fullscreenchange` listeners and exposes `fullscreenElement` and `exitFullscreen()`. Create a fake stage with `requestFullscreen()` and a fake button with `setAttribute()`. Assert these behaviors:

```javascript
const controller = createFullscreenController({ documentRef, target, button });
await controller.toggle();
assert.equal(target.requestCount, 1);

documentRef.fullscreenElement = target;
documentRef.emit("fullscreenchange");
assert.equal(button.attributes["aria-pressed"], "true");
assert.equal(button.title, "Exit fullscreen comparison");

await controller.toggle();
assert.equal(documentRef.exitCount, 1);

documentRef.fullscreenElement = null;
documentRef.emit("fullscreenchange");
assert.equal(button.attributes["aria-pressed"], "false");
assert.equal(button.title, "Enter fullscreen comparison");
```

Also assert that a rejected `requestFullscreen()` calls `onError` and that `destroy()` removes the `fullscreenchange` listener.

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
node python/sglang/multimodal_gen/apps/realtime_webui/fullscreen_controller_test.js
```

Expected: FAIL with `Cannot find module './fullscreen_controller.js'`.

- [ ] **Step 3: Implement the minimal controller**

Export the controller for both browsers and CommonJS:

```javascript
(function (global) {
  function createFullscreenController({
    documentRef = document,
    target,
    button,
    onError = () => {},
  }) {
    const sync = () => {
      const active = documentRef.fullscreenElement === target;
      button.setAttribute("aria-pressed", String(active));
      button.title = active
        ? "Exit fullscreen comparison"
        : "Enter fullscreen comparison";
      button.setAttribute("aria-label", button.title);
    };
    const toggle = async () => {
      try {
        if (documentRef.fullscreenElement === target) {
          await documentRef.exitFullscreen();
        } else {
          await target.requestFullscreen();
        }
      } catch (error) {
        onError(error);
      }
    };
    button.addEventListener("click", toggle);
    documentRef.addEventListener("fullscreenchange", sync);
    sync();
    return {
      toggle,
      sync,
      destroy() {
        button.removeEventListener("click", toggle);
        documentRef.removeEventListener("fullscreenchange", sync);
      },
    };
  }

  global.SGLangFullscreen = { createFullscreenController };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { createFullscreenController };
  }
})(typeof globalThis !== "undefined" ? globalThis : window);
```

- [ ] **Step 4: Run the controller test and confirm it passes**

Run the command from Step 2. Expected: `fullscreen controller ok`.

- [ ] **Step 5: Commit the controller**

```bash
git add python/sglang/multimodal_gen/apps/realtime_webui/fullscreen_controller.js \
  python/sglang/multimodal_gen/apps/realtime_webui/fullscreen_controller_test.js
git commit -m "feat(realtime): add comparison fullscreen controller"
```

### Task 2: Fullscreen Stage Integration

**Files:**
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/dual_model_contract_test.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/index.html`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/app.js`
- Modify: `python/sglang/multimodal_gen/apps/realtime_webui/styles.css`

- [ ] **Step 1: Extend the DOM contract with failing assertions**

Assert a single accessible button and the new controller script:

```javascript
assert.equal((html.match(/id="fullscreenBtn"/g) || []).length, 1);
assert.match(html, /id="fullscreenBtn"[\s\S]*aria-label="Enter fullscreen comparison"/);
assert.match(html, /fullscreen_controller\.js\?v=dual-fullscreen-v1/);
assert.match(app, /createFullscreenController/);
assert.match(css, /\.stage:fullscreen\s*\{/);
assert.match(css, /\.stage:fullscreen[\s\S]*height:\s*100vh/);
```

- [ ] **Step 2: Run the contract and confirm it fails**

Run:

```bash
node python/sglang/multimodal_gen/apps/realtime_webui/dual_model_contract_test.js
```

Expected: FAIL because `fullscreenBtn` is absent.

- [ ] **Step 3: Add the icon-only button and controller script**

Place the button after the preview scale control so it remains in the topbar in fullscreen:

```html
<button
  id="fullscreenBtn"
  class="fullscreen-button"
  type="button"
  aria-label="Enter fullscreen comparison"
  aria-pressed="false"
  title="Enter fullscreen comparison"
>
  <span class="fullscreen-icon" aria-hidden="true"></span>
</button>
```

Load `fullscreen_controller.js?v=dual-fullscreen-v1` before `app.js`.

- [ ] **Step 4: Wire the controller in `app.js`**

Instantiate it after the stage is resolved:

```javascript
const fullscreenController = window.SGLangFullscreen?.createFullscreenController?.({
  documentRef: document,
  target: stage,
  button: $("fullscreenBtn"),
  onError: (error) => addHistory(`fullscreen unavailable: ${error.message || error}`),
});
```

The controller remains alive for the page lifetime; browser Escape triggers `fullscreenchange` and updates the button automatically.

- [ ] **Step 5: Add stable fullscreen layout rules**

Use the existing stage instead of creating a second player tree:

```css
.fullscreen-button {
  position: relative;
  width: 30px;
  height: 30px;
  flex: 0 0 30px;
}

.stage:fullscreen {
  width: 100vw;
  height: 100vh;
  max-width: none;
  border: 0;
  border-radius: 0;
  grid-template-rows: auto minmax(0, 1fr) auto auto auto;
}

.stage:fullscreen .model-player-grid,
.stage:fullscreen .model-player,
.stage:fullscreen .preview-frame {
  min-height: 0;
}

.stage:fullscreen .preview-frame {
  width: 100%;
  height: 100%;
}

.stage:fullscreen canvas {
  width: 100%;
  height: 100%;
  max-height: none;
  object-fit: contain;
}
```

At `max-width: 900px`, keep the existing one-column player grid so each video remains inspectable. Do not add filters, opacity layers, sharpening, color correction, or model-specific rendering branches.

- [ ] **Step 6: Run all WebUI unit and contract tests**

Run:

```bash
for test in python/sglang/multimodal_gen/apps/realtime_webui/*_test.js; do node "$test"; done
```

Expected: every test exits 0, including `fullscreen controller ok` and `dual model DOM contract ok`.

- [ ] **Step 7: Commit the integration**

```bash
git add python/sglang/multimodal_gen/apps/realtime_webui/index.html \
  python/sglang/multimodal_gen/apps/realtime_webui/app.js \
  python/sglang/multimodal_gen/apps/realtime_webui/styles.css \
  python/sglang/multimodal_gen/apps/realtime_webui/dual_model_contract_test.js
git commit -m "feat(realtime): fullscreen dual model comparison"
```

### Task 3: Align MinWM Temporal Defaults

**Files:**
- Modify: `deploy/k8s/overlays/minwm-lingbot2-dual/test_profile.py`
- Modify: `deploy/k8s/overlays/minwm-lingbot2-dual/workloads.yaml`

- [ ] **Step 1: Add a failing deployment contract**

Parse `REALTIME_UI_CONFIG_JSON` from the gateway environment and assert:

```python
self.assertEqual(config["sinkSize"], 8)
self.assertEqual(config["windowFrames"], 32)
```

- [ ] **Step 2: Run the profile test and confirm it fails**

Run:

```bash
python -m unittest deploy/k8s/overlays/minwm-lingbot2-dual/test_profile.py -v
```

Expected: FAIL with `KeyError: 'sinkSize'`.

- [ ] **Step 3: Publish the matching UI defaults**

Add these keys to `REALTIME_UI_CONFIG_JSON`:

```json
{"sinkSize":8,"windowFrames":32}
```

Keep the existing server flags `--realtime-causal-sink-size 8` and `--realtime-causal-kv-cache-num-frames 32` unchanged.

- [ ] **Step 4: Run the deployment test and confirm it passes**

Run the command from Step 2. Expected: all profile tests pass.

- [ ] **Step 5: Commit the deployment contract**

```bash
git add deploy/k8s/overlays/minwm-lingbot2-dual/test_profile.py \
  deploy/k8s/overlays/minwm-lingbot2-dual/workloads.yaml
git commit -m "fix(deploy): align MinWM temporal defaults"
```

### Task 4: Build and Deploy the Stateless Gateway

**Files:**
- Modify: `deploy/k8s/overlays/minwm-lingbot2-dual/workloads.yaml`

- [ ] **Step 1: Build and push the updated immutable gateway image**

Use the deployment repository's existing ECR build workflow to build the SGLang worktree commit. Record the resulting digest in the gateway container image and update `seedleap.ai/code-git-ref` to the exact SGLang commit SHA.

- [ ] **Step 2: Validate rendered Kubernetes objects before applying**

Run:

```bash
kubectl kustomize deploy/k8s/overlays/minwm-lingbot2-dual >/tmp/minwm-lingbot2-dual-rendered.yaml
kubectl apply --dry-run=server -f /tmp/minwm-lingbot2-dual-rendered.yaml
```

Expected: server dry-run succeeds without mutating the cluster.

- [ ] **Step 3: Roll only the gateway deployment**

Apply the updated gateway Deployment object and leave MinWM, LingBot2, and VAE GPU workloads untouched. Wait for both gateway replicas:

```bash
kubectl -n minwm-realtime rollout status deployment/minwm-realtime-gateway --timeout=10m
```

Expected: `deployment "minwm-realtime-gateway" successfully rolled out` and no GPU pod restart timestamps change.

- [ ] **Step 4: Commit the immutable digest**

```bash
git add deploy/k8s/overlays/minwm-lingbot2-dual/workloads.yaml
git commit -m "deploy: publish dual comparison fullscreen"
```

### Task 5: End-to-End Browser Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Verify the default page contract**

Open the public WebUI with Playwright. Assert the size input is `1280x704`, sink is `8`, window is `32`, both labels are visible, and the fullscreen button has `aria-pressed=false`.

- [ ] **Step 2: Verify native fullscreen behavior**

Click the fullscreen button and assert `document.fullscreenElement` is the stage, both player bounding boxes are non-zero and visible, the shared control block is visible, and the button reads `aria-pressed=true`. Press Escape and assert the state returns to false.

- [ ] **Step 3: Verify dual I2V generation**

Select the shared reference image, click Generate, and wait for both canvases to contain non-black pixels. Hold `W` long enough to produce action events and confirm both sessions remain live and both canvases continue updating.

- [ ] **Step 4: Reconfirm the MinWM haze diagnosis**

Capture early and long-horizon MinWM frames without applying CSS filters. Confirm early output is clear enough to rule out a static frontend overlay, while any later haze/corruption is reported as MinWM autoregressive latent drift. Record that the standalone TAEHV repeated fixed-latent test remained numerically stable, so the deployment change must not claim to cure model drift.

- [ ] **Step 5: Push both branches**

```bash
git push origin codex/minwm-lingbot2-dual-webui
git push origin codex/minwm-lingbot2-dual-h100
```

Report the public URL, deployed image digest, commits, browser evidence, and the MinWM drift diagnosis.
