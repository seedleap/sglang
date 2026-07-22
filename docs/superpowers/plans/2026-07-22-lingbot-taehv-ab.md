# LingBot TAEHV A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in TAEHV realtime decode path and produce a test-only, 100-video B300 A/B comparison without changing production decoder selection.

**Architecture:** The SGLang realtime VAE stage chooses TAEHV only when a non-empty `taehv_checkpoint_path` is passed to `sglang serve`; the normal path remains byte-for-byte the existing causal VAE invocation. The offline runner translates a test-only environment variable into that one CLI argument. A dedicated B300 Job runs baseline then candidate sequentially against the same first 100 `testset100-v2` cases, and a local report builder pairs artifacts by case ID.

**Tech Stack:** Python, PyTorch, SGLang realtime video WebSocket, Bash, Docker, Kubernetes, AWS S3/FSx, `taehv==0.1.0` from `madebyollin/taehv@093b918971d59001a0bad6dfd6e0409b5e1752cf`.

---

### Task 1: Add opt-in decoder configuration and streaming decode

**Files:**
- Modify: `python/sglang/multimodal_gen/configs/models/vaes/base.py`
- Modify: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/realtime/vae.py`
- Modify: `python/sglang/multimodal_gen/test/unit/realtime/test_realtime_vae.py`

- [ ] **Step 1: Write the failing configuration and streaming tests**

Append a CLI parsing test and a fake-module streaming test. The streaming test must assert that `NCTHW -> NTCHW -> NCTHW` preserves shape, `frames_to_trim` applies only to the first chunk, a model is cached once, and a session decoder resets once:

```python
def test_causal_vae_decoding_stage_can_use_streaming_taehv(monkeypatch):
    fake_taehv = ModuleType("taehv")
    fake_taehv.TAEHV = _TAEHV
    fake_taehv.StreamingTAEHV = _StreamingTAEHV
    monkeypatch.setitem(sys.modules, "taehv", fake_taehv)
    monkeypatch.setattr(realtime_vae, "get_local_torch_device", lambda: torch.device("cpu"))
    stage = CausalVaeDecodingStage.__new__(CausalVaeDecodingStage)
    state = RealtimeVAEDecodeState()
    args = SimpleNamespace(pipeline_config=_PipelineConfig("/tmp/taew2_1.pth"))
    first = stage.decode_taehv_streaming(torch.arange(6.).reshape(1, 2, 3, 1, 1), args, state, first_chunk=True)
    second = stage.decode_taehv_streaming(torch.arange(6., 12.).reshape(1, 2, 3, 1, 1), args, state, first_chunk=False)
    assert tuple(first.shape) == (1, 2, 9, 1, 1)
    assert tuple(second.shape) == (1, 2, 12, 1, 1)
    assert _TAEHV.init_count == 1
    assert state.taehv_streaming_decoder.reset_calls == 1
```

- [ ] **Step 2: Verify the test fails before implementation**

Run:

```bash
pytest -q python/sglang/multimodal_gen/test/unit/realtime/test_realtime_vae.py
```

Expected: failure because `decode_taehv_streaming` and `taehv_checkpoint_path` do not exist.

- [ ] **Step 3: Add `taehv_checkpoint_path` to `VAEConfig`**

Add the field after `auto_parallel_decode_min_latent_elements_per_rank` and expose it through the existing nested CLI convention:

```python
taehv_checkpoint_path: str | None = None

parser.add_argument(
    f"--{prefix}.taehv-checkpoint-path",
    type=str,
    dest=f"{prefix.replace('-', '_')}.taehv_checkpoint_path",
    default=None,
    help="Path to a TAEHV checkpoint for realtime preview decode",
)
```

- [ ] **Step 4: Add the lazy, cached TAEHV path**

Extend `RealtimeVAEDecodeState` with `taehv_streaming_decoder` and `taehv_output_queue`. In `CausalVaeDecodingStage`, add `_taehv_checkpoint_path`, `_get_or_create_taehv_streaming_decoder`, and `decode_taehv_streaming`. The implementation must:

```python
latents_ntchw = latents.to(get_local_torch_device(), dtype=vae_dtype)
latents_ntchw = latents_ntchw.permute(0, 2, 1, 3, 4).contiguous()
target_frames = max(1, latents_ntchw.shape[1] * int(decoder.taehv.t_upscale)
                    - (int(decoder.taehv.frames_to_trim) if first_chunk else 0))
```

For every one-latent timestep, call `decoder.decode(latent_t)` and drain `decoder.decode()` until it returns `None`. Append outputs to the session queue, take exactly `target_frames`, retain the remainder, and return `frames_ntchw.permute(0, 2, 1, 3, 4).contiguous()`. When `taehv_checkpoint_path` is absent, retain the existing `decode_causal` call unchanged. Missing package or path configuration must raise a precise `RuntimeError`; it must never silently use a different decoder.

- [ ] **Step 5: Run focused tests and formatter checks**

Run:

```bash
pytest -q python/sglang/multimodal_gen/test/unit/realtime/test_realtime_vae.py
ruff check python/sglang/multimodal_gen/configs/models/vaes/base.py python/sglang/multimodal_gen/runtime/pipelines_core/stages/realtime/vae.py python/sglang/multimodal_gen/test/unit/realtime/test_realtime_vae.py
python -m black --check python/sglang/multimodal_gen/configs/models/vaes/base.py python/sglang/multimodal_gen/runtime/pipelines_core/stages/realtime/vae.py python/sglang/multimodal_gen/test/unit/realtime/test_realtime_vae.py
```

Expected: all pass.

- [ ] **Step 6: Commit decoder implementation**

```bash
git add python/sglang/multimodal_gen/configs/models/vaes/base.py python/sglang/multimodal_gen/runtime/pipelines_core/stages/realtime/vae.py python/sglang/multimodal_gen/test/unit/realtime/test_realtime_vae.py
git commit -m "feat: add opt-in TAEHV realtime decode"
```

### Task 2: Make the immutable runner image and launcher test-capable

**Files:**
- Modify: `benchmark/lingbot2_offline_batch/Dockerfile.video-runner`
- Modify: `benchmark/lingbot2_offline_batch/run_capacity_smoke_720p.sh`
- Create: `benchmark/lingbot2_offline_batch/test_taehv_runner_contract.py`

- [ ] **Step 1: Write the failing runner-contract tests**

Add source-contract tests that read `run_capacity_smoke_720p.sh` and assert the candidate-only argument is constructed as an array and is absent when the environment is empty:

```python
def test_runner_only_adds_taehv_flag_when_checkpoint_is_set():
    script = Path(__file__).with_name("run_capacity_smoke_720p.sh").read_text()
    assert 'taehv_args=()' in script
    assert '[[ -n "${taehv_checkpoint_path}" ]]' in script
    assert '--vae-config.taehv-checkpoint-path' in script

def test_runner_image_installs_pinned_taehv():
    dockerfile = Path(__file__).with_name("Dockerfile.video-runner").read_text()
    assert 'madebyollin/taehv.git@093b918971d59001a0bad6dfd6e0409b5e1752cf' in dockerfile
```

- [ ] **Step 2: Verify runner-contract tests fail**

Run:

```bash
pytest -q benchmark/lingbot2_offline_batch/test_taehv_runner_contract.py
```

Expected: both assertions fail because no TAEHV image dependency or optional CLI argument exists.

- [ ] **Step 3: Pin TAEHV in the runner image**

Extend the image install layer with the immutable source revision:

```dockerfile
&& python3 -m pip install --no-cache-dir \
    'taehv @ git+https://github.com/madebyollin/taehv.git@093b918971d59001a0bad6dfd6e0409b5e1752cf' \
    --root-user-action=ignore \
&& python3 -m pip install --no-cache-dir boto3 --root-user-action=ignore
```

- [ ] **Step 4: Add a test-only optional launcher flag**

At the top of `run_capacity_smoke_720p.sh`, read `TAEHV_CHECKPOINT_PATH`. Before `sglang serve`, build an argument array:

```bash
taehv_checkpoint_path=${TAEHV_CHECKPOINT_PATH:-}
taehv_args=()
if [[ -n "${taehv_checkpoint_path}" ]]; then
  [[ -r "${taehv_checkpoint_path}" ]] || { echo "TAEHV checkpoint is not readable: ${taehv_checkpoint_path}" >&2; exit 2; }
  python3 -c 'import taehv' || { echo "TAEHV package is unavailable" >&2; exit 2; }
  taehv_args=(--vae-config.taehv-checkpoint-path "${taehv_checkpoint_path}")
fi
```

Append `"${taehv_args[@]}"` to only the `sglang serve` invocation. Write `taehv-runtime.json` under the result root with `enabled`, `checkpoint_path`, `checkpoint_sha256`, and `pip show taehv` output. Do not read this environment variable anywhere in the controller.

- [ ] **Step 5: Verify contract tests and runner syntax**

Run:

```bash
pytest -q benchmark/lingbot2_offline_batch/test_taehv_runner_contract.py
bash -n benchmark/lingbot2_offline_batch/run_capacity_smoke_720p.sh
docker build --check -f benchmark/lingbot2_offline_batch/Dockerfile.video-runner .
```

Expected: all pass without building or pushing an image.

- [ ] **Step 6: Commit runner support**

```bash
git add benchmark/lingbot2_offline_batch/Dockerfile.video-runner benchmark/lingbot2_offline_batch/run_capacity_smoke_720p.sh benchmark/lingbot2_offline_batch/test_taehv_runner_contract.py
git commit -m "feat: support test-only TAEHV video runner"
```

### Task 3: Create the deterministic 100-case A/B manifest and comparison report

**Files:**
- Create: `benchmark/lingbot2_offline_batch/k8s-taehv-ab-testset100-b300.yaml`
- Create: `benchmark/lingbot2_offline_batch/build_taehv_ab_report.py`
- Create: `benchmark/lingbot2_offline_batch/test_build_taehv_ab_report.py`

- [ ] **Step 1: Write the failing report pairing test**

Create fixture summaries with two overlapping `sample_id` values and assert the report model retains both decoder videos plus image, prompt, and action metadata:

```python
def test_pair_cases_requires_identical_case_ids(tmp_path):
    rows = pair_cases(_summary("baseline", ["case-0", "case-1"]), _summary("taehv", ["case-0", "case-1"]))
    assert [row["sample_id"] for row in rows] == ["case-0", "case-1"]
    assert rows[0]["baseline"]["video_s3_uri"].endswith("case-0.mp4")
    assert rows[0]["taehv"]["video_s3_uri"].endswith("case-0.mp4")

def test_pair_cases_rejects_mismatched_case_ids():
    with pytest.raises(ValueError, match="case IDs differ"):
        pair_cases(_summary("baseline", ["case-0"]), _summary("taehv", ["case-1"]))
```

- [ ] **Step 2: Verify report test fails**

Run:

```bash
pytest -q benchmark/lingbot2_offline_batch/test_build_taehv_ab_report.py
```

Expected: import error because `build_taehv_ab_report.py` does not exist.

- [ ] **Step 3: Implement report model and standalone HTML**

Implement `pair_cases`, `compute_metrics`, and `build_html` in
`build_taehv_ab_report.py`. Pair solely by `sample_id`, reject any missing or
duplicated IDs, surface image URI/prompt/action trajectory/action list, and
render two `<video>` elements per row with labels `原 VAE` and `TAEHV`.
The summary must calculate both arms' successful/failed count, first persisted
video, steady throughput from p10-to-p90 persisted timestamps, total wall time,
mean end-to-end time, upload lag, and tail duration. Embed the exact image
digest, SGLang commit, TAEHV source commit, checkpoint SHA-256, GPU inventory,
and 8x1 topology.

- [ ] **Step 4: Add test-only B300 Job manifest**

The manifest creates no Deployment or controller patch. It defines two
sequential `batch/v1 Job` objects, each requesting exactly `8` GPUs and pinned
to:

```yaml
nodeSelector:
  eks.amazonaws.com/capacityType: CAPACITY_BLOCK
  eks.amazonaws.com/nodegroup: wan22-cb-p6b300-0715-20c
  node.kubernetes.io/instance-type: p6-b300.48xlarge
```

Both set `SGLANG_VIDEO_TOPOLOGY=8x1`, `SGLANG_VIDEO_WIDTH=832`,
`SGLANG_VIDEO_HEIGHT=480`, `SGLANG_VIDEO_FPS=16`, `STREAM_UPLOAD=true`, and
use `benchmark_evalset.py --limit 100`. Baseline omits `TAEHV_CHECKPOINT_PATH`.
Candidate's init container downloads the `taew2_1.pth` file from commit
`093b918971d59001a0bad6dfd6e0409b5e1752cf` into an `emptyDir`, calculates
`sha256sum`, and mounts it read-only at `/models/taehv/taew2_1.pth`.
Each arm writes only to the fixed test prefix
`world-model/eval/lingbot2/taehv_ab/taehv-ab-20260722/{baseline,candidate}`
and has no callback URL or LWDP generation ID.

- [ ] **Step 5: Verify report tests and manifest rendering**

Run:

```bash
pytest -q benchmark/lingbot2_offline_batch/test_build_taehv_ab_report.py
python -m py_compile benchmark/lingbot2_offline_batch/build_taehv_ab_report.py
kubectl --context leap-world-aws03-usw2 apply --dry-run=client -f benchmark/lingbot2_offline_batch/k8s-taehv-ab-testset100-b300.yaml
```

Expected: tests pass and client-side Kubernetes validation produces no changes.

- [ ] **Step 6: Commit A/B tooling**

```bash
git add benchmark/lingbot2_offline_batch/k8s-taehv-ab-testset100-b300.yaml benchmark/lingbot2_offline_batch/build_taehv_ab_report.py benchmark/lingbot2_offline_batch/test_build_taehv_ab_report.py
git commit -m "feat: add LingBot TAEHV B300 A/B tooling"
```

### Task 4: Build, run, and report the isolated A/B test

**Files:**
- Output: `/tmp/lingbot-taehv-ab-20260722.html`

- [ ] **Step 1: Build and inspect the immutable candidate image**

Build from the worktree, push only the candidate test tag, then capture its
digest. Before running a GPU Job, start a non-GPU image check that prints:

```bash
python3 -c 'import taehv; print(taehv.__file__)'
python3 -c 'from taehv import TAEHV, StreamingTAEHV; print("ok")'
```

- [ ] **Step 2: Obtain explicit confirmation for the external writes**

Show the precise image registry tag, two Job names, two FSx directories, two S3
prefixes, report prefix, expected `200` MP4 writes plus metadata/report files,
and the cleanup procedure. Do not create Jobs, write S3 data, or push an image
until the human confirms this exact scope.

- [ ] **Step 3: Run baseline then candidate, with production untouched**

Apply only the test manifest. Wait for the baseline Job to complete before
creating the candidate Job. Read production controller Deployment before and
after and verify its image, `SGLANG_VIDEO_TOPOLOGY=8x1`, backend list, and lack
of `TAEHV_CHECKPOINT_PATH` are unchanged.

- [ ] **Step 4: Generate and validate the comparison report**

Download only the two test reports/manifests, run
`build_taehv_ab_report.py`, inspect that the HTML contains exactly 100 paired
case cards, and upload the final HTML/report to the dedicated test prefix.
Verify all 100 image URIs, prompt/action records, and both video URLs are
present. Report visual quality differences as observed facts, and report
performance measurements with the sampling basis and any failed cases.

- [ ] **Step 5: Leave production configuration unchanged**

Do not set `TAEHV_CHECKPOINT_PATH` in `sglang-video-controller`, do not change
its image, and do not roll out its Deployment. Delete only the two completed
test Jobs after their outputs and report are independently verified; retain the
candidate image tag and test S3 outputs for review.
