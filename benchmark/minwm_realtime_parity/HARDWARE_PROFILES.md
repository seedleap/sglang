# MinWM hardware deployment profiles

This is the stable source for humans and Codex when generating MinWM benchmark or
deployment manifests. Do not use old dated YAML as a template.

## Generation workflow

1. Read the actual GPU name, total memory, and compute capability:

   ```bash
   nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
   ```

2. Select a profile by **physical memory first**. A product label never overrides
   the reported memory.
3. Copy `k8s/minwm_hardware_job.template.yaml` to `k8s/generated/<run>.yaml`
   and replace every `REPLACE_WITH_...` token.
4. Keep the generated file untracked. Validate it with:

   ```bash
   kubectl apply --dry-run=server -f k8s/generated/<run>.yaml
   ```

5. Record the detected hardware, selected profile, SGLang/model commits, image
   digest, and result URI with the run artifacts.

## RTX 6000 labels and `vae_cpu_offload`

Do not derive memory policy from the string `6000`. NVIDIA's full-card published
capacities differ by generation: RTX 6000 Ada has 48 GB, while the RTX PRO 6000
Blackwell Server, Workstation, and Max-Q editions have 96 GB. A cloud device or
partition labelled `6000` can expose a different amount, including about 32 GiB.
The value reported inside the workload is authoritative.

Use this decision table when Codex renders a job:

| Visible memory and architecture | `vae_cpu_offload` | Decision |
| --- | --- | --- |
| SM120 and <=36,864 MiB | `true` | Required by the current 32 GiB candidate |
| SM120 and >=65,536 MiB | omitted / `false` | High-memory speed default |
| Any other capacity or compute capability, including a 48 GB Ada card | `true` initially | Experimental; do not disable until the same device passes the acceptance gates |

Always record the queried GPU name, total MiB, compute capability, and the final
offload value. A familiar product name is not evidence that offload can be
disabled. The conservative `true` setting for unclassified hardware is a first
validation candidate, not a claim that the hardware profile is production-ready.

## Checkpoint and request invariants

These values align the current MinWM checkpoint with the Tianpeng setting. They
are not hardware tuning knobs and must remain identical across profiles:

| Setting | Required value |
| --- | --- |
| local attention / sliding window | 32 frames |
| attention sink | 8 frames |
| RoPE position mode | `block_relative` |
| RoPE maximum frame gap | 12 |
| prompt first-frame pin | enabled |
| request cache window / sink | 32 / 8 |
| cache growth | disabled for the bounded request |

Use these conversion arguments:

```text
--local-attn-size 32
--sink-size 8
--sliding-window-num-frames 32
--rope-position-mode block_relative
--rope-max-frame-gap 12
--prompt-first-frame-pin-enabled
```

The request must set `realtime_causal_kv_cache_num_frames=32` and
`realtime_causal_sink_size=8`. At runtime, verify the one-time
`MINWM_RUNTIME_ALIGNMENT` log rather than assuming the requested values took
effect.

## Hardware profiles

### `blackwell-32g`

Select when an SM120 GPU reports at most 36,864 MiB. This includes the 32 GiB
5090/production target described for this work. If a machine is labelled
"6000" but reports 32 GiB, this profile still wins.

Add these explicit server settings:

```text
--vae-config.taehv-checkpoint-path <taehv-checkpoint>
--vae-cpu-offload true
--text-encoder-cpu-offload true
--dit-cpu-offload true
--dit-layerwise-offload true
--dit-offload-prefetch-size 0.0
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This is the
lowest-memory candidate, not a speed recommendation.

Validation status:

- Physical fit passed on an RTX PRO 4500 Blackwell Server Edition reporting
  32,623 MiB: 17,411 MiB peak at 832x480 and 25,505 MiB at 1248x704.
- The same-SKU performance gate has **not** passed for the 32 GiB 5090/6000
  production target. Do not claim production readiness from the fit result.

### `blackwell-high-memory`

Select when an SM120 GPU reports at least 65,536 MiB. The measured
RTX PRO 6000 Blackwell Server Edition node reported 96 GiB.

Add:

```text
--performance-mode speed
--vae-config.taehv-checkpoint-path <taehv-checkpoint>
```

Do not add CPU/layerwise offload unless a separate memory goal requires it. The
96 GiB characterization measured 30.045 FPS at 832x480 and 11.356 FPS at
1248x704 for local TAEHV, but those numbers do not prove a 32 GiB deployment.

### Unclassified hardware

Do not silently choose a nearby profile for non-SM120 GPUs or memory between the
two ranges. Generate a manifest only after marking the profile experimental and
defining an explicit memory/performance validation.

## Acceptance gates

For a 32 GiB production candidate:

- physical process peak <= 30,720 MiB;
- complete 1089 ordered frames at both 832x480 and 1248x704;
- at least three repetitions per resolution with no OOM;
- headline FPS CV <= 3%;
- <= 5% throughput regression against the accepted baseline on the same GPU SKU
  and software environment.

Fit on a different SKU is evidence for memory feasibility only.
