# Repository instructions

## MinWM hardware-specific jobs

When a task creates or changes a MinWM deployment or benchmark manifest:

1. Read `benchmark/minwm_realtime_parity/HARDWARE_PROFILES.md`.
2. Start from `benchmark/minwm_realtime_parity/k8s/minwm_hardware_job.template.yaml`.
3. Select the profile from measured GPU memory and compute capability. Never infer
   capacity from a product name alone.
   - Record the `vae_cpu_offload` decision explicitly. A GPU reporting at most
     36,864 MiB requires it for the current 32 GiB candidate. A GPU reporting at
     least 65,536 MiB may omit it only when the high-memory profile applies.
     Treat intermediate capacities as experimental and start with offload enabled
     until same-device validation proves it can be disabled.
4. Render one-off manifests under
   `benchmark/minwm_realtime_parity/k8s/generated/`. This directory is ignored;
   do not commit dated, region-specific, run-specific, or instance-specific YAML.
5. Preserve the MinWM checkpoint/request invariants in the profile document.
6. Validate generated YAML before applying it and report the selected profile,
   detected hardware, memory gate, and any unvalidated assumptions.

Only update the committed template when the reusable schema changes. Only update
the hardware profiles when new evidence changes a stable decision or validation
status.
