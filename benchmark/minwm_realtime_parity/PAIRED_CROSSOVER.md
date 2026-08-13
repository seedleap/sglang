# Paired-crossover runner

This runner compares two explicit server commands without changing a product
code path. It runs each pair on two isolated GPU/CPU/NUMA slots and swaps the
GPU assignment on every repetition.

Copy `paired_crossover.example.json`, replace the immutable commit, paths, CPU
sets, NUMA nodes, and server commands, then inspect the complete schedule:

```bash
python3 benchmark/minwm_realtime_parity/run_paired_crossover.py \
  --config /path/to/paired.json --dry-run
```

Run it from the checkout containing commit `c1381ba984f7d4d3908e4b32449641989010d908`
or a descendant:

```bash
python3 benchmark/minwm_realtime_parity/run_paired_crossover.py \
  --config /path/to/paired.json
```

The calibration first runs control and candidate alone, then concurrently. If
either steady client or scheduler FPS slows by more than `concurrency_threshold`
(default 2%), paired repetitions retain both servers but run clients
sequentially; `calibration.json` marks concurrent measurements exploratory.

Each variant receives its own `CUDA_VISIBLE_DEVICES`, port, CPU set, NUMA
binding, HOME, temporary directory, and compiler caches. A successful repetition
is atomically moved to `artifact_root`, optionally uploaded with
`upload_command`, and receives `COMPLETE` and `UPLOADED.json`. Those repetitions
are skipped on restart. SIGTERM/SIGINT stops process groups and copies current
logs to `artifact_root/interrupted` before exit, without marking the repetition
complete. `{source}` and `{relative}` placeholders are available to the upload
command.

The environment directory contains topology, P2P, CPU layout, and full
`nvidia-smi -q`; every run also records one-second power, clock, utilization,
memory, and temperature samples in `gpu-telemetry.csv`.
