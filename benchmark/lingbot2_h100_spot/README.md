# LingBot-World 2.0 H100 Spot performance experiment

This directory contains the isolated AWS EKS/direct-EC2 deployment, fixed
realtime benchmark, and compact result records used to measure the BF16,
fixed-KV performance ceiling of LingBot-World 2.0 on one `p5.48xlarge` node.

The deployment deliberately owns its EC2NodeClass, NodePool, Service, node,
host cache, and result path. It does not use the neighboring `lingbot2-h100`
or `lingbot-nsys-profile` resources.

Fixed experiment contract:

- 8 H100 GPUs maximum;
- BF16 causal-fast checkpoint, four DMD steps;
- no FP8, FP4, frame interpolation, or upscaling;
- interactive KV policy remains enabled with the repository defaults;
- moving camera action throughout the measured interval;
- 832x480, 9 requested decoded frames per chunk (12 steady-state frames);
- the moving action is held with a level-triggered realtime state event rather
  than a finite script, so runs longer than 171 chunks cannot fall back to
  still mode;
- first 20 chunks are warmup and excluded.

Cluster context:

```bash
CTX='arn:aws:eks:us-east-2:829115578968:cluster/leap-world'
kubectl --context "$CTX" apply --server-side --dry-run=server -f k8s.yaml
kubectl --context "$CTX" apply -f k8s.yaml
```

If EKS has no p5 capacity, `ec2_apse2_spot.json`,
`prepare_ec2_runtime.sh`, `run_ec2_server.sh`, and `run_ec2_benchmark.sh`
describe the independently pinned direct-EC2 fallback. RunInstances was
dry-run validated before launch, and the experiment never runs an EKS GPU
Deployment at the same time as the direct instance.

Run the client from a separate CPU pod in the cluster so that transport work
does not consume server CPU. Save raw per-chunk stats for every run.

Delete only the resources declared in `k8s.yaml` after copying results. The
NodePool consolidates an empty node after ten minutes and expires nodes after
eight hours as a safety net.

Final measured result (2026-07-13): the best correctness-preserving 200-chunk
run is 23.238 generated FPS. An invalid-output ceiling probe that removes all
Ulysses A2A, followed by perfect VAE/output overlap, still has a fastest
measured DiT mean of 310.541 ms per 12-frame chunk (38.642 FPS). This is an
empirical ceiling for the current BF16 kernels and tested FA3 split surface,
not a mathematical limit for future kernels. See
`docs/diffusion/lingbot_world_v2_h100_40fps_experiment_log_zh.md` for the full
protocol and scope.
