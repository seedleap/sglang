# Realtime Zing deployment notes

This document captures the current single-Zing realtime deployment shape used
for the B300/B200 lab chain. It is intentionally narrow: production services may
use different account, namespace, and admission settings.

## B300 single-node baseline

- Public WebUI: `https://zing-b300-world-studio.loopit.me/`
- Namespace: `minwm-zing-b300-sp1-20260819`
- Transport: WebSocket + direct VAE H.264 encode, no bridge service in the
  active media path.
- Zing denoiser: 7 replicas, one GPU per replica, `sp=1`.
- VAE: 1 replica on the same B300 node, one GPU, async output path.
- WebUI/gameplay: 1280x704, 24 FPS, `sink=8`, `window=32`,
  `sessionMaxLifetimeSeconds=70`.
- H.264 profile used by the lab WebUI: 3000 kbps, CRF 20, `fast`, GOP 2s,
  VBV buffer 250 ms.
- Effective admission while only the B300 node is present: 7 denoiser slots and
  7 VAE slots.

## B200 Spot expansion target

The optional B200 Spot expansion keeps the same code path and adds:

- Zing denoiser: 7 additional replicas, one GPU per replica, `sp=1`.
- VAE: 1 additional GPU, capacity aligned to the 7 extra denoiser sessions.
- WebUI admission target: 14 sessions after the B200 node is actually Ready.

Do not count the B200 capacity as available until the node has joined the
cluster and the `zing-denoiser-b200-*` and `zing-vae-b200-*` pods are Ready.
During Spot `UnfulfillableCapacity`, the Kubernetes resources may exist but the
actual serving capacity remains the B300 baseline.

## Consumed-reservation watchdog

Realtime workers should not let consumed reservations leak forever. A stale
consumed token is unsafe to simply delete because the corresponding session may
still be executing on GPU. The worker-local watchdog therefore:

1. Keeps normal unconsumed reservation expiry unchanged.
2. Tracks `oldest_consumed_age_s`, `stale_consumed_reservations`, and
   `last_progress_age_s` in the worker snapshot and heartbeat.
3. Marks the worker lifecycle as `failed` once a consumed reservation exceeds
   `max_consumed_age_s`.
4. Exposes failed lifecycle through health checks so the container supervisor
   can restart the worker instead of admitting more work onto a half-dead
   process.

Recommended value for the current 70-second game sessions:

```text
max_consumed_age_s = 120
```

This keeps a 50-second grace window after normal session lifetime. Denoiser
workers should use this strictly because capacity is one session per process.
VAE workers can use the same limit; if stale VAE reservations accumulate, the
worker fails health and restarts rather than silently over-admitting sessions.

## Rolling-update compatibility

Gateway and coordinator assignment parsing must tolerate additional heartbeat
or worker-slot fields. During rolling deploys, the coordinator may return a
newer payload schema before every gateway replica has the matching dataclass.
Filtering unknown fields avoids transient WebSocket init failures such as:

```text
WorkerSlot.__init__() got an unexpected keyword argument 'oldest_consumed_age_s'
```
