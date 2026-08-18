# MinWM 统一镜像发布记录

## 2026-08-18：H200 / B200 验证通过

状态：**Accepted**（H200、B200）。H100 与 B300 共用相同 family dispatch，
但尚未做本次 digest 的 SKU 级真机门禁，不能把本记录当作这两种 SKU 的
生产验收。

- Source commit：`0b28ce762b6f1d8c722e7925aa3f3b3fb4c39744`
- Tag：`minwm-cu130-torch211-0b28ce762b6f-20260818T094900Z`
- Immutable image：
  `829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-runtime@sha256:7a34f507564a51500cff85d58ed861e1841f29c5e9891b74062d2e9577da26ed`
- ECR image size：`12,262,108,019` bytes
- Linux/amd64 manifest：
  `sha256:21d9d45e07111e22d27d4e55381f41b52b8bdcae4f75ea520064ac2b48748eef`
- Attestation manifest：
  `sha256:d51355c43c5e9ec5c40fb4f51b23168416e8a4f7a2d1fd372db971c2f6a95472`
- SPDX SBOM layer：
  `sha256:12639ce13735b633345f8d2e82c91e852fc15215aad4f5c66ec11557772168b6`
- SLSA provenance layer：
  `sha256:29193ad3c5fe5ce482c1eadfef0c0f141f58102408816637fede30f3ac9390c3`

构建前、推送后两次 `pip check` 与 software contract 均通过。核心合同为
Torch `2.11.0+cu130`、CUDA `13.0`、FA4 `4.0.0b15`、kernels `0.14.1`，
Hopper FA3 锁定到 revision
`15c17db0bf9ce6599db795fa02a8f27467c92860`；MoviePy、NIXL 与 classic
`flash-attn` distribution 均不在 runtime 中。OCI index 同时绑定 SPDX SBOM
与 SLSA provenance。

### GPU 功能门禁

| GPU | 实际路径 | 结果 |
|---|---|---|
| H200 / CC 9.0 | dense + packed FA3；provider=`kernels-community`，锁定 revision 命中 | self/cross attention 4 项与 online/static FP8 FFN 2 项全部通过，无 fallback |
| B200 / CC 10.0 | dense + packed FA4；active module=`flash_attn.cute` | self/cross attention 4 项与 online/static SM100 FP8 FFN 2 项全部通过 |

两边 Pod 的 kubelet `imageID` 均精确等于上述 top-level digest。

### 720p speed 门禁

固定合同：1248×704、KV45、SP1、BF16、`packed-fast`、
`performance_mode=speed`、packed deterministic=false、segment compile、
whole-DiT compile=false、20 warmup + 100 measured chunks。Job 预留整台 8 GPU
节点以隔离其他 GPU workload，推理进程只暴露 GPU0。checkpoint、首帧、cases
以及 12 个 donor 文件均按固定 VersionId/字节哈希验收；checkpoint-native
action encoder 为 `primitive_rope_token_residual`，Realtime API workload 合同为
`primitive_token_residual` / `primitive_float`，两者不可混写。

| GPU | backend | scheduler FPS | client FPS | 相对既有基线 | scheduler/client gap | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|
| H200 | locked FA3 | 9.501709058 | 9.490211946 | +20.32% / +20.27%（旧基线是 packed FA2） | 0.1210% | 61,803 MiB |
| B200 | FA4 | 14.167187296 | 14.149351395 | -1.588% / -1.582% | 0.1259% | 61,962 MiB |

两边都高于当次 97% 下限。H200 的旧基线来自 FA2，故从下一候选开始将本次
锁定 FA3 的 9.501709058 / 9.490211946 FPS 晋升为 scheduler/client 回归基线；
B200 保留更高的既有 14.395795593 / 14.376812178 FPS 基线，不因本次轻微下降
而放宽门禁。

结果哈希：

- H200 server log：
  `89c526ed330596e5facf11897d3923d5e39d09a72dea72833d6bca668b8998e6`
- H200 throughput JSON：
  `3c23184dedff448816da6cab67c439cdaa8f5d42c28ae46a9aa6b3b1b18f3082`
- H200 measured payload：
  `4b8f130364d419def7e80db1b9076624fa6b69aa7ed65d752311ca5a1ad81099`
- B200 server log：
  `01221dbec1cd127b72f0b095f7ef2361f0e7a1344659fca27443cdac4695ed7b`
- B200 throughput JSON：
  `f889d295f980240dc243493d82b53ee4156dc4313131b262834106e06ec5eab5`
- B200 measured payload：
  `f4f0500285a75a57f612d604dddc305f386c2b0a88d24c89adfccb675b970ce1`

ECR repository 已启用 immutable tags 与 scan-on-push；当前发布身份无
`DescribeImageScanFindings`，因此本记录不把“已读取漏洞扫描结果”列为已完成
门禁。
