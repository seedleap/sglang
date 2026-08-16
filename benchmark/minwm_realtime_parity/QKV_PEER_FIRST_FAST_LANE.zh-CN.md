# MinWM QKV projection 与 peer-first A2A fast lane

日期：2026-08-07；续跑：2026-08-14、2026-08-16

任务：S4 / 点子 6

状态：**6a+6b 的 H200 CUDA、micro、Nsight Compute、704p 最终视频质量、重复确定性、
SP2/SP4 20+200 反向 ABBA 均已完成。SP2 Client 为 −0.001%（噪声），SP4 Client 为
+1.695%、Scheduler +1.681%、同步 trace DiT wall −1.740%；所有重复 CV <0.7%，8 次 SP4
200-chunk payload SHA 完全一致。默认开启仍为 NO-GO（SP2 无收益且 DiT 未到 2%），但
作为两个独立、默认关闭的低侵入 fast lane 是 GO。latent/denoiser-forward dump、最终视频、
同机重复确定性和真实 prompt-switch 服务端轨迹均已通过；prompt-switch 的旧客户端统计
后处理失败单独记录。SP4 20+400 长跑完整通过，最后扩容后连续 272 chunk 显存水位不变、
无地址漂移/rank hang；不宣称 bitwise parity contract。**

## 2026-08-16 权威结果（后文早期“待执行”表为历史计划）

### 结论先行

6a 单独把每 chunk 的 self-QKV projection 从 450 个 GEMM 合并为 150 个，但新增 300 个
strided V materialization kernel，SP2 的 projection CUDA 基本不变。这个实测证据满足“V
copy/pack 是显著新瓶颈”后才实现 6b。6b 让原 peer-first Triton pack 直接读取 Q/K/V
stride，消除 V copy；Nsight Compute 证明 strided pack 本身只增加 0.03 µs，而每次省掉
7.39 µs copy kernel 和 launch。真实 SP2 micro 每次省 28.086 µs，150 次/chunk 的局部上限
是 4.213 ms/chunk。

端到端存在明确的 SP shape 分界：SP2 的 Client/Scheduler 完全不变，DiT 只改善 0.446%；
SP4 的 Client/Scheduler 改善约 1.7%，DiT 改善 1.74%。因此最终决策是：

- **6a 质量/确定性：GO；性能默认开启：NO-GO。**
- **6b 设计与实测依据：GO；作为与 6a 组合的 opt-in fast lane 保留。**
- **全局默认：保持两个开关均关闭。** SP2 主验收未满足“Client 不回退且 DiT >=2%”；
  SP4 有稳定收益但仍略低于 2% DiT 门槛。
- 保留理由不是“headline 达标”，而是代码边界小、输出 contract 不变、明确减少 300 次
  launch 和 2,635,776 B 临时 allocation，并为后续 graph/pack 布局解锁。任何 unsupported
  dtype/shape/backend 继续安全回原 eager/torch pack。

### 最终 profiler-off / 同步 trace A/B

统一为 MinWM 5B step-3200、1248×704、BF16、4 DMD + 1 clean-cache、rolling KV45。
每个位置均独立启动服务，20 warmup + 200 measured，反向 ABBA
`candidate/control/control/candidate`；Client/Scheduler headline 取
`SGLANG_REALTIME_TRACE_SYNC_CUDA=0`，DiT/VAE wall 只取同步 trace，后者不作 headline。

| SP | 指标 | control mean | 6a+6b mean | candidate 变化 | 两次重复最大 CV |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | Client FPS（profiler-off） | 12.830956 | 12.830823 | **−0.001%** | 0.153% |
| 2 | Scheduler FPS（profiler-off） | 12.845364 | 12.845324 | **−0.0003%** | 0.155% |
| 2 | DiT wall（同步 trace） | 746.276 ms | 742.948 ms | **−0.446%** | 0.203% |
| 2 | VAE wall（同步 trace） | 425.239 ms | 425.330 ms | +0.021% | 0.087% |
| 4 | Client FPS（profiler-off） | 9.889596 | 10.057258 | **+1.695%** | 0.165% |
| 4 | Scheduler FPS（profiler-off） | 9.897438 | 10.063838 | **+1.681%** | 0.159% |
| 4 | DiT wall（同步 trace） | 780.452 ms | 766.876 ms | **−1.740%** | 0.684% |
| 4 | VAE wall（同步 trace） | 423.614 ms | 423.528 ms | −0.020% | 0.036% |

SP2 的 no-sync raw `chunk wall` 曾显示 candidate 慢 1.681%，但同一组 Client/Scheduler
稳定到 0.001%，根因是 42 MiB raw RGB payload 的异步写回边界漂移，不能当模型时间；同步
trace 给出可归因的 DiT −0.446%。相反，SP4 两种测量都给出同方向的约 1%～1.7% 收益，
且所有 8 个 measured payload SHA 都是
`d6ece46ada98f531176a768cf85cb285b21b77e8e3a6b68a6b078390e4a1c434`。

SP2 产物：

- `pvc://default/minwm-s4-qkv-ncu-results-20260814-v03/results/attempts/`
  `minwm-s4-qkv-ncu-h200-20260814-03/perf-nosync/sp2/s4-qkv-abba-summary.json`，
  SHA256 `4ecb0d2bd85c85c5987addb344d87678469ca21ab1ac2f0806037735847f5236`；
- 同 root `perf/sp2/s4-qkv-abba-summary.json`，SHA256
  `e88a65623a75ede12986026c2adcd83ab0133466252c6f64a4f4db238ca1231d`。

SP4 产物位于同 PVC 的
`/results/attempts/minwm-s4-qkv-sp4-final-h200-20260816/`：

- `perf-nosync/sp4/s4-qkv-abba-summary.json`，SHA256
  `6c6757496217a0759d73c28459968d2c7b99a1d5098a1642fa687c2ce5413ef4`；
- `perf-sync/sp4/s4-qkv-abba-summary.json`，SHA256
  `00d108d5e6beb4d8f51240359c6306e7e8895637ea87907d1b9a6784a82fb487`。

两种 profiler-off 的峰值 sampling：SP2 control 48,843 MiB、candidate 48,867 MiB
（+24 MiB，为 1 秒采样/allocator 水位差）；SP4 两边均 49,397 MiB。它不是 micro 的精确
allocation 差；精确 micro 是 baseline +2,635,776 B、candidate +0 B。

### latent、最终视频、确定性与 prompt switch

新增 SP2、1248×704、BF16、KV45、8 chunk 的同机 control/candidate latent dump：

- 16 组 chunk latent，共 10,543,104 个元素：exact fraction 1.0、max_abs 0、RMSE 0；
- 82 组 denoiser forward 输出，共 53,044,992 个元素：exact fraction 1.0、max_abs 0、
  RMSE 0；
- 两边最终 lossless RGB frame SHA256 均为
  `b2a2f33343f013120ebddd8301c57a2f944fcd20ee01d91e247c568c50912a87`；
- `latent-quality-sp2/report.json` SHA256 为
  `f822fae0b0768e9f818343ee0d2ab56e0d3365afe7f9c8ce5d0013b0711a8659`。

旧节点 candidate 与新节点 candidate 使用同一代码、镜像、依赖、checkpoint 和输入，但跨
两台物理 H200 的 BF16 结果不是 bitwise：max_abs 7、RMSE 0.152662、exact fraction
0.976810、SSIM mean 0.999777 / min 0.999707，仍通过 max_abs <=8、RMSE <=1.0、
SSIM >=0.995 门槛。这个结果限制了确定性声明：**同一节点独立重启可 bitwise，跨物理节点
只声明质量门槛通过**。旧节点 control/candidate/candidate-restart 的 frame SHA 完全一致，
人工抽查 1/32/64/96/128 帧无可见差异；不把这些实测升级成通用 bitwise contract。

prompt-switch 用 832×480 回归 lane（不是 headline）发送 event `1101`：服务端 8 个 chunk
全部完成，切换首次且只在 chunk 1 应用，condition 同时包含
`minwm_condition_switch` 与 `minwm_prompt_updated`。服务端轨迹判定通过，摘要
`prompt-switch-regression-sp2/server-trace-summary.json` SHA256 为
`b3d122ba2f3eb4b8b4cba7105e7a53192401ff682de1274c0daa85fb0a29b147`。客户端已接收完整
输出，但旧 `run_sglang_api.py` 在收尾时仍索引已移除的 legacy `chunk_stats`，抛
`KeyError: 0`；因此客户端后处理明确记为失败，不能把它写成整条脚本通过。

### 420-chunk 饱和稳定性与地址复用

SP4、1248×704、BF16、KV45 候选使用 profiler-off 独立运行 20 warmup + 400 measured；
`realtime_session_max_lifetime_s=1800`，不把该单 lane FPS 当 paired headline。完整 run 的
PyTorch peak-memory 台阶为：

```text
chunk 0/1/2: 39884/40330/40436 MB
chunk 12/18/30/41: 40716/40718/40720/40722 MB
chunk 88/121/147: 40842/40844/41404 MB
chunk 148..419: 41404 MB（连续 272 chunk 无变化）
```

结构化断言核对 420 个 peak-memory sample、420 个 `server.chunk_complete`、420 个
`server.scheduler_forward_done`，索引均为 0..419；客户端 20+400、6,400 measured frames
完整，payload SHA256
`4e6399e5eb95cfa27efe18e3d7880858600a429dd503c740d6a041844dcf2ece`。逐秒 GPU memory
峰值为 rank 0..3 的 49,597 / 49,657 / 49,317 / 49,317 MiB；这是整进程采样峰值，allocator
判漏使用上面的逐 chunk 41,404 MB 水位。服务端 Traceback/CUDA/NCCL/session-lifetime 错误
计数均为 0，Pod 0 restart，无 rank hang。

`_MinWMUlyssesWorkspace` 的 peer-first input/output buffer data pointer 复用及布局 round-trip
门禁为 3 passed、123 deselected；结合长跑证明没有地址漂移或每 chunk 重分配。growing、
saturated、clean commit、cache eviction、prompt switch、scene cut、非均匀 shard 已由 14-test
稳定性 gate 和完整运行共同覆盖。结果在专用 PVC
`minwm-s4-qkv-stability-results-20260816`：
`/results/attempts/minwm-s4-qkv-sp4-final-h200-20260816/stability-long-sp4-candidate/`
`stability-summary.json`，SHA256
`36d4a3f9b318cf7d987a1efc5348d6c257de6a645e192fe82f8c8b1d2bdbe09f`。

### 来源、提交与独立开关

已有线索分支 `codex/minwm-s4-qkv-measure-25cc` 经审计后，只安全引入其两个 6a commit，
没有直接修改其他 worktree：

- `55f9a98b92`：6a single-GEMM QKV；
- `a342097595`：6a lint；
- `2d3b9ac852`：本任务 6b direct strided peer-first pack；
- `05158e858e`：真实布局 micro/NCU driver。

当前分支 `codex/minwm-qkv-fusion-benefit`，基线 `origin/main=d9d3d87fdb`。开关：

交付 checkpoint：H200 acceptance 文档提交 `a754257ab3`；独立 draft PR
[`seedleap/sglang#33`](https://github.com/seedleap/sglang/pull/33)。正式 H200 运行代码仍精确 pin
到 `05158e858e`，后续提交只更新实测记录与 PR 链接，没有改变被测 fast-path 代码。

| 开关 | 默认 | 作用 | 回退 |
| --- | --- | --- | --- |
| `MINWM_FUSED_QKV_PROJECTION` | `0` | 三个 Q/K/V GEMM 合一 | quantized、未知 linear/gather 布局自动三 GEMM |
| `MINWM_STRIDED_QKV_PACK` | `0` | peer-first Triton 直接读 strided V | 非 CUDA/HIP、unsupported shape/dtype 或 kernel 异常走既有 torch pack |

6b 只有和 6a 同时为 `1` 时才消除目标 V copy。开关均在 import/model build 前读取；不能在
同一进程中途切换。本项明确是 fast lane，不把本次 bitwise 实测提升为 bitwise contract。

### 2026-08-16 H200 Job 与硬件 profile

按 `HARDWARE_PROFILES.md` 从唯一模板
`k8s/minwm_hardware_job.template.yaml` 生成忽略文件
`k8s/generated/minwm-s4-qkv-sp4-final-h200-20260816.yaml`，两次均先执行：

```bash
kubectl --context codex-minwm-test-phx2 apply --dry-run=server \
  -f benchmark/minwm_realtime_parity/k8s/generated/minwm-s4-qkv-sp4-final-h200-20260816.yaml
kubectl --context codex-minwm-test-phx2 apply \
  -f benchmark/minwm_realtime_parity/k8s/generated/minwm-s4-qkv-sp4-final-h200-20260816.yaml
```

检测为 4×NVIDIA H200、每卡 143,771 MiB、compute capability 9.0，节点
`i-0976fa6e9175c7fe5`；Spot 回收后的长稳定性节点为 `i-0cf2fb89f9840c963`，硬件门禁相同。
SM90 不属于仓库两种 SM120 stable profile，因此明确标记
`experimental-h200-143g`；本 workload 实测峰值约 49 GiB，`vae_cpu_offload=false`。
仓库 profile 文档的 KV32/sink8 是另一部署合同；本任务的用户验收合同 KV45 优先，日志已
核对 `request_window_size=45 allow_growth=False`，不把 H200 manifest 宣称为生产 profile。

H200 CUDA gate 实际命令及结果：

```bash
PYTHONPATH=python pytest -q \
  python/sglang/multimodal_gen/test/unit/test_minwm_qkv_projection.py \
  test/registered/jit/diffusion/test_ulysses_qkv_pack.py
# 21 passed, 21 warnings in 3.12s

PYTHONPATH=python pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py \
  -k 'prompt_switch or scene_cut or raw_k_cache_overwrites or cache_plan_is_shared \
      or cache_append_does_not_repack or fixed_shape_metadata or default_kv_horizon \
      or model_bounded_window or unbounded_kv_policy or nonuniform_sp8 \
      or causal_attention_packs_one_ulysses_collective'
# 14 passed, 112 deselected
```

### 2026-08-16 新增失败记录

- 默认 kube context 是 `codex-seed-leap-use1`，首次查询 v03 PVC 得到 NotFound；核对 context
  列表后改用 `codex-minwm-test-phx2`，PVC `Bound` 且旧数据完整。错误查询不是产物缺失。
- 模板的 `nvidia-smi ... | head -n1` 在 4 GPU + `set -o pipefail` 下令 producer 收到
  SIGPIPE，首个 Job 在写完 `hardware.csv` 后 exit。把本次忽略 manifest 改为
  `sed -n '1p'`，精确删除失败 Job、再次 server dry-run 后通过；没有改 committed template。
- 本地重跑一度用了不存在的旧测试路径
  `python/.../test_ulysses_qkv_pack.py`，ruff/pytest 均报 file not found；修正为
  `test/registered/jit/diffusion/test_ulysses_qkv_pack.py`。随后直接在无 CUDA 的 macOS 运行
  registered CUDA test 得到 8 个 `Torch not compiled with CUDA`，该组不是失败的 H200
  数据点；H200 正式命令为上述 21 passed。
- 最终本地 ruff 命令又误用了不存在的旧模块路径 `runtime/distributed/{usp,triton/qkv_pack}`，
  得到两个 E902；用 `rg --files` 核对后改为 `runtime/layers/usp.py` 与
  `jit_kernel/diffusion/triton/ulysses_qkv_pack.py`，最终 ruff、compileall、
  `git diff --check` 全部通过，CPU QKV 单测为 12 passed、1 deselected。
- 4×H200 Job 的 6 小时 deadline 在性能 ABBA 全部落盘后触发，Pod 被 controller 删除；
  第一次复制 latent runner 因旧 Pod NotFound 失败。`/results` PVC 的性能产物完整，`/work`
  emptyDir 丢失。随后把同一忽略 manifest 的 deadline 改为 2 小时，重新 dry-run/apply 和
  staging，只补 latent/prompt-switch，不重复或混用性能 run。
- 第一次 2 小时重建的 pip build dependency 下载遇到 PyPI HTTP 502，setup 失败；在忽略
  manifest 中加入最多 3 次的有界 setup retry 后成功。PVC 上还残留前一 Pod 的 `ready`
  marker，曾让只读轮询提前报告 ready；没有据此启动测试，随后在每次 setup 开头精确
  `unlink` 该 marker，只有新 setup 完成后重建。
- latent/forward dump 完成后，prompt-switch 服务端完成 8 chunk，但客户端收尾因 legacy
  `chunk_stats` 缺失报 `KeyError: 0`；服务端结构化轨迹独立生成并校验，客户端失败原样保留。
- 追加饱和显存长跑前，H200 Spot 节点 `i-0976fa6e9175c7fe5` 对主 Pod 与只读 Nsight Pod
  同时报 `Evicted: Forceful Termination`；长跑尚未启动，PVC 已落盘数据无损。精确删除两个
  Job 后重新 server dry-run/apply，仅重建主 Job 执行长跑。
- 新节点第一次重建持续报旧 v03 RWO 盘 `VolumeInUse`；本机尝试只读
  `aws ec2 describe-{instances,volumes}` 又因没有 AWS credentials 失败，未绕过权限或强制
  detach。为不覆盖/删除旧证据，创建专用 PVC
  `minwm-s4-qkv-stability-results-20260816`（100 GiB，同 storage class），先 server dry-run
  再 apply；首次修改忽略 manifest 时因 URI 实际带引号导致 `apply_patch` context mismatch，
  无文件改变，核对原文后重试成功。新盘在新 H200 节点成功 Bound，只保存长稳定性结果。
- 首条 20+400 长跑沿用了 server 默认 `realtime_session_max_lifetime_s=600`，在约 600 秒、
  chunk 371 被 watchdog 正常以 WebSocket 1000 关闭；客户端报
  `ConnectionClosedOK: maximum session lifetime reached`，没有 `complete` 标记。它已证明
  chunk 148–370 共 223 个连续 chunk 的 allocator 水位不变，但仍按未完成处理，目录保留为
  `stability-long-sp4-candidate-failed-max-lifetime600`。两次期间的只读进度查询还瞬时收到
  `kubectl EOF`，Pod 随后确认 0 restart 且 run 继续；两者都未写成通过。重试显式设置
  `--realtime-session-max-lifetime-s 1800`，完整 420 chunk 通过，不修改候选代码。生成 summary
  前第一次 `kubectl exec` heredoc 忘记 `-i` 因而无输出、无写入；补 `-i` 后完成硬断言。
- setup 的 pip resolver 报 `open-clip-torch/timm` 与 `wandb/protobuf` 警告；这两个包不在
  本 MinWM serving/测量调用链，服务成功加载并完成 200 chunks，故记录为环境 warning，
  未把它当成通过项或静默删除。

## 2026-08-14 续跑：预估、证据与当前候选

已审计的 v12/v13 exact-window Nsight（H200、BF16、1248×704、KV45）给出以下每 rank、
每 stable chunk 结果：

| SP | lane | projection GEMM | projection CUDA / trace wall | V layout CUDA / wall | peer pack CUDA | A2A wall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | control | 450 | 19.303 / 33.317 ms | 0 / 0 | 3.221 ms | 11.849 ms |
| 2 | 6a | 150 | 19.305 / 19.305 ms | 2.802 / 3.541 ms | 3.099 ms | 11.765 ms |
| 4 | control | 450 | 11.363 / 30.063 ms | 0 / 0 | 1.660 ms | 12.469 ms |
| 4 | 6a | 150 | 10.359 / 10.359 ms | 1.713 / 2.675 ms | 1.435 ms | 12.255 ms |

这证明 6a 少掉的 300 个 GEMM launch 被 300 个 V materialization kernel 完全补回；SP2
projection CUDA 本身没有下降，收益只来自约 14 ms 的 launch span。若 6b 只消除 V copy，
理论上限约为 SP2 DiT 0.50% / SP4 0.36%。提交前先验预估为 SP2 Client +0.2%～+0.6%；
若 strided pack 同时减少同步/launch gap，目标区间为 DiT +1%～2%、Client +0.5%～+1.2%。
这些是预估，不是实测。

当前 6b 候选只做三件事：Triton pack 接受 Q/K/V 的四维 stride；USP CUDA 路径在
`MINWM_STRIDED_QKV_PACK=1` 时允许 non-contiguous 输入；fused QKV 不再执行
`value.contiguous()`。6b 开关独立且默认 `0`；只有同时启用 6a/6b 才进入目标路径。
它不修改 GEMM backend、collective、wire layout、Q/K norm、RoPE 或 cache。CPU 逻辑测试已覆盖真实 strided QKV
view，15 passed、1 CUDA skipped；随后 CUDA kernel、质量和性能已按顶部结果在 H200 完成。

基础设施失败也保留：第一次 reader 复用旧固定 hostname `i-06888dc1ca88547e1`，但该节点
已被回收，Pod 90 秒 Ready wait 超时后被精确删除。随后提交的 Phoenix Local Zone Spot
2-GPU Pod `minwm-s4-qkv-ncu-h200-20260814-01` 被 Karpenter nominate 后因当时无 Spot
capacity 保持 Pending。当准备切到 us-west-2c、已创建 `...-02` 和 v02 PVC 时，Phoenix
节点意外 Ready；为避免同时占两台 H200，只删除了本任务尚 Pending 的 v02 Pod/PVC，
没有抢占或修改其他任务。最终运行 Pod 为 `minwm-s4-qkv-ncu-h200-20260814-03`，节点
`i-018819c4fb79acba6`，2×H200、0 restart；结果 PVC 为
`minwm-s4-qkv-ncu-results-20260814-v03`。旧 v12/v13 PVC 只做读取，未改内容。

### 2026-08-14 H200 环境与 CUDA gate

- 容器：`829115.../minwm-training@sha256:bedc07ea...53ef5f2a`；
- SGLang：`05158e858e09271d76408b92b40fd28a78ff8444`；运行目录因 setup 脚本移除未使用
  Rust extension，只留下预期的 `python/pyproject.toml` runtime dirty，不回写本 worktree；
- MinWM：`2efc6485f65e8fcab506665efde79bc41406385e`；
- converted checkpoint SHA256：
  `1dc42d498cad84349987db2015120ce4d77e6b641f7f38c75ec9df3f942a7975`；
- setup 日志：`/results/attempts/minwm-s4-qkv-ncu-h200-20260814-03/setup.log`；
- CUDA gate：QKV unit + registered Triton peer pack，`21 passed, 21 warnings in 2.85s`；
  日志为同目录 `cuda-qkv-gate.log`。

核心 CUDA gate 命令为：

```bash
pytest -q \
  python/sglang/multimodal_gen/test/unit/test_minwm_qkv_projection.py \
  python/sglang/multimodal_gen/test/unit/test_ulysses_qkv_pack.py
```

### 真实布局 micro 与 Nsight Compute

micro 使用 SP2 真实布局 `[1,429,24,128]`、BF16、stride
`[3953664,9216,384,1]`，100 warmup + 5000 measured：

| 路径 | wall / 次 | 5000 次期间额外 peak allocation |
| --- | ---: | ---: |
| 6a：`V.contiguous()` + contiguous pack | 64.650 µs | 2,635,776 B |
| 6a+6b：direct strided pack | 36.564 µs | 0 B |
| 差值 | **−28.086 µs（−43.44% runtime）** | **−2,635,776 B** |

每 chunk 150 次对应约 **4.213 ms/chunk** 的局部 wall 上限；输出 bitwise equal。产物：
`micro/sp2-pack-micro.json`，SHA256
`15ba666ea6013080876af12b28cea5a9c3796dfbe0c7f2c693ca45ceb83c5ced`。kubectl stdout
在 JSON 已落盘后发生 TCP read timeout，随后确认 pod 内进程已结束且 SHA 正确；该连接错误
不改变测量，但已保留，不能把终端 exit 1 写成测试通过。

复现命令：

```bash
python3 benchmark/minwm_realtime_parity/profile_strided_qkv_pack.py \
  --device cuda --dtype bf16 --sp-degree 2 --warmup 100 --iterations 5000 \
  --output /results/attempts/minwm-s4-qkv-ncu-h200-20260814-03/micro/sp2-pack-micro.json
```

Nsight Compute 2025.1.1 使用 detailed set、kernel replay 和 NVTX filter；profiler 下 wall
不作 headline：

| lane / kernel | Duration | DRAM Throughput | Memory Throughput | Compute (SM) | SM Busy | Achieved Occupancy | L2 hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline V copy | 7.39 µs | 7.35% | 11.94% | 30.80% | 44.82% | 67.39% | 55.54% |
| baseline pack | 7.55 µs | 21.41% | 27.27% | 23.75% | 33.08% | 74.55% | 52.37% |
| candidate strided pack | 7.58 µs | 21.43% | 27.20% | 23.79% | 33.47% | 72.58% | 52.60% |

strided 读取只给 pack 增加约 0.03 µs，实质消除一枚 7.39 µs copy kernel 和一次 launch；
短 kernel/launch wall 才是 micro 28.09 µs 差值大于 CUDA duration 的原因。报告和 SHA：

- `ncu/sp2-baseline.ncu-rep`：5,542,269 B，
  `cf858d2c44f22c1c3ede2d375c889b1ba151340ed05ee216aaf11b4d5b607e4a`；
- `ncu/sp2-candidate.ncu-rep`：211,720 B，
  `ab646fe671b338b5d3dff8a305778d80e42089eb9da9f8eaee25386fe39260a1`；
- `ncu/sp2-detailed-summary.json`：
  `04b128579e4d501b7233eef2a95a9653274581699944a86d1b383bc5031b1c54`。

容器没有 `jq`，第一次解析命令失败后改用 Python CSV reader；一次直接输出 NCU raw CSV
产生过大且被终端截断，之后只从 `.ncu-rep` 定向提取上述 metric。这些失败没有作为数据点。

v12/v13 Nsight Systems 是 6a-only 的独立 20 precondition + 1 discard + 10 stable capture，
包含所有 active rank；profiler 下 FPS 不作 headline：

| SP/lane | DiT CUDA | VAE CUDA | kernels / launches（全 rank/chunk） | <10 µs | 10–50 µs | GPU busy | SM Active | Tensor Active | DRAM read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SP2 control | 739.788 ms | 442.967 ms | 34,608 / 34,608 | 18,217 | 12,283 | 76.84% | 61.79% | 28.35% | 8.30% |
| SP2 6a | 713.533 ms | 442.617 ms | 34,608 / 34,608 | 18,554 | 11,661 | 77.90% | 63.56% | 29.21% | 8.59% |
| SP4 control | 788.943 ms | 254.195 ms | 69,202 / 69,202 | 49,085 | 13,416 | 67.45% | 37.31% | 15.72% | 4.34% |
| SP4 6a | 743.812 ms | 254.479 ms | 69,202 / 69,202 | 50,145 | 11,760 | 66.51% | 38.34% | 16.24% | 4.48% |

总 kernel/launch 数没有随 450→150 GEMM 下降，正是因为 6a 又增加 300 次 V copy；短 kernel
只在 bucket 间重新分布。SP2/SP4 的 SM/Tensor Active 略升但绝对值仍低，尤其 SP4 的
Tensor Active 仅约 16%，证明 rank-local short-M/launch 主导而非 FLOP 饱和。DRAM read
几乎不变，A2A wall 仍是 11.765–12.255 ms；因此 6b 应消除 copy/launch，不能期待算力或
通信级跃升。Nsight summary SHA256：control v12
`8cdb183996ca00580f8b98e6c878520283da3e8a046711fce90dd30239b20a97`，candidate v13
`4107adb1382587940e84cff2e7cdea2734b3586e265a17a6bcdeaef82842336f`。

### 704p 最终视频质量与重复确定性

SP2、1248×704、BF16、KV45、seed 42，control、candidate 和 candidate 重跑都独立重启
服务，生成 8 chunk / 129 lossless RGB frames。候选日志明确包含
`single-gemm-fast-lane` 以及 `strided_qkv_pack=True`。三份 NPY 均为 340,015,232 B、SHA256
`38e7ef07cffb7e8df2e59323dcbd9dacda92d31ab4a268d1276b554b7f3e833b`：

- control vs candidate：bitwise equal，max_abs 0，RMSE 0，SSIM 1.0；
- candidate restart vs candidate restart：bitwise equal，重复确定性通过；
- 人工抽查 frames 1/32/64/96/128，baseline/candidate 无可见差异，16× diff 全黑。

结果位于 `quality-control-vs-candidate/report.json`、
`quality-candidate-determinism/report.json` 和 `quality-control-vs-candidate/montage.png`。
第一次服务清理 `pkill -f` 匹配到当前 shell，命令以 143 结束；第二次把进程检查和后续
server 命令放在同一个 shell 仍自匹配为 `stale-server`；另一次检查发现容器没有 `rg`。
三次均发生在 GPU 推理前，拆开只读检查和启动后完成了上述有效结果。

### profiler-off 20+200 执行记录（最终聚合见顶部）

SP2 使用反向 ABBA `candidate control control candidate`，每个位置独立启动服务，显式
`--realtime-session-idle-timeout-s 1800`，profiler 环境变量 unset，并逐秒采集 GPU
utilization、SM clock、power、temperature 和 memory。第一次 position-01 没有覆盖当前主线
60 秒 session idle 默认值，在 chunk 50 被服务端以 close 1000 正常关闭；无 200-chunk
结果，部分产物保留在 `perf/sp2/position-01-candidate-invalid-idle60`，不会进入聚合。一次
本地进度查询因 zsh 展开未转义 glob 而未执行，也不构成测量。

## 结论与开关

6a 是显式 fast lane，不是 bitwise parity 修复。设置
`MINWM_FUSED_QKV_PROJECTION=1` 后，每个 MinWM self-attention block 用一个
`to_qkv` 线性层取代 `to_q`、`to_k`、`to_v` 三个线性层。权重只在模型构造/加载时
稳定打包；forward 只执行一次 linear 后用 view/chunk 切分，绝不在 forward 中拼权重。

默认值是 `0`。回滚只需取消该环境变量或设为 `0`，无需转换 checkpoint。量化权重、
未知 linear 子类和不安全的 column-parallel gather 布局会告警并自动走原三 GEMM 路径。

单 GEMM 可能因为 BF16 GEMM shape、reduction/bucket 或 cuBLASLt 算法选择变化而产生
数值差异。因此在完成 layer probe、latent、最终视频质量和确定性 A/B 前，本开关不会
作为默认路径。本次 H200 实测通过质量门槛，但这不改变 fast lane 的数值 contract。

## 范围和非目标

本改动只触碰 self-attention QKV projection 及其到既有 peer-first pack 的布局边界：

```text
norm hidden states
  -> [6a] one to_qkv GEMM
  -> q/k/v views
  -> existing Q/K norm and RoPE/cache logic
  -> V contiguous for 6a；6b opt-in 直接保留 strided view
  -> existing peer-first Triton pack
  -> existing input A2A / attention / output A2A
```

以下不在 6a 中修改：RMSNorm、RoPE、KV cache 的数值/所有权语义，attention backend，
A2A collective，cross-attention QKV 和 FFN。6b 的 GEMM epilogue、定制输出布局或支持
strided V 的 pack kernel 必须等 6a profile 后再决定。

## 假设与预期

MinWM 5B 的 hidden size 是 3072。原路径的 self-attention projection 是三个
`M x 3072 @ 3072 x 3072` GEMM；6a 变为一个
`M x 3072 @ 3072 x 9216` GEMM。1248x704、4 latent frames/chunk 的 nominal token
数为 3432，因此 sequence shard 后大致是：

| SP | 每 rank 的 nominal M | 原输出 N | 6a 输出 N |
| ---: | ---: | ---: | ---: |
| 1 | 3432 | 3 次 3072 | 1 次 9216 |
| 2 | 1716 | 3 次 3072 | 1 次 9216 |
| 4 | 858 | 3 次 3072 | 1 次 9216 |

每个 chunk 有 30 blocks ×（4 DMD + 1 clean-cache）= 150 组 self QKV。忽略启动、
图捕获和后端融合时，projection GEMM kernel 的理论计数从 450 降到 150。预期收益来自：

- 每 chunk 少 300 次 GEMM launch；
- SP2/SP4 较短 M 下，用更宽的 N 聚合工作，可能提高 Tensor Core 利用率；
- 权重总元素数和数学 FLOPs 基本不变，所以如果原 GEMM 已经足够大且 Tensor Active
  已饱和，收益可能很小；
- A2A 数量和 payload 不变，所以通信主导时 headline FPS 不会按 kernel 数同比提升。

一个已知抵消项是 V：packed GEMM 的 Q/K/V 是最后一维的三个 view。Q/K norm 会产生
连续输出，但 V 仍是 strided view。SP2/SP4 的既有 Triton peer-first pack 要求三者连续，
所以 6a 在该边界显式做一次 `value.contiguous()`。这会增加一个 copy kernel；它是 6b
要用数据判断是否值得消除的主要候选，不应隐藏在“单 GEMM”收益里。

## 实现与兼容性

实现位于
`python/sglang/multimodal_gen/runtime/models/dits/minwm.py`：

- `MinWMCausalTransformerBlock.__init__` 在开关打开且布局安全时构造一份物理
  `ReplicatedLinear(3072, 9216, output_sizes=[3072] * 3)`；安全的非 gather
  column-parallel 路径使用 `MergedColumnParallelLinear`。
- 创建 `to_qkv` 后立即删除三个旧 module，因此不会同时常驻两份权重。
- `_project_qkv` 的 fast 分支只有一次 `self.to_qkv(hidden_states)` 和
  `qkv.chunk(3, dim=-1)`；权重 merge 只发生在 loader/pre-hook。
- 原生 checkpoint 的 `blocks.N.self_attn.{q,k,v}.*` 先沿用现有映射到
  `blocks.N.to_{q,k,v}.*`，再按固定 q/k/v 顺序在 load 时合到
  `blocks.N.to_qkv.*`。
- 普通 `load_state_dict` 的 block pre-hook 支持 split state dict -> fused model，也支持
  fused state dict -> fallback model。
- component/FSDP loader 的 `preprocess_loaded_state_dict` 支持把保存的 fused state dict
  拆回 fallback keys。设备和 dtype 移动由注册的单一 parameter 自然继承。
- parity dump 打开时，fused hook 仍按原文件名导出 Q/K/V 输出和 Q 权重，便于做同层对照。

state_dict 的 key 合同是“跨开关可加载”，不是“开关两边 key 文本相同”：

| 模式 | self-attention key |
| --- | --- |
| 默认/fallback | `blocks.N.to_q.*`、`to_k.*`、`to_v.*` |
| 6a fast lane | `blocks.N.to_qkv.*` |

所有非空 `quant_config` 目前都保留原格式感知的三个量化 module。这是有意的安全 fallback：
独立 Q/K/V 可能各自带 scale、zero point 或 packed metadata，未经对应格式的真实 checkpoint
验证，不把三个量化 parameter 强行拼成一份。本次 H200 真机验证 BF16；static FP8/NVFP4
没有可用的同 checkpoint 实测，只验证 `quant_config` 非空时安全 fallback，不伪装成量化快路。

当前 MinWM causal Ulysses 本来就拒绝 TP>1 与 SP>1 组合，也拒绝 SP>1 + whole-DiT
`torch.compile`。本 PR 不扩大这些既有并行边界：SP2/SP4 在 TP1 验证，TP2 在 SP1 做
兼容 smoke；SP1 运行 compile off/on，SP2/SP4 的 compile-on 继续验证为明确拒绝。

## 本地验证边界

本地结果只证明 CPU 语义和静态正确性，不代表 CUDA、Triton、BF16 或真实 compile：

| 环境/检查 | 结果 | 边界 |
| --- | --- | --- |
| Codex Python 3.12.13 `compileall` + AST | 通过 | 无 torch/pytest |
| ruff format/check、`git diff --check` | 通过 | 静态检查 |
| Python 3.11.13、torch 2.13.0、pytest 9.1.1 | 12 passed，1 skipped | CPU 语义；CUDA compile 用例按设计跳过 |

macOS 本地 torch 在导入仓库的 eager `torch.compile` decorator 时会触发其自带
Inductor/Triton typing 错误。CPU 回归使用 `/tmp/codex-minwm-s4-cpu-site` 中的临时
`sitecustomize` 将 `torch.compile` 替换为 identity，并只在该临时目录补了 `uvicorn`；
仓库环境没有改变。真实 `torch.compile` 用例已包含在 H200 的 21-test CUDA gate 中。

本地测试覆盖：真实 block 只有一个物理 QKV parameter、forward projection 无 cat、
多维/多 sequence shape、SP1/SP2/SP4 peer-first 线布局、原生与内部 checkpoint key、
跨开关严格 load、保存后反向加载、dtype move、量化 fallback。GPU 侧已覆盖 BF16、registered
Triton contiguous/strided 多 shape 与真实 compile；层/latent dump 最终结果见顶部续写。

## H200 验收矩阵（2026-08-07 历史计划；权威状态见顶部）

统一 workload：MinWM 5B step-3200，1248x704，BF16，4 DMD + 1 clean-cache，
16 frames/chunk。headline 必须是 profiler-off 的 20 warmup + 200 measured；Nsight 在
外部 20 chunks precondition 后丢 capture session 首 chunk，保留至少 10 个 steady chunks，
且不与 torch.profiler 同跑。

所有 control/candidate 的正式稳态 run 固定
`MINWM_S0_KV_CACHE_NUM_FRAMES=45`（client 等价参数
`--kv-cache-num-frames 45`），不得随 `max_chunks` 扩张。这是 rolling-window
steady-state contract。首块、短程 append/recompute、cache growth 和尚未发生淘汰的
数值行为另跑短程质量检查，不与 20+200 headline 混合归因。

| 项目 | control | candidate | 状态 |
| --- | --- | --- | --- |
| SP2 主验收 | `MINWM_FUSED_QKV_PROJECTION=0` | `=1` + 6b | 完成；Client −0.001% |
| SP4 复验 | `=0` | `=1` + 6b | 完成；Client +1.695% |
| SP1 eager / compile | `=0` | `=1` | CUDA compile unit 通过；未跑 headline |
| TP2 + SP1 smoke | `=0` | `=1` | 非本次 headline；loader/layout unit 通过 |
| static FP8 | 原量化三 projection | 请求 6a 后安全 fallback | 逻辑 fallback；无真 checkpoint 实测 |
| NVFP4/不支持设备 | 原设备合同 | 同样拒绝或 fallback | 逻辑 fallback；无 H200 快路声明 |
| layer/forward probe | Q/K/V、norm 后、block output | 同输入/权重/seed | denoiser-forward dump exact；未单列每层 raw Q/K/V |
| latent/最终视频 | lossless latent 与 frame metrics | 同 case/seed/backend | 完成；latent/frame 同机 exact，跨节点通过质量门槛 |
| 确定性 | candidate 重复运行 | candidate 重复运行 | 完成；frame SHA bitwise equal |

质量先遵循现有 contract：parity lane 要求 bitwise；本 BF16 fast lane 至少满足
`max_abs <= 8`、`RMSE <= 1.0`、`SSIM >= 0.995`。是否接受即便门槛内但有可见时序漂移，
仍需结合 latent、最终视频和 deterministic replay 审阅，不只看 FPS。

## S0 测量契约与复现

H200 临时测量分支只允许临时引入：

- S0 branch：`origin/codex/minwm-fused-ops-s0`
- S0 commit：`e75e9e24b5`（包含 `411d9b9ec4`）
- draft PR：#19

旧的 `30cb16708f` / `8e06ab2fc3` / `411d9b9ec4` 不再作为 clean runner 的最终 pin。
S0 未合并前，测量分支可以在 S4 实现 commit 上 cherry-pick `e75e9e24b5`；S4 PR 对
main 的最终 diff 必须移除 S0 基础设施。
入口使用 `benchmark/minwm_realtime_parity/run_s0_measurement.sh`，结果再经同一 commit 的
`measurement_tool.py` validate/merge-nsys/aggregate。

若 raw capture 是由 `411d9b9ec4` 启动，可以保留 `.sqlite`，但最终 JSON 必须用
`e75e9e24b5` 的工具重新 merge，并记录实际 checkout SHA，不能沿用旧 schema 的 JSON。

每个 JSON 必须记录实际 SGLang SHA、minWM SHA、镜像、GPU、SP、精度和 UTC 时间。
`provenance.gpu.count` 是 active GPU（SP2=2、SP4=4），整机隔离的 8 卡写入
`allocated_count=8`。Nsight kernel/短 kernel 保留 raw total、per-device、
per-stable-chunk；CUDA API/launch 的精确字段是 `raw_total`、`total_per_chunk`、
`per_rank_per_chunk`。只有 SQLite 能证明覆盖全部 rank 时，最后一项才 available；否则
写 unavailable 和 evidence。

S0 runner 的 `MINWM_S0_KV_CACHE_NUM_FRAMES` 必须显式记录为 45；control 和 candidate
必须完全相同。若结果使用了增长到完整 200-chunk horizon 的 cache，该结果不具备本任务
headline 资格。

profiler-off server 不打开 layerwise NVTX。单独的 Nsight server 打开
`--enable-layerwise-nvtx-marker`，用 `to_q/to_k/to_v` 或 `to_qkv` range 归因 projection；
peer-first Triton kernel 和 NCCL A2A 按 kernel/API 名归因。NVTX 只用于 profile 证据，
不进入 headline 路径。

## 实际 A/B（早期空表；已由顶部最终表取代）

### Provenance

| 字段 | control | candidate |
| --- | --- | --- |
| SGLang commit | `05158e858e` | `05158e858e` |
| minWM commit | `2efc6485f6` | `2efc6485f6` |
| container image | `sha256:bedc07ea...53ef5f2a` | 相同 |
| GPU active / allocated | SP2=2 / SP4=4 | 相同 |
| SP / precision / UTC | 2/4 / BF16 / 2026-08-14~16 | 相同 |

### Profiler-off headline（20 + 200）

完整均值、delta、CV 与 profiler 边界见“2026-08-16 权威结果”，不在此重复一张可能漂移的表。

### Nsight steady state（20 precondition + 1 discard + >=10）

该历史空表已被顶部“真实布局 micro 与 Nsight Compute”及 v12/v13 Nsight Systems 实测表
取代；其中保留 DiT/VAE CUDA、kernel/launch、短 kernel、GPU busy、SM/Tensor Active、
DRAM、A2A 与证据 SHA。profiler 下 FPS 没有被用作 headline。

## 与预期不符处

macOS torch 2.13.0 会在 compile decorator 初始化时失败，因此本地 compile 明确跳过；
转移到 H200 后通过，这不是实现失败。

实测低于先验的原因已经闭环：GEMM kernel 数确实 450→150，但 6a 新增 300 个 V copy，
SP2 projection CUDA 19.303→19.305 ms，省下的主要是约 14 ms launch span；6b 消除 V
copy 后，micro 上限仍只有 4.213 ms/chunk，按约 746 ms DiT 的 Amdahl 上限约 0.56%，实测
0.446%吻合。A2A 维持约 11.8–12.3 ms且 payload/collective 数不变，VAE 完全不受影响。
SP4 的 rank-local M 更短，launch/调度占比更高，因此相同 kernel reduction 转化为 1.74%
DiT 与 1.70% Client；没有证据支持 5% 级收益。BF16 输出在本 workload 实测 bitwise equal，
未观察到 GEMM bucket 引发的 video 漂移，但仍不提升为 bitwise contract。

## 证据与决策过程

1. 先选择“构造/加载时只有一份 packed parameter”，避免 forward cat 和双份权重常驻。
2. 保留环境变量默认关闭，因为 GEMM shape 改变本身就是数值契约变化。
3. 所有量化格式先 fallback；等格式逐一证明 packed scale/metadata 合同后才可能放行。
4. SP fast path 先显式 materialize V，保证继续命中已有 peer-first Triton pack；这让 6a
   的收益/代价可单独归因。
5. 只有 6a 后 pack + V copy 仍占显著 steady kernel/wall，并且可维护实现有正收益时，
   才进入 6b。若收益落入噪声、A2A 主导或实现需要侵入 GEMM backend，6b 结论就是“不做”。

## 尝试后放弃或暂缓的方案

- **forward 中 `torch.cat([Wq, Wk, Wv])`**：每次重复分配/复制权重，直接违反目标，未采用。
- **保留三个 module，再在 post-load 复制一份 fused weight**：会常驻双份约 3×3072²
  参数，并给 FSDP/device move/save 制造双源真相，未采用。
- **直接融合所有量化 QKV**：独立 scale/packed metadata 未验证，改为安全 fallback。
- **在 6a 同时写 GEMM epilogue/定制布局**：无法区分 GEMM 聚合与布局优化收益，暂缓到 6b。
- **为 profile 默认加入 NVTX**：会污染 profiler-off headline，改成只在独立 Nsight 运行打开。

## 6b go/no-go

6b 当前为 **MINIMAL IMPLEMENTATION / H200 VALIDATED / OPT-IN GO**。用户在 2026-08-14
明确要求继续推进；以下保留条件已经用顶部证据核对：

1. 6a 已通过兼容性和质量门槛；
2. SP2 主验收中 V copy + peer-first pack 仍占可重复的显著 DiT CUDA 或 wall；
3. profiler 证明瓶颈不是 A2A wait 或其他串行阶段；
4. 候选方案能保持 fallback 和清晰接口，不要求维护私有 GEMM backend fork；
5. 6b 独立 A/B 在 profiler-off 也有收益，而非只在 Nsight 下好看。

不满足时不实现或回滚 6b，并在本节保留测量证据。

## 风险、回滚与验收状态

主要风险是 BF16 数值轨迹、FSDP/TP packed load、compile graph、量化 metadata 和 V copy
抵消收益。回滚路径始终是 `MINWM_FUSED_QKV_PROJECTION=0`；保存于任一开关下的 state
dict 都有反向加载路径。若 fast lane 启动日志没有出现 `single-gemm-fast-lane`，该次结果
必须视为 fallback，不得计入 candidate。

当前验收状态：

- 6a 实现/CPU 语义/静态检查：通过；
- H200 BF16、SP2/SP4、CUDA compile、量化 fallback：通过（量化只声明 fallback）；
- latent/denoiser-forward/final video/determinism：通过；同机 exact，跨节点通过质量门槛；
- profiler-off / Nsight A/B：通过并完成归因；
- SP4 20+400 饱和稳定性：通过；最后扩容后 272 chunk 水位不变，地址复用 gate 通过；
- 6b：独立默认关闭的 strided pack 已实现，H200 SP4 有收益、SP2 为噪声；
- 默认开关：保持关闭。

## 给负责人掌握代码的检查题

1. **开关在哪里读，默认是什么？**

   `minwm.py` 顶部 `_MINWM_FUSED_QKV_PROJECTION`，默认 `False`。

2. **为什么说没有 forward cat？**

   `_project_qkv` fast 分支只调用 `to_qkv` 后 `chunk`；cat 只在 loader/pre-hook 合并权重。

3. **原生 `self_attn.q.weight` 如何进入 fused parameter？**

   `_minwm_fused_qkv_param_names_mapping` 先复用现有 `self_attn.q -> to_q` 链，再按
   merge index 0/1/2 合到 `to_qkv.weight`。

4. **fast state_dict 为什么能被默认模型加载？**

   普通 `load_state_dict` 由 `_minwm_qkv_load_state_dict_pre_hook` 拆分；component/FSDP
   loader 由 `preprocess_loaded_state_dict` 在 name mapping 前拆分。

5. **peer-first wire layout 是什么？**

   `_usp_pack_peer_first_qkv` 输出
   `[destination_peer, batch, local_sequence, local_heads, 3 * head_dim]`，最后一维为 Q/K/V。

6. **为什么 6a 仍有一次 V copy？**

   packed GEMM 的 V 是最后一维 strided view；Q/K norm 会物化，V 不会。SP>1 时显式
   contiguous 才能继续命中现有 Triton pack。这是 6b 的独立候选，不是隐藏融合。

7. **为什么不能把 single GEMM 称作 bitwise parity？**

   输出 N 从 3072 变 9216，BF16 GEMM bucket/algorithm/reduction 可能改变；必须看 layer
   probe、latent、最终帧和 replay，而不是只比较数学公式。

8. **projection kernel 理论上每 chunk 从多少降到多少？**

   30 blocks × 5 forwards 下从 450 降到 150；实际数以 Nsight stable-chunk 归一化为准。

9. **看到 FPS 没提升时先查什么？**

   查 projection CUDA/wall 是否真降、V copy/pack 是否增加、A2A wait 是否主导，以及
   SM/Tensor Active 和 SP 后 GEMM M shape；不能只用 kernel 数解释。

10. **什么条件下 6b 应直接放弃？**

    pack/V copy 不显著、收益落入 repeat CV 噪声、A2A 才是主瓶颈，或方案需要难维护的
    私有 GEMM backend 且 profiler-off 无收益时，保留证据并不实现。
