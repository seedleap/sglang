# LingBot-World 2.0 H100 40 FPS 实验计划与执行记录

> 最后更新：2026-07-13 07:44 EDT
> 状态：当前正确性最佳正式长跑为 23.238 FPS；当前实现与已测 FA3 split 配置的实测乐观天花板为 38.642 FPS，仍低于 40。E14/E15 已完成；全部任务自有 EC2、卷、ENI、SG、Deployment、Service、Pod、NodePool 与 NodeClass 已清理，当前占用 0 张 H100。
> 本文档是本任务的单一事实源。先更新本文档，再对外报告阶段性结果。

## 1. 当前最需要知道的信息

- **目标**：将 LingBot-World 2.0 在最多 8×H100 上的真实 generated FPS 从当前约 20 提高到 40，并用隔离变量实验得到同模型约束下的实测天花板。
- **正式基线**：8-way、20 warmup + 200 measured、纯 moving：generated/delivered=`22.401/22.380 FPS`，scheduler P50/P95/P99=`536/541/544 ms`。
- **当前分段事实**：8-way 纯 moving 中 DiT `361.53 ms`、VAE `133.75 ms`、camera condition `10.36 ms`；当前结构完全隐藏 VAE/输出后的乐观上限是 `12/0.36153=33.19 FPS`。
- **稳态 profile 事实**：在 20 个 warmup chunk 后截取的单 rank 中间 chunk，DiT NVTX 墙钟约 `503.9 ms`（profiler 扰动后），GPU kernel busy `480.9 ms`；其中 NCCL SendRecv `167.1 ms`、FA3 `150.2 ms`、GEMM `108.0 ms`。这些数只用于归因，不替代 profiler-off 基线。
- **当前最佳允许配置**：TP1×SP8、eager、`performance_mode=speed` 使 VAE 常驻；两次 20 warmup + 40 measured 分别为 `23.423/23.337 FPS`，scheduler mean=`512.33/514.20 ms`。chunk 20/21 与原 eager 基线逐字节一致。
- **当前最佳正式长跑**：在上述配置上强制 `NCCL_PROTO=Simple`，20 warmup + 200 measured 为 generated/delivered=`23.238/23.213 FPS`，scheduler mean/P50/P95/P99/max=`516.39/516.5/525/528/536 ms`，generated FPS 95% CI=`[23.205, 23.271]`。
- **当前关键上限证据**：同 probe SHA、同 stage logging、同 speed/TP1×SP8/Simple 的 A2A-on/理想零 A2A 完整链路分别为 `23.518 FPS / 510.25 ms` 与 `25.290 FPS / 474.50 ms`，端到端差 `35.75 ms`。进一步对 FA3 `num_splits=2` 做 3 次独立装载、每次 2 个 20-warmup + 40-measured run；最快零 A2A DiT 为 `310.541 ms`，即使 VAE、输出、A2A 与 layout 全部免费也只有 `38.642 FPS`。
- **FA3 split 结论已反转**：固定 shape 微基准曾预测 `num_splits=2` 可省约 `19.17 ms/chunk`；真实 A2A 的完整 DiT A/B 为 `375.292/375.617 ms`，split2 反而慢 `0.325 ms`。理想零 A2A 的 6 个 split2 run 也只在 `310.541–320.676 ms`，没有出现微基准预测的 `291.68 ms`。因此该配置不进入正确性候选。
- **达到 40 需要三项新增实现，不是继续调开关**：当前最佳正确完整链路需从 `516.39` 降到 `300 ms/chunk`；即使使用最快理想 compute 高点仍先缺 `10.541 ms (3.39%)` 的 DiT 计算优化，然后还要近乎消除或隐藏真实 Ulysses A2A/layout 的 `35.75 ms` 端到端代价，并把约 `110–140 ms` VAE 与 DiT 做真正流水。三项缺一都不能到 40。
- **并行度 sweep**：2/4/8-way generated FPS=`7.51/14.25/22.40`；三次都是相同的 20 warmup + 200 measured，220/220 服务日志均为 moving。
- **本轮不会做**：FP8、FP4、减少 denoising steps、调整 KV window、插帧或超分。
- **下一步**：若继续冲 40，先做新的 causal attention compute kernel，入口门槛是零 A2A DiT `<300 ms`；再开发无需每层两次同步 A2A 的 causal ring/context-parallel 路径；最后实现 DiT/VAE 双缓冲流水并做 10 分钟有界队列验收。当前配置调优阶段已经结束。
- **资源安全**：所有 GPU 资源都使用 `codex-lingbot2-h100-perf-*` 前缀；不修改、复用或删除隔壁 `lingbot2-h100`、`lingbot-nsys-profile`、`lingbot-hf-cache` 等资源。

## 2. 用户约束与验收口径

### 2.1 不可改变的约束

1. 最多使用 8 张 H100，即至多一台 `p5.48xlarge`。
2. 暂不使用 FP8、FP4 或其他低精度/量化方案。
3. 推理时 KV window 大小和既有动态策略不能调整。
4. 允许独立部署、重启服务及运行 profiler。
5. 可以参考隔壁部署经验，但本实验不能与隔壁部署资源或结果耦合。

### 2.2 主指标

- 主指标为服务端真实生成、完成 VAE decode 的 `generated FPS`。
- 不以浏览器 render FPS、播放标签 FPS、插帧后 FPS 作为达标证据。
- 固定 workload 下，steady chunk 实际输出 12 帧：

```text
20 FPS = 600 ms/chunk
40 FPS = 300 ms/chunk
```

实际计算以服务返回的 `chunk_stats.num_frames` 为准，不假定永远为 9。

### 2.3 40 FPS 验收条件

- 10 分钟有界队列运行期间无持续积压；
- steady-state 总 generated FPS ≥ 40；
- 同时报 P50/P95/P99 chunk latency、首帧延迟和 delivered FPS；
- 关闭 frame interpolation 与 upscaling；
- 维持同 checkpoint、BF16、4 DMD steps、固定 KV 策略和同一测试输入。

## 3. 固定实验身份

| 项目 | 固定值 |
| --- | --- |
| 正确性基线 SGLang commit | `196f3df97cdaabbdd1d840c52c32b4c15ddbf7b5` |
| benchmark-only probe commit | `38191bd71f95875b95aeb3abe889f56555f3c60a`；只增加理想 A2A 旁路与测试 |
| FA3 split tuning commit | `afc619cc79a2960f9cab53b3823d904672f3c5c0`；增加 LingBot causal scoped split 配置与测试 |
| Git branch | `codex/speedup-lingbot2` |
| 模型 | `robbyant/lingbot-world-v2-14b-causal-fast-diffusers` |
| 模型 revision | `59cccf49f2d2dd27418ae7a04b82b10868d455c2` |
| 容器 | `lmsysorg/sglang:dev@sha256:8f78575e03ab59a39191a4a6f718bbbe1726fa940f72a86a187a3f1628ada9a7` |
| 原基线 AMI | `ami-0cba08f75775ef406` |
| E14/E15 AMI | `ami-05d137128918f2c04`；Deep Learning Base OSS Nvidia Driver GPU Ubuntu 22.04 20260710 |
| 实例 | `p5.48xlarge` Spot，严格单机 8×H100 |
| 原基线区域/AZ | `us-east-2c` |
| E14/E15 区域/AZ | `ap-southeast-2b`；单机 NVLink/NVSwitch 实验不使用跨节点网络 |
| 精度 | BF16 |
| DMD steps | 4 |
| 分辨率 | 832×480 |
| decoded frames/chunk | 请求值 9；首 chunk 9，steady chunk 实测 12 |
| KV 策略 | interactive KV 开启；配置 moving=12、still=3；checkpoint sink=9、current block=3；有效 moving cache=24；不做 sweep |
| 基线并行 | Ulysses=8 |
| 基线 compile | false |
| 基线 VAE | spatial parallel decode |

## 4. 独立线上资源

| 类型 | 名称/身份 | 当前状态 |
| --- | --- | --- |
| AWS profile/account | `wms` / `829115578968` | 已验证 |
| EKS cluster | `arn:aws:eks:us-east-2:829115578968:cluster/leap-world` | 可访问 |
| EC2NodeClass | `codex-lingbot2-h100-perf-ec2` | 已删除 |
| NodePool | `codex-lingbot2-h100-perf` | 已删除；实验期限制 8 GPU/192 CPU |
| NodeClaim | `codex-lingbot2-h100-perf-p84wb` | 已因 Spot no-capacity 回收并删除 |
| EC2 | `i-07331e6e9d21cb89c` | 已终止；原为 Spot us-east-2c |
| Kubernetes node | `ip-172-31-83-83.us-east-2.compute.internal` | 已删除 |
| Deployment/Service | `codex-lingbot2-h100-perf` | us-east-2 已删除 |
| benchmark client | `codex-lingbot2-h100-perf-client` | us-east-2 已删除 |
| 公网入口 | 无 | 只走 ClusterIP |
| fallback EKS | `minwm-test-phx2` / us-west-2 | 可访问；EKS Auto Mode |
| fallback NodeClass/NodePool | `codex-lingbot2-h100-perf-auto` / `codex-lingbot2-h100-perf` | us-west-2 已删除 |
| fallback GPU Pod | `codex-lingbot2-h100-perf-*` | us-west-2 已删除；2a/2b 均已证实无容量 |
| fallback CPU client | `codex-lingbot2-h100-perf-client` | us-west-2 已删除 |
| 当前独立 EC2 | `i-07ea1cee39552512f` / `sir-5mtzh9fp` | 已终止/取消；实验期为 ap-southeast-2b 8×H100 80GB HBM3 |
| 当前源码/容器 | archive SHA-256 `43ab521e…` / 固定 image digest | 源码 SHA=`afc619…`；H100 单测 6 passed |
| 当前访问边界 | `sg-0c97b903f90a2535a` + EC2 Instance Connect | 专属 SG 已删除；实验期仅本机公网 `/32` SSH，无持久私钥或共享 SG 修改 |

实验期 NodePool 设有空闲 consolidation 与 8 小时 expiry 安全网；最终没有依赖安全网，而是逐项显式删除。三地域按任务 tag 扫描无 pending/running/stopping/stopped/shutting-down 实例，两套 EKS context 按对象前缀扫描无残留。

## 5. 天花板定义与判定方法

分别测量：

- `T_DiT`：4-step denoise + clean-context cache update；
- `T_VAE`：persistent causal VAE decode；
- `T_output`：D2H、物化、编码和 WebSocket 传输；
- `T_e2e`：完整连续服务。

令每 chunk 的实际 decoded frame 数为 `F`：

```text
乐观流水线天花板 = F / max(T_DiT_min, T_VAE_min, T_output_min)
当前串行预测     = F / (T_DiT + T_VAE + T_output)
```

判定规则：

- 若所有允许配置中 `T_DiT_min > F/40`，则同模型、BF16、固定 KV、8×H100 即使完全隐藏 VAE/输出也无法达到 40 FPS。
- 若 `T_DiT_min < F/40`，且 VAE/输出也各自小于预算，则 40 FPS 结构上可达；还需用真实有界流水线验证资源竞争后的可持续吞吐。
- 理论硬件峰值不作为结论；以真实 kernel replay、NCCL trace 和完整连续运行作为上限证据。

当前已实测的 tuned ceiling：

```text
split2 三次独立装载、每次两轮的零 A2A DiT mean：
  320.676 / 319.994 ms
  313.820 / 310.541 ms
  318.615 / 319.641 ms

最快直接观测 DiT      = 310.541 ms
完全隐藏其他阶段的上限 = 12 / 0.310541 = 38.642 FPS
距 40 FPS compute 预算 = 310.541 - 300 = 10.541 ms（3.39%）
```

这里取全部直接观测中的最快均值，主动把进程间方差朝“更容易达到 40”的方向处理。它是**当前代码、当前 BF16 kernel 与已测 split 配置的经验乐观天花板**，不是对任何未来新 kernel 的硬件数学上限。零 A2A probe 不保留模型语义，只能用于计时上界，不能作为画质样本。

## 6. 实验矩阵与进展

| ID | 实验 | 唯一变化 | 状态 | 结果 |
| --- | --- | --- | --- | --- |
| E0 | 独立环境身份核验 | 无 | 完成 | SHA/model/依赖/topology/PID tree/8 rank 均已验证 |
| E1 | BF16 baseline | Ulysses=8，compile=false | 完成 | 正式纯 moving 22.40 FPS；旧 22.65 混合结果作废 |
| E2 | baseline Nsight/NCCL | 只开启 profiler | 完成 | 20 warmup 后截取 3 个 DiT chunk；单 rank 中间 chunk：SendRecv/FA3/GEMM=`167.1/150.2/108.0 ms` |
| E3 | DiT/VAE/output 分段 | 只增加计时 | 完成 | 8-way 纯 moving：DiT 361.53 ms；VAE 133.75 ms；condition 10.36 ms |
| E4 | Ulysses sweep | 2 / 4 / 8；KV 不变 | 完成 | 7.51 / 14.25 / 22.40 FPS；8-way 最优 |
| E5 | compile sweep | off / on；使用 E4 最佳并行度 | 完成 | max-autotune/default 热重放 14.08/14.74 FPS，均淘汰 |
| E6 | VAE 执行方式 | auto layerwise offload → speed resident | 完成 | 两次 23.42/23.34 FPS；VAE 单次日志 112.2 ms；RGB bitwise 一致 |
| E7 | 天花板 replay | 零 A2A + 完全隐藏 VAE/output 的乐观上界 | 完成 | 最快 DiT 310.541 ms，对应 38.642 FPS |
| E8 | 10 分钟最终验收 | 仅候选达到 40 后进入 | 未进入 | 当前无候选通过 `<300 ms` compute gate，避免把未达标配置包装成验收 |
| E9 | 2D 混合并行 | 8 卡不变；TP1×SP8 → TP2×SP4 | 完成/淘汰 | 18.59 FPS、645.48 ms；慢约 20.5%，chunk20/21 PSNR 仅约 7.3 dB |
| E10 | NCCL protocol | TP1×SP8 speed 不变；只强制 protocol | 完成 | LL128 两次 23.84/23.86 FPS 但 RGB 漂移；Simple 23.47 FPS 且 bitwise 一致 |
| E11 | 最佳正确配置正式长跑 | speed + TP1×SP8 + Simple | 完成 | 23.238 FPS，516.39 ms；95% CI [23.205,23.271] |
| E12 | 理想零 A2A 上限 | benchmark-only，保持 shape/FA/GEMM/VAE | 完成 | 完整 25.290 FPS；DiT 310.85 ms；完全隐藏 VAE 后 38.60 FPS |
| E13 | A2A-on 配对基线 | 同 probe SHA/stage logging，只关闭旁路 | 完成 | 23.518 FPS、510.25 ms；相对零 A2A 的端到端可回收量为 35.75 ms |
| E14 | LingBot causal FA3 split tuning | 只令 causal self-attention `num_splits=2` | 完成/淘汰 | 真实 A2A warmed DiT split0/split2=`375.292/375.617 ms`；无收益且 AR RGB 漂移 |
| E15 | tuned compute ceiling | 零 A2A，split0/split2，多次独立装载 | 完成 | split2 6 个 run 最快 DiT 310.541 ms；经验乐观上限 38.642 FPS |

通用协议：探索候选先做 20 warmup + 40 measured 并保存原始 `chunk_stats`；只有通过性能与正确性门槛的候选才晋级 200 measured 与 10 分钟稳定性验收。E11 已完成 200 measured；E14 因无性能收益且数值漂移被淘汰；E15 是无模型语义的 ceiling probe，以多次独立装载替代长跑。

## 7. 已验证结论与下一阶段入口

1. **先过 compute gate**：当前最快理想 DiT 是 `310.541 ms`，高于 40 FPS 的 `300 ms` 预算。下一阶段必须开发新的 causal attention/MLP/launch 融合实现，并先在 zero-A2A 完整 DiT 上稳定做到 `<300 ms`；单算子微基准不再作为晋级证据。
2. **再改通信架构**：当前 40 层、5 forwards/chunk 产生 400 次同步 A2A。真实配对的完整 scheduler 可回收量为 `35.75 ms`，DiT 阶段归因为 `51.17 ms`。代码中的通用 ring attention 不能直接用于 LingBot causal path，需要新增保持固定 KV 语义的 ring/context-parallel 实现。
3. **最后做 VAE 流水**：resident parallel VAE 仍约 `110–140 ms`，但它不是 compute gate。只有 DiT 本身低于 300 ms、通信路径也压下后，才值得用双缓冲把 chunk N+1 的 DiT 与 chunk N 的 VAE decode 重叠，并验证两者争用后的真实吞吐。
4. **已经关闭的路径**：全局 torch.compile 热态只有 `14.08/14.74 FPS`；TP2×SP4 只有 `18.59 FPS`；LL128 虽约快 2% 但改变 AR 输出；FA3 split2 在真实 A2A DiT 中无收益。这些配置不再重复消耗 H100。
5. **最终验收仍未发生**：当前最佳正确长跑为 `23.238 FPS`，不是 40 FPS 候选。只有新实现先通过正确性、`<300 ms` compute gate 和 200 measured run，才进入 10 分钟有界队列验收。

## 8. 与原设想不一致的发现

### D1：隔壁部署仍在运行且占用两台 p5

- 原先历史复盘称 GPU 已释放；实时检查发现 `lingbot2-h100` 和 `lingbot-nsys-profile` 正分别占用一台 p5。
- 处理：新建第三套独立 NodeClass/NodePool/host cache/Service，不复用这两台节点。

### D2：AWS CLI profile 的默认 output 配置损坏

- `aws sts get-caller-identity --profile wms` 因默认 output 值异常报错，但显式 `--output json` 后身份和权限正常。
- 处理：所有 AWS 命令显式指定 `--profile wms --region us-east-2 --output json`。

### D3：CPU client 最初无法调度

- 原因：platform 节点带 `workload-type=platform:NoSchedule` taint。
- 处理：只给 CPU client 添加精确 toleration；不添加到 GPU server。

### D4：集群默认注入 OpenTelemetry init containers

- 新 server/client 首次创建时被自动注入四类 instrumentation init container，拖慢启动并改变实验环境。
- 处理：在独立 Pod annotations 中显式关闭 Java/Node/Python/.NET auto-annotation 与 injection，随后重建尚未启动的 Pod。

### D5：客户端不应使用 SGLang GPU 镜像

- 初始客户端清单会在小型 CPU 节点拉取大型 GPU 镜像，既慢又无必要。
- 处理：改为轻量 `python:3.11-slim`，仅安装固定版本 `websockets==15.0.1` 和 `msgspec==0.19.0`。

### D6：固定镜像与固定代码 SHA 的 FlashInfer 依赖不一致

- 固定镜像内置 `flashinfer-jit-cache 0.6.14+cu130`；对固定代码 SHA 执行 editable install 后，`flashinfer_python` 与 `flashinfer_cubin` 被解析为 0.6.12。
- import 阶段报错：`flashinfer-jit-cache version (0.6.14+cu130) does not match flashinfer version (0.6.12)`。
- 没有采用 `FLASHINFER_DISABLE_VERSION_CHECK=1`，因为绕过二进制版本检查会污染性能结论。
- 已确认官方 cu130 wheel index 存在 `flashinfer-jit-cache==0.6.12+cu130`；处理方式是显式安装匹配版本，使 Python、cubin、JIT cache 全部为 0.6.12。

### D7：SGLang 成为容器 PID 1 后主动误杀 worker

- 依赖修复后，rank 0 连续三次在 server 启动约 14 秒后以 `-9` 退出；无 Python 异常，Pod cgroup 的 `oom_kill=0`，内核也没有 OOM 记录，因此不是 1 TiB 内存上限导致。
- 代码证据：`kill_itself_when_parent_died()` 设置 `PR_SET_PDEATHSIG=SIGKILL` 后，若发现 `os.getppid() == 1` 就主动 `SIGKILL` 当前 worker。
- 清单脚本最后使用 `exec sglang serve`，使 SGLang 成为容器 PID 1；其 worker 的合法父进程因而恰好是 PID 1，命中上述孤儿进程保护逻辑。
- 处理：不再 `exec`，由 Bash 作为 PID 1 后台启动并显式 `wait` SGLang，同时转发 TERM/INT；修复后需用实际进程树和完整启动日志复核。

### D8：请求 9 帧不等于 steady chunk 输出 9 帧

- 首 chunk 返回 9 帧，之后每个 steady chunk 都返回 12 帧；这是 causal VAE 的 chunk 边界行为，不应把 40 FPS 预算按 9 帧误算成 225 ms。
- 处理：所有 steady-state FPS 和天花板都用服务返回的 `num_frames=12` 计算，因此 40 FPS 预算为 300 ms/chunk。

### D9：KV 日志中的有效窗口不是简单的 moving=12

- pipeline 配置仍是用户要求保持不变的 moving=12、still=3；checkpoint 实际 `sink_size=9`、`num_frames_per_block=3`。
- moving 模式的完整有效窗口按 `sink + sample + current = 9 + 12 + 3 = 24` 记录，日志也实测 `window_frames=24`、`cache_frames=24`。
- 处理：不改任何 KV 参数；文档同时记录配置窗口和完整 cache 窗口，避免把不同单位混用。

### D10：长动作脚本被 512 项队列截断，导致第 172 chunk 起切到 still

- 原 benchmark 把整个运行编码成 `[["w"]] * (chunks * 3 + 12)`，但 `LingBotWorldRealtimeState` 的 script queue `maxlen=512`；每 chunk 消耗 3 项，因此最多只覆盖 171 个完整 chunks。
- 服务日志精确显示旧 4-way 长跑在 chunk 171 出现 `still_chunks=1`，chunk 172 起切为 still，KV 有效窗口也从 moving 24 变为 still 15；scheduler 随即从约 840 ms 降至约 756 ms。
- 影响：旧 8-way 和 4-way 的 220-chunk 结果都不是纯 moving；旧 E1 的 measured 200 chunks 中后 28 个已被污染，旧 E3 的分段均值也混合了两种 workload。
- 处理：init 只放 4 个 bootstrap moving chunks，随后立即发送 `mode=state` 的 level-triggered `W` 事件；状态会持续保持到显式 release，不受 512 项 script 上限影响。正式长跑还必须验证 `chunk_stats.event_id=1` 且服务日志全程为 moving。

### D11：默认 max-autotune torch.compile 对 LingBot realtime DiT 是严重负优化

- `--enable-torch-compile true` 实际选择 DiT `max-autotune-no-cudagraphs`、VAE `default`；首次请求产生约 `2.3 GiB` Inductor/Triton cache，并多次报告部分 Triton GEMM 配置超出 H100 shared-resource 限制后被忽略。
- 第一次 22-chunk replay 在 warmup 20 后仍仅 `4.05 FPS`，因为前序各形状持续编译/autotune；同一进程立即重放相同 22 chunks 后，已命中热 cache，但仍只有 `14.08 FPS`。
- 热重放分段：DiT `704.3 ms`，比 eager `361.5 ms` 慢 `1.95x`；VAE 从 `133.7 ms` 改善到 `116.5 ms`，不足以抵消 DiT 回退。
- compile-on 与 eager 的 chunk 20/21 RGB replay 只有约 `17.70/17.24 dB PSNR`，20-chunk autoregressive 累积后数值漂移显著；该模式同时不满足速度与保真要求，直接淘汰。
- `SGLANG_TORCH_COMPILE_MODE=default` 热重放也只有 `14.74 FPS`，DiT/VAE=`635.3/148.8 ms`；说明回退不是 max-autotune 单一选型造成。只保留 VAE compile 作为可选的单独代码候选，不再全局编译 DiT。

### D12：Nsight 的自动 NVTX capture 没有按预期启动，且首份手工报告不是稳态 KV

- 清单使用 `--capture-range=nvtx --nvtx-capture=stage_LingBotWorldCausalDMDDenoisingStage`，但 Nsight 2026.3.1 一直报告 `application launched; use start`，没有自动进入 collection；改用同一 session 的 `nsys start/stop` 才得到报告。
- 第一份手工报告只跑 6 chunks，KV cache 尚在填充：第二个 chunk 的 FA3 GPU 时间仅 `43.6 ms`，第六个已增至 `116.5 ms`，不能代表固定 KV window 的稳态上限。
- 处理：重新发起 40-chunk 请求，先完整运行 20 个 warmup chunks，再手工截取约 2 秒。第二份报告覆盖 3 个完整 DiT chunk（24 个 rank-range）和边界上的 4 个 VAE decode，作为 E2 正式归因报告。
- Nsight report 参数也有版本差异：`cuda_gpu_kern_sum:range-name` 只是给 kernel 名加 NVTX 前缀，并不是过滤；正式统计改用 `--filter-nvtx stage_LingBotWorldCausalDMDDenoisingStage/8`。

### D13：Nsight 对该热路径的扰动约 40%，不能直接拿报告墙钟当基线

- profiler-off 的 8-way DiT 均值是 `361.53 ms`；稳态报告中 24 个 rank-range 的平均 DiT NVTX 是 `505.58 ms`，单 rank 中间 chunk 的 GPU span 为 `503.91 ms`。
- 因此 `167.1 ms` SendRecv、`150.2 ms` FA3、`108.0 ms` GEMM 只用于确认组成和优先级；不会把其中任一项直接从 `361.53 ms` 相减来宣称无通信天花板。

### D14：`vae_cpu_offload=false` 不等于 VAE 常驻

- `performance_mode=auto` 的启动参数最终仍是 `layerwise_offload_components=['image_encoder', 'vae']`，运行时确实为 Wan VAE 创建了 layerwise offload manager；这是与最初“已关闭 VAE offload”设想不一致的独立机制。
- 改为 `performance_mode=speed` 后，显式 `enable_torch_compile=false` 仍优先生效，最终参数变为 `layerwise_offload_components=null`，其余精度/steps/KV/并行均不变。
- 两次 60-chunk 实验为 `23.423/23.337 FPS`，对同次 TP1×SP8 auto 回归 `22.800 FPS` 提升约 `2.4%–2.7%`；单次 VAE stage 日志为 `112.2 ms`，方向与完整 scheduler 的约 12–14 ms 改善一致。
- speed 与原 eager 的 chunk20/21 原始 RGB SHA256 完全相同，确认该收益没有改变数值输出。

### D15：TP2×SP4 的 2D 混合并行不仅更慢，AR 输出也明显漂移

- 参数与加载均合法：`tp_size=2, sp_degree=4, ulysses_degree=4`；transformer 每卡模型占用由约 `41.92 GiB` 降到 `22.98 GiB`，不存在显存压力。
- 实测只有 `18.591 FPS`，scheduler mean/P50/P95=`645.48/644.5/656 ms`；比 speed TP1×SP8 的两轮均值约 `513.3 ms` 慢约 `25.7%`。更小 Ulysses 组的收益被 TP2 的同步规约成本压倒。
- chunk20/21 对 TP1 eager 的 PSNR 仅 `7.32/7.29 dB`、MAE 约 `82.8/83.0`；这是 20 个 autoregressive chunks 后的累积差异，不能单独证明单步错误，但它进一步排除了把该配置作为性能候选。

### D16：LL128 有小幅吞吐收益，但全局 protocol 强制改变了确定性输出

- `NCCL_PROTO=LL128` 的两次 60-chunk 为 `23.843/23.863 FPS`，scheduler mean=`503.30/502.88 ms`；相对未强制 protocol 的 speed 两轮约 `513.3 ms` 快约 `2.0%`。
- 同一 LL128 进程第三次 22-chunk replay 与第二次的 chunk20/21 SHA256 完全一致，说明 LL128 自身可重复；但它们对原 eager 基线的 PSNR 仅 `20.39/20.52 dB`、MAE=`14.27/14.39`。
- `NCCL_PROTO=Simple` 的 60-chunk 为 `23.472 FPS`、scheduler `511.25 ms`，与默认 speed 在短跑噪声内一致；chunk20/21 与原 eager SHA256 完全相同。
- 解释：全局 NCCL protocol 同时影响 Ulysses A2A、encoder folding 和 VAE collective；浮点规约路径变化可能被 20 个 AR chunks 放大。没有完成 component-local protocol 隔离前，不把 LL128 当作正确性候选。

### D17：正确性最佳的短跑收益在 200-chunk 长跑中略有回落

- speed + Simple 的短跑为 `23.472 FPS`，正式 200 measured 为 `23.238 FPS`；后者 95% CI 很窄，因此不以短跑高点作为可持续性能。
- 与原正式 eager/auto `22.401 FPS` 相比，正确性最佳长跑的提升是约 `3.74%`，scheduler mean 从 `535.70 ms` 降到 `516.39 ms`（约 `3.60%`）。

### D18：Nsight 中 167 ms 的 SendRecv residency 不等于 167 ms 可回收墙钟

- benchmark-only 旁路同时删除每层 input/output Ulysses `all_to_all_single` 及相邻的 permute/contiguous copy。严格配对的 A2A-on/zero-A2A scheduler mean 为 `510.25/474.50 ms`，端到端只回收 `35.75 ms`。
- 旁路 measured DiT mean/P50/P95=`310.85/309.99/314.95 ms`，VAE mean/P50/P95=`140.07/141.08/143.76 ms`；完整链路 `25.290 FPS`。
- 这解释了为何不能从 Nsight 的 profiler-inflated 单 rank kernel residency 中直接减掉 `167.1 ms`；正式差值以同 SHA、同 stage logging 的配对完整 scheduler 为准。
- 更强的上限结论：即使 Ulysses 通信、layout copy、VAE 和输出都免费，当前保留 shape/FLOPs 的 DiT compute 仍需 `310.85 ms`，对应 `38.60 FPS`，低于 40。要跨过 40，除通信外至少还需 `10.85 ms (3.49%)` 的 compute 优化。

### D19：A2A 阶段差不能直接当作端到端收益，VAE 在配对运行间反向漂移

- A2A-on measured 40 chunks：generated/delivered=`23.518/23.489 FPS`，scheduler mean/P50/P95/P99/max=`510.25/509/520/525/525 ms`；协议核验为 chunk0 `event_id=null`、其余 59 个 chunks `event_id=1`，measured chunks 全部 12 帧。
- 同步 stage logging 中，A2A-on/zero-A2A 的 DiT mean=`362.02/310.85 ms`，差 `51.17 ms`；但 VAE mean 同时从 `117.59 ms` 反向变为 `140.07 ms`，差 `-22.48 ms`，其余与调度开销也有约 6–7 ms 差异。
- 因此正式可回收墙钟采用完整 scheduler 的 `35.75 ms`；`51.17 ms` 只作为 A2A 对 DiT 的阶段归因。阶段计时不能跨运行简单相加成新的端到端预测。

### D20：E14 rollout 恰好撞上 Spot no-capacity 回收，不能只看 EC2 表层原因

- Spot request `sir-91gzj3qp` 的最终状态是 `instance-terminated-no-capacity`，更新时间=`09:30:31 UTC`；Karpenter 随后给自有节点加 `karpenter.sh/disrupted`，NodeClaim deletionTimestamp=`09:30:33 UTC`，并在 `09:30:38 UTC` drain/终止 EC2。
- EC2 `StateTransitionReason` 显示 `User initiated`，但这是 Karpenter 执行终止动作的表层记录；以 Spot request 的具体 status 为准，本次根因是云端无匹配 Spot 容量，不是 Recreate 空窗触发 consolidation，也不是新代码启动失败。
- NodePool GPU limit=8，旧 claim 完成终止前替代 claim 不能创建；新 Pod 已被 Karpenter nomination。关键 JSON、performance log、Nsight 报告均已事先复制到本地，不会随 hostPath 节点丢失。
- 自有 NodePool 仍新增实验期 disruption budget：禁止 `Empty/Underutilized` 自动回收，保留 8 小时 expiry 与任务结束显式清理。它不能阻止 Spot no-capacity 回收，但能排除后续 rollout 中的 consolidation 干扰变量。

### D21：Spot 回收后，us-east-2 三个 AZ 与 us-west-2a 都没有 p5 替代容量

- us-east-2 Spot placement score 三个 AZ 均降到 1；Karpenter 在 2a/2b/2c 多次得到 Spot `UnfulfillableCapacity`，on-demand 也分别得到 `InsufficientInstanceCapacity`。这排除了单一 AZ 或 Spot-only 的问题。
- 旧 terminating claim 一度仍占 NodePool 8-GPU/192-CPU 记账额度；曾临时把自有 NodePool limit 提到 16/384 以允许替代 claim，但旧机 allocatable GPU=0、仅有一个 Pod 请求 8 GPU，实际使用从未超过 8 张。旧 claim 删除后已立刻恢复 8/192。
- 新增独立 us-west-2 EKS Auto Mode 路径；没有复用现有 `minwm-test-phx2-p5e-spot`，因为它是 p5e/H200 Local Zone，不符合 H100 实验身份。自有 NodeClass/NodePool 只允许 p5.48xlarge。
- us-west-2a 的 Spot/on-demand 同样失败；当前改为 us-west-2b on-demand-only，但 EKS Auto 暂沿用 p5 offering 负缓存而未发出新 CreateFleet。保留唯一 Pending Pod 等缓存过期，不再用重建制造无效重试。

### D22：40 FPS 要求把 400 次同步 A2A 几乎压到物理极限，单做 FA tuning 不够

- 每个 12-frame chunk 是 4 次 DMD forward 加 1 次 clean-cache forward；40 层 causal self-attention 合计 `200` 层调用，每层各一次 input/output A2A，共 `400` 次同步 collective。
- 稳定 Nsight 采样区间里出现 `187` 个长 causal FA3 kernel 与 `374` 个 NCCL SendRecv kernel，严格保持 `1:2`；另有 `187` 个约 22 us 的短 cross-attention kernel，不能与 causal kernel 混算。
- 单 rank 每层 input packed-QKV A2A 为 `[1,585,40,384]` BF16=`17,971,200 bytes`，output A2A 为 `[1,4680,5,128]` BF16=`5,990,400 bytes`；一个完整 chunk 的 200 层合计处理约 `4.792 GB/rank` 的 A2A tensor，尚未包含 layout-copy 流量。
- 配对实验的 DiT 阶段差是 `51.17 ms`，完整 scheduler 差是 `35.75 ms`。E14 已否定“FA split 先回收 19.17 ms”的微基准外推：真实 A2A 的 warmed DiT 中 split2 比 split0 慢 `0.325 ms`；理想零 A2A 的最快 split2 DiT 也仍为 `310.541 ms`。
- 因而现有每层双 A2A 架构没有可供 40 FPS 使用的正通信预算：compute 本身已经超出 300 ms。顺序必须是先用新 compute kernel 至少再省 `10.541 ms`，再开发能近乎消除或隐藏 A2A 的 causal ring/context-parallel 路径，最后把约 110–140 ms VAE 与 DiT 真正流水化。

### D23：美区容量耗尽后，最高分 H100 Spot 在澳洲可用，但部署面必须从 EKS 降到直接 EC2

- 为严格维持最多 8 张 H100，先后把 us-east-2 与 us-west-2 的自有 GPU Deployment 都缩到 0，才申请下一地域；CPU client 保留不影响 GPU 上限。
- `ap-southeast-2b` 的 p5.48xlarge Spot placement score=`9/10`，该区域 P Spot/On-Demand 配额均为 `768 vCPU`。创建独立 EKS Auto 集群的 dry-run 配置本身正确，但 IAM 在第一步 `cloudformation:CreateStack` 返回 AccessDenied；因此控制面、VPC 与节点均未创建，无残留集群。
- 该区默认 VPC 有 2b 公网子网，EC2 `RunInstances --dry-run` 通过；随后单台 Spot `i-07ea1cee39552512f` 成功获得 8×H100 80GB HBM3。当前 IAM 不能调用 SSM 控制面，因此创建任务专属 SG，仅允许本机公网 `/32` SSH，并通过 EC2 Instance Connect 注入 60 秒临时公钥；没有修改共享 default SG。
- 固定源码以 `git archive afc619…` 生成，archive SHA-256=`43ab521e…`，传输后再次校验；固定容器 digest 与 FlashInfer 0.6.12 不变。派生镜像在依赖已安装后长时间停在 Docker `exporting layers`，故取消非实验性导出，改为长生命周期 runtime container 安装一次依赖后重启 server 进程。
- H100 runtime 内 `test_flash_attention_num_splits.py` 与 `test_usp_benchmark_bypass.py` 共 `6 passed`。E14 日志验证 TP1/SP8/ring1、compile=false、offload=false、NCCL Simple、`lingbot_causal_fa_num_splits=2`。固定模型首次下载 26 个文件、约 81 GB，受 gp3 吞吐限制约 11 分钟；该时间未计入 FPS。
- 因 E13 与 E14 不在同一物理实例，split 因果实验在澳洲实例执行独立 server restart 的 A/B/A，并用 split0 的 raw SHA 与原美东正确基线对齐。E15 又对 split0/split2 做 3 次独立装载，显式量化进程间方差。

### D24：FA3 单算子 11.4% 收益没有转化成真实 DiT 收益

- E14 顺序为 split2 A1 → split0 B → split2 A2，每轮 20 warmup + 40 measured。A1 是该实例首个真实请求，DiT=`429.801 ms`，仍受一次性初始化影响，不用于性能因果结论；warmed B/A2 DiT=`375.292/375.617 ms`，split2 反而慢 `0.325 ms`。
- B/A2 scheduler mean=`518.175/525.800 ms`，表面差 `7.625 ms`；但 VAE 同时从 `109.477` 漂到 `115.458 ms`，所以不能把 scheduler 差归给 causal FA。阶段和完整链路都没有 split2 正收益。
- split0 B 的 chunk20/21 SHA256=`40fa3431…/581cbe9c…`，与原美东正确 eager/Simple 基线逐字节相同，证明跨区域、AMI 与驱动更换后协议仍对齐。split2 A1/A2 自身也逐字节可重复，但相对 split0 的 chunk20/21 PSNR=`17.98/18.05 dB`、MAE=`18.71/18.69`、变化比例约 `95.8%/96.0%`。
- 结论：微基准只覆盖单个 FA3 kernel，没有覆盖 200 层调用周围的 A2A、layout、launch 与共享资源竞争，不能乘以 200 后外推完整 DiT。split2 同时无真实性能收益且改变 AR 轨迹，因此淘汰。

### D25：严格上限必须使用直接观测的最快高点，而不是跨实例拼接收益

- E15 的 zero-A2A probe 保留 attention/GEMM/MLP 等 shape 和 FLOPs，但用本地 reshape 代替 400 次 A2A；输出不保留模型语义，只用于回答“通信、VAE 与输出都免费时 compute 最快能到哪里”。
- split0 的 3 次独立装载共有 4 个 measured run，DiT mean=`322.946/324.690/318.677/319.857 ms`。split2 的 3 次独立装载各跑两轮，DiT mean=`320.676/319.994`、`313.820/310.541`、`318.615/319.641 ms`，说明进程装载之间存在约 10 ms 的不可忽略方差。
- 不采用“旧实例最快 split0 减去新实例 split2 差值”的拼接法；直接取全部完整 DiT 运行中的最快均值 `310.541 ms`，对应经验乐观天花板 `12/0.310541=38.642 FPS`。这已经主动选择最有利于 40 的高点，但仍比 300 ms 预算慢 `10.541 ms (3.39%)`。
- 最快 run 内 40 个 measured chunks 的 DiT sd=`1.583 ms`；把 chunks 当观测值的 t-based 95% CI 为 `[310.035,311.047] ms`，即便取区间内最乐观的低时延端也只有 `38.705 FPS`、仍差 `10.035 ms`。这不是跨进程总体置信区间，所以正式结论仍同时报告全部 3 次装载范围，而不靠单个 CI 夸大确定性。
- 相邻 split0/split2 遥测的 busy samples 都几乎 100% power-throttle；split0/split2 平均功耗约 `609.6/642.3 W`、平均 pclk 约 `1758/1682 MHz`。最快 split2 不是靠更高 GPU 时钟得到，因此不会再用 boost 差解释或修正其结果。
- 结论边界：`38.642 FPS` 是当前代码、BF16、固定 KV、当前 FA3/GEMM 与已测 split 面的经验上界；它足以判定“继续调现有开关”不能到 40，但不排除未来新 kernel 把 DiT 降到 300 ms 以下。

### D26：组合式 `kubectl delete` 没有删除后续异构 resource type

- 清理时曾使用一条命令同时列出 Deployment、Service、Pod，另一条同时列出 NodePool、NodeClass；实际输出只确认了第一类对象，后续扫描发现两区 Service/Pod 与 NodeClass 仍存在。
- 没有把“命令 exit 0”当作清理完成；改为逐个 `resource/name` 精确删除，再扫描两套 context 中所有 task prefix。最终 Deployment、Service、Pod、NodePool、EC2NodeClass/NodeClass 均无残留。
- 直接 EC2 的 graceful termination 长时间停在 shutting-down；本地证据已校验后，先取消 Spot request，再对同一实例 dry-run 并执行 force/skip-OS-shutdown。最终状态 terminated，DeleteOnTermination 根卷不存在、ENI 为空、专属 SG 删除成功。

## 9. 关键文件

- 独立 K8s 清单：`benchmark/lingbot2_h100_spot/k8s.yaml`
- 集群内 benchmark client：`benchmark/lingbot2_h100_spot/client.yaml`
- us-west-2 EKS Auto fallback：`benchmark/lingbot2_h100_spot/eks_auto_usw2.yaml`、`client_eks_auto_usw2.yaml`
- ap-southeast-2 直接 EC2 固定部署：`benchmark/lingbot2_h100_spot/ec2_apse2_spot.json`、`prepare_ec2_runtime.sh`、`run_ec2_server.sh`、`run_ec2_benchmark.sh`
- 固定 realtime benchmark：`benchmark/lingbot2_h100_spot/benchmark_realtime.py`
- 原始 JSON、stage log、Nsight 与遥测：`benchmark/lingbot2_h100_spot/results/`
- 既有分析假设：`docs/diffusion/lingbot_world_v2_performance_analysis_zh.md`

## 10. 执行日志

### 2026-07-13 01:55–02:10 EDT

- 验证 AWS 身份为 account `829115578968`，Spot placement score 在 us-east-2a/2c 为 9。
- 只读检查隔壁部署、NodePool、EC2NodeClass、Pod、日志和 topology；未修改任何隔壁资源。
- 创建并通过 server-side dry-run：独立 EC2NodeClass、单节点 NodePool、ClusterIP Service、Deployment、CPU client。
- Karpenter 成功获取 Spot `i-07331e6e9d21cb89c`，节点 Ready 且归属于独立 NodePool。
- 修复 platform taint 与 OpenTelemetry 自动注入问题。
- 当前 server Pod Running、尚未 Ready；等待代码安装与模型加载。

### 2026-07-13 02:10–02:15 EDT

- server 首次启动在 import 阶段失败，未开始加载 checkpoint、未产生性能样本。
- 根因是镜像内置 FlashInfer JIT cache 0.6.14 与代码依赖解析出的 FlashInfer 0.6.12 不匹配。
- 验证匹配的 `flashinfer-jit-cache==0.6.12+cu130` wheel 存在，更新独立清单后重建。

### 2026-07-13 02:15–02:24 EDT

- 匹配 FlashInfer JIT cache 后，server 成功进入 scheduler 启动，但 rank 0 连续三次以 `SIGKILL(-9)` 退出。
- 读取 Pod cgroup 和节点内核日志，排除 cgroup/主机 OOM；GPU 驱动日志也没有与失败时刻对应的致命 Xid。
- 将固定约 14 秒的退出与 worker 的 `PPID==1` 自杀保护代码对应，确认是清单中 `exec sglang serve` 造成的容器 PID 1 语义冲突。
- 清单已改为 Bash 保持 PID 1、后台运行并等待 SGLang；下一步先验证进程树，再等待 checkpoint 完整加载。

### 2026-07-13 02:24–02:42 EDT

- 修复后实际进程树为 Bash PID 1 → SGLang PID 4925 → 8 个 scheduler workers；超过原 14 秒失败窗口且服务最终 Ready、restart=0。
- 验证代码 SHA `196f3d...`、模型 revision `59cccf...`、SGLang editable 包及 FlashInfer Python/cubin/JIT cache 0.6.12 身份一致。
- E0 smoke 揭示首 chunk 9 帧、steady chunk 12 帧，并独立复现约 23 FPS 量级。
- E1 正式跑完 20 warmup + 200 steady chunks：22.65 generated FPS，scheduler mean 529.735 ms、P95 537 ms；均值换算 FPS 的 95% CI 为 `[22.625, 22.681]`。
- E3 从同一基线最后 200 chunks 的 stage 日志统计：DiT 359.644 ms、VAE 131.270 ms、camera condition 10.290 ms。
- 获取 2 份 steady-chunk rank-0 PyTorch CUDA trace；trace 显示每 chunk 约 399 次 all-to-all 与 99 次 coalesced NCCL，但 profiling 将 chunk 放大到约 8.7 秒，因此只用来判断结构，不拿其绝对耗时替代无 profiler 基线。

### 2026-07-13 02:42–03:02 EDT

- 完成旧协议的 4-way 220-chunk 探索跑；整体混合值为 14.58 FPS，但日志发现 chunk 172 起从 moving 切到 still。
- 用不会触发 512 项上限的 60-chunk pilot 复核 4-way 纯 moving：20 warmup + 40 measured 为 14.28 FPS，scheduler P50/P95/P99=`840/849/872 ms`。
- GPU 固定最大 graphics clock 到 1980 MHz 后，负载下实际时钟仍因约 700 W power limit 在 1530–1815 MHz 波动，且纯 moving pilot 无收益；时钟锁定不是此前时延变化的根因。
- 根因确定为动作脚本耗尽；修正 benchmark 为 bootstrap script + 持续状态事件，并将所有受影响的旧长跑结论降级为探索性数据。
- 修正协议 smoke 中，chunk 0 由 bootstrap 保持 moving，chunk 1 起 `event_id=1`；服务日志 8/8 chunks 均为 `mode=moving`、`window_frames=24`、`still_chunks=0`。
- 恢复默认 GPU application clock 后完成 4-way 正式纯 moving 长跑：20 warmup + 200 measured，generated/delivered=`14.250/14.241 FPS`，scheduler mean/P50/P95/P99/max=`842.09/842/850/855/862 ms`。
- 4-way 协议核验：JSON 中 chunk 0 `event_id=null`、其余 219 个 chunks `event_id=1`；服务日志精确计数 220 moving、0 still。对应 measured stage 均值约为 DiT `567.75 ms`、VAE `233.55 ms`、camera condition `10.21 ms`。
- 2-way 正式纯 moving 长跑：generated/delivered=`7.511/7.508 FPS`，scheduler mean/P50/P95/P99/max=`1597.64/1597.5/1607/1616/1618 ms`；JSON 事件与 220 moving/0 still 日志核验同样通过。
- 2-way measured stage 均值约为 DiT `1150.63 ms`、VAE `405.26 ms`、camera condition `10.38 ms`。从 2 到 4 卡，DiT 加速 `2.03x`、VAE 加速 `1.74x`，完整 scheduler 加速 `1.90x`；尚不能据此外推 8-way，需实测通信拐点。
- 8-way 正式纯 moving 长跑：generated/delivered=`22.401/22.380 FPS`，scheduler mean/P50/P95/P99/max=`535.70/536/541/544/554 ms`；JSON 事件与 220 moving/0 still 日志核验通过。
- 8-way measured stage 均值约为 DiT `361.53 ms`、VAE `133.75 ms`、camera condition `10.36 ms`。4→8 卡的完整 scheduler 加速只有 `1.57x`，确认已经偏离线性扩展，但 8-way 仍是现有并行度中的最优点。
- 按 12 steady frames/chunk，当前 DiT 单项上限为 `33.19 FPS`，低于 40；compile-on 必须先把 DiT 从 361.53 ms 降到 300 ms 以下，异步 VAE 才有可能继续推进到 40。
- 为 compile 正确性对比保存 compile-off 的 chunk 20/21 原始 RGB，各 `14,376,960 bytes`，SHA256 分别为 `40fa3431...`、`581cbe9c...`。
- 全局 compile 的首次 22-chunk replay 在 measured chunk 20/21 仅 `4.05 FPS`；同一进程热 cache 第二次 replay 为 `14.08 FPS`，DiT/VAE=`704.3/116.5 ms`。因此不是只有冷编译成本，编译后 DiT kernel 本身也明显慢于 eager。
- eager 对 compile 的 chunk 20/21 原始 RGB 对比：MAE=`20.12/21.43`，PSNR=`17.70/17.24 dB`，变化像素比例约 `96.5%/96.8%`；注意这是 20 个 autoregressive chunks 后的累积差异，不等同于单步算子误差。
- 较保守的全局 `compile=default` 首次 replay 为 `4.26 FPS`、同进程热重放为 `14.74 FPS`；热分段 DiT/VAE=`635.3/148.8 ms`，仍全面落后于 eager，E5 至此关闭。

### 2026-07-13 04:00–04:19 EDT

- 为独立 server 增加可控的 Nsight wrapper 和 layerwise NVTX marker；自动 NVTX capture 未触发后，使用同一 Nsight session 手工 `start/stop`，没有修改模型、精度、steps 或 KV 配置。
- 第一份 6-chunk 报告因 KV 尚未稳态被降级；报告与客户端 JSON 已复制到本地，SHA256 分别为 `e717568f...`、`ee7dcf06...`。
- 第二份请求先跑 20 warmup chunks，再截取稳态窗口；正式报告包含 24 个 DiT rank-range，即 8 ranks × 3 chunks。报告与客户端 JSON 已复制到本地，SHA256 分别为 `36de72a9...`、`2298369b...`。
- 单 rank 中间 DiT chunk 的 8,354 个 GPU 操作全部落在 H100(5) 上，kernel busy=`480.917 ms`、首末 GPU op span=`503.905 ms`。组成：NCCL SendRecv=`167.128 ms (34.75%)`、FA3=`150.239 ms (31.24%)`、GEMM=`108.000 ms (22.46%)`、其他 kernel=`51.124 ms`。
- 代码映射确认 LingBot V2 是 40 层、40 heads、head_dim=128；sequence-shard self-attention 每层先对 packed QKV 调一次同步 `all_to_all_single`，attention 后再调一次。当前基线是 TP1×SP8；代码与参数校验允许 TP2×SP4，已将独立清单增加显式 `TP_SIZE` 变量，准备做单变量实测。

### 2026-07-13 04:20–04:41 EDT

- 恢复 TP1×SP8、eager、profiler-off 后做 60-chunk 回归：`22.800 FPS`、scheduler mean/P50/P95=`526.33/524/540 ms`，确认 profile rollout 未污染基线。
- 发现 auto 模式仍对 VAE 做 layerwise offload；切换到 speed 且显式保持 compile=false 后，两次独立 60-chunk 为 `23.423/23.337 FPS`，scheduler mean=`512.33/514.20 ms`。
- speed 常驻 VAE 的 chunk20/21 与原 eager 对照 SHA256 完全一致，逐字节差异为 0；保留为当前新底座。
- TP2×SP4 完整加载并服务成功，但 60-chunk 仅 `18.591 FPS`、scheduler mean=`645.48 ms`，直接淘汰。该配置的 chunk20/21 与 TP1 的 AR 结果也出现明显漂移，保留原始 RGB 与 JSON 供复核。

### 2026-07-13 04:42–05:05 EDT

- LL128 两次短跑稳定在 `23.84–23.86 FPS`，但与 eager 的 AR RGB 非 bitwise；同一 LL128 进程 replay 可重复，因此暂不接受为正确性优化。
- Simple 短跑 `23.472 FPS` 且 RGB bitwise；正式 20 warmup + 200 measured 为 generated/delivered=`23.238/23.213 FPS`，scheduler mean/P50/P95/P99/max=`516.39/516.5/525/528/536 ms`。
- 正式 JSON 协议核验：chunk0 `event_id=null`，其余 219 个 chunks `event_id=1`；200 个 measured chunks 全部 `num_frames=12`。
- 新增 benchmark-only 的 Ulysses A2A 理想旁路：只把本地 tensor reshape 到 collective 后的相同 shape，保留 downstream FA/GEMM/VAE 工作量但明确不保留模型语义。ruff、diff check 和 H100 容器内定向 pytest 通过（2 tests）。
- 旁路代码单独提交为 `38191bd71f...` 并推送到 `origin/codex/speedup-lingbot2`；K8s 清单将固定该 SHA，开启 stage logging 与旁路后测“通信完全免费”的乐观上限。

### 2026-07-13 05:06–05:14 EDT

- probe SHA、speed、TP1×SP8、Simple、stage logging 下完成零 A2A 60-chunk：20 warmup + 40 measured generated/delivered=`25.290/25.263 FPS`，scheduler mean/P50/P95/P99=`474.50/474/478/490 ms`。
- performance.log 与客户端 JSON 一一对应，共 60 条请求；measured 40 条的 DiT mean/P50/P95=`310.85/309.99/314.95 ms`，VAE=`140.07/141.08/143.76 ms`。
- 完全重叠 VAE/输出后的乐观上限按 DiT mean 为 `38.604 FPS`；这已在目标 40 以下。当前正用同 SHA、同 logging、同 Simple，只关闭旁路，补 A2A-on 配对基线。

### 2026-07-13 05:15–05:20 EDT

- 完成严格 A2A-on 配对：20 warmup + 40 measured generated/delivered=`23.518/23.489 FPS`，scheduler mean/P50/P95/P99/max=`510.25/509/520/525/525 ms`。
- 与零 A2A 的 `474.50 ms` 相比，完整链路可回收 A2A+layout 墙钟为 `35.75 ms`，不是 Nsight 中 profiler-inflated 的 `167.1 ms` kernel residency。
- measured stage 的 A2A-on/zero-A2A DiT=`362.02/310.85 ms`，但 VAE=`117.59/140.07 ms` 出现跨运行反向漂移；正式预算采用完整 scheduler 差值，阶段差只用于归因。
- 零 A2A 且完全隐藏 VAE/输出后仍需再回收 `10.85 ms (3.49%)` compute 才能达到 40 FPS；这证明当前计算路径的上限低于目标，但尚不把它夸大为任何未来 kernel 实现都无法跨越的硬件绝对上限。

### 2026-07-13 05:21–05:35 EDT

- 代码审计确认通用 ring attention 虽存在，但 LingBot causal sequence-shard 在 model 与 stage 两处都显式要求 `ring_degree=1`；当前不能作为无需开发的配置实验。
- 当前 causal path 已把 Q/K/V 合成一次 input A2A，每层再做一次 output A2A；“把三次 Q/K/V collective 合并”为已经存在的优化。
- 在空闲 H100 GPU 上按稳态 `Q=[1,4680,5,128]`、`K/V=[1,37440,5,128]` 做 FA3 100 次 CUDA-event 微基准：`num_splits=0/1/2/4/8/16` mean 分别为 `0.841/0.846/0.746/0.790/0.825/0.890 ms`，选择 2。
- 新增显式 `attention_backend_config.lingbot_causal_fa_num_splits`，只影响 LingBot causal self-attention，不改变 text/cross-attention；ruff、format、diff check 通过，本机 pytest 因既有环境缺 `pybase64` 未启动。
- 代码与测试提交 `afc619cc79a2960f9cab53b3823d904672f3c5c0` 已推送；8×H100 manifest 经 server-side dry-run 核对后 rollout。旧节点同时遭遇 Spot `instance-terminated-no-capacity` 回收，正在等待替代容量。

### 2026-07-13 05:36–06:02 EDT

- 通过 Spot request、Karpenter interruption events 与 EC2 状态交叉确认：原 `i-07331e6e9d21cb89c` 是 `instance-terminated-no-capacity`，不是 rollout 或代码导致。
- us-east-2 依次验证 2c、2a、2b 的 Spot/on-demand，均无 p5.48xlarge；NodePool 已恢复严格 8-GPU/192-CPU 上限。
- 只读发现 us-west-2 EKS Auto 集群；现有 p5e/H200 NodePool 不符合硬件身份，因此创建完全独立的 H100 NodeClass/NodePool、Service/Deployment 与 CPU client。两组大 GPU manifest 均通过 server-side dry-run。
- us-west-2a p5 Spot/on-demand 仍无容量；切换到 2b on-demand-only 后暂被 offering 负缓存短路，保留 Pending Pod 等待控制器自动重试。

### 2026-07-13 06:02–06:41 EDT

- 为防止两区同时补到容量导致超过 8 张上限，先把 us-east-2 自有 Deployment 缩为 0；us-west-2b 控制器随后真实返回 `InsufficientInstanceCapacity`，再将西部 Deployment 缩为 0。
- us-east-1 现有 EKS Auto 集群对当前 IAM 不开放 Kubernetes 访问；没有新增相邻集群管理员权限。独立 ap-southeast-2 EKS dry-run 通过，但 CloudFormation 创建权限被拒且没有产生资源。
- 对任务专属直接 EC2 清单执行 `RunInstances --dry-run` 后，成功获取 ap-southeast-2b Spot `i-07ea1cee39552512f`；验证 8 张 H100 80GB HBM3、Docker 与 582 GiB 根盘。
- 通过任务专属 SSH SG、EC2 Instance Connect 和持久控制连接部署固定源码 archive；容器依赖身份保持 FlashInfer Python/cubin/JIT cache 0.6.12，H100 测试 `6 passed`。
- 启动 E14 真实 A2A + FA3 split=2；server 参数身份已核验，固定模型 revision 正从公开 HF Hub 下载。下一检查点是服务 Ready 后同实例 `split=2/0/2` A/B/A、20 warmup + 40 measured 与 raw 输出正确性比对。

### 2026-07-13 06:41–07:02 EDT

- 固定模型 26 个文件、约 81 GB 完成首次下载；gp3 吞吐约 125 MiB/s，使下载耗时约 11 分钟，但加载/下载均不计入稳态 FPS。
- E14 split2 A1 是实例上的首个真实请求，虽然已有 20 warmup，measured DiT 仍为 `429.801 ms`；后续独立重启的 split0 B 与 split2 A2 为 `375.292/375.617 ms`，确认 A1 还有一次性初始化污染。
- B/A2 generated FPS=`23.158/22.822`、scheduler mean=`518.175/525.800 ms`；split2 没有完整链路或 DiT 收益。split0 B 的 raw SHA 与原正确基线一致，split2 自身可重复但与 split0 的 20-chunk AR 输出明显分叉，因此 E14 淘汰。

### 2026-07-13 07:02–07:31 EDT

- E15 先复现 zero-A2A split0：generated=`25.143 FPS`、scheduler=`477.275 ms`，与旧实例 E12 的 `25.290 FPS/474.50 ms` 接近，证明直接 EC2 路径可复现实验量级。
- 为控制进程装载方差，split0 做 3 次独立装载、4 个 measured run；split2 做 3 次独立装载、6 个 measured run。所有 JSON 均核验 20 warmup + 40 measured、measured 全为 12 帧、chunk1–59 event_id=1。
- split2 的 6 个 DiT mean 为 `320.676/319.994`、`313.820/310.541`、`318.615/319.641 ms`。最快完整 DiT 均值 `310.541 ms` 对应 `38.642 FPS`；即使 A2A、layout、VAE 与输出全部免费，仍未跨过 300 ms。
- 在相邻 split2/split0 run 期间记录 8 卡功耗、SM、时钟与 power violation；两者都处于功耗限制，最快 split2 不是较高 boost 时钟造成。保留 `.dmon`、performance JSONL 与客户端 JSON 供复核。

### 2026-07-13 07:31–07:34 EDT

- E14/E15 原始结果已复制到本地；E15 每个 JSON 的协议断言全部通过，结果目录总计约 487 MB。
- 发起终止直接 EC2 `i-07ea1cee39552512f`；根卷 `vol-02b3002d87d405b8a` 为 DeleteOnTermination。实例仍在 AWS shutting-down，专属 ENI/SG 待释放后继续清理。
- 已发起 us-east-2 与 us-west-2 两套自有对象清理；首轮输出确认 Deployment 与 NodePool 删除，没有操作隔壁 LingBot 资源。其余类型由下一轮最终扫描核验。

### 2026-07-13 07:34–07:44 EDT

- graceful termination 长时间停在 shutting-down；取消 one-time Spot request，分别对 force 与 skip-OS-shutdown 做权限 dry-run 后精确作用于 `i-07ea1cee39552512f`。最终实例为 terminated，根卷查询返回 NotFound，任务 ENI 为空。
- 删除任务专属 SG `sg-0c97b903f90a2535a` 并以 `InvalidGroup.NotFound` 复核；没有删除实例同时挂载的 default SG。
- 清理后扫描发现组合式 kubectl 命令遗漏后续异构类型，遂逐个删除两区遗留 Service/Pod/NodeClass。最终 us-east-2、us-west-2、ap-southeast-2 的任务 tag 活跃实例列表均为空，两套 EKS context 的 `codex-lingbot2-h100-perf` 对象列表均为空。
