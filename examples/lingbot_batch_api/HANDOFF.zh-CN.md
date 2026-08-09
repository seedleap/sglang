# LingBot2 推理服务交接手册

本文面向最终接管 LingBot2 数据生成服务的同事。目标不是只接手一次批次，
而是逐步接管任务 API、SGLang 调用、视频编码、S3 结果、GPU 部署、扩缩容和日常运维。

> 安全说明：`seedleap/sglang` 是公开仓库。本文只记录可公开的协议、代码位置、
> 架构和操作清单。AWS 账号、EKS context、真实负载均衡域名、IP 白名单、IAM Role、
> ECR 地址、S3 写权限和密钥必须通过公司内部安全渠道交接，不得提交到 Git。

## 1. 最终职责边界

最终由接手方拥有整个业务链路：

```text
业务调用方
    |
    | HTTP: 提交 task_id、查询状态
    v
接手方的任务服务
    |-- 队列、限流、幂等、重试
    |-- 调用 SGLang WebSocket
    |-- raw RGB -> MP4
    |-- 上传 S3
    `-- webhook / 查询接口
    |
    v
接手方维护的 SGLang GPU 池
    |-- Kubernetes Deployment
    |-- B300 Spot
    |-- Karpenter / 扩缩容
    `-- 监控、升级、回滚、故障处理
```

原维护方在最终交接完成后，不再负责：

- 任务 API、`task_id`、webhook 或任务状态；
- SGLang 服务的日常扩缩容；
- Spot 中断后的任务重试；
- S3 结果完整性和补跑；
- 镜像升级、模型升级和线上告警。

## 2. 交接分两个阶段

| 阶段 | 接手方负责 | 原维护方暂时负责 | 完成标志 |
|---|---|---|---|
| 阶段 1：消费现有实例 | 任务 API、队列、WebSocket 客户端、MP4、S3、webhook、重试 | 提供固定版本的 SGLang 实例和容量说明 | 接手方可以独立完成端到端批次，实例故障可自动重试 |
| 阶段 2：接管部署运维 | 阶段 1 全部内容，加镜像、EKS、B300 Spot、扩缩容、监控、升级回滚 | 只提供必要答疑 | 接手方能独立部署、增减实例、处理 Spot 和完成值班 |

不要把阶段 1 做成依赖临时 Pod IP 的一次性脚本。阶段 1 应使用稳定的 Service/LB
地址，并把 SGLang 当作可能中断、需要重试的下游服务。

## 3. 当前状态快照

以下状态核查于 2026-07-16 04:02 UTC，仅用于说明起点，交接当天必须重新确认。
当前有两套容易混淆的环境：

**Capacity Block 批推理环境：**

- 两台 `p6-b300.48xlarge` 正在运行推理，每台 8 GPU；
- 节点属于 Capacity Block nodegroup `wan22-cb-p6b300-0715-20c`；
- 当前 Job 是 `codex-lingbot2-tpvremain3699x5-720p`，`parallelism=2`；
- Job 共 75 个 indexed shards；每个 Pod 独占一台 B300，在 Pod 内启动四个 2-GPU
  SGLang 进程，因此两台主机合计同时处理 8 条视频；
- 这是直接批处理 Job，不是由稳定 Kubernetes Service 暴露的长期在线推理池；Pod
  完成 shard 后会退出，不能把 Pod IP 当作阶段 1 的交付地址。

**历史 H100 serving 环境：**

- 已有一个保留 WebSocket Upgrade 的 Nginx 网关，网关副本为 `2/2`；
- 网关后面的 H100 SGLang Deployment 当前缩容到 `0`，因此该网关当前没有推理容量；
- H100 Deployment 是调试/Profiling 遗留形态，包含开发镜像和 Nsight 启动方式，
  不能直接作为长期生产基线。

另外：

- 当前没有作为长期 Service 运行的 B300 SGLang Deployment；
- 本目录的 Helm Chart 是参考实现，默认把 HTTP adapter 与 SGLang 放在同一个 Pod，
  尚未部署为正式公网 API；
- 现有批处理 Job 和结果脚本位于 `benchmark/lingbot2_offline_batch/`，它们是历史
  数据生成、当前 Capacity Block 批次和性能证据，不是稳定在线服务。

因此，“两台 B300 正在推理”和“同事已有稳定 WebSocket 服务可接入”是两件事。
阶段 1 开始前需要原维护方把同一套已验证的 4x2-GPU runtime 整理成稳定 Service，
或者与接手方共同把其 worker 放进 B300 Pod；同时交付稳定地址、可用时间、实例数量、
单实例并发和停机通知方式。

### 阶段 1 线下交付清单

以下信息不得填入本公开文档，应通过内部安全渠道交接：

- AWS 账号、Region、EKS cluster/context 和 Namespace；
- WebSocket 的 `wss://` 地址或内部 `ws://` 地址；
- 网络接入方式：VPC、PrivateLink、VPN 或公网出口 IP 白名单；
- 鉴权方式和密钥轮换负责人；
- 输入 S3 读取范围、输出 S3 写入范围和 IRSA/IAM Role；
- ECR 仓库、镜像 digest、代码 commit 和模型 revision；
- 当前副本数、每副本 GPU 数、并发上限和预计下线时间；
- CloudWatch/Grafana dashboard 与告警群；
- 成本归属标签、预算负责人和 Spot/On-Demand 策略。

## 4. SGLang WebSocket 协议

SGLang 提供的是低层视频流接口，不提供 `task_id`、队列、webhook、MP4 或 S3：

```text
WS /v1/realtime_video/generate
```

可直接复用的客户端参考实现：

- [`lingbot_batch_api/realtime_client.py`](lingbot_batch_api/realtime_client.py)：
  建连、MessagePack、raw RGB、ffmpeg；
- [`lingbot_batch_api/actions.py`](lingbot_batch_api/actions.py)：动作生成和帧到 latent
  action 的量化；
- [`client_example.py`](client_example.py)：HTTP 重试和退避参考；
- [`lingbot_batch_api/server.py`](lingbot_batch_api/server.py)：完整同步 HTTP adapter
  参考。最终是否使用该 HTTP 形态由接手方决定。

### 4.1 请求流程

1. 客户端建立 WebSocket；一条连接就是一条生成 session。
2. 客户端发送一个二进制 MessagePack `init` 消息。
3. 服务端返回 MessagePack 控制消息。
4. 收到 `frame_batch_header` 后，下一条 WebSocket 消息是 raw RGB bytes。
5. 客户端将视频帧送入 ffmpeg，得到 MP4。
6. 客户端上传 S3，更新任务状态，然后发送 webhook 或允许调用方查询。

当前 720p/5 秒生产参数：

| 字段 | 值 |
|---|---:|
| 分辨率 | `1280x720` |
| FPS | `24` |
| 最终保留帧数 | `129` |
| 视频时长 | `5.375s` |
| 推理 steps | `4` |
| guidance scale | `1.0` |
| max chunks | `11` |
| 输出格式 | RGB24 raw frames |

参考 `init`：

```python
init_payload = {
    "type": "init",
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "first_frame": presigned_https_url,
    "size": "1280x720",
    "fps": 24,
    "num_frames": 9,
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "seed": video_seed,
    "max_chunks": 11,
    "realtime_output_format": "raw",
    "output_compression": 95,
    "realtime_output_pacing": False,
    "enable_upscaling": False,
    "enable_frame_interpolation": False,
    "profile": False,
    "profile_all_stages": False,
    "condition_inputs": {"camera_actions": latent_actions},
}
```

注意事项：

- `first_frame` 是必需条件。S3 URI 应由调用侧转换为有时效的 HTTPS presigned URL；
- 消息使用 MessagePack，不是 JSON；
- raw payload 大小必须等于 `num_frames * 1280 * 720 * 3`；
- `frame_batch_header` 与紧随其后的 raw payload 必须成对读取；
- `error` 消息应转成可重试或不可重试的任务错误；
- 当前客户端同时等待每个 chunk 的 frame 和 stats 完成，最后只保留 129 帧；
- 参考 ffmpeg 参数是 `libx264 / veryfast / crf 18 / yuv420p / +faststart`；
- 不应把 raw RGB 长期写入 S3，只保存编码后的 MP4 和小型任务元数据。

### 4.2 动作约定

一条视频的原始动作是：

```text
57 帧：wasd 中一个键
15 帧：noop
57 帧：ijkl 中一个键
```

模型控制序列为：

```text
1 个 reference noop + 14 个 movement + 4 个 noop + 14 个 camera = 33 controls
```

每张图生成 5 个 variant 时，应从 `4 * 4 = 16` 个 `(wasd, ijkl)` 组合中无放回
选择 5 个。无需依赖 `trajs.jsonl`；当前确定性算法见 `actions.py`。同一个输入、
`image_index`、`variant_slot` 和 `action_seed` 必须产生相同动作，方便重试和复现。

## 5. 阶段 1：接入现有 SGLang

### 5.1 推荐由接手方实现的最小任务系统

如果任务约 20--60 秒且调用方可以保持 HTTP 请求，最小方案是同步 HTTP：

```text
POST /tasks/generate
    -> 调用 SGLang WebSocket
    -> 编码并上传 S3
    -> HTTP 200 返回 task_id + s3_uri
```

如果必须立即返回并通过 webhook 通知，则使用异步模式：

```text
POST /tasks -> 202 + task_id
             -> durable queue
             -> worker -> SGLang -> MP4 -> S3
             -> status store
             -> webhook dispatcher
```

异步模式在 Spot 环境中不能只靠进程内 background task。至少应有持久队列；否则
Pod/实例被回收后，已经返回 `202` 的任务会永久丢失。

建议的业务接口最少包含：

- `POST /tasks`：提交并幂等返回 `task_id`；
- `GET /tasks/{task_id}`：查询 `queued/running/succeeded/failed`；
- 可选 webhook：完成和永久失败通知；
- 管理接口：重试、取消尚未开始的任务、按批次统计。

### 5.2 幂等和重试

推荐使用调用方给出的 `task_id` 作为全局幂等键，并派生确定性输出路径：

```text
s3://<bucket>/<prefix>/<sha256(task_id)[:2]>/<sha256(task_id)>.mp4
```

执行顺序建议为：

1. 检查 `task_id` 是否已有成功结果；
2. 记录请求 fingerprint，拒绝同一个 `task_id` 携带不同 payload；
3. 调用 SGLang；
4. MP4 完整编码后再上传正式 key；
5. `HEAD` 校验对象存在、大小大于零；
6. 将任务标记为 `succeeded`；
7. 异步发送 webhook，回调失败不能触发重新推理。

以下错误通常可使用同一 `task_id` 重试：

- WebSocket 建连失败或意外断开；
- `429/502/503/504`；
- Spot 中断；
- 服务端重启；
- S3 临时错误。

请求校验失败、同一 `task_id` payload 冲突等错误不应无限重试。所有重试都必须有
指数退避、随机 jitter、最大次数和 DLQ。

### 5.3 webhook

webhook 是通知，不应成为唯一结果来源。必须同时提供任务查询接口。

- webhook URL 最好预注册或绑定租户，不允许任意 URL，避免 SSRF；
- 使用 HTTPS、HMAC 签名、timestamp 和唯一 `event_id`；
- 接收方按 `event_id` 去重；
- 回调设置短超时并独立重试；
- payload 至少包含 `event_id/task_id/status/s3_uri/attempt`；
- 推理完成与 webhook 投递必须解耦，回调失败不能重复占用 GPU。

### 5.4 并发和背压

当前 SGLang realtime 路径的一个实例只应运行一个 active session。不要把
`_wait_for_active_session_slot` 当服务端任务队列：

- 阶段 1 只有一个 SGLang 实例时，接手方并发上限先设为 `1`；
- 有多个实例时，简单的 Service 负载均衡可能把新连接发到忙实例；
- 生产方案优先使用“一份任务 worker + 一份 SGLang、同 Pod localhost 调用”，
  由队列给空闲 worker 分配任务；
- 如果 worker 在集群外，则需要显式维护后端 slot/lease，而不是盲目提高连接数；
- 记录 `queued/running` 数和端到端吞吐，根据成功率逐步增加生产流量。

### 5.5 阶段 1 验收

- [ ] 10 条固定 smoke case 能稳定得到 1280x720、24fps、129 帧 MP4；
- [ ] 相同 `task_id` 重试不会产生不同动作或不同 S3 key；
- [ ] 模拟 WebSocket 中断后任务可以自动重试成功；
- [ ] 模拟 webhook 失败不会重新推理；
- [ ] 同一 `task_id` 不同 payload 返回冲突；
- [ ] 队列、运行中、成功、失败、DLQ 数量可观测；
- [ ] 连续跑至少 100 条，结果数、任务数和 S3 对象数一致；
- [ ] 原维护方只需要保证 SGLang endpoint，不参与批次补跑。

## 6. 阶段 2：接管 SGLang 部署和运维

### 6.1 推荐最终形态

对于“异步任务 + Spot + webhook”，推荐每个 GPU Pod 同时包含任务 worker 和
SGLang，worker 只通过 localhost WebSocket 调用该 Pod 内的 SGLang：

```text
HTTP API -> SQS -> [worker + 2-GPU SGLang] x N -> S3
                    |                    |
                    `-- one task slot ---'
                   KEDA/HPA scales Pods
                   Karpenter scales B300 Spot nodes
```

这样可以避免负载均衡把连接随机送到忙实例，并把“一个 worker = 一个推理 slot”变成
明确约束。接手方也可以采用等价设计，但必须证明 slot 分配和 Spot 重试正确。

### 6.2 固定生产版本

生产部署必须固定以下版本：

- SGLang Git commit；
- ECR image digest，禁止浮动 `latest/dev`；
- 模型 ID 和 model revision；
- CUDA、PyTorch、Flash Attention backend；
- SGLang 启动参数和环境变量；
- ffmpeg 版本及编码参数；
- Helm Chart/manifest commit。

当前 2-GPU B300 参考启动参数位于
[`deploy/helm/templates/deployment.yaml`](deploy/helm/templates/deployment.yaml)，核心是：

```text
--pipeline-class-name LingBotWorldCausalDMDPipeline
--num-gpus 2
--performance-mode speed
--tp-size 1
--sp-degree 2
--ulysses-degree 2
--dit-cpu-offload false
--text-encoder-cpu-offload false
--vae-config.use-parallel-decode true
--vae-config.parallel-decode-mode spatial
--enable-torch-compile false
--attention-backend-config lingbot_causal_fa_num_splits=0
```

以及：

```text
SGLANG_LINGBOT_LAZY_VAE_ENCODE_BLACK_FRAMES=60
SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW=true
```

升级任一版本或参数前，都要运行固定 smoke case 和小批量吞吐回归；不要只验证
`/health` 返回 200。

### 6.3 B300 容量基线

当前测得的 720p/24fps/129f 基线：

| 项目 | 基线 |
|---|---:|
| 拓扑 | 每个 SGLang replica 使用 2 GPU |
| 单台 B300 | 4 replicas / 4 concurrent videos |
| 单视频延迟 | 约 26 秒，尾延迟需重新测量 |
| 单台吞吐 | 约 556 videos/hour |
| 两台理论日产量 | 约 26,688 videos/day |

因此两台 B300 在当前实测吞吐下达不到 30,000 条/天。30,000 条/天需要每台至少
`625 videos/hour`，比当前基线高约 12.4%；生产规划应按三台或按可接受的安全余量
配置，而不是按两台的理论满载值承诺。

吞吐定义和历史测试入口见
[`benchmark/lingbot2_offline_batch/README.md`](../../benchmark/lingbot2_offline_batch/README.md)。
扩容后必须重新测量 videos/hour、P50/P95 latency、成功率和 GPU 利用率。

### 6.4 扩缩容

GPU 工作负载不应只按 API sidecar CPU 扩容。推荐信号优先级：

1. 队列中可见任务数和最老任务等待时间；
2. running task 数与可用推理 slot 数；
3. GPU 利用率用于诊断，不单独作为任务容量信号。

建议：

- KEDA/HPA 根据 queue depth 扩 SGLang/worker Pod；
- Pending GPU Pod 触发 Karpenter 创建 B300 Spot node；
- 一台 B300 最多放四个 2-GPU Pod，并验证 CPU、内存、`/dev/shm` 也足够；
- 设置较长 scale-down stabilization，避免频繁加载模型；
- 缩容前只终止空闲 worker；活跃 WebSocket 需要 drain 或允许任务重试；
- 设置 PodDisruptionBudget，但不要认为它可以阻止 Spot interruption；
- 设置最大副本数和每日成本告警；
- Spot 无容量时让任务留在持久队列，不能在网关内无限等待。

[`deploy/karpenter-b300-spot.example.yaml`](deploy/karpenter-b300-spot.example.yaml)
仅为模板，引用的 EC2NodeClass 必须替换成目标集群真实对象后才能使用。

### 6.5 健康检查和流量

必须区分三种状态：

- 进程存活：liveness；
- 模型加载完成、可以建连：readiness；
- 当前有空闲 session slot：capacity/busy。

网关自身 `/healthz` 为 200 不代表后端模型可用。阶段 2 应确保 LB 只把新连接发到
模型 ready 的后端，并让任务 worker 在消费队列消息前确认本地 SGLang ready。

WebSocket 建立后天然固定在一个 Pod。Spot 中断会断开连接，调用方必须按同一
`task_id` 重跑整个视频。不要尝试把一条进行中的 session 迁移到另一 Pod。

### 6.6 监控与告警

最低监控集合：

| 层 | 指标/事件 |
|---|---|
| 任务 | queued、running、success、failed、DLQ、oldest age |
| 接口 | 请求量、4xx/5xx、提交延迟、查询延迟、webhook 成功率 |
| 推理 | videos/hour、P50/P95 latency、WS connect/error/disconnect |
| 视频 | frame count、分辨率、FPS、MP4 bytes、ffmpeg failure |
| S3 | upload latency/error、结果对象缺失、重复 key 冲突 |
| GPU | utilization、memory、temperature、Xid、OOM |
| Kubernetes | Pending、CrashLoop、readiness、restart、eviction |
| Spot | interruption notice、node churn、扩容等待时间、无容量 |
| 成本 | node-hours、每千条成本、空闲 GPU 时间 |

至少对以下情况告警：队列最老任务超阈值、连续推理失败、DLQ 非零、无 ready GPU、
S3 上传失败、Spot 大面积中断、24 小时产量低于计划。

### 6.7 常用 Runbook

接手方应在自己的运维仓库补齐并演练：

- 从零部署一个固定版本 SGLang replica；
- 安全地将 replicas 从 0 扩到 N、从 N 缩到 0；
- 查看模型加载进度和 readiness 失败原因；
- 查看某个 `task_id` 的完整日志链路；
- 处理 WebSocket 断开、GPU OOM/Xid 和 ffmpeg 失败；
- 处理 S3 写入失败和同 key payload 冲突；
- 处理 Spot interruption 和 Spot 无容量；
- 镜像/模型升级、灰度、回滚；
- 对 DLQ 任务执行幂等补跑；
- 停止服务并确认不再产生 GPU 和 LB 成本。

## 7. 阶段 2 验收和最终移交

- [ ] 接手方从空集群状态独立部署一套固定版本服务；
- [ ] 接手方独立将容量从 0 扩到一台 B300，再扩到两台或更多；
- [ ] 新节点 ready 后 smoke case 自动通过，失败时不会接流量；
- [ ] 主动删除一个运行中 Pod，任务能从队列自动恢复；
- [ ] 模拟 Spot 中断，任务没有永久丢失且不会重复交付不同结果；
- [ ] 完成一次版本升级和回滚演练；
- [ ] dashboard、告警、DLQ 和成本告警由接手方账号拥有；
- [ ] EKS、ECR、S3、IAM、DNS/证书权限不再依赖原维护方个人身份；
- [ ] 接手方独立值守至少一个完整生产批次；
- [ ] 原维护方退出日常扩缩容、补跑和故障处理群。

## 8. 代码地图

| 路径 | 用途 |
|---|---|
| `examples/lingbot_batch_api/lingbot_batch_api/realtime_client.py` | SGLang WebSocket 和 raw frame 客户端 |
| `examples/lingbot_batch_api/lingbot_batch_api/actions.py` | 动作组合、129 帧和 33 controls |
| `examples/lingbot_batch_api/lingbot_batch_api/server.py` | 同步 HTTP、S3、幂等参考实现 |
| `examples/lingbot_batch_api/client_example.py` | 429/5xx 退避重试参考 |
| `examples/lingbot_batch_api/deploy/helm/` | API sidecar + 2-GPU SGLang 参考 Chart |
| `examples/lingbot_batch_api/deploy/karpenter-b300-spot.example.yaml` | B300 Spot NodePool 模板 |
| `benchmark/lingbot2_offline_batch/` | 历史批处理、性能测试、输入输出工具 |
| `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py` | WebSocket 服务端协议与 active session slot |

## 9. 交接会议建议议程

1. 用一条固定 case 现场演示 WebSocket -> raw RGB -> MP4 -> S3；
2. 现场解释 `task_id` 幂等、Spot 中断和 webhook 失败的处理顺序；
3. 线下交付 AWS/EKS/ECR/S3/IAM/网络访问清单；
4. 由接手方现场完成一次阶段 1 smoke；
5. 对齐阶段 2 的队列、worker/SGLang Pod、KEDA/Karpenter 方案；
6. 确定阶段 1 和阶段 2 的负责人、截止日期和验收时间；
7. 最终以第 7 节 checklist 全部完成作为运维责任转移标准。
