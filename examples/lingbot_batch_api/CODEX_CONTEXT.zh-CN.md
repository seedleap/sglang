# Codex 运行上下文：LingBot2 推理交接

本文供后续 Codex 会话快速恢复上下文。它不是实时状态数据库。任何关于“当前在跑什么、
还有多少节点、Job 是否成功”的回答，都必须先执行只读查询，不能直接复述本文快照。

## 1. 当前目标

协助一位同事分两阶段接管 LingBot2 数据生成：

1. 先基于原维护方提供的现有 SGLang runtime，完成任务 API、队列、WebSocket 调用、
   MP4、S3、webhook、重试和业务流控；
2. 再接管 SGLang 镜像、EKS、B300 Spot/Capacity Block、扩缩容、监控和全部运维。

最终责任边界是接手方独立拥有整个系统，原维护方不再负责扩缩容、补跑和日常故障。
面向人的完整交接手册见 [`HANDOFF.zh-CN.md`](HANDOFF.zh-CN.md)。

## 2. Git 与 worktree

- GitHub repo：`seedleap/sglang`，仓库是 **public**；
- 相关 worktree：`/Users/chenshengdong/workspace/sglang-lingbot-batch-api`；
- 相关 branch：`codex/lingbot-batch-api`；
- 本分支起始功能 commit：`0dc586baa3`，内容是轻量 HTTP adapter、Helm、B300 Spot
  模板以及离线 batch 工具；
- 主 worktree `/Users/chenshengdong/workspace/sglang` 可能有用户的未提交改动，禁止
  为完成交接任务而清理、reset 或覆盖它。

开始修改前执行：

```bash
git worktree list --porcelain
git -C /Users/chenshengdong/workspace/sglang-lingbot-batch-api status --short --branch
git -C /Users/chenshengdong/workspace/sglang-lingbot-batch-api remote -v
```

仓库公开，因此禁止提交 AWS account、真实 ELB、IP 白名单、token、secret、私有 IAM
Role、私有 ECR 地址或临时 presigned URL。真实接入信息必须线下交接。

## 3. 两套环境，不要混淆

### 3.1 当前 B300 Capacity Block 批处理

2026-07-16 04:02 UTC 的已核查快照：

```text
kubectl context: leap-world-aws03-usw2
namespace:       default
nodegroup:       wan22-cb-p6b300-0715-20c
instance type:   p6-b300.48xlarge
active hosts:    2 x 8 GPU
Job:             codex-lingbot2-tpvremain3699x5-720p
Job mode:        Indexed, completions=75, parallelism=2
execution mode:  direct-two-hosts
```

每个 indexed Pod 独占一台 8-GPU B300，并在本机启动四个 2-GPU SGLang server：

```text
GPU 0,1 -> ws://127.0.0.1:30000/v1/realtime_video/generate
GPU 2,3 -> ws://127.0.0.1:30100/v1/realtime_video/generate
GPU 4,5 -> ws://127.0.0.1:30200/v1/realtime_video/generate
GPU 6,7 -> ws://127.0.0.1:30300/v1/realtime_video/generate
```

两台主机合计 8 个并发视频。SGLang 只监听 localhost，Pod 由 Job 管理并在 shard 完成
后退出；它们不是可以直接交付同事的稳定 WebSocket endpoint。

当前 Job 固定的重要版本：

```text
code git ref:  d9a7e0e6630ea8aea135191115a13e6451618a6f
model:         robbyant/lingbot-world-v2-14b-causal-fast-diffusers
model revision:59cccf49f2d2dd27418ae7a04b82b10868d455c2
runtime image: lmsysorg/sglang:dev@sha256:8f78575e03ab59a39191a4a6f718bbbe1726fa940f72a86a187a3f1628ada9a7
```

不要把 Job 的 `lmsysorg/sglang:dev` 字符串误判为浮动运行时：当前 Pod YAML 已固定 image
digest；但长期交接仍应构建 seedleap 自己的不可变 ECR image，避免 Pod 启动时 clone 和
`pip install -e`。

### 3.2 历史 H100 serving

另一套资源位于 `leap-world-use2`：

```text
Deployment: lingbot2-h100          replicas=0
Deployment: lingbot2-gateway       ready=2/2
Service:    lingbot2-h100-backend
Service:    lingbot2-h100-public
```

网关存在不等于后端有容量。该 H100 Deployment 还是 Profiling/调试形态，不能作为当前
B300 Capacity Block 作业的证据，也不能未经恢复验证就交给调用方。

## 4. 每次状态问题的只读核查顺序

先确认 context，再看 Job、Pod 和 node；不要只看一个名称或旧日志：

```bash
kubectl --context leap-world-aws03-usw2 \
  -n default get job codex-lingbot2-tpvremain3699x5-720p -o wide

kubectl --context leap-world-aws03-usw2 \
  -n default get pods \
  -l job-name=codex-lingbot2-tpvremain3699x5-720p -o wide

kubectl --context leap-world-aws03-usw2 get nodes \
  -l eks.amazonaws.com/nodegroup=wan22-cb-p6b300-0715-20c \
  -o custom-columns='NAME:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,READY:.status.conditions[?(@.type=="Ready")].status,GPUS:.status.allocatable.nvidia\.com/gpu'
```

查看某个 active Pod 的启动、进度和错误：

```bash
pod=$(kubectl --context leap-world-aws03-usw2 -n default get pod \
  -l job-name=codex-lingbot2-tpvremain3699x5-720p \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')

kubectl --context leap-world-aws03-usw2 -n default logs "$pod" --tail=200
kubectl --context leap-world-aws03-usw2 -n default describe pod "$pod"
```

如果 Job 名改变，先列出资源，不要继续用旧名字：

```bash
kubectl --context leap-world-aws03-usw2 -n default get jobs \
  -l seedleap.ai/owner=chenshengdong -o wide
```

状态判断必须区分：

- Job `.status.active/succeeded/failedIndexes`；
- Pod `Running/Pending/Failed/ContainerStatusUnknown`；
- node `Ready` 和 `allocatable nvidia.com/gpu`；
- S3 已上传对象数或 FSx `progress.jsonl`；
- “Job 存在”“Pod Running”和“持续产生有效 MP4”不是同一件事。

## 5. 关键代码地图

```text
examples/lingbot_batch_api/
  HANDOFF.zh-CN.md                    面向同事的完整交接手册
  CODEX_CONTEXT.zh-CN.md              本文件
  lingbot_batch_api/realtime_client.py WS、MessagePack、raw RGB、ffmpeg
  lingbot_batch_api/actions.py         16 组动作、129 帧、33 controls
  lingbot_batch_api/contracts.py       HTTP request 校验
  lingbot_batch_api/server.py          同步 HTTP、幂等、S3 参考 adapter
  client_example.py                    429/5xx backoff 参考
  deploy/helm/                         API sidecar + 2-GPU SGLang 参考 Chart
  deploy/karpenter-b300-spot.example.yaml

benchmark/lingbot2_offline_batch/
  run_capacity_smoke_720p.sh           每台 8-GPU 主机启动 4x2-GPU SGLang
  run_capacity_shard_720p.sh           indexed shard、resume、上传摘要
  benchmark_evalset.py                 多 WS URL 调度和 MP4 生成
  k8s-thirdperson-remaining3699x5-720p-b300.yaml
  README.md                            指标和运行入口

python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/
  realtime_video_api.py                WS 协议和 active session slot
```

## 6. 协议事实

- Endpoint：`/v1/realtime_video/generate`；
- 传输：WebSocket + MessagePack，不是 JSON streaming；
- 一条 WebSocket 连接是一条 session；
- 当前一个 SGLang process 只运行一个 active session；
- `frame_batch_header` 的下一条消息是 RGB24 raw bytes；
- 生产输出：1280x720、24fps、129 frames、约 5.375 秒；
- `first_frame` 必需，参考客户端将 S3 URI 转成 HTTPS presigned URL；
- 11 chunks，最终保留 129 帧；
- raw frame 由调用侧 ffmpeg 编码 MP4，再上传 S3；
- SGLang 不负责 `task_id`、队列、S3、webhook 或任务状态。

动作：57 帧 movement (`wasd`) + 15 帧 noop + 57 帧 camera (`ijkl`)；模型输入为
1 reference noop + 14 movement + 4 noop + 14 camera = 33 controls。每张图的 5 个 variant
使用确定性、无重复的 `(wasd, ijkl)` 组合；不要再要求 `trajs.jsonl`。

## 7. 架构决策

阶段 1 可以让接手方在集群外调用原维护方提供的稳定 SGLang endpoint，但当前 batch
Job 本身不是这个 endpoint。必须先完成以下之一：

1. 将已验证的 4x2-GPU runtime 部署成稳定 Service；或
2. 将接手方 worker 与四个 SGLang server 放在同一 B300 Pod，由 worker 调 localhost。

最终异步/Spot 形态推荐：

```text
HTTP API -> durable queue -> [worker + 2-GPU SGLang] x N -> S3
                              KEDA/HPA -> Pods
                              Karpenter -> B300 Spot nodes
```

原因：一个 worker 对应一个 session slot，避免普通 Service 将 WS 随机路由到忙 Pod；
Spot 中断时消息仍在持久队列中，可以用同一 `task_id` 重试。

如果使用 webhook：回调是通知，不是唯一真相。必须还有任务查询接口、HMAC、event_id
去重、独立回调重试和 SSRF 防护。回调失败不能触发重新推理。

## 8. 性能和容量基线

720p/24fps/129f、4x2-GPU/B300 的已测基线：

```text
single-video latency: about 26 seconds
one B300 host:        about 556 videos/hour
two hosts ideal:      about 1,112 videos/hour = 26,688 videos/day
```

两台按当前基线达不到 30,000/day。达到目标需要每台至少 625 videos/hour，或准备三台
主机/更大的安全余量。数字可能因代码、模型、Flash Attention 和输入变化而漂移；任何
新承诺都要重新实测，不得把历史峰值直接当 SLA。

## 9. 修改和运维边界

- 用户只问状态、原因或方案时，保持只读；
- 用户明确要求部署、扩缩容、提交 Job、停止实例或修改代码时才执行 mutation；
- Capacity Block/Spot 资源可能正被其他批次使用，扩缩容前先确认 active Pod 和节点占用；
- 不要删除、重启或 unsuspend 后续 Job，除非用户当前明确授权；
- 不要下载大量视频做状态检查，优先看 `progress.jsonl`、Job indexes 和 S3 object count；
- 任何 Git 修改只处理当前目标，保留其他 worktree 的用户改动；
- commit 前运行文档链接/格式检查和相关单测；push 到 `origin` 的
  `codex/lingbot-batch-api`，除非用户明确指定别的分支。

## 10. 后续 Codex 的第一步

接到新任务时先分类：

- **当前状态**：执行第 4 节只读命令；
- **任务 API/业务层**：从 `server.py`、`realtime_client.py` 和人的交接文档开始；
- **SGLang 协议问题**：读 `realtime_video_api.py` 和 `realtime_client.py`；
- **吞吐问题**：读 benchmark summary/progress，报告 videos/hour 和 videos/day；
- **部署接管**：先确认目标是 Capacity Block、Spot 还是常驻 H100，不能混用 context；
- **正式交接完成**：以 `HANDOFF.zh-CN.md` 第 7 节 checklist 为准。
