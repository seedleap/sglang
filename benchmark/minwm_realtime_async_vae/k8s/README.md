# MinWM Realtime Production Topology

本目录是可重复部署的生产链路，不是直连 Denoiser 的验证拓扑：

```text
NLB -> Gateway CPU Pool -> Coordinator CPU Pool -> one H100 Spot node
                                             \-> L4 TAEHV (On-Demand preferred, Spot fallback)
TAEHV -> owning Gateway -> Browser
```

Gateway 和 Coordinator 各至少两个跨 AZ CPU Pod。Coordinator 使用 DynamoDB
On-Demand 保存短期用户、Session 和 Worker slot Lease；GPU KV、latent history 和
TAEHV context 只保存在绑定 Worker 本地。H100 与 L4 使用独立 NodePool；生产 GPU
workload 有最小副本保护，禁止缩容到 0。L4 不满足延迟门禁时，使用不在 base
kustomization 中的 `l40s-vae.yaml`。

GPU 基线为一台 8 卡 H100 Spot `p5.48xlarge`。单机部署两个 Zing SP=2 worker
（每个 1 个 Session）和一个 LingBot2 SP=4 worker（4 个 Session），共占满 8 卡，
全局容量为 Zing 2 个 Session + LingBot2 4 个 Session。Coordinator 会按 Worker 的
有效占用率、队列深度和服务耗时选择 slot。

### H100 本地 NVMe 约束

- H100 NodePool 必须引用 `minwm-async-denoiser-8gpu-nvme-ec2`，并设置
  `instanceStorePolicy: RAID0`。AL2023 会在 kubelet 启动前把 p5 的本地 NVMe 组成
  `/dev/md/0`，挂载到 `/mnt/k8s-disks/0`。
- Zing 与 LingBot2 的共享模型缓存固定使用
  `/mnt/k8s-disks/0/minwm-model-cache`，禁止回退到根盘的
  `/var/lib/minwm-model-cache`。原始模型仍从同区域只读 S3 serving artifact staging；
  节点回收后本地缓存丢失是预期行为。
- NodePool 和所有 H100 Pod 必须同时带
  `seedleap.ai/model-cache-storage=local-nvme`；该调度标签是硬门禁，禁止新 Pod 落回
  尚未启用 Instance Store 的旧节点。
- gp3 根盘只承载操作系统，固定为 100Gi。发布和验收必须检查 Node 的
  `ephemeral-storage` 接近 p5 本地盘总容量，并在 Denoiser 内确认
  `findmnt -T /model-cache` 的底层设备为 `/dev/md/0`，不能仅凭 Linux 中包含
  `nvme` 的设备名判断——EBS 同样会显示为 NVMe 设备。
- 修改 NodeClass 会令旧 NodeClaim 进入 Drift。在线切换前必须先保护当前 Pod，且
  Denoiser 保持 `updateStrategy: OnDelete`；只有下一次节点替换或明确维护窗口才删除
  旧 Pod，禁止为切换缓存盘主动中断当前用户。

## 不可变依赖与模型

- Gateway、Coordinator、Denoiser、VAE 都使用 ECR digest，禁止使用可变 tag。
- 容器启动时禁止 `pip install`、`git clone`、`curl` 或外网下载。
- TAEHV 包和校验过的 `taew2_2.pth` 在 VAE 镜像构建时固化。
- 原始 checkpoint 先由一次性 CPU Spot Publisher 转换为版本化、带 SHA-256 manifest
  的独立 S3 serving artifact；`_READY` 最后写入。Denoiser 只挂载只读 serving
  artifact，启动时不再转换 checkpoint。
- 生产 serving artifact 从
  `leap-world-model-serving-829115578968-us-east-2` 只读挂载，避免 H100 冷启动跨区读取。
- 集群复用已有 NVIDIA device plugin、S3 CSI Driver 和 EC2NodeClass，不重复安装集群级
  组件。

## 标准发布顺序

所有 AWS/Kubernetes 写操作执行前，必须先展示精确资源、范围、费用影响和清理方案，并按
仓库规则获得当次人工确认。

1. `provision_aws.sh` 创建 CloudFormation 控制面：DynamoDB、5 天 CloudWatch Log
   Group、不可变 ECR 和最小权限 IRSA。
2. `docker/build_and_push.sh` 构建并推送四个角色镜像，生成 `.env.images` digest 清单。
3. `publish_model_artifact.sh` 在一次性 `r7i.8xlarge Spot` 节点发布模型制品并写
   `_READY`；Publisher 使用独立 300Gi 加密 gp3 NodeClass，结束后随 NodePool 删除。
4. `deploy_production.sh` 只读检查 DDB、Log Group、模型 `_READY` 和所有镜像 digest，
   然后 server-side apply 完整拓扑。
   两个 Denoiser StatefulSet 固定使用 `podManagementPolicy: Parallel` 和
   `updateStrategy: OnDelete`。扩容时先等待新增副本全部 Ready，再按默认每批 1 个
   Pod 轮换旧 revision，任何时刻都保留另一台/另一个副本提供服务。
5. 从 NLB 运行 `browser_probe.cjs` 与 `e2e_production_chain.py`；测试必须证明真实
   Coordinator 配对、VAE 直传、独立 Trace HTTP 查询和 Display Lag 门禁。
6. 本轮测试完成后保持服务运行供人工验证，不调用清理脚本。收到明确清理指令后才运行
   `cleanup_production.sh --execute`，并验证 GPU Node、NLB 和命名空间全部消失。版本化
   模型制品和 ECR 镜像默认保留，便于下次复用。

## 回滚

部署脚本在写入前保存 workload spec。发布失败时使用原 spec 执行原地 Server-Side Apply；
Deployment 保持原地滚动恢复，Denoiser StatefulSet 则恢复原 spec 后按相同的 2 Pod
批次回滚。模型版本
由 `MODEL_ARTIFACT_REVISION` 固定。DynamoDB Schema 保持向后兼容；若控制面发布失败，现有
Session 不迁移，受影响用户重试。Spot Worker 不复制状态，节点中断时仅终止绑定 Session，
Coordinator TTL 自动回收 Lease。

## B300 Capacity Block 经验

`aws03-west2-haoze-20260813.yaml` 是一次隔离的 us-west-2 B300 Capacity Block 部署记录。
该拓扑复用一台 8 卡 B300 节点：Zing denoiser 使用 2 卡 SP=2，LingBot2 denoiser
使用 4 卡 SP=4，剩余 2 卡分别作为 Zing 与 LingBot2 的异步 TAEHV VAE worker。
如果 L4 VAE 无法调度，这种切分可以保持异步 VAE 架构，但要显式检查节点
`Allocated resources` 为 `nvidia.com/gpu 8/8`，避免 VAE 与 denoiser 抢卡或漏卡。

- B300 节点需要独立 service label/taint，例如
  `seedleap.ai/service=minwm-west2-haoze-20260813`，并让本次 namespace 的 GPU pod
  都精确绑定到该节点；这样不会影响现有线上服务。
- LingBot2 冷启动 warmup 在 B300 上可能超过 60s。实测 720p startup warmup 约
  84.6s，其中 text encoding 约 47.2s、aux encoding 约 7.9s、VAE encode 约 9.4s、
  denoise 约 18.7s。要把内部 `--realtime-session-idle-timeout-s` 放宽到 180s 或更高，
  同时继续保留用户侧 `--realtime-session-max-lifetime-s 90`；这两个参数不要混在一起。
- Zing 与 LingBot2 的 TAEHV checkpoint 不同：Zing 使用 `taew2_2.pth`，LingBot2 使用
  `taew2_1.pth`。如果镜像没有内置 `taew2_1.pth`，需要通过 init container 下载或从
  可信缓存挂载，并做 SHA-256 校验。
- denoiser 与 VAE heartbeat 都需要 `WORKER_EPOCH_FILE` 和共享 `worker-epoch` volume；
  否则 worker 可能主容器已健康但 heartbeat CrashLoop，Coordinator 看不到可用 worker。
- B300 `compute_103` 上可能出现自定义 CUDA JIT 编译失败并 fallback 到 Triton，例如
  `Unsupported gpu architecture 'compute_103'`。只要后续阶段继续完成，这是非致命启动噪声；
  但首次 warmup 会因此更慢，watchdog 和 startup timeout 要留余量。
- Denoiser StatefulSet 使用 `updateStrategy: OnDelete` 时，apply 新模板不会自动替换已有
  pod。修改启动参数后要显式删除对应 pod，并等同名新 pod 全部 Ready。
- 验收时至少做三类检查：WebUI 首页返回并显示 90s 倒计时；Coordinator 收到四路 heartbeat
  真实 websocket smoke 对 Zing 与 LingBot2 I2V 各完成一个 chunk。

### GPU 分批重启与冷加载

- 默认 `DENOISER_RESTART_BATCH_SIZE=2`，同批两个 Pod 并行重建，批次之间有严格
  Ready 屏障。
- 发布期间临时给承载 Denoiser 的节点加 `karpenter.sh/do-not-disrupt`，完成后移除，
  避免 Pod 短暂减少时触发 Spot 节点整机回收。
- 宿主机模型制品只由一个 `model-stager` 写入，其他 Pod 通过 `flock`
  复用本地缓存，避免重复下载和覆盖。
- 同批最多两个 Pod 进入冷加载，槽位锁在 Denoiser `/health` 成功后立即释放。
- 验收时必须检查每批两个 Pod 都恢复 Ready，最终 8 个 Pod 全部位于新 revision，
  restart count 为 0。

## 验收门禁

- 单用户和 2/4/8 并发 Session 全部完成且错误率为 0；当部署 8 个 H100 Denoiser
  Worker 和 2 个 L4 VAE Worker 时，32 并发应全部准入，超过 32 的请求明确返回
  `CAPACITY_EXHAUSTED`。
- 视频 WebSocket 不包含 Trace payload；Trace 仅通过 OTLP 和独立 HTTP Query API。
- Warm session Display Lag P95 不高于 250ms。
- 每个 Session 的 latent、Gateway 输出队列和 Trace 查询并发均有硬上限。
- 容量控制器能够从 Coordinator 的 shared-capacity 快照扩容，并在 active、queued 或
  draining 非零时拒绝缩容；定时 scale-to-zero 与事件扩容可以独立暂停。
- 故障测试完成后记录 H100/L4 Pod、Node、Spot 生命周期、NLB 与访问 URL，并保持服务运行
  供人工验证。只有收到明确清理指令后，才验证标签
  `seedleap.ai/test-run=minwm-async-vae-benchmark` 的 Node 归零。
