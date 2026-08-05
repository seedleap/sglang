# minWM SGLang 量化第一轮执行记录

更新时间：2026-08-05（Asia/Shanghai）

## 目标与边界

本轮只回答一个问题：在固定 720p MinWM realtime 请求上，现有量化路径与 whole-DiT compile 组合能把吞吐上限推到哪里。

本轮不是质量放行实验，不做多 case 视觉回归，也不据此直接决定生产默认值。任何性能胜出项都必须进入后续质量轮才能放行。

## 固定实验合同

| 项目 | 固定值 |
|---|---|
| 硬件 | AWS Spot `p6-b200.48xlarge`，单 Pod、单 GPU、所有 lane 串行 |
| 集群 / 可用区 | `codex-seed-leap-use1` / `us-east-1-atl-2a` |
| NodePool | `minwm-test-atl2-p6-spot` |
| 模型代码 | `seedleap/minWM@2efc6485f65e8fcab506665efde79bc41406385e` |
| checkpoint | `global_step_003200/ema_student/model.pt`，S3 VersionId `wduScksw2f3yPErnG9lBioOuE2AToyAP` |
| 请求 case | `cases_720p_compile_smoke.json` / `00_forward_080_pottery_720p` |
| 分辨率 | `1248x704`（项目中的 720p 合同） |
| realtime 参数 | 4 latent frames/chunk，KV cache 45 frames，4 denoise steps |
| 测量窗口 | warmup 20 chunks + measured 200 chunks |
| attention / components | dense、SGLang optimized components |
| 主指标 | client steady received FPS（ratio of sums） |
| 次指标 | scheduler forward FPS、chunk p50/p95/p99、失败类型、峰值显存 |

公平性规则：除量化方法、NVFP4 kernel backend、whole-DiT compile 开关外，其余输入和服务参数保持不变。每个量化配置先测 eager，再测 compile。

## 第一轮矩阵

| # | 权重 / 激活 | kernel backend | compile | 状态 | client FPS | scheduler FPS | 相对 BF16 同 compile |
|---:|---|---|---|---|---:|---:|---:|
| 1 | BF16 | BF16 GEMM | off | 待运行 | - | - | 基线 |
| 2 | BF16 | BF16 GEMM | on | 待运行 | - | - | 基线 |
| 3 | online FP8 W8A8 dynamic | SGLang FP8 | off | 待运行 | - | - | - |
| 4 | online FP8 W8A8 dynamic | SGLang FP8 | on | 待运行 | - | - | - |
| 5 | calibrated FP8 W8A8 static | ModelOpt static FP8 | off | 待运行 | - | - | - |
| 6 | calibrated FP8 W8A8 static | ModelOpt static FP8 | on | 待运行 | - | - | - |
| 7 | ModelOpt NVFP4 W4A4 | `flashinfer_trtllm` | off | 待运行 | - | - | - |
| 8 | ModelOpt NVFP4 W4A4 | `flashinfer_trtllm` | on | 待运行 | - | - | - |
| 9 | ModelOpt NVFP4 W4A4 | `flashinfer_cutlass` | off | 待运行 | - | - | - |
| 10 | ModelOpt NVFP4 W4A4 | `flashinfer_cutlass` | on | 待运行 | - | - | - |
| 11 | ModelOpt NVFP4 W4A4 | `flashinfer_cudnn` | off | 待运行 | - | - | - |
| 12 | ModelOpt NVFP4 W4A4 | `flashinfer_cudnn` | on | 待运行 | - | - | - |

## 预期与判读

- BF16 compile 是当前已知强基线；量化必须与相同 compile 状态比较，不能把 compile 收益误归因给量化。
- online FP8 的动态 activation scale 有运行时成本，可能降低理论收益；它的价值是零离线校准成本。
- calibrated static FP8 消除动态 activation scale，预期比 online FP8 更接近 FP8 kernel 上限。
- NVFP4 显著降低 DiT 线性层带宽，但 realtime 端到端仍包含 VAE、调度和传输，因此 DiT kernel 收益不会等比例变成 client FPS。
- 三个 NVFP4 backend 先全部试跑；backend 不支持、数值或 compile 失败也属于本轮有效结论，不做静默回退。

## 已知限制

- MinWM NVFP4 builder 当前只使用校准 dump 的 `forward_000.pt`。这足够用于本轮固定 case 性能探索，但不能代表多状态质量覆盖。
- 静态 FP8 与 NVFP4 只量化 30 个 transformer block 中的大型 linear（共 300 个目标层）；action encoder、embedding、norm、scheduler、VAE 保持 BF16。
- 单次测量主要用于找上限。胜出组合后续仍需重复测量、误差条和多 case 质量回归。

## 执行日志、问题与决策

### 2026-08-05：运行前盘点

1. 发现仓库旧 B200 量化 YAML 使用 `ray` namespace、`minwm-test-b200-spot` NodePool、`minwm-test-b200-karpenter` capacity label 和 `seedleap.ai/workload=wan22-ti2v` toleration。
2. 当前集群没有 `ray` namespace；可用输入 PVC、GitHub secret 和 ServiceAccount 位于 `default` namespace。旧 NodePool 也不存在。
3. 当前可用 B200 Spot NodePool 是 `minwm-test-atl2-p6-spot`，固定 `us-east-1-atl-2a`，capacity label / taint 为 `minwm-test-atl2-karpenter`，从零扩容，8 小时到期。
4. 决策：新建第一轮专用清单，不修改或误用旧集群清单；namespace 改为 `default`，使用当前 NodePool 与 taint，输入只读挂载 `s3-claim`，结果写入专用 EBS PVC。
5. 发现 throughput client 默认读取 `cases.json` 的 832x480 case；而静态 FP8 / NVFP4 校准使用 1248x704 case。若直接复用旧入口，会产生“720p 校准、480p 测速”的不可比结果。
6. 决策：入口新增显式 throughput cases/case/warmup/measured 参数。本轮校准和测速统一为同一个 1248x704 case。
7. 决策：所有 lane 放在同一 Pod 串行运行，请求 1 张 GPU。BF16 首次完成 staging、依赖安装和模型转换；后续 lane 复用 Pod 内输入与 BF16 模型，避免重复 S3 staging。NVFP4 只导出一次，三个 backend 复用同一份量化权重。

### 2026-08-05：首次提交与 Spot 调度

1. `kubectl apply --dry-run=client`、`kubectl apply --dry-run=server` 和清单内嵌脚本的 `bash -n` 均通过后，创建专用 PVC 与 Job。
2. Karpenter 正确为 Pod 提名 `minwm-test-atl2-p6-spot` NodeClaim，证明 namespace、selector、toleration 和资源请求已经进入预期调度路径。
3. AWS Fleet 随后连续返回 `UnfulfillableCapacity` / `InsufficientCapacityError`，失败发生在创建 `p6-b200.48xlarge` Spot 实例阶段。PVC 的 `WaitForFirstConsumer` 是该 Pending 状态的结果，不是独立存储故障。
4. 决策：先保留 Pending Job 让 Karpenter 自动重试，不改成 B300 或 On-Demand。否则硬件变化会使本轮与既定 B200 合同不可比。若最终决定切硬件，必须新建单独矩阵并明确标注，不能覆盖本轮身份。
5. B200 在容量失败缓存窗口后仍无可行 offering。为了让第一轮量化方法探索继续推进，决策新增独立的 `p6-b300.48xlarge` Spot fallback Job/PVC；它保持相同 12-lane 请求合同，但使用独立 matrix id，数据不得标为 B200。
6. B200 Job 先保留 Pending 继续争取容量；如果 B300 已 Running，则暂停 B200 Job，避免同一第一轮在容量恢复后意外双跑。暂停是可恢复状态，不删除已有 Job/PVC 或诊断事件。

### 2026-08-05：识别到正确的 east2 Spot 控制面

1. use1 的 B300 fallback 也连续收到 `UnfulfillableCapacity`，说明 ATL2 单可用区 P6 池当时同时缺 B200 与 B300。
2. 进一步检查本机 kubeconfig 后发现 `minwm-spot` context 指向 us-east-2 的 `leap-world` 集群。这里正好存在旧量化清单引用的 `ray` namespace、`minwm-test-b200-spot`、`s3-claim` 和 `github-token`。
3. east2 的 B200 Spot NodePool 跨常规三 AZ，且当天已有 static-FP8 与 NVFP4 Job 成功完成，说明它比 ATL2 单可用区池更符合本轮运行预期。
4. 决策：暂停 use1 的 B200 / B300 Pending Job，保留 Job、PVC 和事件用于审计；没有创建 GPU、没有结果数据被中止。第一轮转到 east2 的 B200 Spot 池，结果写到独立 S3 前缀。

### 2026-08-05：读取旧量化结果与 compile 失败证据

1. 旧结果合同明确是 `832x480 / 00_forward_pottery`，不是本轮固定的 `1248x704`。eager client FPS 分别为 BF16 `23.183`、online FP8 `26.653`、static FP8 `24.859`、NVFP4 `20.173`。
2. 因此旧数据只能形成先验：online FP8 约比 BF16 快 `15.0%`，static FP8 约快 `7.2%`，该次 NVFP4 反而慢约 `13.0%`；不能把它当作本轮 720p 结论。
3. 旧 BF16 / online FP8 compile lane 均失败。日志显示 whole-DiT 在 KV 长度增长阶段反复产生冷编译：BF16 后段单 chunk 曾达到 `337–391 s`，online FP8 首 chunk 约 `150 s`；client 最终收到 WebSocket `1012 service restart`。这更像“长冷编译暴露在 Spot / Job 生命周期内”而不是稳态 FPS 失败。
4. 决策：本轮仍保留 20 个 warmup chunk，让 KV45 前的形状编译不进入 measured 200 chunks；east2 Job deadline 从 4 小时延长到 7 小时。各 lane 结果直接写独立 S3 目录，即使 Spot 中断也保留已完成证据，后续只补缺失 lane。
5. 决策：将运行顺序从“每种量化 eager+compile 连跑”改为“先完成 6 个 eager，再运行 6 个 compile”，且 compile 阶段按旧先验把 online FP8 放在最前。这样长冷编译或 Spot 中断不会阻止其余量化方法先拿到 eager 上限。

### 2026-08-05：增加 west2d 独立 P6 Spot 供给路径

1. east2 B200 Job 持续收到 `InsufficientInstanceCapacity`；NodePool 的另一名称虽然不同，但与现有池使用相同的三组 subnet、security group 和 AMI，不构成新的 AWS 容量路径，因此没有并行提交重复 Job。
2. `codex-minwm-test-phx2` 集群存在 `minwm-sp12-usw2d-p6-spot`，位于 `us-west-2d`，允许 `p6-b200.48xlarge` 与 `p6-b300.48xlarge` Spot，是真正独立于 east2 / ATL2 的供给路径。另一个名称相似的 `minwm-test-usw2d-p6-spot` 当前 GPU limit 已为 0，不能使用。
3. 该集群的 `s3-claim` 是只读卷，不能像 east2 一样把每条 lane 直接写回 S3。决策：结果写入专用 `minwm-dmd-0724-p6-gp3` RWO PVC；矩阵末尾仍把完整 summary 打到 Job log，PVC 保留逐 lane 原始数据。
4. 同集群刚完成的 H200 BF16 / online FP8 作业使用旧 SHA `7cb482cc...`、832x480 默认 case 且仅含 eager，因此只作为供给与执行链路旁证：client FPS 分别为 `18.958` 和 `21.271`（FP8 `+12.2%`），不能并入固定 720p 正式矩阵。
5. 决策：west2d Job 不锁死 B200 或 B300，让 Karpenter 在该 NodePool 的合法 P6 offering 中选择实际可获得的 Spot 型号；最终结果必须绑定实际 node / instance type / GPU 名称，不与 B200 基线混写。east2 B200 在 west2d Job 获得 GPU 后暂停，避免两个完整矩阵意外同时运行。

## 作业证据

运行前待填写：

- SGLang immutable SHA：`ca2509f03432d07f183f8f1816c1ae1f218ec6a0`
- Job：`minwm-quant-r1-b200-20260805-01`；Pod / Node 待调度后填写
- image digest：`sha256:bedc07ea3ba55059a8c1c569c3b177c4d00d41f37d4fa9105375531534ef5f2a`
- GPU request：1 x B200
- 结果 PVC：`minwm-quant-r1-results-20260805`

## 结果结论

待第一轮完成后填写。结论必须区分：

1. 量化本身对 eager 的收益；
2. compile 本身在各精度上的收益；
3. 量化 + compile 的全局最优；
4. scheduler FPS 与 client FPS 的差距，判断收益是否被 VAE / 传输吞掉；
5. 失败 backend 与明确错误，不把失败项包装成“无收益”。
