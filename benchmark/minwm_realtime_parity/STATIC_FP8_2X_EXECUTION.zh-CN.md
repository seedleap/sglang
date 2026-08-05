# MinWM static FP8 2× 调查与执行记录

更新时间：2026-08-05

## 1. 验收合同

- 正式硬件：B200 Spot；只使用 AWS `spot` profile / Kubernetes `minwm-spot` context。
- 输入：固定 `1248x704`、4 DMD steps、KV45、每 chunk 4 latent / 16 pixel frames。
- 测量：20 个 warmup chunks + 200 个 measured chunks。
- 对照：同 checkpoint、同 commit、同输入、同服务参数；唯一实验变量必须被显式记录。
- 正确性：不允许静默 fallback；输出帧数、分辨率、协议和量化覆盖率必须满足相同合同。
- 目标：static FP8 的客户端端到端稳态 FPS 至少为 BF16 的 `2.0x`。

## 2. 第一轮事实

固定 720p B200 Spot 第一轮结果：BF16 `14.079` client FPS，online FP8
`14.122`，static FP8 `14.158`。static FP8 只提高 `0.56%`，没有达到目标。

同一轮 server trace 的代表性 steady chunk：

| 阶段 | BF16 | static FP8 | 结论 |
|---|---:|---:|---|
| DiT denoise | 568.381 ms | 571.671 ms | static FP8 没有加速 DiT |
| VAE decode pipeline | 468.084 ms | 467.182 ms | 未量化路径，基本相同 |
| scheduler total | 1126.456 ms | 1129.172 ms | 与 DiT/VAE 事实一致 |

这排除了“客户端传输吞掉了一个很大的 DiT 收益”：static FP8 在模型层就没有拿到收益。

## 3. 与“调研 minWM 量化吞吐方案”任务的差异

隔壁任务旧 B200 矩阵为 `832x480`：BF16 `23.183`、online FP8 `26.653`、
static FP8 `24.859` client FPS。它与本轮不能直接横比：

1. `1248x704` 的像素数和 DiT token 数约为 `832x480` 的 `2.20x`。
2. 旧矩阵的 online FP8 与 static FP8 使用不同 SGLang SHA；Job annotation 还出现第三个 ref，
   因而不是只改变 activation scale 来源的同构 A/B。
3. 旧矩阵使用 step-3200 checkpoint 和默认 480p case；本轮固定 720p 的正式矩阵使用另一套
   checkpoint/case 合同。
4. 两轮 static FP8 都进入 diffusion ModelOpt 的旧实现，尚未使用主 SRT 已有的 SM100
   per-tensor FlashInfer GEMM。因此 `+7.2%` 与 `+0.56%` 都不是 B200 static FP8 的合理上限。

后续会在同一 immutable SHA、同一 B200 Spot、同一输入上重跑分层 A/B；不再用旧矩阵外推。

## 4. 根因与修复决策

代码审计发现，主 SRT 的 `ModelOptFp8LinearMethod` 已在 SM100 使用
`apply_fp8_linear_bmm_flashinfer`：保留 scalar weight/input scale，并调用 FlashInfer per-tensor
FP8 BMM。diffusion 复制版没有同步该实现：它把 scalar weight scale 扩成 per-channel scale，
随后调用通用 CUTLASS `apply_fp8_linear`。

第一项修复：让 diffusion static FP8 在 SM100 + FlashInfer 可用时复用同一 per-tensor 快路径；
其他硬件和 fallback 行为保持不变。验证分三层：

1. 单元/正确性：scalar scale、快路径路由、数值误差。
2. Spot GEMM/DiT：证明不是 silent fallback，并量化 Linear 与完整 denoise 的收益。
3. Spot 端到端：复用固定 720p 的 20+200 合同。

## 5. Amdahl 上限与待决事项

当前顺序流水线中，BF16 代表 chunk 的 DiT 约 `568 ms`、VAE 约 `468 ms`。即使把 DiT
时间降为零，其他阶段仍接近 `558 ms`；客户端理论上限约 `28.7 FPS`，只略高于
`2x BF16 = 28.158 FPS`。因此单独修 GEMM 几乎不可能稳定满足端到端 2×。

这不是提前降低验收标准。先实测快路径后的 DiT/端到端数字；若 Amdahl 瓶颈按预期转移到
VAE/调度，则继续做不降低画质的流水化或阶段并行，并分别报告：

- static FP8 本身的隔离收益；
- 后续流水线优化的隔离收益；
- 组合方案相对当前 BF16 生产基线的端到端收益。

## 6. 执行日志

### 2026-08-05：代码审计

- 发现 diffusion 与主 SRT 的 ModelOpt FP8 实现漂移。
- 决策：先同步 SM100 per-tensor FlashInfer 路径，不扩大到 NVFP4 或 Attention。
- 偏离预期：第一轮 static FP8 的 DiT 比 BF16 略慢；问题不是动态 scale 开销，而是 GEMM
  backend/scale layout 选择错误。
- 新增对照要求：所有结论必须同时标注分辨率、checkpoint、SGLang SHA 和计时边界。

### 2026-08-05：本地验证

- `py_compile`、`ruff check`、`ruff format --check` 和 `git diff --check` 通过。
- 新增 CPU 路由测试：mock SM100 + FlashInfer 时必须调用 per-tensor BMM，generic FP8 路径若被调用则失败。
- 更新原有 CUDA correctness test：SM100 快路径必须保留 scalar weight scale；其他 CUTLASS 路径仍验证
  per-channel scale。
- 偏离预期：本机默认 Python 3.9 不满足仓库 3.10+ 类型语法；本机 Python 3.11 的 Torch/Triton
  组合又在 pytest collection 阶段发生模块/类型冲突。它们发生在目标测试执行前，因此不能作为测试通过证据；
  完整 correctness test 移到固定镜像的 B200 Spot 上执行。

### 2026-08-05：B200 Spot 微基准 `-01`

- profile/context：`spot` / `minwm-spot`；Pod 落到已有 `minwm-test-b200-spot` 的
  `p6-b200.48xlarge`，没有落到 on-demand。
- 结果：pytest collection 在导入 SGLang 时因镜像缺少 `orjson` 失败；CUDA test 与 microbenchmark
  均未开始，不能用于判断修复有效性。
- 决策：保留 `-01` Job 和 S3 日志；新建 `-02`，复用首轮正式入口的 SGLang diffusion extra 与
  FlashInfer JIT cache 安装方式，不覆盖旧 run id。

### 2026-08-05：B200 Spot 微基准 `-02`

- 结果：原有 B200 Spot 节点在提交窗口内被回收；Pod 未进入 Running，等待 30 分钟后触发
  `DeadlineExceeded`。因此依赖安装、CUDA test 和 microbenchmark 都未执行。
- 证据：NodePool `minwm-test-b200-spot` 仍 Ready，但节点数已变为 0；Job event 是
  `FailedScheduling`，不是容器错误。
- 决策：保留 `-02`，以 `-03` 重提并把 deadline 扩为 60 分钟，给 Spot 补充容量更充分的时间；
  仍不切到 `aws03` 或 on-demand。

### 2026-08-05：B200 Spot 微基准 `-03`

- 结果：Karpenter 先后尝试多个 NodeClaim，最终在 east2b 创建 B200 Spot；Pod 拉取镜像并启动后，
  节点再次被回收，事件为 `TaintManagerEviction`。Pod 随节点消失，未产生有效测试结果。
- 偏离预期：完整 diffusion extra 安装扩大了“拿到 Spot 到开始测试”的脆弱窗口，且结果目录建在安装后，
  因此这次没有持久化安装日志。
- 决策：`-04` 把 S3 provenance/job log 前置，并改为已在 RTX preflight 使用过的最小 runtime
  依赖集合；不重新解析完整 diffusion extra，不构建 kernel，目标是把 B200 上的准备时间压到数分钟。
