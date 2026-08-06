# MinWM static FP8 2× 调查与执行记录

更新时间：2026-08-06

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

同一轮 server trace 的全部 200 个 measured chunks 聚合（不是挑单个最优 chunk）：

| 阶段 | BF16 mean / p50 | static FP8 mean / p50 | static 相对 BF16 |
|---|---:|---:|---:|
| DiT denoise | 566.045 / 565.869 ms | 569.782 / 569.182 ms | `-0.66%`（变慢） |
| VAE decode pipeline | 468.655 / 468.592 ms | 466.961 / 467.032 ms | `+0.36%` |
| scheduler total | 1135.394 ms | 1129.063 ms | `+0.56%` |
| chunk total | 1173.071 ms | 1160.667 ms | `+1.07%` |

BF16 denoise 的 min/max 为 `560.364/579.700 ms`，static 为 `566.532/574.714 ms`；两组分布
整体重叠但 static 均值更慢。客户端的 `+0.56%` 不能归因于 FP8 DiT：它来自 VAE、raw frame
构造、WebSocket 写入及运行间噪声的合计差异。代表性 chunk 219（BF16/static 的 denoise
`568.381/571.671 ms`）与全量聚合结论一致。

这排除了“客户端传输吞掉了一个很大的 DiT 收益”：旧 static FP8 在模型层就没有拿到收益。

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

隔壁任务还有一个容易造成“static 看起来很差/online 看起来很好”的计时事实：在它的 480p
实验里，online/static 的 denoise 分别为 `357.15/402.55 ms`，VAE 为
`219.97/219.26 ms`。约 30 blocks × 10 Linear × 4 steps，即约 1200 次 Linear/chunk；
`45.4 ms / 1200 ≈ 37.8 µs/Linear`，正好符合旧 static 路径每层额外 scale layout、Triton
static quant 和 generic scaled-mm dispatch 开销的量级，而不是 FP8 Tensor Core 的硬件极限。

另外，隔壁 A/B 的 online SHA 为 `47a8bcbf...`，static 实际 SHA 为 `4a66501d...`，Job
annotation 还记录 `afbee9c...`；static builder、权重量化尺度、calibration 分辨率和吞吐
case 也没有完全对齐。它适合发现候选方向，不足以给出“量化方法本身”的因果排序。本轮差异大，
主要是口径（720p vs 480p）和实验同构性不同；两边共同指向的代码根因反而是一致的。

后续会在同一 immutable SHA、同一 B200 Spot、同一输入上重跑分层 A/B；不再用旧矩阵外推。

## 4. 根因与修复决策

代码审计发现，主 SRT 的 `ModelOptFp8LinearMethod` 已在 SM100 使用
`apply_fp8_linear_bmm_flashinfer`：保留 scalar weight/input scale，并调用 FlashInfer per-tensor
FP8 BMM。diffusion 复制版没有同步该实现：它把 scalar weight scale 扩成 per-channel scale，
static activation 又先进入 Triton `_static_quant_fp8`；因为 generic CUTLASS 路径只接收
per-token activation scale，还会把 scalar input scale 物化为 `(M, 1)`。最后才调用通用
`apply_fp8_linear` / `fp8_scaled_mm`。online FP8 则已经使用 CUDA per-token quant，所以旧 static
并不天然比 online 少开销。

最初假设是让 diffusion static FP8 在 SM100 + FlashInfer 可用时复用主 SRT 的 per-tensor
`bmm_fp8` 路径。但 `-08` 的真实 B200 数据否定了“主 SRT 路径也适合 DiT 大 M 矩阵”这一
假设：数值正确不等于性能正确。随后在完全相同的已量化输入、权重和 scalar scale 上增加
`torch._scaled_mm` 对照；`-09` 证明 cuBLASLt 才是当前 minWM 形状的较优后端。因此正式修复改为：

- SM100 保留 scalar weight/input scale；
- static activation quant 不再物化 `(M, 1)` scale；
- 对 16 对齐的 minWM 主矩阵调用 native `torch._scaled_mm`；
- 对其他 diffusion 模型可能出现的非 16 对齐 `K/N`，仅在该分支 pad 到 16 后裁剪输出；
- 其他硬件仍保留原 generic CUTLASS 路径。

验证分三层：

1. 单元/正确性：scalar scale、快路径路由、数值误差。
2. Spot GEMM/DiT：证明不是 silent fallback，并量化 Linear 与完整 denoise 的收益。
3. Spot 端到端：复用固定 720p 的 20+200 合同。

## 5. Amdahl 上限与待决事项

按全部 200 chunks 的 BF16 scheduler 均值，DiT 为 `566.045 ms`、VAE 为 `468.655 ms`、
其余调度约 `100.695 ms`。客户端基线 `14.079 FPS` 的 2× 目标要求 chunk interval 不超过
`16 / 28.158 = 568.22 ms`。当前顺序执行即使把 DiT 降到零，VAE + 其他调度仍约
`569.35 ms`，已略高于目标。因此只修 static FP8 GEMM 在物理上无法稳定满足客户端端到端 2×。

若把 VAE 放到同一台 p6 主机的另一张 GPU，并与下一 chunk 的 DiT 重叠，稳态 interval 近似
`max(DiT + 其他调度, VAE)`。这时要满足 `568.22 ms`，DiT 必须低于约 `467.53 ms`，即相对
BF16 至少加速 `1.211x`（denoise 时间至少降低 `17.4%`）。这给下一阶段建立了明确的进入条件：
先用 B200 microbenchmark/完整 denoise 证明新路径能越过这个阈值，再实现和测量 VAE overlap；
不能把两项收益混在一起归因给 static FP8。

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

- 结果：Karpenter 先后尝试多个 NodeClaim，最终在 east2b 创建 B200 Spot；完整依赖安装完成，
  但 pytest collection 因残留 `peft==0.17.0` 导入已从 `transformers==5.12.1` 移除的
  `HybridCache` 而失败。CUDA test 与 microbenchmark 未执行。
- 证据纠偏：Pod 消失后只看 Kubernetes 事件，曾把 `TaintManagerEviction` 误判为运行主因；S3 中实际
  留有 provenance 和 `pytest.log`，证明主因是 PEFT/Transformers API 不兼容。最终判断以 S3 日志为准。
- 决策：暂停尚未拿到节点的 `-04`，避免它用同样环境失败；`-05` 沿用短安装路径并显式
  `pip uninstall -y peft`。保留 suspended `-04`，不删除旧 Job。

### 2026-08-06：B200 Spot 微基准 `-05`

- profile/context：仍严格为 `spot` / `minwm-spot`；代码固定在 `761b76f520...`，镜像固定为
  `sha256:bedc07ea...`。
- 结果：多轮 `UnfulfillableCapacity` 后，NodeClaim `minwm-test-b200-spot-7hsjl` 在 east2b
  成功启动 `p6-b200.48xlarge` Spot。依赖安装继续推进，但 pytest collection 在导入 IPython 时
  因缺少 `traitlets` 失败；CUDA test 与 microbenchmark 均未执行，不能作为性能证据。
- 现场变量：同集群同时存在另一个请求 B200 的任务，会影响获取节点的等待时间；它不会共享本 Pod
  的 GPU，也不会进入正式计时窗口，因此节点成功独占后不影响测量合同。
- 决策：Pending 不产生 GPU 费用；为避免低 placement-score 时被 1 小时 deadline 人为截断，
  把 `activeDeadlineSeconds` 延长到 4 小时。此次变更不改变 profile、实例类型、镜像、代码或计时合同。
- 决策：保留 `-05` S3 日志；`-06` 使用新 run id，在最小依赖后让 pip 补齐 IPython 的传递依赖，
  并升级到含 quant/GEMM 分解计时的 immutable SHA `aae6435e9160...`。复用当前 Spot 节点，不覆盖旧结果。

### 2026-08-06：B200 Spot 微基准 `-06` / `-07`

- `-06` 补齐 IPython 后，pytest collection 继续暴露 `transformers==5.12.1` 与镜像原生
  `huggingface_hub` 的 API 不兼容（缺少 `is_offline_mode`）；CUDA test 与 microbenchmark
  仍未执行。
- 偏离预期与纠偏：顶层包逐个 `--no-deps` 升级破坏了固定镜像原有的一致环境。停止继续追补依赖；
  `-07` 回到镜像原生 Transformers/Diffusers/PEFT，只安装 `-01` 已确认缺失的 `orjson` 和固定
  FlashInfer JIT cache。新 run id 保留 `-06` 失败证据，并继续复用已启动的 B200 Spot 节点。
- `-07` 进一步证明训练镜像并不是完整 SGLang 开发环境：补完 `orjson` 后，collection 下一个缺失项是
  `pybase64`。因此 `-08` 不再逐项试错，恢复 `-03` 已经验证能完整安装的
  `pip install -e python[diffusion]`，并在安装结束后立即卸载导致 `HybridCache` 冲突的 PEFT；这是目前
  唯一同时覆盖完整依赖和已知兼容性修正的配方。

### 2026-08-06：B200 Spot 微基准 `-08` / `-09`

- `-08` 首次跑通完整证据链：固定镜像和 SHA、B200 SM100、5/5 correctness tests、四个
  minWM Linear 形状以及 S3 JSON 全部成功。
- 偏离预期：FlashInfer `bmm_fp8(backend="cublas")` 并不是这些单矩阵 DiT 形状的快路径。
  对 `(M,N,K)=(3432,3072,3072)` 与 `(13728,3072,3072)`，完整 static Linear 相对 BF16
  只有 `0.317x/0.728x`；两个大 MLP 形状也只有 `1.309x/1.245x`。纯 FP8 GEMM 已经偏慢，
  因此不能继续把问题只归因于 static quant 或 repeated scale。
- `-09` 在同一输入上加入 scalar `torch._scaled_mm`：四个形状的完整 Linear 相对 BF16 p50
  分别为 `0.804x/1.382x/1.482x/1.334x`。在 720p 4-frame 主形状上，它稳定优于 FlashInfer；
  输出相对 legacy 的 L2 差异只有 `1.28e-5` 到 `1.54e-5`。
- 决策：撤销 diffusion 对 FlashInfer BMM 的选择，改为 SM100 scalar cuBLASLt scaled-mm；
  FlashInfer helper 保留给主 SRT，不扩大改动范围。

### 2026-08-06：B200 Spot 微基准 `-10` / `-11`

- `-10` 的路由和两个 aligned correctness case 通过，但原有非对齐投影用例
  `(M,N,K)=(19,150,80)` 被 cuBLASLt 拒绝：`mat2 shape (80x150) must be divisible by 16`；
  总结果为 4 passed / 1 failed，因此没有进入 microbenchmark，也不能验收。
- 纠偏：只在 `K` 或 `N` 非 16 对齐时构造 column-major padded weight 和 padded activation，
  scaled-mm 后裁剪到原输出宽度；minWM 全部主矩阵是 16 对齐，正式热路径不承担 padding 成本。
- `-11` 使用 immutable SHA `3c910b87bc...` 在同一台 B200 Spot 上完成复测：5/5 correctness
  全部通过，包括 `-10` 失败的非 16 对齐投影；没有进入 generic FP8 fallback。
- 修复后实际 helper 相对 BF16 的 p50 speedup，按 `(M,N,K)` 分别为：
  `(3432,3072,3072) 0.743x`、`(13728,3072,3072) 1.301x`、
  `(13728,13824,3072) 1.578x`、`(13728,3072,13824) 1.268x`。输出相对 legacy
  static FP8 的 L2 差异为 `1.28e-5` 到 `1.54e-5`。
- 结论纠偏：cuBLASLt 修复显著抬高了三个大 M 热点形状的上限，但小 M 形状仍慢于 BF16；
  因此不能从单个 MLP 的 `1.578x` 外推整个 DiT，更不能外推端到端 `2x`。
- 完整产物：`s3://leap-world-us-east-2/world-model/evals/minwm/quantization/20260805/static-fp8-2x/minwm-static-fp8-fastpath-b200-20260806-11/`。

### 2026-08-06：固定 720p 成对 E2E `-02`

- 已提交 Job `minwm-static-fp8-fastpath-e2e-b200-20260806-02`；AWS profile/context 为
  `spot` / `minwm-spot`，node selector 强制 `p6-b200.48xlarge` + Spot。
- BF16 和 static FP8 在同一 Pod 顺序运行，固定 SGLang SHA `3c910b87bc...`、minWM SHA
  `2efc6485f6...`、checkpoint version、输入 case、20 warmup + 200 measured chunks；static lane
  复用 BF16 lane 的输入和转换后模型，避免重新取样或模型转换差异。
- 结果根目录：`s3://leap-world-us-east-2/world-model/evals/minwm/quantization/20260805/static-fp8-2x/e2e-paired-02/`；状态待回填。
