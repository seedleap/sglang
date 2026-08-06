# minWM static FP8 端到端 2x：掌握度考察

版本：第一轮专项验收卷。建议闭卷 60 分钟，总分 100 分。

## 使用方式

先独立作答，再对照文末评分要点。计算题必须写公式；性能结论必须指出证据层级和实验身份。只有结论、没有证据或归因的答案不给满分。

本卷考察的不是“记住某个 FPS”，而是能否独立判断 static FP8 为什么没有达到 2x、怎样公平地修复和验收，以及怎样解释本任务与“调研 minWM 量化吞吐方案”的结果差异。

## A. 实验合同与证据（20 分）

1. 写出一组可做 strict A/B 的最小实验合同，至少包括 10 个字段。解释为什么清单 commit、Pod checkout 的 immutable SGLang SHA、MinWM SHA 是三个不同身份。（6 分）
2. 为什么 20 个 warmup chunk 不能随意减少？以 KV45 下前 4 个 chunk 和 chunk 10–19 的耗时变化说明风险。（4 分）
3. 对 `micro kernel -> whole DiT -> scheduler -> client FPS` 四层证据排序，并说明每层能证明什么、不能证明什么。（5 分）
4. Spot Job 的 Pod 处于 `Pending`，事件显示 Karpenter 多次 `Nominated` 但没有节点。此时能否判断代码正确或性能不达标？是否已经产生 B200 GPU 费用？正式操作必须使用哪个 AWS profile/context？（5 分）

## B. static FP8 机制与瓶颈（25 分）

5. B200 static FP8 W8A8 路径中，分别说明权重离线量化、activation 量化、`torch._scaled_mm` 的输入/scale/输出职责。为什么“B200 FP8 峰值是 BF16 的 2x”不等于一次 linear 或端到端服务必然 2x？（7 分）
6. 旧通用 static FP8 activation helper 在约 1200 次 linear 调用的一个 denoise 中多出约 45.4 ms。估算每次调用的额外开销，并列出至少 3 个来源。（5 分）
7. 为什么把 activation quantization 改为已注册的向量化 CUDA custom op，可能同时改善 eager 和 `torch.compile`？为什么在 B200 测试完成前仍不能宣称 compile 问题已解决？（5 分）
8. 若 micro benchmark 显示大 M 的 FP8 GEMM 加速明显、小 M 反而变慢，设计一个 hybrid 路由；同时说明额外 BF16 权重副本的显存代价和必须重新验证的质量/性能项目。（4 分）
9. 已知旧路径的纯 `scaled_mm` 大形状上限约 1.69x，而实际 helper 在三个代表形状约为 0.74x、1.30x、1.58x。解释两组数字的差别，以及为何不能只优化 GEMM。（4 分）

## C. 端到端 2x 与 Amdahl 定律（20 分）

10. 720p、20+200 的同构 A/B 中，BF16 client 为 13.9848 FPS，static FP8 为 13.9918 FPS；BF16/static 的 DiT 均约 571–576 ms，VAE 均约 469–470 ms。计算 static FP8 client 加速比，并解释为何该结果不是测量出错就能一笔带过。（5 分）
11. BF16 scheduler 每 chunk 约 1143 ms，其中 VAE 约 470 ms。假设其它部分不变，即使把 DiT 时间降为 0，单 GPU串行路径理论上最快约多少 FPS？为什么 strict 2x 的目标约 27.97 FPS 不可能只靠当前串行 static FP8 达到？（5 分）
12. 画出或文字描述“双 GPU 原始完整 VAE 精确重叠”时间线。为什么必须逐 chunk 一一对应、最终 chunk 只 flush 一次？一 chunk lag 或双 flush 会引入哪些正确性风险？（5 分）
13. 四条验收 lane 为 BF16-local、static-local、BF16-overlap、static-overlap。分别写出它们回答的问题，并说明最终 combined speedup 为什么不能全部记为 static FP8 收益。（5 分）

## D. 与隔壁任务结果的差异（20 分）

14. 隔壁旧 832x480 结果约为 BF16 23.183 FPS、online FP8 26.653 FPS、static FP8 24.859 FPS；本任务 720p BF16/static 约为 13.985/13.992 FPS。至少列出 6 个不能直接横比的因素。（7 分）
15. 为什么分辨率从 832x480 提高到 1248x704，会改变 quant GEMM 形状、attention/非 linear 占比、VAE 占比和端到端 Amdahl 上限？“像素/token 约 2.20x”能否直接推出 FPS 降为 1/2.20？（5 分）
16. 解释下列 SHA 问题为什么会污染归因：隔壁 online/static lane 的 SGLang SHA 不同，且清单注释 SHA 与实际 checkout SHA 不一致；calibration 720p 而吞吐 case 为 480p；checkpoint/case 也不同。（4 分）
17. 集群已有用 L4 + TAEHV 的异步 VAE 方案。为什么本轮不能直接采用并声称“static FP8 达到 2x”？什么条件下它可以作为另一个产品方案评测？（4 分）

## E. 独立验收与决策（15 分）

18. 写一份不超过 15 行的 Spot 真机执行清单，覆盖：context/profile、server dry-run、immutable SHA、同构四 lane、测试门禁、日志/事件、S3 证据和失败保全。不要写删除命令。（5 分）
19. 假设 optimized static-local 为 16 FPS，BF16-overlap 为 23 FPS，static-overlap 为 29 FPS。分别计算 static 本地收益、VAE 重叠收益、combined 收益还缺哪些基线；指出哪一个数字可以与 strict 27.97 FPS 门槛比较。（5 分）
20. 最终结果未过 2x 时，给出一个证据驱动的下一步决策树，至少区分 activation quant、FP8 GEMM、attention/非 linear、VAE server、序列化/传输、同步空洞、Spot 中断和计时合同污染。何时应该做 Nsight Systems profiling？（5 分）

## 评分要点

### A 部分

- 合同至少覆盖：GPU 型号/数量、Spot、镜像 digest、SGLang immutable SHA、MinWM SHA、checkpoint、case/seed/action、分辨率、steps、KV、warmup/measured chunks、attention/components、量化配置、compile 状态、VAE 拓扑。
- 运行时代码由 Pod checkout 的 SHA 决定；清单 commit 是控制面审计身份；MinWM SHA 决定模型基线与配置语义。
- KV cache 在前若干 chunk 尚未进入稳态；只看首 4 chunk 会把 cache 填充阶段误当成 steady state。20 warmup 是当前合同的一部分。
- micro 只证明代表形状；whole DiT 包含调用频次与非 GEMM；scheduler 包含 VAE 等服务关键路径；client 才是 strict 端到端验收。
- `Pending` 只能证明没有容量/尚未调度，不能证明代码或性能；未分配 GPU 时没有 B200 GPU 运行费用。正式路径只能是 `AWS_PROFILE=spot`、context `minwm-spot`。

### B 部分

- 权重预先量化并保存 FP8 + scale；activation 每次 forward 根据约定 scale 转 FP8；`_scaled_mm` 完成 FP8 Tensor Core GEMM 并按 scale 输出目标 dtype。
- 峰值比不是有效吞吐比：形状、tile/occupancy、内存流量、量化 reduction/cast、launch、padding、epilogue、非 GEMM 均会损失收益。
- `45.4 ms / 1200 ~= 37.8 us/call`。可能来源包括 Python/dispatcher、amax reduction、标量 host/device 交互、临时张量、额外 launch、通用 Triton helper、padding/布局转换。
- custom op 可消除旧 inline-asm 路径并让 Dynamo 看到稳定边界；但 KV 形状推进仍可能反复编译，必须以 B200 的测试和完整 20+200 lane 为准。
- hybrid 可让大 M 走 FP8、小 M 走 BF16；代价是保留第二份 BF16 权重及更复杂的路由/缓存，需复测显存、数值、compile graph、真实形状分布和 client FPS。

### C 部分

- 旧同构结果的 client speedup 为 `13.9918 / 13.9848 ~= 1.0005x`，约 `+0.05%`；它与 DiT 没有加速、VAE 基本相同互相印证。
- 即便 DiT 为 0，仍约有 `1143 - 571 ~= 572 ms/chunk`，以每 chunk 16 帧计约 `16 / 0.572 ~= 27.97 FPS`，已经贴住目标且没有余量；任何非重叠开销都会使其低于 2x。
- 精确重叠必须把 chunk N 的完整原始 VAE decode 放到 GPU1，同时 GPU0 计算 chunk N+1 的 DiT，并保持请求、事件、chunk index 和最终态一一对应。
- static-local 隔离量化收益；BF16-overlap 隔离拓扑收益；static-overlap 是联合上限；BF16-local 是统一基线。联合收益必须拆分归因，不能都记到 static FP8。

### D 部分

- 不可直接横比的因素包括：832x480 对 1248x704、不同 SGLang SHA、注释 SHA 与实际 SHA 不一致、不同 checkpoint/case、calibration 与 throughput 分辨率不一致、warmup/测量口径、compile/attention/components、单次波动、VAE 占比与形状分布。
- 像素/token 增长会非线性改变 GEMM M、attention、VAE 和内存压力；FPS 不按单一像素比线性缩放。
- 隔壁旧实验仍可作为“旧通用 static helper 有每-call 固定税”的旁证，但不能作为本轮修复后的 strict A/B 结果。
- TAEHV 是近似解码器，改变画质和模型语义；可另立质量阈值、同构基线和产品合同评测，但不能混入“原始完整 VAE + static FP8”的验收。

### E 部分

- 必须明确 `spot`、`minwm-spot`、不可变 SHA、四 lane 同构、B200 测试门禁、20+200、事件/日志和 S3 原始证据；失败 lane 不能补零或覆盖旧 run id。
- 例子里只有 static-overlap 的 29 FPS 能直接与 27.97 FPS strict 门槛比较；另外三个数字用于拆分量化与拓扑贡献，仍需统一 BF16-local 基线计算完整比值。
- 决策必须从层级数据定位：micro 新旧 helper、whole-DiT、scheduler 分段、client、远端 VAE decode/传输时间，再决定是否 hybrid、减少序列化、修同步或重新校准。
- 当日志只能看到“整体慢”、无法定位 GPU kernel/CPU launch/空洞/跨 GPU 重叠时，再用 Nsight Systems；先做一次不计入结果的 warmup，并把 primary 与 decoder 的时间线分开归因。

## 掌握等级

- 90–100：能独立设计、运行和审计 strict 2x 实验，能拆分 static FP8 与拓扑收益，并识别隔壁结果的合同污染。
- 75–89：能复现主流程和核心计算，少量 kernel、异步正确性或 Spot 运维细节需要查文档。
- 60–74：会运行任务，但证据层级、Amdahl 归因或同构 A/B 仍不稳定。
- 60 以下：建议先用执行文档完整演练一次“合同冻结 -> micro -> 四 lane -> S3 审计 -> 归因”。
