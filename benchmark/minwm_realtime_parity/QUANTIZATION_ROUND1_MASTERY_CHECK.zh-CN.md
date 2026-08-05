# minWM SGLang 量化第一轮掌握度考察

版本：第一轮执行配套卷，建议闭卷 60 分钟，总分 100 分。

## 使用方式

先独立作答，再对照本文末尾评分要点。计算题必须写公式；判断题必须说明证据，只有结论不给满分。

本卷考察的是能否独立复现、解释和诊断本轮工作，不考死记命令。允许引用执行记录里的 immutable SHA、Job 名和结果路径，但不能把“旧结果”“本轮结果”和“推断”混在一起。

## A. 实验身份与公平性（20 分）

1. 写出至少 8 个必须固定或记录的实验合同字段，并说明其中哪 3 个最容易造成“看起来是量化收益，实际不是”的混淆。（8 分）
2. 为什么“用 1248x704 case 做 calibration，再让 throughput client 使用默认 832x480 case”不能作为有效量化对比？（4 分）
3. 为什么每种精度都要同时保留 eager 与相同口径的 whole-DiT compile lane？（4 分）
4. 区分以下三个身份：清单所在 commit、Pod 实际 checkout 的 SGLang SHA、MinWM SHA。哪一个真正决定运行时代码？（4 分）

## B. 量化机制与覆盖范围（25 分）

5. 对比 online FP8 W8A8 dynamic、calibrated static FP8 W8A8、ModelOpt NVFP4 W4A4 的权重格式、activation scale 来源、离线准备成本和主要运行时成本。（9 分）
6. 本轮 MinWM builder 量化哪些模块、保留哪些模块为 BF16？为什么端到端 FPS 不会按 DiT linear 的位宽缩减比例增长？（5 分）
7. NVFP4 为什么要分别测 `flashinfer_trtllm`、`flashinfer_cutlass`、`flashinfer_cudnn`？遇到 backend 不支持时为什么不能静默回退再记成该 backend 的结果？（5 分）
8. 当前 NVFP4 calibration 只消费 `forward_000.pt`。这对“固定 case 性能上限”和“生产质量放行”分别意味着什么？（6 分）

## C. 数据计算与解释（20 分）

9. 旧 832x480 eager 结果为：BF16 `23.183`、online FP8 `26.653`、static FP8 `24.859`、NVFP4 `20.173` client FPS。分别计算三个量化方案相对 BF16 的百分比变化。（6 分）
10. 上题数据为什么只能作为本轮先验，不能回答 1248x704 的最终收益上限？至少列出 3 条原因。（4 分）
11. 对任意一条本轮 lane，同时拿到 client FPS 和 scheduler FPS 后，如何利用二者差距判断收益是否被 VAE、输出物化或 WebSocket 吞掉？（4 分）
12. 从执行记录的最终表中指出：最佳 eager、最佳 compile、全局最佳，以及各自相对同 compile 状态 BF16 的收益。若某 lane 失败，必须写失败类型而不是补零。（6 分，待正式矩阵完成后作答）

## D. Spot 运维与故障诊断（25 分）

13. 解释 `WaitForFirstConsumer`、`FailedScheduling`、`Nominated`、`InsufficientCapacityError` 的因果顺序。为什么本轮最初的 PVC Pending 不是独立存储故障？（6 分）
14. 最初 use1 清单与 east2 旧清单分别依赖哪些 namespace / NodePool / taint / 存储资源？如何用只读命令证明是“控制面不匹配”而不是代码错误？（6 分）
15. 为什么在发现 ATL2 B200 和 B300 都无容量后，没有把 B300 结果伪装成 B200？为什么最终又转向 east2 B200 三 AZ pool？（4 分）
16. 旧 compile 日志中出现单 chunk `150–391 s`，随后 WebSocket `1012 service restart`。给出一个证据驱动的诊断树，区分冷编译、稳态性能差、Spot 中断和应用异常。（5 分）
17. 为什么正式矩阵改成“先 6 个 eager，再 6 个 compile”？Spot 中断后应怎样只补缺失 lane，避免覆盖已有 S3 证据？（4 分）

## E. 独立复现实操（10 分）

18. 写一份不超过 15 行的执行清单，覆盖：确认 context、确认资源、server dry-run、immutable SHA、提交、监控、结果合同校验、失败保全。不要写删除命令。（5 分）
19. 给定一个 `throughput.json`，写出你会检查的最小字段集合，证明它确实来自固定 1248x704 case、20+200 chunks、KV45、4 steps，并可参与对比。（5 分）

## 评分要点

### A 部分

- 合同至少应覆盖：GPU 型号/数量、Spot/On-Demand、SGLang SHA、MinWM SHA、checkpoint version、case/seed/action、分辨率、steps、KV window、warmup/measured chunks、attention/components、compile、量化/backend、镜像 digest。
- 常见混淆包括：不同分辨率、把 compile 收益算给量化、不同 checkpoint、不同 GPU、不同 KV 或 native components。
- 运行时代码由 Pod checkout 的 immutable SGLang SHA 决定；清单 commit 提供控制面审计，MinWM SHA 决定基线代码和配置语义。

### B 部分

- online FP8 在加载/运行时量化，activation scale 动态计算；static FP8 使用校准得到的静态 input scale；NVFP4 使用 group size 16 的 4-bit 权重与 4-bit activation 路径，需要离线校准/导出。
- 当前目标是 30 个 block 的大型 linear，共 300 个目标层；action encoder、embedding、norm、scheduler、VAE 等仍为 BF16。
- 端到端还包含非量化模块、attention、VAE、输出物化与传输，Amdahl 定律限制收益。
- 单 forward 校准可用于本 case 的探索，不足以覆盖多 prompt/action/KV 状态和质量尾部。

### C 部分

- 旧 eager 结果约为 online FP8 `+15.0%`、static FP8 `+7.2%`、NVFP4 `-13.0%`。
- 不可外推原因至少包括分辨率不一致、旧 compile 缺失/失败、单次运行无误差条、NVFP4 backend 未完整 sweep、旧代码 SHA 不同。
- scheduler 与 client 接近表示模型关键路径基本传导到客户端；scheduler 明显更高则应检查 VAE、raw RGB 物化、D2H/序列化与网络写入。

### D 部分

- 正确链路是 Pod 等待调度，Karpenter 提名 NodeClaim，AWS Fleet 创建实例失败；动态 PVC 因 Pod 未落区而继续 `WaitForFirstConsumer`。
- use1 是 `default / minwm-test-atl2-p6-spot / minwm-test-atl2-karpenter / 专用 EBS PVC`；east2 是 `ray / minwm-test-b200-spot / wan22-ti2v / s3-claim`。
- `1012` 只是连接终止表象；必须结合 compile chunk 时间、server shutdown、Pod/NodeClaim/Spot 事件和是否存在 steady measured chunks判断。
- 先 eager 是故障域排序：短路径先覆盖方法空间，长冷编译后置；补跑使用新 run id，只读取旧证据，不覆盖旧目录。

### E 部分

- 必须出现 dry-run、不可变 SHA、load-bearing 字段复核、事件与日志、结果合同校验；只写 `kubectl apply` 不及格。
- 最小数据字段包括 `comparison_contract.size/case/steps/kv_cache_num_frames`、`warmup_chunks`、`measured_chunks`、profile、client/server FPS；同时验证失败列表和量化/backend 身份。

## 掌握等级

- 90–100：可独立设计下一轮并值守 Spot 重试，能识别合同污染和错误归因。
- 75–89：能复现与解释主流程，少量运维或量化细节需要查文档。
- 60–74：会跑命令但实验公平性、失败保全或数据解释仍不稳定。
- 60 以下：建议先按执行记录重新做一次“从 dry-run 到结果合同校验”的演练。
