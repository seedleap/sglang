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
