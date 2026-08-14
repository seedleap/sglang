# MinWM 后处理小算子融合：正确性根因与负结论（S2）

首次调查：2026-08-07；根因复核：2026-08-14

## 范围与最终决策

本任务调查 5.3 中的三个候选：

1. self-attention residual/gate 与随后 affine LayerNorm；
2. cross-attention residual 与 AdaLN；
3. FFN residual/gate。

最终三项均不落地，产品代码、默认路径和环境变量相对 `main` 均无变化：

- self 候选在 H200 上确实能把两个 Triton segment/launch 变成一个。旧 proposal 的首个
  错误是 Inductor 消除了 graph 内看似显式的 BF16 cast，使 LayerNorm 读取未舍入的 FP32
  residual。用整数位操作强制 BF16 RNE 后修复了这层错误，但真实 `D=3072` 上 fused 与
  baseline 会独立 autotune 到 `R0_BLOCK=1024/2048`，Welford combine tree 仍可能不同。
  当前公开 compile API 不能让一个 graph 跟随另一个 graph 的动态 winner；固定任一配置
  也都不能稳定满足现行两段 baseline 的 bitwise 合同。因此不发布单段实现。
- cross residual/AdaLN 已是一个 reduction Triton kernel/launch，不再包装等价融合层。
- FFN residual/gate 已是一个 pointwise Triton kernel/launch，不再包装等价融合层。

PR 只保留本调查文档、一个 Torch-only CUDA golden repro 及其纯 CPU 位级回归。脚本在
benchmark 侧局部重建已被否决的 proposal、显式 bit-RNE 候选和 reduction 配置因果实验，
既不修改也不暴露产品路径。现有两段实现就是正确性优先的最小可行替代：保留
residual/gate 的 pointwise 融合，保留独立 compiled LayerNorm 边界。

## 调用链与数值语义

`MinWMCausalTransformerBlock.forward` 的相关顺序是：

1. self-attention output projection；
2. `_minwm_adaln(..., y=attn_output, m_gate=..., e_gate=...)` 以 FP32 计算
   `hidden + attn_output * (model_gate + timestep_gate)`，再舍入回 BF16；
3. `_minwm_layer_norm` 从已经舍入的 BF16 residual 开始 FP32 affine LayerNorm，再舍入
   回 BF16；
4. cross-attention；
5. 一次 `_minwm_adaln(..., r=cross_output, shift/scale=...)` 完成 cross residual、
   LayerNorm 和 AdaLN；
6. FFN；
7. 一次 `_minwm_adaln(..., y=ff_output, gates=...)` 完成 FFN residual/gate。

旧 self proposal 在 Python 表达式中写了 `.type_as(hidden).float()`，但生成的单核直接把
FP32 residual 寄存器送进 `welford_reduce`；真实 BF16 store/load 舍入没有发生。旧单核
因此不是步骤 2 与 3 的等价合并。

第二版 golden repro 用 FP32 位模式实现 BF16 round-to-nearest-even，再把舍入后的位模式
解释回 FP32。生成核中的 integer mask、lower-half、tie-to-even 与 NaN canonicalization
位于 `welford_reduce` 之前，因而编译器不能消掉物化语义。它在相同 reduction 配置下与
“先返回 BF16 residual，再独立 LayerNorm”bitwise exact；但 Inductor 对两个 graph 分别
运行 `dynamic_scale_rblock`。`R0_BLOCK=1024` 把 `D=3072` 合成三个 1024 tile，`2048`
则合成 2048 + 1024(masked) 两个 tile，两种 FP32 Welford combine tree 的最终 BF16 affine
结果可能不同。Causal rollout 会将每个 chunk 的 latent 作为后续 chunk 的条件，局部
1 ULP 级差异不能按独立误差看待。

## 假设与预期

| 候选 | 调查前假设 | CUDA 生成代码结论 | 最终决策 |
| --- | --- | --- | --- |
| self residual/gate → affine LN | 存在真实双 segment 边界，少一次 launch 可能有收益 | 2→1 kernel 成立；旧 proposal 漏真实 BF16 舍入；bit-RNE 修复后仍受独立 reduction autotune 影响 | 无稳定 bitwise 单核，保留现有两段 |
| cross residual/AdaLN | 可能仍有 residual 与 norm/modulation 边界 | 已是 `triton_red_fused__to_copy_add_mul_native_layer_norm_0` 单 kernel | 不重复实现 |
| FFN residual/gate | 可能是多个 eager pointwise launch | 已是 `triton_poi_fused__to_copy_add_mul_0` 单 kernel | 不重复实现 |

预期中“launch 变少可能带来小收益”只在 micro 层面成立；“Python 中写出 BF16 cast 就
足以保留数值语义”不成立；“显式 bit-RNE 后自然与独立 LN 一致”也只在 reduction 配置
相同时成立。端到端质量门是性能 A/B 的全局前置条件，因此旧候选质量失败后未消费
H200 时间测无资格发布的 headline；根因复核也只做单卡 correctness，没有进入性能测量。

## 最小 golden repro 与首差定位

根因复核固定 Torch `2.12.1+cu130`、CUDA `13.0`、Triton `3.7.1`、H200 SM90，不安装
MinWM/diffusion 依赖。输入生成器固定 seed `20260807`，真实维度 `D=3072`；覆盖
`B1/S1`、`B2/S7`、真实 SP4 local row count `B1/S3432`、BF16 autocast on/off、
contiguous/non-contiguous hidden/update、gate `[D]`/`[1,D]`/`[B,1,D]`，以及真实
`[B,S,6,D].select(2,2)` 的非连续 timestep gate。每个 comparison 报 changed fraction、
max absolute error、BF16 ULP 和首差坐标。

| 步骤/候选 | 布局与 shape | residual | LayerNorm/首差 | 结论 |
| --- | --- | --- | --- | --- |
| 旧单核 proposal | B1/S7/D3072 contiguous | bitwise | 473/21,504 changed，首差 `[0,0,30]`，max_abs 0.03125，max_ulp 648 | LN 读取 pre-round FP32 residual |
| bit-RNE，小 shape | S1、B2/S7；contig/noncontig；autocast on/off；三种 gate | bitwise | bitwise，max_ulp 0 | BF16 RNE 实现与广播/stride 正确 |
| bit-RNE，真实 shape，默认 autotune | B1/S3432/D3072 | bitwise | 同 winner 时 0；winner 不同时 203/10,543,104 changed，首差 `[0,0,637]`，max_abs 0.03125，max_ulp 43 | 物化必要但不充分 |
| bit-RNE 固定初始 R-block | 同上，contiguous，6 个冷进程 | bitwise | baseline 选 R2048 的 2 次为 0；选 R1024 的 4 次均精确复现 203 个差异 | 差异由 reduction tree 唯一解释 |
| 同上，non-contiguous，2 个冷进程 | B1/S3432/D3072 | bitwise | 一次 0；一次 170 changed，首差 `[0,24,601]`，max_abs 0.015625，max_ulp 8 | 非连续布局也会独立选 winner |
| fixed fused vs fixed 独立 LN | 上述全部 8 个真实 shape 冷进程 | bitwise | 全部 bitwise，max_ulp 0 | 排除 eps、affine、promotion、RNE 本身 |

生成代码的静态元数据对照为：二者均
`size_hints={x:4096,r0_:4096}`、`ReductionHint.INNER`，初始配置均为
`XBLOCK=1,R0_BLOCK=2048,num_warps=16,num_stages=1`。baseline 有 4 loads、2
reductions；bit-RNE fused 有 7 loads、2 reductions，均未触发 `load+reduction>=10` 的
静态降块阈值。实际分叉来自 `dynamic_scale_rblock` 按各自寄存器占用加入 R1024 候选，
再独立 benchmark/cache。`TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1` 直接确认同一输入的
baseline 可跨冷进程选择 R1024 或 R2048；fused 关闭动态缩放后恒为 R2048。

这也回答了几个容易混淆的点：`eps=1e-6`、BF16 affine 参数转 FP32、LayerNorm 输入 FP32
promotion、输出 BF16 舍入在 fixed-vs-fixed 中完全相同；同一 graph 中普通 BF16 cast 会被
折叠，而整数 bit-RNE 确实在 reduction 前发生。剩余差异不是“数值容差”，而是 Welford
combine 顺序。

## CUDA 取证与探索性测试

H200 micro 取证使用 `B=1,S=6864,D=3072,BF16`（1248×704、4 latent frames 的 SP2
local sequence）。该节点同时有其他微任务，时间只解释 kernel 边界，不是 headline：

| 候选 | kernel 名 | launch | profiler CUDA | 10 次循环均值 | 对 eager |
| --- | --- | ---: | ---: | ---: | --- |
| self baseline | `triton_poi_fused__to_copy_add_mul_0` + `triton_red_fused__to_copy_native_layer_norm_0` | 2 | 65.152 + 95.455 µs | 0.238882 ms | max_abs 0.015625 |
| self proposal | `triton_red_fused__to_copy_add_mul_native_layer_norm_0` | 1 | 136.095 µs | 0.160144 ms | max_abs 0.125 |
| cross 现状 | `triton_red_fused__to_copy_add_mul_native_layer_norm_0` | 1 | 133.471 µs | 0.169013 ms | max_abs 0.125 |
| FFN 现状 | `triton_poi_fused__to_copy_add_mul_0` | 1 | 66.015 µs | 0.144963 ms | bitwise |

真实旧双 compiled segment 对 proposal 单 segment 的小形状 CUDA 对照中，self residual
仍 bitwise，LayerNorm 在 autocast off/on 分别有 8.7%/9.7% 元素改变，max_abs=0.03125；
首差为 1 BF16 ULP，但近零位置的 ordered-bit 最大距离更大。这里没有把 trace 相对 eager 的 `max_abs=0.125`
反向写成“刚好通过”的 tolerance，也没有用 12.5% changed-fraction guard 代替质量合同。

探索性测试曾覆盖：

- `B1/S1`、`B2/S7` 边界；
- SP2/SP4 local sequence 6864/3432；
- gate shape `[D]`、`[1,D]`、`[B,1,D]`；
- 真实 `[B,S,6,D].select(-2, index)` 非连续 timestep gate，以及非连续 hidden/update；
- autocast 与 segment compile on/off；
- 非 CUDA/compile-off fallback。

这些测试验证了 proposal 的形状安全性，却不能证明 causal rollout 的质量。由于产品
proposal 已删除，相应产品测试也从最终 diff 删除，避免给未发布 API 留永久维护负担。

## 实际端到端 A/B

质量 A/B 使用同一 H200 Job、MinWM 5B step-3200、1248×704、BF16、SP2、4 DMD +
1 clean-cache、16 frames/chunk、seed 42。baseline 是产品旧双 segment，candidate 是
proposal 单 segment；两边固定 `MINWM_S0_KV_CACHE_NUM_FRAMES=45`，运行 8 个 rolling
chunks，共比较 129 帧（首帧加 8×16 输出帧）和两 rank 的 16 个逐 chunk latent 文件。

沿用仓库已有 `bf16_backend_candidate` generated-frame 合同：`max_abs<=8`、
`RMSE<=1.0`、`SSIM>=0.995`。仓库没有已评审 latent 阈值，因此 latent 只要求文件、
shape、dtype 对齐与 finite，并完整报告漂移，不根据本次结果杜撰阈值。

| 指标 | 观测 | 合同/解释 | 结果 |
| --- | ---: | --- | --- |
| frame max_abs | 255 | <=8 | 失败 |
| frame RMSE | 27.4112707 | <=1.0 | 失败 |
| frame SSIM | 0.8589902 | >=0.995 | 失败 |
| min per-frame SSIM | 0.5127644 | 诊断 | 严重偏离 |
| frame changed fraction | 65.2035% | 诊断 | 非局部差异 |
| frame PSNR | 19.3722 dB | 诊断 | 严重偏离 |
| latent files | 16/16 finite、shape/dtype 对齐 | 必须满足 | 通过结构门 |
| latent changed fraction | 95.8299% | 诊断 | 几乎全变 |
| latent max_abs | 6.40625 | 诊断 | 随 rollout 放大 |
| latent aggregate RMSE | 0.4075434 | 诊断 | 随 rollout 放大 |
| max latent relative-L2 | 0.5554221 | 诊断 | 严重累积 |
| min latent cosine | 0.8455656 | 诊断 | 严重累积 |

同一 rank 的 relative-L2 随 chunk 演化如下；两个 SP rank 的记录一致：

| chunk | max_abs | RMSE | relative-L2 | cosine |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.44043 | 0.02790 | 0.02768 | 0.99962 |
| 1 | 1.58008 | 0.05108 | 0.04829 | 0.99883 |
| 2 | 3.49219 | 0.08146 | 0.07729 | 0.99701 |
| 3 | 4.19531 | 0.20061 | 0.19054 | 0.98187 |
| 4 | 5.09375 | 0.37203 | 0.35912 | 0.93567 |
| 5 | 4.89062 | 0.46507 | 0.42467 | 0.91035 |
| 6 | 6.21875 | 0.56882 | 0.44587 | 0.90071 |
| 7 | 6.40625 | 0.77473 | 0.55542 | 0.84557 |

这不是临界失败：视频三项 normative 指标同时大幅越界，latent 还显示明确的逐 chunk
累积趋势。Job 因质量门返回 exit 1，未进入 profiler-off 20+200，也从未启动 Nsight。
因此没有 Client/Scheduler FPS、DiT/VAE wall/CUDA、kernel/launch headline 可以诚实
报告；micro 的 2→1 launch 不能冒充正式性能收益。后续发现的旧 Nsight 11-chunk/除以
10 exact-window 问题也没有污染 S2 数据，因为本 Job 在任何 capture 前已经止损。

## Provenance 与失败证据

正式质量 attempt：

- kube context：`codex-minwm-test-phx2`（所有读写命令显式指定）；
- region / NodePool / instance：`us-west-2` /
  `minwm-test-phx2-p5e-spot` / `p5e.48xlarge`（8×H200）；
- Job / Pod：`minwm-s2-postproc-ab-h200-phx2-20260807-01` /
  `minwm-s2-postproc-ab-h200-phx2-20260807-01-4kncp`；
- live `backoffLimit=0`，Pod restart=0，最终 exit=1；
- SGLang 临时 runner：`4a35ed30d28704c8550d1607671b729a48978513`；
- S0 schema/count/audit 工具：`b178572f84521fa44400670fa76a29ad40c433d7`
  （包含 `b9240233b2`）；最终实现 PR 不带入 S0 diff；
- minWM：`2efc6485f65e8fcab506665efde79bc41406385e`；
- container：`829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-training@sha256:bedc07ea3ba55059a8c1c569c3b177c4d00d41f37d4fa9105375531534ef5f2a`；
- PVC：`minwm-s2-postproc-ab-h200-results-20260807`，200 GiB，保留；
- `quality-comparison.json` SHA-256：
  `6eefed64904216caa9957bc16d3158dd46ef6e3f1fb9292406b73d25a10b3e6b`；
- root `invalid/attempt.json` SHA-256：
  `66aad640b831d6b57ff355db65818672865e55e7a5130f12c9195081d02bd216`。

质量是所有性能 lane 的全局前置条件，所以本次 root marker 范围正确。marker 记录
`inventory_status=complete`，共 470 个文件，0 个缺 SHA-256，`physically_deleted=false`，
并明确可通过任务 PVC 恢复。没有 profiler-off 合格 lane，因此 root marker 没有误伤
headline。若未来是已验证 profiler-off 后才发生 Nsight/另一 lane 失败，marker 必须放在
失败 lane 目录，不能用 attempt-root marker 作废兄弟 lane；只有 setup/质量/parity 等全局
前置失败才使用 root marker。

任何失败、旧契约或 partial 产物都不得物理删除。聚合器应沿当前 JSON 的 parent 向上到
最近 measurement root 检查 marker，sibling lane marker 不得互相影响。只允许精确删除
自己创建的 Job/Pod 控制对象止损。旧质量 Job/Pod 控制对象在根因复核收尾时按完整名称
精确删除，PVC、root marker、470 个文件及其 SHA 证据仍保留。

## 与预期不符处、失败 attempt 与审计

源码和 CUDA 生成代码共同推翻了“需要实现三项融合”的预期：cross/FFN 已是单 segment，
只有 self 有真实边界。self 的错误也不是最初概括的单一“reduction schedule 改变”：旧
proposal 先漏了真正的 BF16 store/load 舍入；修复该层后，才显露独立动态 reduction
winner 的第二层问题。

2026-08-07 的四次 micro Job 均保留原结论：

- `minwm-s2-postproc-trace-h200-phx2-20260807-01` 使用不存在的完整 checkout SHA，未进入
  CUDA 测试。
- `...-02` 正确 checkout 后在 pytest 收集阶段发现镜像缺 `orjson`，未进入 candidate。
- `...-03` 如实暴露 `B1/S1` singleton transpose 可能仍 contiguous，以及 proposal
  LayerNorm 相对 eager 非 bitwise；`set -e` 在 trace 前停止。
- `...-04` 改为真实旧双 compiled segment 对 proposal 单 segment。pytest 正确报告
  non-bitwise，runner 保存测试状态后仍完成 self baseline/proposal、cross、FFN 四项
  BEGIN/END、结果和 `cuLaunchKernel` 证据，最后按 `test_status=1` 退出。它是正确性
  暴露，不是基础设施或上传失败。

2026-08-14 根因复核的 attempt 如下；全部使用显式 kube context、`backoffLimit=0`、
单卡 H200、无 SP。每次结束后只精确删除自己的 Job/Pod 控制对象，PVC 证据保留：

| attempt | 状态 | 结果与审计 |
| --- | --- | --- |
| `minwm-s2-self-rounding-h200-phx2-20260814-01` | failed | source-tree import 缺 `orjson`，exit=1；root marker 含 3 个文件和 SHA，未进 candidate |
| `...-02` | 强制止损/invalid | 完整 diffusion 安装开始把原生 Torch 2.12.1+cu130、cuDNN/NCCL/cuBLAS/Triton 替换为另一套版本；立即删除 Job，`runner.log` 70,845 bytes，SHA `0e34ab20…`；任何输出不得用于正确性 |
| `...-03` | correctness failure | 原生 runtime 契约通过；定位旧 proposal 漏 BF16 物化；bit-RNE 小 shape 全 exact，但 S3432 有 203 differences；exit=1 marker 完整，12 文件均有 SHA |
| `...-04` | diagnostic success | 保存 baseline 与 bit-RNE 生成源码；静态 decorator 相同，但 bit-RNE 当次 exact，暴露动态 winner 假设 |
| `...-05` | diagnostic success | 6 contig + 2 noncontig 冷进程，baseline/fused 当次都选 X1/R1024/W16/S1，8/8 exact；说明配置相同时可 exact，但不能证明稳定 |
| `...-06` | causal success | fused 禁用动态缩放固定 R2048；baseline 冷进程在 R1024/R2048 间变化，输出严格随 winner 在“固定差异指纹/0 差异”间切换；完成后 exit=0，控制对象删除，PVC 保留 |

`-06` 的关键证据 SHA：`all-launch-params.txt` 为
`040b429cbb373ed150937949540bdd15aab0fc5da9ea4003f5c1afc4efdc61cc`；首个 203-diff
log 为 `6d6b554df9caced795ec7964c6dd973a54250760d9afee36cc6974c9373a259a`；runtime
contract 为 `41a1a10f8f72817077be98bc4584042445db8e9c9d128df2c971693331465df3`。

桌面默认 kube context 曾漂移到 `codex-seed-leap-use1`；误投对象始终 Pending、未启动，
随后只按完整名称精确删除。此后所有命令固定
`--context codex-minwm-test-phx2`。Pod 内若运行仓库 registered/unit tests，
`PYTHONPATH` 必须包含 `/workspace/sglang/python`，并在模型或 client 前先跑
`python -c 'import sglang.test.ci.ci_register'`；本轮 golden repro 刻意保持 Torch-only，
不导入 `sglang`，从而无需改变镜像依赖。

## 证据与决策过程

1. 源码定位 self 的两个相邻 `_MinWMSegmentCompile.get`，确认真实优化边界。
2. 源码和 H200 生成代码确认 cross/FFN 各自已是单 segment，排除重复实现。
3. 在 benchmark 侧重建旧 self proposal，确认 graph 内普通 BF16 cast 被折叠；LayerNorm
   读取 pre-round residual，这是第一个首差边界。
4. 用整数位操作显式实现 BF16 RNE；小 shape、autocast、gate 广播和非连续布局均 exact，
   真实 shape 仍偶发 203 differences。
5. 比较生成代码与实际 launch winner；固定 fused 为 R2048，并用 fixed 独立 LN 做交叉
   对照，严格复现“winner 不同就固定指纹、相同就 0 差异”。
6. 审计 PyTorch 2.12.1 的公开 compile options：没有稳定的 per-segment R-block API，也
   没有让 fused graph 跟随 standalone graph autotune winner 的合同。
7. 沿用已有视频 fast-lane 合同复核旧 proposal 的 rolling A/B，三项 normative 指标同时
   严重失败，并由逐 chunk latent 证明误差累积；保留 470 文件及哈希，未启动
   profiler-off/Nsight。
8. 删除产品实现、开关和测试。当前数值语义下，5.3 的这三个点均不落地；最小正确替代
   是 main 已有的 pointwise segment + 独立 LayerNorm segment。

## 尝试后放弃的方案

- 不复用通用 `ScaleResidualLayerNormScaleShift` CUDA kernel：它包含额外 scale/shift
  接口并可能 materialize contiguous 输入，扩大数值与布局审计面。
- 不发布显式 bit-RNE 的 Inductor 单 graph：它解决真实 BF16 materialization，却不能让
  reduction autotune 与 standalone baseline 同步。
- 不用 `dynamic_scale_rblock=False` 作为产品修复：它只固定初始 R2048；现行 baseline
  可选 R1024，`-06` 已证明会稳定产生 203/170 个差异。
- 不使用 `autotune_lookup_table`、`fixed_config` 或 `strict_reduction_rblock`：前者按
  Triton source hash/version/layout 绑定且 miss 时静默回退，后两者是 Inductor 内部
  codegen/IR 接口，不是产品合同。
- 不复制 fixed-R1024/R2048 custom Triton：任一个都只能匹配 baseline 的一种动态 winner；
  要在运行时先执行/探测 standalone LN 再分派会恢复额外 kernel、依赖私有状态并失去本次
  优化边界。未来只有先把 baseline reduction 另立为公开 canonical contract 后才可重开。
- 不为 cross/FFN 增加“新融合”开关：它们已经是单编译段，任何对照只会关闭 main 的
  既有优化后再冒充收益。
- 不把 self proposal 作为默认关闭的隐藏 fast lane：质量失败不是默认值能消除的风险，
  留下死代码还会增加未来维护和误开启概率。
- 不放宽视频或 latent tolerance：视频合同是已有评审口径，且结果远非阈值附近；latent
  没有既有合同，所以只报告而不反向拟合一个会通过的数值。
- 不在质量失败后继续跑 20+200 或 Nsight：launch 数减少已由 micro 证明，性能数据无法
  改变发布资格。

## 风险、回滚与复现

最终产品 diff 没有运行时改动，因此回滚面为零；默认 parity 路径保持 main 的双 segment。
调查脚本是 Torch-only 的 source-shaped copy，只用于同一 checkout 的 CUDA 诊断，不导入
`sglang`、不安装 diffusion 依赖，也不定义 headline schema。

复现单项生成代码/launch：

```bash
export PYTHONPATH="$PWD/python"
TORCH_LOGS=output_code,kernel_code \
TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1 python3 \
  benchmark/minwm_realtime_parity/trace_postprocess_fusions.py \
  --candidate self-bitquant-no-dynamic-scale \
  --batch-size 1 --sequence-length 3432 --hidden-size 3072 \
  --gate-shape row --input-layout contiguous --detailed-correctness
```

`self-baseline`、`self-proposed`、`self-bitquant`、
`self-bitquant-no-dynamic-scale`、`cross`、`ffn` 应分别启动冷进程，避免
`_minwm_adaln_op` specialization 的编译日志混合。`self-proposed` 在脚本内局部定义，
比较对象是产品真实旧双 compiled segment；`self-bitquant-no-dynamic-scale` 纯为因果
实验，不能复制到产品路径。检查同目录生成的 `.launch_params`：fused 应为 R2048；
baseline 若为 R1024 则应精确出现对应 difference 指纹，若为 R2048 则应为 0。脚本输出
只解释 kernel 边界和数值漂移。

S0 canonical 在本调查期间经历 `59aa68a382` 的 stage-trace 修复、`b9240233b2` 的 latency
count 强约束与 `b178572f84` 的 invalid-attempt 审计。正式 S2 Job 使用后两者；由于质量
门在任何 measurement client 前失败，S2 没有旧契约结果需要混入或删除。

## 给负责人掌握代码的检查题

1. **self 的真实边界在哪里？** 参考：`MinWMCausalTransformerBlock.forward` 先调用
   `_minwm_adaln` 更新 residual，再调用 `_minwm_layer_norm`。
2. **旧 proposal 的首差具体在哪里？** 参考：graph 内 `.to(BF16).float()` 被折叠，生成核
   将 residual 的 FP32 寄存器直接送入 `welford_reduce`；首差在 LayerNorm，而返回
   residual 本身仍 bitwise。
3. **为什么整数 bit-RNE 修复后仍不能发布？** 参考：它保证 LN 输入值与 BF16
   materialization 相同，却不能固定 Welford combine tree；baseline 与 fused 独立
   autotune R1024/R2048。
4. **R1024 与 R2048 为什么会改变结果？** 参考：D3072 分别是 3×1024 与
   2048+1024(masked) 的 Welford tile/combine 顺序；`-06` 在 winner 不同时精确复现
   203/170 differences，相同时为 0。
5. **真实 timestep gate 为什么非连续？** 参考：它来自
   `[B,S,6,D].select(-2,index)`，token stride 是 `6*D`；只测 contiguous 会漏布局风险。
6. **cross 为什么不实现？** 参考：`r=cross_output`、LayerNorm、shift/scale 已在一次
   `_minwm_adaln_op` specialization，H200 是单 reduction kernel/launch。
7. **FFN gate accumulation 的 dtype 和边界是什么？** 参考：hidden、FFN output 和两个
   gate 提升到 FP32 乘加，随后 `type_as(hidden_states)`；现状已是单 pointwise kernel。
8. **micro 2→1 launch 为什么不能证明可发布？** 参考：它不覆盖 30 blocks、4 DMD +
   1 clean-cache 和 causal feedback；本次 frame SSIM 只有 0.859，latent relative-L2 随
   chunk 增至 0.555。
9. **为什么旧质量 Job 的 root invalid marker 合理？** 参考：质量是全局前置条件且没有任何
   profiler-off 合格 lane；若只坏 Nsight，应改用 lane marker，不能误伤 sibling headline。
10. **最终代码如何保证 parity，何时可重开？** 参考：proposal、环境开关和产品测试都已
    删除，运行时文件与 `origin/main` 一致；只有 baseline reduction 先获得公开、稳定、
    可复用的 canonical config 合同，才值得重开单核实现和后续 SP2/SP4 性能 A/B。
