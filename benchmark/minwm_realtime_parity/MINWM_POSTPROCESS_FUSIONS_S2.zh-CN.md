# MinWM 后处理小算子融合调查：负结论（S2）

日期：2026-08-07

## 范围与最终决策

本任务调查 5.3 中的三个候选：

1. self-attention residual/gate 与随后 affine LayerNorm；
2. cross-attention residual 与 AdaLN；
3. FFN residual/gate。

最终三项均不落地，产品代码、默认路径和环境变量相对 `main` 均无变化：

- self 候选在 H200 上确实能把两个 Triton segment/launch 变成一个，但改变了
  LayerNorm reduction 的舍入结果。微算子误差在 causal rolling rollout 中逐 chunk
  放大，端到端视频质量灾难性越过已有 BF16 fast-lane 合同；因此删除 proposal、开关和
  产品测试，不保留 opt-in fast lane。
- cross residual/AdaLN 已是一个 reduction Triton kernel/launch，不再包装等价融合层。
- FFN residual/gate 已是一个 pointwise Triton kernel/launch，不再包装等价融合层。

PR 只保留本调查文档和一个 CUDA 取证脚本。脚本在 benchmark 侧局部重建已被否决的
self proposal，既不修改也不暴露产品路径。

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

self proposal 保留了步骤 2 与 3 之间显式 BF16 materialization，却让 Inductor 在一个
graph 内重新选择 LayerNorm Welford reduction schedule。表达式看似保留了 cast，CUDA
reduction 的加法树仍不相同，所以旧双 compiled segment 与 proposal 单 segment 并非
bitwise exact。Causal rollout 会将每个 chunk 的 latent 作为后续 chunk 的条件，局部
1 ULP 级差异不能按独立误差看待。

## 假设与预期

| 候选 | 调查前假设 | CUDA 生成代码结论 | 最终决策 |
| --- | --- | --- | --- |
| self residual/gate → affine LN | 存在真实双 segment 边界，少一次 launch 可能有收益 | 2→1 kernel 成立；真实旧双 segment 与 proposal 非 bitwise | 端到端质量失败，删除实现 |
| cross residual/AdaLN | 可能仍有 residual 与 norm/modulation 边界 | 已是 `triton_red_fused__to_copy_add_mul_native_layer_norm_0` 单 kernel | 不重复实现 |
| FFN residual/gate | 可能是多个 eager pointwise launch | 已是 `triton_poi_fused__to_copy_add_mul_0` 单 kernel | 不重复实现 |

预期中“launch 变少可能带来小收益”只在 micro 层面成立；“保留 BF16 cast 就足以保留
数值语义”不成立。端到端质量门是性能 A/B 的全局前置条件，因此质量失败后不再消费
H200 时间测无资格发布的 fast lane。

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
每个改变不超过该位置一个 BF16 ULP。这里没有把 trace 相对 eager 的 `max_abs=0.125`
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
自己创建的 Job/Pod 控制对象止损；本次只删除了临时只读 evidence-reader Pod，失败 Job
和 PVC 证据均保留。

## 与预期不符处及四次 micro Job

- 源码和 CUDA 生成代码共同推翻了“需要实现三项融合”的预期：cross/FFN 已是单
  segment，只有 self 有真实边界。
- self 保留显式 BF16 materialization 仍不能保住 reduction 加法树；一个 BF16 ULP 的
  局部偏差经 causal feedback 放大成明显视频差异。
- `minwm-s2-postproc-trace-h200-phx2-20260807-01` 使用了不存在的完整 checkout SHA，未
  进入 CUDA 测试。
- `...-02` 正确 checkout 后在 pytest 收集阶段发现镜像缺 `orjson`，未进入 candidate。
- `...-03` 如实暴露 `B1/S1` singleton transpose 可能仍 contiguous，以及 proposal
  LayerNorm 相对 eager 非 bitwise；`set -e` 在 trace 前停止。
- `...-04` 改为真实旧双 compiled segment 对 proposal 单 segment。pytest 正确报告
  non-bitwise，runner 保存测试状态后仍完成 self baseline/proposal、cross、FFN 四项
  BEGIN/END、结果和 `cuLaunchKernel` 证据，最后按 `test_status=1` 退出。它是正确性
  暴露，不是基础设施或上传失败；证据完整，因此没有第五次 micro trace。
- 桌面默认 kube context 曾漂移到 `codex-seed-leap-use1`；误投对象始终 Pending、未启动，
  随后只按完整名称精确删除。此后所有命令固定 `--context codex-minwm-test-phx2`。
- Pod 内运行仓库 registered/unit tests 时，`PYTHONPATH` 必须包含
  `/workspace/sglang/python`，并先执行
  `python -c 'import sglang.test.ci.ci_register'`。S2 micro runner 已设置路径但当时未单列
  preflight；以后收集失败应在模型/client 前止损。

## 证据与决策过程

1. 源码定位 self 的两个相邻 `_MinWMSegmentCompile.get`，确认真实优化边界。
2. 源码和 H200 生成代码确认 cross/FFN 各自已是单 segment，排除重复实现。
3. 在 benchmark 侧构造最小 self proposal，保留 residual 后 BF16 cast，确认 2→1 launch。
4. 用真实旧双 compiled segment 比较，发现 LayerNorm 非 bitwise；不允许进入 parity lane。
5. 沿用已有视频 fast-lane 合同运行 rolling quality A/B，观察三项 normative 指标同时
   严重失败，并由逐 chunk latent 证明误差累积。
6. 在全局质量门停止，保留 470 文件及哈希，不启动 profiler-off/Nsight。
7. 删除产品实现、开关和测试。当前数值语义下，5.3 的这三个点均不落地。

## 尝试后放弃的方案

- 不复用通用 `ScaleResidualLayerNormScaleShift` CUDA kernel：它包含额外 scale/shift
  接口并可能 materialize contiguous 输入，扩大数值与布局审计面。
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
调查脚本依赖 MinWM 私有 benchmark 接口，只用于同一 checkout 的 CUDA 诊断，不定义
headline schema。

复现单项生成代码/launch：

```bash
export PYTHONPATH="$PWD/python"
TORCH_LOGS=output_code,kernel_code python \
  benchmark/minwm_realtime_parity/trace_postprocess_fusions.py \
  --candidate self-proposed --sequence-length 6864 --profile-kernels
```

`self-baseline`、`self-proposed`、`cross`、`ffn` 应分别启动进程，避免
`_minwm_adaln_op` specialization 的编译日志混合。`self-proposed` 在脚本内局部定义，
比较对象是产品真实旧双 compiled segment；脚本输出只解释 kernel 边界和数值漂移。

S0 canonical 在本调查期间经历 `59aa68a382` 的 stage-trace 修复、`b9240233b2` 的 latency
count 强约束与 `b178572f84` 的 invalid-attempt 审计。正式 S2 Job 使用后两者；由于质量
门在任何 measurement client 前失败，S2 没有旧契约结果需要混入或删除。

## 给负责人掌握代码的检查题

1. **self 的真实边界在哪里？** 参考：`MinWMCausalTransformerBlock.forward` 先调用
   `_minwm_adaln` 更新 residual，再调用 `_minwm_layer_norm`。
2. **为什么保留中间 BF16 cast 仍不 bitwise？** 参考：单 graph 让 Inductor 重新选择
   LayerNorm Welford reduction schedule；cast 保留值域，不能固定 FP32 reduction 加法树。
3. **1248×704 的 SP2/SP4 local sequence 为什么是 6864/3432？** 参考：每帧
   `78×44=3432` tokens，4 latent frames 共 13728，再按 sequence shard 除以 2/4。
4. **真实 timestep gate 为什么非连续？** 参考：它来自
   `[B,S,6,D].select(-2,index)`，token stride 是 `6*D`；只测 contiguous 会漏布局风险。
5. **cross 为什么不实现？** 参考：`r=cross_output`、LayerNorm、shift/scale 已在一次
   `_minwm_adaln_op` specialization，H200 是单 reduction kernel/launch。
6. **FFN gate accumulation 的 dtype 和边界是什么？** 参考：hidden、FFN output 和两个
   gate 提升到 FP32 乘加，随后 `type_as(hidden_states)`；现状已是单 pointwise kernel。
7. **micro 2→1 launch 为什么不能证明可发布？** 参考：它不覆盖 30 blocks、4 DMD +
   1 clean-cache 和 causal feedback；本次 frame SSIM 只有 0.859，latent relative-L2 随
   chunk 增至 0.555。
8. **为什么本次 root invalid marker 合理？** 参考：质量是全局前置条件且没有任何
   profiler-off 合格 lane；若只坏 Nsight，应改用 lane marker，不能误伤 sibling headline。
9. **为什么没有 SP2/SP4 headline A/B？** 参考：quality gate 先失败，继续性能测量不能
   改变 fast lane 的发布资格；Job 在任何 profiler-off/on client 前 exit 1。
10. **最终代码如何保证 parity？** 参考：proposal、环境开关和产品测试都已删除，运行时
    文件与 `origin/main` 完全一致；PR 只留下调查文档和 benchmark 取证脚本。
