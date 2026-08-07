# MinWM 后处理小算子融合（S2）

日期：2026-08-07

## 范围与结论状态

本任务审计三个候选：

1. self-attention 的 residual/gate 与随后 affine LayerNorm；
2. cross-attention 的 residual 与 AdaLN；
3. FFN 的 residual/gate。

当前实现只改动候选 1，并由 `MINWM_FUSE_SELF_ATTN_POST_FAST` 独立控制，默认关闭。
候选 2 和 3 的 H200 生成代码已确认各自是单 Triton kernel/单 launch，因此不新增等价
融合层。候选 1 的单 kernel 会改变 LayerNorm reduction，只能作为显式 fast lane，
不能成为 parity 默认路径。统一测量 canonical 为 `b9240233b2`：它修复最后一条
DiT/VAE stage trace 的 199/200 竞态，且所有 available wall/CUDA latency 都显式写
`count == workload.measured_chunks`；失败产物审计再叠加 `b178572f84`。S0 合并前只在
临时测试分支叠加其提交链，本任务对 `main` 的实现 diff 不复制测量基础设施。
旧 `59aa68a382` 因 latency summary 漏 `count` 而不具备最终验收资格；S2 没有用它启动
正式 client，也没有旧结果需要冒充或删除。

## 调用链与数值合同

`MinWMCausalTransformerBlock.forward` 的相关顺序是：

1. self-attention output projection；
2. `_minwm_adaln(..., y=attn_output, m_gate=..., e_gate=...)` 计算
   `hidden + attn_output * (model_gate + timestep_gate)`；
3. `_minwm_layer_norm` 做带 weight/bias 的 affine LayerNorm；
4. cross-attention；
5. 一次 `_minwm_adaln(..., r=cross_output, shift/scale=...)` 完成 cross residual、
   LayerNorm 和 AdaLN；
6. FFN；
7. 一次 `_minwm_adaln(..., y=ff_output, gates=...)` 完成 FFN residual/gate。

候选 1 的新编译段 `_minwm_self_attn_post_op` 严格保留两处边界：

- residual/gate 计算使用 FP32，随后显式转回输入 dtype；
- LayerNorm 再从已舍入的 BF16 residual 提升到 FP32，weight/bias 先按旧路径转成
  hidden dtype，再在归一化中提升到 FP32。

因此它不是把所有表达式自由重排成一个 FP32 公式。非 CUDA、关闭
`MINWM_SEGMENT_COMPILE` 或关闭候选开关时均有 PyTorch eager/原路径回退。

## 假设与预期

| 候选 | 审计前假设 | CUDA 取证结论 |
| --- | --- | --- |
| self residual/gate → affine LN | 两个独立 segment，中间有 BF16 materialization 和一次 launch gap | 2→1 kernel 成立，但 reduction 非 bitwise；只保留待验收 fast lane |
| cross residual/AdaLN | 可能仍有 residual 与 norm/modulation 边界 | 已是一个 reduction Triton kernel；不新增实现 |
| FFN residual/gate | 可能是若干 eager pointwise launch | 已是一个 pointwise Triton kernel；不新增实现 |

性能预期必须由 H200 profiler-off 和独立 Nsight 稳态窗口验证。即使 microbenchmark
减少一个 launch，也不能预设端到端一定提升；kernel 变大、寄存器压力、编译 cache 或
通信隐藏关系都可能抵消收益。

## 测试矩阵

CPU、compile-off 与安全 fallback 直接比较新算子和旧的 `_minwm_adaln_op` +
`_minwm_layer_norm_op`，要求 `rtol=0, atol=0`。CUDA compile-on 比较真实旧双 compiled
segment 与新单 compiled segment：residual 必须 bitwise；LayerNorm fast 输出逐元素最多
偏离旧基线 1 个该位置的 BF16 ULP，并同时记录 `max_abs` 与 changed fraction。changed
fraction 的 12.5% regression guardrail 来自 H200 autocast off/on 的 8.7%/9.7% 首次观测
并预留约 29% 相对幅度；它只是微算子回归界，不替代端到端质量验收。

这里没有把 micro trace 相对 eager 的 `max_abs=0.125` 反向写成全局 tolerance。CUDA
约束按旧双 compiled segment 的每个输出位置计算向正/负方向的 BF16 ULP，并要求每个
非零差异都不超过该位置 1 ULP；同时单独打印/限制 changed fraction。端到端视频沿用
仓库已评审的 `bf16_backend_candidate` 合同：generated frames 的 `max_abs<=8`、
`RMSE<=1.0`、`SSIM>=0.995`。仓库没有既有 latent 阈值，所以 latent 只强制文件集合、
shape、dtype 一致且值 finite，并完整报告 bitwise、changed fraction、max_abs、RMSE、
relative-L2 和 cosine；不会根据本次观测杜撰一个“刚好通过”的 latent 门槛。

- batch/sequence 边界：`B1/S1`、`B2/S7`；
- 1248×704、4 latent frames 的 SP2/SP4 local sequence：6864/3432 tokens；
- gate shape：`[D]`、`[1,D]`、`[B,1,D]`，以及非连续 `[B,S,D]` timestep gate；
- 非连续 hidden/attention output；
- CUDA 上 `MINWM_SEGMENT_COMPILE` on/off 与 autocast on/off；
- CPU 或 segment compile 关闭时走安全 eager fallback。

完整 CUDA 用例需要 H200/Linux 镜像；本地 macOS 无 CUDA，只执行语法、静态检查和
可用的 CPU 数值用例。

## 实际 A/B

以下表格只接受通过 S0 `b9240233b2` + audit `b178572f84` validator、且 DiT/VAE
latency summary 都显式包含 `count=200` 的 `minwm-realtime-measurement/v1` 产物。
profiler-off headline 使用 20 warmup + 200 measured；Nsight 在外部完成 20 warmup 后
抓至少 10 个 steady chunks，且不同时启用 `torch.profiler`。baseline/candidate 都固定
`MINWM_S0_KV_CACHE_NUM_FRAMES=45`：这是 rolling-window steady-state contract，不能随
`max_chunks` 扩张。首块、短程 append/recompute 与窗口增长的数值检查另跑，不混入
headline 性能窗口。

| SP | 开关 | Client FPS | Scheduler FPS | chunk wall | DiT wall/CUDA | VAE wall/CUDA | kernel/launch | 状态 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | self off | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | Job 等待整机 |
| 2 | self on | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | Job 等待整机 |
| 4 | self off | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | SP4 复验待测 |
| 4 | self on | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | SP4 复验待测 |

正式 Job `minwm-s2-postproc-ab-h200-phx2-20260807-01` 已创建，但唯一 Pod
`...-4kncp` 尚未选节点或启动容器。调度事件是四台匹配节点均
`Insufficient nvidia.com/gpu`，它们当前分别被 S0、S1、S4 和独立训练整机占用；
NodePool 已到当前限额。S2 原地等待整机释放，不新建 NodePool、不抢占或清理他人
资源。因容器尚未启动，本表没有 partial client 数据。

H200 SP2（`S=6864,D=3072,BF16`）micro trace 只用于解释结构；节点上有其他微任务，
以下时间不作为 headline：

| 候选 | kernel 名 | launch | profiler CUDA | 10 次循环均值 | 对 eager |
| --- | --- | ---: | ---: | ---: | --- |
| self baseline | `triton_poi_fused__to_copy_add_mul_0` + `triton_red_fused__to_copy_native_layer_norm_0` | 2 | 65.152 + 95.455 µs | 0.238882 ms | max_abs 0.015625 |
| self fast | `triton_red_fused__to_copy_add_mul_native_layer_norm_0` | 1 | 136.095 µs | 0.160144 ms | max_abs 0.125 |
| cross 现状 | `triton_red_fused__to_copy_add_mul_native_layer_norm_0` | 1 | 133.471 µs | 0.169013 ms | max_abs 0.125 |
| FFN 现状 | `triton_poi_fused__to_copy_add_mul_0` | 1 | 66.015 µs | 0.144963 ms | bitwise |

cross/FFN 已确认是单 kernel，不用“关闭已有优化”的退化结果冒充本 PR 收益；因此三项
独立消融中，这两项是中性审计结论，组合候选也只等于 self fast 开关。

## 与预期不符处

- 源码审计与“需要分别实现三个新融合”的初始预期不符；CUDA 进一步确认 cross 与
  FFN 已分别是单 kernel/单 launch，不需要新代码。
- self 保留了显式 BF16 materialization 语义，但 Inductor 合并后的 LayerNorm Welford
  reduction schedule 仍发生变化。旧双 compiled segment 与新单 segment 在小形状 H200
  上有 8.7%/9.7% 元素非 bitwise，最大差 0.03125；因此不能做 parity lane。
- 第一次 H200 trace Job `minwm-s2-postproc-trace-h200-phx2-20260807-01` 在执行测试前
  checkout 失败：manifest 把临时 cherry-pick `e728e59d9a` 错写成不存在的完整 SHA
  `e728e59d9ad7682dec24b97f7d0007fc0cd0b1c8`。该次没有 CUDA/kernel 结果，不能算作
  候选证据；重试 Job `...-02` 改为固定已推送的实现 commit `6fcf2bef0e21a95b5...`。
- `...-02` 在 H200 上正确 checkout 后，pytest 收集阶段发现镜像未安装 `orjson`，仍未
  进入候选 kernel。`...-03` 复用 S0 已验证的 Python runtime 依赖集合后重跑；没有
  模型 staging、checkpoint 转换或 sglang-kernel 构建。
- `...-03` 的测试如实暴露两类问题：`B1/S1` 的 singleton transpose 可以仍为
  contiguous，旧断言错误；以及 compile-on fast LayerNorm 相对 eager 非 bitwise。
  当次在 pytest 失败后由 `set -e` 停止，没有候选 trace。
- `...-04` 把 CUDA 语义对照改为真实“旧双 compiled segment vs 新单 segment”，并保存
  pytest 状态后继续 trace。四个候选都有唯一 BEGIN/END、结果行和 `cuLaunchKernel`
  profiler 行；容器最后按 `test_status=1` 执行 `exit 1`，所以 Job 正确显示 Failed。
  这是测试暴露 fast lane 非 bitwise，不是上传或基础设施失败；证据完整，不做第五次
  micro trace。
- S0 `59aa68a382` 虽修复了完整 stage-trace 等待，但真机复核发现 latency summary JSON
  没有显式 `count`，无法证明 DiT/VAE 均收齐 200 条。S2 在正式 A/B client 前暂停，
  没有产生需丢弃的 profiler-off/on 数据；随后改用 `b9240233b2`，其 schema 和 custom
  validator 都强制 count 等于 measured chunks。`b178572f84` 又保证原位 partial JSON
  只被最近 `s0-measurement` 根的直属 invalid marker 排除，不误伤兄弟 attempt。
- 提交 `...-02` 时发现桌面默认 kube context 漂移到了 `codex-seed-leap-use1`：该集群
  中对象始终 Pending、未启动容器或占 GPU，随后只按完整 Job 名精确删除。之后所有
  read/dry-run/apply/logs/delete 均显式指定 `--context codex-minwm-test-phx2`。正式记录
  的 region/NodePool 为 us-west-2 / `minwm-test-phx2-p5e-spot`。
- 跨任务发现 source checkout 的 registered/unit tests 可能在收集 `sglang.test.ci` 时因
  `PYTHONPATH` 未暴露 `/workspace/sglang/python` 而提前失败。S2 micro-trace manifest
  已显式 export 该路径，所以四次记录不是这个原因；但当时没有在 pytest 前单独执行
  `python -c 'import sglang.test.ci.ci_register'`。以后凡 Pod 内运行仓库测试，必须先做
  这两个 machine-check，失败则在模型准备/client 前止损。正式 A/B runner 不运行
  registered/unit tests，只运行质量 client、S0 validator 与测量，因此不热修改当前
  Pending Job。

## 证据与决策过程

1. `minwm.py` 中 self residual 与 affine LN 是两个相邻的
   `_MinWMSegmentCompile.get` 调用，存在真实 segment 边界。
2. cross residual、LayerNorm、shift/scale 位于同一次 `_minwm_adaln_op`；FFN
   residual/gate 也位于同一函数的另一 specialization。
3. 因此先只实现 self 组合段，避免为 cross/FFN 再包一层等价函数。
4. self 的 2→1 launch 已证实，但 bitwise 未通过，所以环境变量显式命名为 `_FAST` 且
   永远默认关闭。只有 `b9240233b2` + `b178572f84` 的 profiler-off/Nsight 有显著收益、端到端
   latent 证据完整且视频通过既有 fast-lane 质量合同，才保留 opt-in 实现；否则删除
   实现，只保留负结论文档。

## 尝试后放弃的方案

- 不复用通用 `ScaleResidualLayerNormScaleShift` CUDA kernel：它还包含 scale/shift
  接口，并会对输入做 contiguous materialization；MinWM self 路径要求保留特定的
  BF16 参数 cast 与 residual rounding 顺序，直接套用会扩大改动和数值审计面。
- 不为 cross/FFN 预先增加“新融合”开关：当前源码已经是编译段；先取证再决定，避免
  测到的只是关闭 main 既有优化后的负向对照。

## 风险、回滚与复现

风险包括：Inductor 对动态/non-contiguous stride 重编译；合并 reduction 后寄存器压力
增加；不同 SP local sequence 选择不同 Triton config；已观测的 micro 非 bitwise 可能在
30 blocks ×（4 DMD + 1 clean-cache）forwards 的 causal rollout 中放大为
latent/视频差异。

回滚只需保持 `MINWM_FUSE_SELF_ATTN_POST_FAST=false` 或不设置；该值当前也是默认值，
会即时回到 main 的双 segment 路径。若正式 A/B 或质量验收不成立，PR 会直接删除 fast
实现而不是依赖用户记住关闭开关。

本地检查：

```bash
ruff check python/sglang/multimodal_gen/runtime/models/dits/minwm.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py
ruff format --check python/sglang/multimodal_gen/runtime/models/dits/minwm.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py
```

真机 runner 为临时分支 commit `4a35ed30d2`，包含 S0 canonical `b9240233b2`、audit
`b178572f84` 和 S2 额外 count validator；S0 21 项 + S2 3 项测试共 24 项通过。正式 Job
和独立 200 GiB PVC 分别是 `minwm-s2-postproc-ab-h200-phx2-20260807-01` 与
`minwm-s2-postproc-ab-h200-results-20260807`，manifest 与 live `.spec.backoffLimit` 都是
0，整机请求 8×H200，稳态固定 KV cache 45 帧。产物根按 Pod hostname 隔离为
`/results/attempts/${HOSTNAME}`。

任何失败、旧契约或 partial attempt 都不得物理删除：原地保留或先写 marker 再移动到
attempt 的 `invalid/`；marker 记录原因、UTC 时间、原/保留路径、size、SHA-256 和
recoverability。S0 聚合器排除路径含 `invalid` 的 JSON，也检查该 JSON 最近
`s0-measurement` 根的直属 `invalid-marker*.json`，不向上误伤兄弟 attempt。验收或脚本
失败即停住保留诊断，禁止 controller 自动重跑、覆盖或混合 provenance；只可精确删除
Job/Pod 控制对象止损，任务 PVC 证据始终保留。

不复用或清理 CUDA Graph/S0 任务的 Job、Pod、PVC。所有 Kubernetes 命令显式固定
`--context codex-minwm-test-phx2`，不依赖或切换桌面的 global current-context。

生成代码与 micro kernel 诊断使用独立脚本，它不产生另一套 headline schema：

```bash
TORCH_LOGS=output_code,kernel_code python \
  benchmark/minwm_realtime_parity/trace_postprocess_fusions.py \
  --candidate self-fused --sequence-length 6864 --profile-kernels
```

`self-baseline`、`cross`、`ffn` 分别单独启动进程，以免不同
`_minwm_adaln_op` specialization 的编译日志混在一起。该脚本的时间只用于解释 kernel
边界；正式 FPS/wall 只来自 S0 schema。

## 给负责人掌握代码的检查题

1. **self 候选为什么不能直接把 residual 到 LayerNorm 全程保持 FP32？** 参考答案：
   minWM 在 residual/gate 后先舍入回 BF16，LayerNorm 读取的是这个 BF16 结果；删除
   cast 会改变 causal rollout。定位 `_minwm_self_attn_post_op`。
2. **开关关闭时走哪条调用链？** 参考答案：`_minwm_adaln` 更新 residual，再调用
   `_minwm_layer_norm`；定位 `MinWMCausalTransformerBlock.forward`。
3. **1248×704 的 SP2/SP4 测试为什么是 6864/3432 tokens？** 参考答案：每帧
   `78×44=3432` DiT tokens，4 latent frames 共 13728，再按 sequence shard 除以 2/4。
4. **timestep gate 为什么专门测试非连续 stride？** 参考答案：真实调用来自
   `[B,S,6,D].select(-2, 2)`，token stride 保留 `6*D`；只测 contiguous 会漏掉编译
   specialization 和错误索引风险。
5. **cross 候选为何不需要新实现？** 参考答案：`r=cross_output`、LayerNorm 和
   shift/scale 已在一次 `_minwm_adaln_op` specialization 中；H200 生成代码确认它是
   `triton_red_fused__to_copy_add_mul_native_layer_norm_0` 单 kernel/单 launch。
6. **FFN 候选的 gate accumulation dtype 是什么？** 参考答案：hidden、FFN output、
   两个 gate 都先提升到 FP32，完成乘加后再 `type_as(hidden_states)`。
7. **减少一个 kernel 为什么不等于端到端变快？** 参考答案：DiT 还受 GEMM、attention、
   Ulysses 通信与编译 config 影响；融合 kernel 的寄存器/访存变化也可能抵消 launch
   节省，必须看 S0 profiler-off wall/FPS 和独立 Nsight。
8. **为什么 self fast 即使变快也不能默认开启？** 参考答案：真实旧 compiled 基线与
   单 segment 的 LayerNorm 已确认非 bitwise；它只能在显式 `_FAST` 开关、端到端质量
   通过时 opt-in，默认始终走旧双 segment parity 路径。
9. **为什么 20+200 测量不能把 KV cache 窗口同步扩到 200 chunks？** 参考答案：正式
   契约要测固定 45 帧 rolling window 下的稳态淘汰成本；随测量长度扩窗会混入增长期
   并改变 baseline/candidate workload。短程首块和 append/recompute 正确性另测。
10. **CUDA 微单测为何同时看 ULP 与 changed fraction？** 参考答案：`max_abs` 会随
    LayerNorm weight/bias 的数值尺度变化；逐元素 1 BF16 ULP 是 scale-aware 误差界，
    changed fraction 则防止“每个元素只差一点但几乎全变”的回退。两者仍不能替代
    causal rollout 的 latent/无损 uint8 视频指标。
