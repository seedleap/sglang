# MinWM 后处理小算子融合（S2）

日期：2026-08-07

## 范围与结论状态

本任务审计三个候选：

1. self-attention 的 residual/gate 与随后 affine LayerNorm；
2. cross-attention 的 residual 与 AdaLN；
3. FFN 的 residual/gate。

当前实现只改动候选 1，并由 `MINWM_FUSE_SELF_ATTN_POST` 独立控制，默认关闭。
候选 2 和 3 在 CUDA 生成代码与 launch 证据完成前不改代码。统一性能记录依赖 S0
commit `30cb16708fc768adb063c31c2f1a21eac5a016d2`；S0 合并前只在临时测试
分支叠加该 commit，本任务对 `main` 的实现 diff 不复制测量基础设施。

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

| 候选 | 审计前假设 | 当前预期 |
| --- | --- | --- |
| self residual/gate → affine LN | 两个独立 segment，中间有 BF16 materialization 和一次 launch gap | 合成一个编译段后应从两个 Triton kernel 降到一个，同时保持 bitwise |
| cross residual/AdaLN | 可能仍有 residual 与 norm/modulation 边界 | 源码已在一次 `_minwm_adaln` 调用内；若生成代码也是单 kernel，则不新增实现 |
| FFN residual/gate | 可能是若干 eager pointwise launch | 源码已在一次 `_minwm_adaln` 调用内；若生成代码也是单 kernel，则不新增实现 |

性能预期必须由 H200 profiler-off 和独立 Nsight 稳态窗口验证。即使 microbenchmark
减少一个 launch，也不能预设端到端一定提升；kernel 变大、寄存器压力、编译 cache 或
通信隐藏关系都可能抵消收益。

## 测试矩阵

本地/CI 测试直接比较新算子与旧的 `_minwm_adaln_op` +
`_minwm_layer_norm_op`，要求 `rtol=0, atol=0`：

- batch/sequence 边界：`B1/S1`、`B2/S7`；
- 1248×704、4 latent frames 的 SP2/SP4 local sequence：6864/3432 tokens；
- gate shape：`[D]`、`[1,D]`、`[B,1,D]`，以及非连续 `[B,S,D]` timestep gate；
- 非连续 hidden/attention output；
- CUDA 上 `MINWM_SEGMENT_COMPILE` on/off 与 autocast on/off；
- CPU 或 segment compile 关闭时走安全 eager fallback。

完整 CUDA 用例需要 H200/Linux 镜像；本地 macOS 无 CUDA，只执行语法、静态检查和
可用的 CPU 数值用例。

## 实际 A/B

以下表格只接受 S0 `minwm-realtime-measurement/v1` 产物。profiler-off headline 使用
20 warmup + 200 measured；Nsight 在外部完成 20 warmup 后抓至少 10 个 steady chunks，
且不同时启用 `torch.profiler`。

| SP | 开关 | Client FPS | Scheduler FPS | chunk wall | DiT wall/CUDA | VAE wall/CUDA | kernel/launch | 状态 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | self off | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | self on | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | self off | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 复验 |
| 4 | self on | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 复验 |

cross/FFN 将分别记录其现有 compiled segment 与关闭全局 segment compile 的生成代码
证据，但在确认它们已经是单 kernel 时，不用“关闭已有优化”的退化结果冒充本 PR 收益。

## 与预期不符处

- 源码审计已经与“需要分别实现三个新融合”的初始预期不符：cross 与 FFN 候选看起来
  已由 `_minwm_adaln` 覆盖。最终结论等待 CUDA 生成代码和 kernel launch 名确认。
- self 融合是否能在保持 BF16 rounding boundary 的同时生成单 kernel，等待 H200
  `TORCH_LOGS=output_code,kernel_code` 证据。

## 证据与决策过程

1. `minwm.py` 中 self residual 与 affine LN 是两个相邻的
   `_MinWMSegmentCompile.get` 调用，存在真实 segment 边界。
2. cross residual、LayerNorm、shift/scale 位于同一次 `_minwm_adaln_op`；FFN
   residual/gate 也位于同一函数的另一 specialization。
3. 因此先只实现 self 组合段，避免为 cross/FFN 再包一层等价函数。
4. 开关先默认关闭；只有 bitwise、单 kernel/launch 证据与组合 profiler-off 不回退
   超过 1% 后，才考虑改为默认开启。

## 尝试后放弃的方案

- 不复用通用 `ScaleResidualLayerNormScaleShift` CUDA kernel：它还包含 scale/shift
  接口，并会对输入做 contiguous materialization；MinWM self 路径要求保留特定的
  BF16 参数 cast 与 residual rounding 顺序，直接套用会扩大改动和数值审计面。
- 不为 cross/FFN 预先增加“新融合”开关：当前源码已经是编译段；先取证再决定，避免
  测到的只是关闭 main 既有优化后的负向对照。

## 风险、回滚与复现

风险包括：Inductor 对动态/non-contiguous stride 重编译；合并 reduction 后寄存器压力
增加；不同 SP local sequence 选择不同 Triton config；bitwise micro parity 仍可能在
30 blocks × 5 forwards 的 causal rollout 中放大为视频差异。

回滚只需保持 `MINWM_FUSE_SELF_ATTN_POST=false`；该值当前也是默认值。若最终默认
开启，可用同一环境变量即时回到 main 的双 segment 路径。

本地检查：

```bash
ruff check python/sglang/multimodal_gen/runtime/models/dits/minwm.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py
ruff format --check python/sglang/multimodal_gen/runtime/models/dits/minwm.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py
```

真机测量命令和产物路径将在任务专用 `minwm-s2-postproc-*` Job dry-run 后补充；不复用
或清理 CUDA Graph/S0 任务的 Job、Pod、PVC。

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
5. **cross 候选为何可能不需要新实现？** 参考答案：`r=cross_output`、LayerNorm 和
   shift/scale 已在一次 `_minwm_adaln_op` specialization 中；要用生成代码确认是否
   单 Triton kernel。
6. **FFN 候选的 gate accumulation dtype 是什么？** 参考答案：hidden、FFN output、
   两个 gate 都先提升到 FP32，完成乘加后再 `type_as(hidden_states)`。
7. **减少一个 kernel 为什么不等于端到端变快？** 参考答案：DiT 还受 GEMM、attention、
   Ulysses 通信与编译 config 影响；融合 kernel 的寄存器/访存变化也可能抵消 launch
   节省，必须看 S0 profiler-off wall/FPS 和独立 Nsight。
8. **何时允许默认开启 self 融合？** 参考答案：单测和端到端无正确性回退，有明确
   kernel/launch 或 wall 证据，SP2 主验收和 SP4 复验完成，组合 profiler-off 回退
   不超过 1%；否则保持默认关闭并在本文记录弃用决定。
