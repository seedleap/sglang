# MinWM timestep modulation pass 内复用

状态：实现与 CPU 回归已完成；H200 CUDA compile smoke 已通过。正式 BF16 output
bitwise、A/B 和 Nsight 已按修正后的测量契约重新提交，尚未形成可验收数据。

## 目标与边界

MinWM 的每个 `MinWMCausalTransformerBlock` 原先都会执行一次
`temb[:, frame_index]`。5B 模型有 30 个 block；一个常规实时 chunk 包含 4 次 DMD
forward 和 1 次 clean-latent cache commit forward。因此代码路径上的上限是每 chunk
`5 × 30 = 150` 次相同的 advanced-indexing 物化。这个数字来自调用链，不是 profiler
实测。

本变更只把 block 共享的 timestep modulation 提升到同一次 transformer pass 的 block
循环之前：每个 DMD 或 clean-cache forward 各自重新准备一次，30 个 block 复用该结果。
output head 中的 `temb[:, frame_index]` 使用的是另一份 output timestep embedding，仍只
执行一次，不在本次重复消除范围内。

不在范围内：跨 DMD step 缓存 time embedding、action encoder 复用、QKV 融合、
clean-cache final-block 裁剪、VAE 或输出路径优化。

## 假设与预期

1. eager/segment-compile exact 路径当前没有把 block 外的 advanced indexing 自动公共
   子表达式消除；Nsight 应看到 indexing/materialization kernel 数下降。
2. 单个 pass 理论上从 30 次物化降为 1 次；一个常规 chunk 从 150 次降为 5 次，另加
   output head 每 pass 原有的一次索引。最终必须按 kernel 名和调用栈实测，不能把该上限
   当成结果。
3. 新旧路径生成的 modulation tensor 在 value、dtype、device、shape、stride 上一致，
   所以数值目标是 bitwise exact，而不是 tolerance parity。
4. 节省的是大块 gather/index 带宽和 launch；DiT 主体仍由 GEMM、attention、AdaLN、
   FFN 和 SP collective 主导，因此端到端收益可能小于 1%，甚至落在噪声内。
5. whole-DiT compile 可能把重复索引融合或移动；compiled lane 若收益显著小于 eager，
   需要用 graph/kernel 证据解释。

## 调用链与数据布局

调用链：

```text
MinWMCausalDMDDenoisingStage
  -> 4 × _forward_causal_transformer(DMD timestep)
  -> 1 × _update_causal_context_cache(clean timestep)
  -> MinWMCausalTransformer3DModel.forward
  -> 30 × MinWMCausalTransformerBlock.forward
```

准备函数的输入合同：

- `hidden_states`: `[B, S_local, D]`；必须是 patch/action/SP shard 后的本 rank token。
- `timestep_proj`: `[B, F, 6, D]`；dtype 原样保留，不做 cast。
- `frame_index`: `[S_local]`、`torch.int64`，与前两者在同一 device。
- SP1 使用按 frame 均匀 repeat 的缓存索引；SP2/SP4 使用
  `forward_batch.sequence_shard_frame_indices`，允许 shard 边界落在一帧内部，因而索引
  可以非均匀。
- 输出为 `[B, S_local, 6, D]`。仍使用与旧路径完全相同的 advanced indexing；没有
  新增 `.contiguous()`。各 modulation slice 继续保留 token 维 `6 * D` stride。

该输出是一次 pass 内唯一的大物化。它从第一层活到最后一层，但没有跨 pass、chunk、
session 或 cache growth 保存，也没有第二份同尺寸副本。旧实现中同尺寸 tensor 在每层
block 的大部分生命周期内也存在，因此预计峰值显存不增加；此项仍以 H200
`max_memory_allocated`/系统采样实测为准。

## 开关、默认值与精度合同

- 默认：`MINWM_HOIST_TIMESTEP_MODULATION=1`，使用 pass 内复用路径。
- parity/回滚：`MINWM_HOIST_TIMESTEP_MODULATION=0`，恢复每个 block 独立执行旧物化。
- 两条路径均使用同一 BF16 权重、算术和 stride 边界；验收要求 bitwise exact。
- 开关在模块加载时读取，服务启动后不支持热切换。A/B 必须使用两个独立、其余环境完全
  相同的服务进程。

## 数值与回归验证

最低必要 CPU 单测覆盖：

- 旧的逐 block advanced indexing 与新的 pass-local 结果逐元素、stride 完全相同；
- 非均匀 SP frame index 和帧内 shard 边界；
- 1-frame clean-cache、4-frame DMD 和重复 recompute 各自重新物化，不跨 pass 缓存；
- batch/width/device/index dtype/shape 约束；
- 现有 cache growth、SP metadata 和 output-head 回归。

本地实际结果（2026-08-07，macOS）：

- Codex bundled Python 3.12：三个变更 Python 文件 `compileall` 通过；该环境不含
  torch/pytest，只承担语法检查。
- Homebrew Python 3.11 + 已有 CPU torch/pytest，临时环境只补 `uvicorn`：
  新增 CUDA-only 用例前 `test_minwm_realtime.py` 为 `123 passed`；新增后目标集合为
  `8 passed, 1 skipped`，skip 原因是本机无 CUDA。
- `ruff format --check` 与 `ruff check`：通过。
- `test_lingbot_causal_denoising.py`：`25 passed, 2 failed`；两项失败均是未构造
  `stage.transformer` 的既有 cache-config fixture，失败文件不在本 diff，和新增的
  causal-Wan no-op hook 无调用关系。H200 镜像仍会重跑相关集合。

本机 CPU 结果不覆盖 CUDA advanced-indexing kernel、BF16 segment compile、SP
collective、lossless frame bitwise 或峰值显存；这些结论只接受 H200 容器结果。

H200 镜像已运行 CUDA-only smoke：BF16、非均匀 SP frame index 下，pass 外准备结果与
`torch.compile(fullgraph=True)` 编译的旧 advanced indexing 逐 bit、stride 相同，结果为
`1 passed`。正式 SP2/SP4 workload 按 MinWM 现有约束关闭 whole-DiT compile；该 smoke
只证明编译消费者兼容性，不作为 headline 性能数据，也不能据此声称 compile 已消除重复。

H200 数值验收将同时保存旧/新 lossless latent/frame 校验；若出现任何差异，默认不放宽
阈值，先定位第一个不同的 block 输入和 modulation stride，并回退默认开关。

## 统一测量契约

测量工具来自 S0 canonical commit
`b9240233b2438829cbd72ee3dfbc1d37ed675560`（PR #19）：

- schema：`benchmark/minwm_realtime_parity/measurement_schema.json`
- profiler-off：`benchmark_realtime_throughput.py`
- validate/CV/Nsight merge：`measurement_tool.py`
- Nsight SQL 指标：`nsys_metrics.py`

该 pin 包含 `59aa68a382` 的完整 stage-trace 等待，并修复 59aa runner 重复实现
`latency_summary` 时漏写 `value.count` 的问题。schema 现在要求所有可用 wall/CUDA
latency 显式带 count，自定义 validator 强制 count 等于
`workload.measured_chunks`（profiler-off 为 200，profiler-on 为 10）。临时 S1 runner
还会独立递归断言所有 `ms_per_chunk` 指标的 count。S0 未合并前，只在临时测量分支
叠加该工具；S1 最终对 main 的实现 diff 不复制 S0 基础设施。

本地对 b924 工具边界的回归为：S0 `test_measurement.py + test_common.py` 19 passed；
S1 额外 count 断言 3 passed，合计 22 passed。

S0 后续审计层 `2f15c29471` 不改变 b924 schema，但规定失败/中止时逐文件记录原路径、
保留路径、size、SHA256 和 recoverability，并把旧结果移入 attempt 内的 `invalid/`；聚合器
排除该目录。`-03` 已按 b924 创建后收到此规则，依 S0 指示不热切换：成功结果仍有效；
若失败则按 2f15 规则后处理且不删除 PVC，新的 retry 才 pin 2f15。

固定 workload：MinWM 5B step-3200、1248×704、BF16、16 pixel frames/4 latent
frames per chunk、4 DMD + 1 clean-cache，20 warmup + 200 measured。SP2 是主验收，
SP4 复验。profiler-off 与 Nsight 分开运行；Nsight 先外部 warmup 20 chunks，capture
丢弃 1 个 session 首 chunk后保留至少 10 个 steady chunks，不同时启用
`torch.profiler`。

吞吐 A/B 显式固定 `realtime_causal_kv_cache_num_frames=45`。这是 rolling-window
steady-state contract：避免 220 chunks 的 full-history 序列/显存增长使小算子 A/B
失稳或令 SP2 OOM。旧/新路径使用同一 45 帧窗口；首块、短程 append/recompute、
variable chunk 与无淘汰 cache growth 的语义由独立数值测试覆盖。

## 实际 A/B

以下表格只填写 S0 schema 校验通过的同机 paired run。`待测` 不是零，也不代表不可得。

采集 provenance：kube context `codex-minwm-test-phx2`，AWS region `us-west-2`，NodePool
`minwm-test-phx2-p5e-spot`，实例 `p5e.48xlarge` / NVIDIA H200；整机隔离申请 8 GPU，
JSON 中 active GPU 严格记 SP2=2、SP4=4，`allocated_count=8`。镜像 digest 为
`sha256:bedc07ea3ba55059a8c1c569c3b177c4d00d41f37d4fa9105375531534ef5f2a`，MinWM commit 为
`2efc6485f65e8fcab506665efde79bc41406385e`，checkpoint step 3200。

### Profiler-off headline

| SP | 路径 | repeat | Client FPS | Scheduler FPS | chunk wall | DiT wall | VAE wall | 峰值显存 | CV/结论 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 旧，每层物化 | 1/2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 新，pass 内复用 | 1/2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 旧，每层物化 | 1/2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 新，pass 内复用 | 1/2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

### Nsight steady-state

| SP | 路径 | DiT CUDA | VAE CUDA | kernel 数 | launch/API 数 | 短 kernel 分桶 | GPU busy | SM Active | Tensor Active | DRAM | indexing 归因 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 旧 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 新 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 旧/新复验 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

每次 run 还必须记录：SGLang commit、MinWM commit、镜像、GPU 型号/数量、SP、精度、
fast-lane、UTC 时间和产物路径。Nsight overhead 下的 FPS 不作为 headline。

## 与预期不符处

待实测后填写。重点检查：

- 若 indexing kernel 没有减少：确认 whole-DiT compile 是否已经融合，或 kernel 名是否被
  Inductor/Triton 改写；用调用次数和 bytes 辅助归因。
- 若 kernel 减少但 DiT wall 不降：检查 saved launch 是否被 GEMM/collective 隐藏，以及
  pass-local tensor 长生命周期是否增加 allocator 压力。
- 若 Client/Scheduler 不随 DiT 变化：检查 VAE/output/transport 串行瓶颈。
- 若 SP2 有收益而 SP4 没有：按 local sequence size、all-to-all 和 gather bytes 解释，
  不能把 SP degree 差异归因给同一个索引 kernel。

## 证据与决策过程

目前的静态证据：

1. block 内唯一重复点位于 `runtime/models/dits/minwm.py` 的
   `MinWMCausalTransformerBlock.forward`；output head 是独立的一次索引。
2. 4 次 DMD loop 和 clean-cache commit 位于
   `runtime/pipelines_core/stages/causal_denoising.py`。
3. SP frame metadata 在 MinWM transformer patch/shard 后生成，准备函数在该 metadata
   已确定后执行。
4. 新缓存是 Python 局部变量，没有写入 model、forward batch 或 realtime session。
5. 首次集群 Job `minwm-s1-temb-ab-h200-20260807-01` 在 legacy server 启动前收到 S0
   stage-trace 竞态修复通知；检查确认尚无任何 client JSON 后，只删除了该精确命名 Job，
   保留 PVC 并升级到 canonical `59aa68a382`。重启 Job 为
   `minwm-s1-temb-ab-h200-20260807-02`，临时 runner checkout
   `1cb8d5221e4a4cf91e1aead5517df2f29272310b`。
6. `-02` 已开始 legacy SP2 repeat1 后，真机暴露 59aa JSON 不含显式 latency count；根任务
   主动终止 Job。PVC 核查确认没有任何正式 JSON，因此该 attempt 标记为
   `measurement-contract-invalid`，不能进入 A/B。legacy bitwise correctness 与
   `pod-exit-diagnostic.txt` 保留。
7. 收到“保留 invalid 诊断文件”的纠偏前，我依据更早的“不得保留正式 profiler 结果”
   指令删除了中断的 telemetry、空 client log 和 server log；PVC 文件删除无法撤销。这是
   执行偏差，不能用重建内容掩盖。可核验证据为：telemetry 58,686 B，SHA256
   `ac79a4c4e15f981075a8864bf42dcadd9e6b451716305b613c94b22344866ea1`；client log 0 B，
   SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`；server log
   1,688,485 B，SHA256 `52f1fd75f9b1e371a512dc13bd11d1295b66feb7c1a4e383441367c73ed7104d`。
   invalid marker、原路径/大小/hash 与其余 attempt 证据继续保留，最终聚合不扫描该
   attempt。
8. 修正后的 Job 为 `minwm-s1-temb-ab-h200-20260807-03`，S1 runner commit
   `5f92d276c08086db638f05536a46fa5434ecb169`，S0 tool pin 为 `b9240233b2`；
   `backoffLimit=0`，每个 Pod 结果仍写入 `/results/attempts/${HOSTNAME}`。只有 b924
   validator 与 S1 独立 count 断言均通过的结果才可进入表格。

最终保留或回滚规则：bitwise 不通过则回滚；profiler-off DiT/Client 回退超过 paired
噪声或默认 1% 且无法解释则回滚；若 headline 落在噪声内但能稳定消除预期 launch、
不增峰值显存且代码复杂度低，可以保留并明确“launch 优化、端到端中性”。若 compile 已
完全吞掉重复且 eager 也无可解释 kernel 下降，则删除实现而不以理论估算充当收益。

## 尝试后放弃的方案

1. **跨 DMD step/session 缓存**：timestep、variable chunk、SP shard 和 clean-cache shape
   都可能变化；cache key 和失效逻辑会放大状态风险，放弃。
2. **把 modulation 存到 model attribute**：延长生命周期，并对并发/re-entrant forward
   不安全，放弃。
3. **分别 gather 六个 `[B,S,D]` tensor**：改变旧路径先 gather
   `[B,S,6,D]` 再 select 所形成的 stride，无法先验保证 bitwise，放弃。
4. **用 broadcast/as_strided 免物化**：uniform frame 可特殊化，但非均匀 SP frame index
   不是单一 affine stride；还会改变 compiled AdaLN 输入布局，放弃。
5. **连 output head 一起复用**：output head 使用不同的 `temb` 语义和 shape，不能与
   block modulation 合并，放弃。

## 风险、回滚与复现

主要风险是大 modulation tensor 从单层生命周期延长到完整 pass。其大小为
`B × S_local × 6 × D × element_size`；SP2/SP4 会按 local sequence 缩小。它不跨
pass 保存，但可能影响 allocator reuse。必须比较峰值显存和 OOM 行为。

回滚无需改权重或 cache：以 `MINWM_HOIST_TIMESTEP_MODULATION=0` 重启服务即可恢复旧
路径；代码级回滚只涉及 causal Wan 的 no-op hook、MinWM override 和对应测试。

本地目标测试：

```bash
PYTHONPATH=python TORCHDYNAMO_DISABLE=1 python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py \
  -k 'timestep_modulation or sequence_shard_frame_indices or cache_plan'
```

真机命令以 S0 `measurement_tool.py` 生成/校验的 JSON 为准；旧路径服务进程设置
`MINWM_HOIST_TIMESTEP_MODULATION=0`，新路径设置为 `1`，其他参数和容器保持一致。

## 给负责人掌握代码的检查题

1. **为什么理论上是 150 次，而不是 30 次？**
   参考：每 chunk 有 4 次 DMD transformer forward 加 1 次 clean-cache forward，每次
   都跑 30 层。
2. **为什么 output head 的索引没有一起删除？**
   参考：它消费 output `temb`，不是 block 的 `[B,F,6,D]` modulation，而且本来每
   pass 只有一次。
3. **SP shard 落在帧中间时怎样选择 timestep？**
   参考：`forward_batch.sequence_shard_frame_indices` 为每个 local token 给出 frame；看
   `_minwm_frame_indices`。
4. **新 tensor 的 shape、dtype、device 和 stride 合同是什么？**
   参考：`[B,S_local,6,D]`，保留 `timestep_proj` dtype，同 device，advanced indexing
   后 slice 的 token stride 仍为 `6*D`。
5. **为什么不能跨 4 次 DMD 与 clean-cache 共用同一 tensor？**
   参考：timestep 值和可能的 frame/sequence shape 不同；clean-cache 通常是 clean
   timestep，跨 pass 会复用陈旧条件。
6. **cache growth/recompute 为什么不受影响？**
   参考：modulation 是 forward 局部量；KV attention plan 仍在每次 forward 前独立
   prepare，recompute 只复用自己的 cache plan，不复用 timestep storage。
7. **看到 kernel 数下降但 Client FPS 不变时应怎样归因？**
   参考：先看 DiT CUDA/wall，再看 VAE/output/transport、collective 和 GPU busy；不能
   用理论 145 次 saved indexing 直接宣称端到端收益。
8. **怎样一键恢复旧路径，恢复后数值门槛是什么？**
   参考：设置 `MINWM_HOIST_TIMESTEP_MODULATION=0` 并重启；两条路径目标都是 bitwise
   exact，不能因性能优化放宽 parity 阈值。
