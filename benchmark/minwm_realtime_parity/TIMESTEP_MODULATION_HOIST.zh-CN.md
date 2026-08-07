# MinWM timestep modulation pass 内复用

状态：实现、CPU 回归、H200 CUDA compile smoke、BF16 output bitwise 与正式
profiler-off A/B 已完成。SP4 采用 A1(candidate)→B1(legacy)→B2(legacy)→A2(candidate)
反向顺序 ABBA 后，正式 headline 为 Client/Scheduler 约 `+0.5%`、DiT wall 改善
`0.67%`，均低于默认 1% 噪声门槛。Nsight 必须使用 S0 exact-window canonical
`900b5f279b65b2afcfbe6cc9b36cfa4496b41bc3` 重测，最终保留/回滚尚待 kernel/launch
证据。

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

H200 正式 bitwise 结果已通过：旧/新路径输出 shape 都是
`[129, 704, 1248, 3]`，`array_equal=true`、最大绝对差为 0；两者 frames SHA256 均为
`38e7ef07cffb7e8df2e59323dcbd9dacda92d31ab4a268d1276b554b7f3e833b`。该结果覆盖正式
BF16 causal workload 的 4 DMD + 1 clean-cache/recompute 路径。若后续变更出现任何差异，
仍不放宽阈值：先定位第一个不同的 block 输入和 modulation stride，并回退默认开关。

## 统一测量契约

正式 profiler-off 结果由 S0 commit
`b9240233b2438829cbd72ee3dfbc1d37ed675560`（PR #19）生成；S0 后续更新不改变这些
结果的语义。正式 profiler-on/Nsight 必须改用 exact-window / GPU-target canonical
`900b5f279b65b2afcfbe6cc9b36cfa4496b41bc3`：

- schema：`benchmark/minwm_realtime_parity/measurement_schema.json`
- profiler-off：`benchmark_realtime_throughput.py`
- validate/CV/Nsight merge：`measurement_tool.py`
- Nsight SQL 指标：`nsys_metrics.py`

profiler-off pin 包含 `59aa68a382` 的完整 stage-trace 等待，并修复 59aa runner 重复实现
`latency_summary` 时漏写 `value.count` 的问题。schema 现在要求所有可用 wall/CUDA
latency 显式带 count，自定义 validator 强制 count 等于
`workload.measured_chunks`（profiler-off 为 200，profiler-on 为 10）。临时 S1 runner
还会独立递归断言所有 `ms_per_chunk` 指标的 count。S0 未合并前，只在临时测量分支
叠加该工具；S1 最终对 main 的实现 diff 不复制 S0 基础设施。

`900b` 的 Nsight 硬门是：capture target 覆盖全部 8 个 GPU/进程，但汇总只纳入 active
`pwGpuId`；每条正式结果必须恰好覆盖 10 个 steady ranges，并同时给出 DiT/VAE CUDA、
kernel、CUDA API/launch、短 kernel 分桶、SM Active、Tensor Active 与 target coverage。
任何 coverage 或 range-count 不完整的结果只作诊断，不进入 kernel/launch 结论。

本地对 b924 工具边界的回归为：S0 `test_measurement.py + test_common.py` 19 passed；
S1 额外 count 断言 3 passed，合计 22 passed。

S0 后续审计层 `b178572f84`（包含 `2f15c29471`）不改变 b924 schema，但规定失败/中止时
逐文件记录原路径、保留路径、size、SHA256 和 recoverability；聚合器按当前 JSON 的
parent 到最近 measurement root 检查 marker，避免污染兄弟 attempt。

后续现场审计又明确 marker scope：某 profiler-off lane 已验证后，若 Nsight 或另一 lane
失败，只在失败 lane 写 marker，不得用 attempt-root marker 作废已合格 headline；只有
setup、全局质量或 parity 前置失败才使用 root marker。ABBA runner 把
`CURRENT_LANE_DIR` 精确设为 `${label}/sp4`：position JSON 直接位于该目录，post-run
validate/count 失败只作废当前 position，已完成 sibling 不受影响。

固定 workload：MinWM 5B step-3200、1248×704、BF16、16 pixel frames/4 latent
frames per chunk、4 DMD + 1 clean-cache，20 warmup + 200 measured。SP2 是主验收，
SP4 复验。profiler-off 与 Nsight 分开运行；Nsight 先外部 warmup 20 chunks，exact-window
capture 必须得到恰好 10 个 steady ranges，不同时启用 `torch.profiler`。

吞吐 A/B 显式固定 `realtime_causal_kv_cache_num_frames=45`。这是 rolling-window
steady-state contract：避免 220 chunks 的 full-history 序列/显存增长使小算子 A/B
失稳或令 SP2 OOM。旧/新路径使用同一 45 帧窗口；首块、短程 append/recompute、
variable chunk 与无淘汰 cache growth 的语义由独立数值测试覆盖。

## 实际 A/B

以下 profiler-off 数据都通过 S0 schema、完整 stage trace 和 `count=200` 断言。正式 SP4
headline 使用同一 Pod、每个 position 重启 server 的 ABBA；旧的 legacy→candidate 同序
结果只保留为诊断，不能作为实现收益。

采集 provenance：kube context `codex-minwm-test-phx2`，AWS region `us-west-2`，NodePool
`minwm-test-phx2-p5e-spot`，实例 `p5e.48xlarge` / NVIDIA H200；整机隔离申请 8 GPU，
JSON 中 active GPU 严格记 SP2=2、SP4=4，`allocated_count=8`。镜像 digest 为
`sha256:bedc07ea3ba55059a8c1c569c3b177c4d00d41f37d4fa9105375531534ef5f2a`，MinWM commit 为
`2efc6485f65e8fcab506665efde79bc41406385e`，SGLang commit 为
`5f92d276c08086db638f05536a46fa5434ecb169`，checkpoint step 3200。

### Profiler-off headline

| SP | 路径/ABBA position | repeat | Client FPS | Scheduler FPS | chunk wall ms | DiT wall ms | VAE wall ms | active GPU 显存采样 max MiB | 必需指标 CV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 旧，每层物化（B1/B2） | 2 | 14.63745 | 14.65011 | 1159.115 | 753.6245 | 232.4127 | 41045 | 0.482% / 0.484% / 0.112% / 0.071% |
| 4 | 新，pass 内复用（A1/A2） | 2 | 14.71123 | 14.72657 | 1152.950 | 748.5860 | 232.8149 | 41049 | 1.869% / 1.860% / 0.440% / 0.061% |
| 4 | **正式 improvement** | ABBA mean | **+0.504%** | **+0.522%** | **+0.532%** | **+0.669%** | **-0.173%** | +4 MiB（采样噪声） | 必需 CV 均 ≤3%；端到端 <1% |

CV 顺序为 Client/Scheduler/DiT/VAE。candidate 的非必需 `chunk wall` CV 是 4.738%，未过
3%；legacy 为 0.164%。由于 Client/Scheduler/DiT/VAE 四个必需指标均过 CV 门，ABBA
headline 可用，但 chunk-wall 只能作为带位置漂移的辅助证据。显存列是 `nvidia-smi`
系统采样，不是 `torch.cuda.max_memory_allocated`；4 MiB 差异不能证明 allocator 精确相等，
只说明没有可见的长期大 tensor 增长或 OOM。

各 position 原始值解释顺序效应：

| position | 路径 | Client FPS | Scheduler FPS | DiT wall ms | 首 payload ms |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | 新 | 14.51685 | 14.53284 | 750.9149 | 20679.7 |
| B1 | 旧 | 14.58756 | 14.59994 | 753.0291 | 10235.1 |
| B2 | 旧 | 14.68734 | 14.70028 | 754.2200 | 10017.7 |
| A2 | 新 | 14.90560 | 14.92029 | 746.2572 | 9864.7 |

四次都记录 `enable_torch_compile=false`、`warmup_mode=off`，但 execution profile 都是
`segment_compile=True`，并打印同一个 TorchInductor/Triton disk cache 目录。A1 首 payload
约为其余位置的两倍，且 Client 从 A1 到 A2 持续上升，因此单向 legacy→candidate 会把
热态/位置收益误算给实现；ABBA 首尾对称平均才是正式口径。

SP2 主验收由 `-03` legacy 与 `-04` candidate 的同契约 profiler-off 支撑，但跨 attempt，
所以只作为“无回退”证据：Client `-0.028%`、Scheduler `-0.030%`、chunk wall improvement
`-0.073%`、DiT wall improvement `-0.111%`、VAE wall improvement `+1.668%`。Client 与 DiT
均在 1% 噪声内。旧 `-04` 同序 SP4 曾显示 Client `+2.487%`、Scheduler `+2.480%`、
chunk `+5.206%`、DiT `+3.235%`；它与 ABBA 冲突，现明确降级为顺序/热态诊断数据，
不得出现在收益 headline。

### Telemetry 与产物审计

| position | clock mean / p50 / p95 MHz | power mean W | util mean | temp max | P-state / samples |
| --- | --- | ---: | ---: | ---: | --- |
| A1 新 | 1976.16 / 1980 / 1980 | 398.86 | 68.58% | 71°C | P0 / 860 |
| B1 旧 | 1976.18 / 1980 / 1980 | 413.32 | 69.31% | 71°C | P0 / 820 |
| B2 旧 | 1976.05 / 1980 / 1980 | 416.38 | 70.37% | 71°C | P0 / 812 |
| A2 新 | 1975.95 / 1980 / 1980 | 417.46 | 70.59% | 71°C | P0 / 804 |

所有 active GPU 样本都是 P0，四个 position 的 clock p50/p95 都是 1980 MHz；启动审计中
HW thermal slowdown、HW slowdown 与 HW power-brake slowdown 均未激活。power/util 随
position 逐步上升而不是按 variant 分组，与 A1 cold start 和时间顺序一致，不能解释为
hoist 本身的耗电或加速。

成功 attempt 为 `minwm-s1-temb-abba-h200-20260807-06-p6l6b`，Job exit 0、
`backoffLimit=0`；结果根为 PVC
`/results/attempts/minwm-s1-temb-abba-h200-20260807-06-p6l6b/minwm-s1-temb-abba-20260807-06`。
没有 invalid marker、`.nsys-rep`
或 SQLite。aggregate SHA256 为
`3053998de2f461237bc0ce2425406a1c80646b3e56cab7a574dd2254648f8a16`，candidate/legacy
repeat-summary SHA256 分别为 `b992bc9b6b71631cd1b2756aff99c1d8916b2b42ff0d190d61d65dced6ccd206`
和 `9f1186724b3730e5d20c3b8f3d9b253a31b8393e6fb1ee8802eab6b39597e0b2`；PVC 保留。

### Nsight steady-state

| SP | 路径 | DiT CUDA | VAE CUDA | kernel 数 | launch/API 数 | 短 kernel 分桶 | GPU busy | SM Active | Tensor Active | DRAM | indexing 归因 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 旧 | 待 900b exact-window | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 2 | 新 | 待 900b exact-window | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 4 | 旧/新复验 | SP2 归因后决定 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

每次 run 还必须记录：SGLang commit、MinWM commit、镜像、GPU 型号/数量、SP、精度、
fast-lane、UTC 时间和产物路径。Nsight overhead 下的 FPS 不作为 headline。

## 与预期不符处

1. 静态上限暗示每 chunk 最多可少 145 次 block 内物化，但正式 SP4 ABBA 的 Client 只有
   `+0.504%`、DiT wall 只有 `+0.669%`，都低于 1%。这与“索引只是 5B DiT 中的小算子，
   主体由 GEMM/attention/collective 主导”的保守预期一致，但不足以单凭吞吐证明优化有效。
2. 旧同序 SP4 的 `+2.487%` Client / `+3.235%` DiT 与 ABBA 明显冲突。A1 首 payload
   20.68 s，而后三个位置约 9.86–10.24 s；power/util 也随时间上升，表明位置和共享
   compile cache 是显著混淆。决策是用 ABBA `+0.5%` 替换旧数字作为正式 headline。
3. candidate Client/Scheduler repeat CV 约 1.87%，高于 legacy 的约 0.48%；candidate
   chunk-wall CV 4.74% 甚至未过 3%。虽然必需指标 CV 均通过，仍不能把 A2 对 B2 的局部
   `>1%` 当作实现收益。
4. SP2 Client/DiT 近乎中性，而 SP4 ABBA 略正，不支持“local sequence 越大收益越大”的
   简单模型；可能是 saved launch 被 SP collective 和其他算子隐藏。必须等 900b Nsight
   直接数 indexing/materialization kernel 与 launch，而不是继续从 wall time 反推。
5. `nvidia-smi` active-GPU 显存最大值只差 4 MiB，没有看到长生命周期 tensor 导致的
   material regression，但未采到精确 allocator peak。因此当前结论是“系统层未见增长”，
   不是“严格零字节增加”。

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
8. `-03` 在 legacy SP2 两次 profiler-off 与 repeat-summary 完成后，按 S0 exact-window
   缺口在第一次 `nsys start` 前精确止损；capture 从未启动。已验证的 legacy SP2 保留，
   profiler-on lane 单独标 invalid，不用 attempt-root marker。
9. `-04` 是物理无 Nsight 的 off-only resume：完成 candidate SP2、legacy/candidate SP4、
   bitwise/quality 与 aggregate。SP2 基本中性；SP4 同序出现 `+2.49%` Client 后，结合
   telemetry 与 compile-cache 热态，决定不直接收尾，而设计反向 ABBA。
10. S3/S4 的另一次 H200 运行暴露源码注册测试在 `PYTHONPATH` 未包含 source tree 时会在
    collection 阶段失败。S1 runner 因而在测试/model/client 前显式加入仓库 `python/` 并
    import `sglang.test.ci.ci_register`，失败即止损。
11. 第一次 ABBA Job `minwm-s1-temb-abba-h200-20260807-05` 把上述 import 放在依赖安装前，
    16 秒内因缺 `orjson` exit 1；没有 model/client/Nsight，GPU 显存为 0。该全局 preflight
    失败在唯一 attempt 根写 marker
    `invalid-marker-s1-preflight-orjson-20260807T053630Z.json`，原地保留 4453 B
    `pod-exit-diagnostic.txt`（SHA256
    `9ad150768adca8b2c16c9546db161b91a76b597203132121871b1f4a35a44cc7`）；PVC 未删。
12. 修正后的不可变 manifest commit 是
    `2aaa3b51773f99840c3e8ed3136b7cddf6cd1898`，顺序为 setup-only 依赖安装 → source
    import → 无 Nsight/ABBA 静态检查 → registered CUDA test → server/client。新 Job `-06`
    为 `backoffLimit=0`，H200 registered test `1 passed, 123 deselected`，四个 position
    都通过 schema/count，且没有 marker/Nsight 产物。
13. ABBA 对称均值将旧同序 `+2.49%` 修正为正式 Client `+0.504%`、DiT `+0.669%`。
    这满足“profiler-off 不回退超过 1%”，但收益落在噪声内。当前决策是暂时保留实现与
    parity fallback，不转 ready；只有 900b Nsight 能证明 indexing/materialization kernel
    和 launch 有可解释下降时才保留，否则回滚，不用静态 145 次上限代替实测。

最终保留或回滚规则：bitwise 已通过；profiler-off ABBA 已证明不回退超过默认 1%。下一硬
门是 900b Nsight：若 headline 落在噪声内但能稳定消除预期 indexing/materialization
launch、不增可见峰值显存且代码复杂度低，可以保留并明确“launch 优化、端到端中性”。若
segment compile 已吞掉重复，或 eager 也无可解释 kernel/launch 下降，则删除实现，不以
理论估算充当收益。

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
profiler-off 复现使用 b924 schema；新的 Nsight runner 临时 pin 900b，必须在依赖安装后、
model/client 前完成 source-tree import 与无并发 profiler 检查，并保留 exact 10-range 和
all-target coverage 证据。产品 PR 不带任何 S0 runner/tool diff。

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
9. **为什么旧 SP4 `+2.49%` 不能写成实现收益，ABBA 如何修正？**
   参考：A1 首 payload 是 20.68 s，后三个约 10 s，且 power/util 随位置升高；看
   `temb-hoist-sp4-abba-summary.json` 的 A1/B1/B2/A2 首尾对称均值，正式 Client 只有
   `+0.504%`。
10. **post-run validate 失败时 marker 为什么放 `${label}/sp4`，Nsight 又要过哪些门？**
    参考：position JSON 的直接父目录就是 `sp4`，S0 聚合器从 JSON parent 向上查 marker，
    因此不会误伤 sibling；900b Nsight 还要求 all-8 target coverage、active `pwGpuId`、
    exact 10 ranges、DiT/VAE CUDA、kernel/API/launch 与 SM/Tensor 指标齐全。
