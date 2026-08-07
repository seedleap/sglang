# MinWM QKV projection 与 peer-first A2A fast lane

日期：2026-08-07

任务：S4 / 点子 6

状态：**6a 已完成实现、本地 CPU 语义回归、H200 v09 完整质量矩阵、v11 独立 server
反序 ABBA，以及 v12/v13 的 d5b25227d4 exact-window Nsight control/candidate 对照。v09
四角证明 whole-model compile 的既有质量 blocker 对 control/QKV 完全相同，QKV 没有额外
劣化；compile-off 的 SP1/SP2/SP4 输出、layer probe 和 replay 均 bitwise。v11 headline
显示 SP2 Client FPS −0.771%、DiT wall −0.108%，SP4 Client FPS +5.099%、DiT wall
+4.977%；SP4 chunk wall CV 超 3%，保留为噪声风险。v12/v13 证明 projection GEMM 为
450→150/rank/chunk，SP4 projection CUDA −8.83%，但整个 graph 的 kernel/launch 总数因
V layout materialization 抵消而不变。6b 已完成 go/no-go 评估并决定不实现。**

## 结论与开关

6a 是显式 fast lane，不是 bitwise parity 修复。设置
`MINWM_FUSED_QKV_PROJECTION=1` 后，每个 MinWM self-attention block 用一个
`to_qkv` 线性层取代 `to_q`、`to_k`、`to_v` 三个线性层。权重只在模型构造/加载时
稳定打包；forward 只执行一次 linear 后用 view/chunk 切分，绝不在 forward 中拼权重。

默认值是 `0`。回滚只需取消该环境变量或设为 `0`，无需转换 checkpoint。量化权重、
未知 linear 子类和不安全的 column-parallel gather 布局会告警并自动走原三 GEMM 路径。

当前推荐边界是：SP4 可按需显式打开；SP2 headline 有 −0.771% 回退，不建议打开；任何
其他 shape/设备/量化路径在没有同口径数据前保持默认关闭。该结论不把 profiler-on 的
SP2 正向样本覆盖到 profiler-off headline。

单 GEMM 可能因为 BF16 GEMM shape、reduction/bucket 或 cuBLASLt 算法选择变化而产生
数值差异。因此在完成 layer probe、latent、最终视频质量和确定性 A/B 前，本开关不会
作为默认路径。现阶段也不把 H200 尚未执行的项目写成已通过。

## 范围和非目标

本改动只触碰 self-attention QKV projection 及其到既有 peer-first pack 的布局边界：

```text
norm hidden states
  -> [6a] one to_qkv GEMM
  -> q/k/v views
  -> existing Q/K norm and RoPE/cache logic
  -> V contiguous only for uniform SP>1 peer-first fast path
  -> existing peer-first Triton pack
  -> existing input A2A / attention / output A2A
```

以下不在 6a 中修改：RMSNorm、RoPE、KV cache 的数值/所有权语义，attention backend，
A2A collective，cross-attention QKV 和 FFN。6b 的 GEMM epilogue、定制输出布局或支持
strided V 的 pack kernel 必须等 6a profile 后再决定。

## 假设与预期

MinWM 5B 的 hidden size 是 3072。原路径的 self-attention projection 是三个
`M x 3072 @ 3072 x 3072` GEMM；6a 变为一个
`M x 3072 @ 3072 x 9216` GEMM。1248x704 经 VAE 得到 78x44 latent，再经
`patch_size=[1,2,2]` 得到每 latent frame `39x22=858` 个 transformer token。真机
SP1 layer probe 也直接观测到 QKV 输入 `[1,858,3072]`。sequence shard 后每 rank 的
projection M 是：

| SP | 每 rank 的 nominal M | 原输出 N | 6a 输出 N |
| ---: | ---: | ---: | ---: |
| 1 | 858 | 3 次 3072 | 1 次 9216 |
| 2 | 429 | 3 次 3072 | 1 次 9216 |
| 4 | rank 0/1 为 215，rank 2/3 为 214 | 3 次 3072 | 1 次 9216 |

每个 chunk 有 30 blocks ×（4 DMD + 1 clean-cache）= 150 组 self QKV。SP2 的 429/429
是 uniform peer-first Triton pack 路径。纸面切分的 SP4 是 215/215/214/214；正式 trace
实际在四个 active rank 都观测到现有 peer-first Triton pack，说明运行时 padding/buffer
合同保持了 uniform fast path，不能再把 SP4 结果误记为 varlen fallback。忽略启动、
图捕获和后端融合时，projection GEMM kernel 的理论计数从 450 降到 150。预期收益来自：

- 每 chunk 少 300 次 GEMM launch；
- SP2/SP4 较短 M 下，用更宽的 N 聚合工作，可能提高 Tensor Core 利用率；
- 权重总元素数和数学 FLOPs 基本不变，所以如果原 GEMM 已经足够大且 Tensor Active
  已饱和，收益可能很小；
- A2A 数量和 payload 不变，所以通信主导时 headline FPS 不会按 kernel 数同比提升。

一个已知抵消项是 V：packed GEMM 的 Q/K/V 是最后一维的三个 view。Q/K norm 会产生
连续输出，但 V 仍是 strided view。SP2/SP4 的既有 Triton peer-first pack 要求三者连续，
所以 6a 在该边界显式做一次 `value.contiguous()`。源码上是一次 materialization；正式
d5 trace 中它被编译为每次调用两个布局 kernel。它是 6b 要用数据判断是否值得消除的主要
候选，不应隐藏在“单 GEMM”收益里。

## 实现与兼容性

实现位于
`python/sglang/multimodal_gen/runtime/models/dits/minwm.py`：

- `MinWMCausalTransformerBlock.__init__` 在开关打开且布局安全时构造一份物理
  `ReplicatedLinear(3072, 9216, output_sizes=[3072] * 3)`；安全的非 gather
  column-parallel 路径使用 `MergedColumnParallelLinear`。
- 创建 `to_qkv` 后立即删除三个旧 module，因此不会同时常驻两份权重。
- `_project_qkv` 的 fast 分支只有一次 `self.to_qkv(hidden_states)` 和
  `qkv.chunk(3, dim=-1)`；权重 merge 只发生在 loader/pre-hook。
- 原生 checkpoint 的 `blocks.N.self_attn.{q,k,v}.*` 先沿用现有映射到
  `blocks.N.to_{q,k,v}.*`，再按固定 q/k/v 顺序在 load 时合到
  `blocks.N.to_qkv.*`。
- 普通 `load_state_dict` 的 block pre-hook 支持 split state dict -> fused model，也支持
  fused state dict -> fallback model。
- component/FSDP loader 的 `preprocess_loaded_state_dict` 支持把保存的 fused state dict
  拆回 fallback keys。设备和 dtype 移动由注册的单一 parameter 自然继承。
- parity dump 打开时，fused hook 仍按原文件名导出 Q/K/V 输出和 Q 权重，便于做同层对照。

state_dict 的 key 合同是“跨开关可加载”，不是“开关两边 key 文本相同”：

| 模式 | self-attention key |
| --- | --- |
| 默认/fallback | `blocks.N.to_q.*`、`to_k.*`、`to_v.*` |
| 6a fast lane | `blocks.N.to_qkv.*` |

所有非空 `quant_config` 目前都保留原格式感知的三个量化 module。这是有意的安全 fallback：
独立 Q/K/V 可能各自带 scale、zero point 或 packed metadata，未经对应格式的真实 checkpoint
验证，不把三个量化 parameter 强行拼成一份。H200 上会至少验证 BF16 和 static FP8；
H200 不支持的 NVFP4/设备组合只验“明确拒绝或安全 fallback”，不伪装成已运行的快路。

当前 MinWM causal Ulysses 本来就拒绝 TP>1 与 SP>1 组合，也拒绝 SP>1 + whole-DiT
`torch.compile`。本 PR 不扩大这些既有并行边界：SP2/SP4 在 TP1 验证，TP2 在 SP1 做
兼容 smoke；SP1 运行 compile off/on，SP2/SP4 的 compile-on 继续验证为明确拒绝。

## 本地验证边界

本地结果只证明 CPU 语义和静态正确性，不代表 CUDA、Triton、BF16 或真实 compile：

| 环境/检查 | 结果 | 边界 |
| --- | --- | --- |
| Codex Python 3.12.13 `compileall` + AST | 通过 | 无 torch/pytest |
| ruff format/check、`git diff --check` | 通过 | 静态检查 |
| Python 3.11.13、torch 2.13.0、pytest 9.1.1 | 12 passed，1 skipped | CPU 语义；CUDA compile 用例按设计跳过 |

macOS 本地 torch 在导入仓库的 eager `torch.compile` decorator 时会触发其自带
Inductor/Triton typing 错误。CPU 回归使用 `/tmp/codex-minwm-s4-cpu-site` 中的临时
`sitecustomize` 将 `torch.compile` 替换为 identity，并只在该临时目录补了 `uvicorn`；
仓库环境没有改变。真实 `torch.compile` 用例必须在 H200 镜像重新运行。

本地测试覆盖：真实 block 只有一个物理 QKV parameter、forward projection 无 cat、
多维/多 sequence shape、SP1/SP2/SP4 peer-first 线布局、原生与内部 checkpoint key、
跨开关严格 load、保存后反向加载、dtype move、量化 fallback。GPU 侧还需覆盖 BF16
layer probe、Triton kernel 是否命中、不同 head/sequence shape 和真实 compile。

第一次 H200 attempt `minwm-s4-qkv-h200-20260807-01` 没有进入任何测试函数、模型 client
或 A/B。镜像中 editable wheel 没有暴露源码树的 `sglang.test.ci`，导致 registered test 在
pytest collection 时 `ModuleNotFoundError`。这条结果只证明 runner 启动环境缺口，不是
CUDA/QKV 实现失败。后续所有 H200 attempt 在模型转换和 client 前显式设置：

```bash
export PYTHONPATH="/workspace/sglang/python${PYTHONPATH:+:${PYTHONPATH}}"
python3 -c 'import sglang.test.ci.ci_register'
```

预检失败必须立即止损，不能把 registered/unit test 的收集失败记成实现或质量结果。

## H200 验收矩阵

统一 workload：MinWM 5B step-3200，1248x704，BF16，4 DMD + 1 clean-cache，
16 frames/chunk。headline 必须是 profiler-off 的 20 warmup + 200 measured；Nsight 在
外部 20 chunks precondition 后丢 capture session 首 chunk，保留至少 10 个 steady chunks，
且不与 torch.profiler 同跑。

所有 control/candidate 的正式稳态 run 固定
`MINWM_S0_KV_CACHE_NUM_FRAMES=45`（client 等价参数
`--kv-cache-num-frames 45`），不得随 `max_chunks` 扩张。这是 rolling-window
steady-state contract。首块、短程 append/recompute、cache growth 和尚未发生淘汰的
数值行为另跑短程质量检查，不与 20+200 headline 混合归因。

| 项目 | control | candidate | 状态 |
| --- | --- | --- | --- |
| SP2 主验收 | `MINWM_FUSED_QKV_PROJECTION=0` | `=1` | v09 129 帧/replay bitwise；v11 ABBA Client −0.771%、DiT wall −0.108% |
| SP4 复验 | `=0` | `=1` | v09 129 帧 bitwise；v11 ABBA Client +5.099%、DiT wall +4.977% |
| SP1 eager / compile | `=0` | `=1` | v09 四角：两条同模式 QKV 边 bitwise；既有 whole-model compile blocker |
| TP2 + SP1 smoke | `=0` | `=1` | v09 两侧均被同一既有 S3 RMSNorm TP 路径阻断 |
| static FP8 | 原量化三 projection | 请求 6a 后安全 fallback | v09 明确 fallback；129 帧完成且与 BF16 SP1 control bitwise |
| NVFP4/不支持设备 | 原设备合同 | 同样拒绝或 fallback | generic non-null quant config 单测证明不开 fast lane；H200 static FP8 真机 fallback 通过，不伪造未支持 NVFP4 fast path |
| layer probe | Q/K/V、norm 后、block output | 同输入/权重/seed | v09 69 个 probe 文件；质量汇总通过 |
| latent/最终视频 | lossless latent 与 frame metrics | 同 case/seed/backend | v09 SP1/SP2/SP4 最终 129 帧全部 bitwise |
| 确定性 | candidate 重复运行 | candidate 重复运行 | v09 SP1、SP2 candidate replay 均 bitwise |

质量先遵循现有 contract：parity lane 要求 bitwise；本 BF16 fast lane 至少满足
`max_abs <= 8`、`RMSE <= 1.0`、`SSIM >= 0.995`。是否接受即便门槛内但有可见时序漂移，
仍需结合 latent、最终视频和 deterministic replay 审阅，不只看 FPS。

## S0 测量契约与复现

H200 临时测量分支只允许临时引入 S0 工具，且 profiler-off 与后续 Nsight 分别固定：

- S0 branch：`origin/codex/minwm-fused-ops-s0`
- profiler-off canonical commit：`b9240233b2438829cbd72ee3dfbc1d37ed675560`（包含
  `59aa68a382`、`25cc42ef8c`、`e75e9e24b5` 与 `411d9b9ec4`）
- exact-window / GPU-target Nsight canonical commit：
  `d5b25227d4487d113e62c86a0fb572a62d6bcc5b`
- draft PR：#19

旧的 `30cb16708f` / `8e06ab2fc3` / `411d9b9ec4` / `e75e9e24b5` /
`25cc42ef8c` / `59aa68a382` 不再作为新 clean runner 的最终 pin。S0 未合并前，临时
profiler-off 分支从 `b9240233b2438829cbd72ee3dfbc1d37ed675560` checkout 后叠加 S4
实现 commit；独立 Nsight 分支必须改从
`d5b25227d4487d113e62c86a0fb572a62d6bcc5b` 叠加。S4 PR 对 main 的最终 diff 必须
移除全部 S0 基础设施。
入口使用 `benchmark/minwm_realtime_parity/run_s0_measurement.sh`，结果再经同一 commit 的
`measurement_tool.py` validate/merge-nsys/aggregate。

`59aa68a382` 修复了 client 在最后 payload/stats 到达后过早退出、遗漏最后一条 DiT/VAE
stage trace 的竞态。正式 client 必须传
`--require-complete-stage-trace`，按 `expected_indices = 0..N-1` 等待全部合法 index；超时
诊断必须列出各 selector 的 missing/unexpected 以及 stats/payload 缺失。但该版本的实际
latency summary JSON 漏写了 `value.count`，所以其 client 结果不能作为最终 A/B。
`b9240233b2438829cbd72ee3dfbc1d37ed675560` 保持 schema v1，补齐所有 wall/CUDA latency
的显式 count；机器 schema 要求 count 存在，自定义 validator 还要求每个 available latency
的 `value.count == workload.measured_chunks`。旧工具产生的 raw `.sqlite` 可以重新 merge，
但任何 `59aa68a382` client 结果均无效；正式 profiler-off 的 DiT/VAE wall 必须都是
`status=available,count=200`。所有最终 JSON 必须通过
`b9240233b2438829cbd72ee3dfbc1d37ed675560` 的 validator，并记录实际 checkout SHA。

2026-08-07 真机审计进一步确认：b924 的 Nsight capture 虽含 1 个 discarded chunk + 10 个
stable chunks，`merge-nsys` 却把 raw 全表按 10 归一化，不能证明 kernel/API/launch 和 GPU
metric 只覆盖 exact stable window。因此 b924 **只用于 profiler-off**；S4 v03 runner 已物理
移除 `nsys launch`/`nsys start`，只跑 SP2/SP4 的 20+200。任何 b924 profiler-on 结果均不得
作为正式证据。S0 当前 exact-window canonical 为
`d5b25227d4487d113e62c86a0fb572a62d6bcc5b`；它包含 API start-time half-open 归属、
API/launch 独立边界证据、all-target active mapping 与流式 GPU metrics。旧的
`900b5f279b65b2afcfbe6cc9b36cfa4496b41bc3` 不再用于新正式 Nsight。
Nsight 必须以新 Job/attempt 单独执行。采集时 target 全部 8 张 GPU，汇总时只纳入 active
`pwGpuId`；exact 10 ranges、target coverage、DiT/VAE CUDA、kernel/API/launch、SM/Tensor
必须全部通过，任一缺失都使该 lane invalid。S0 -08 已失败；S4 不复用其任何产物，也未
趁 profiler-off Job 运行时混入 Nsight。正式 Nsight 仍以独立 Job/attempt 排队。

每个 JSON 必须记录实际 SGLang SHA、minWM SHA、镜像、GPU、SP、精度和 UTC 时间。
`provenance.gpu.count` 是 active GPU（SP2=2、SP4=4），整机隔离的 8 卡写入
`allocated_count=8`。Nsight kernel/短 kernel 保留 raw total、per-device、
per-stable-chunk；CUDA API/launch 的精确字段是 `raw_total`、`total_per_chunk`、
`per_rank_per_chunk`。只有 SQLite 能证明覆盖全部 rank 时，最后一项才 available；否则
写 unavailable 和 evidence。

S0 runner 的 `MINWM_S0_KV_CACHE_NUM_FRAMES` 必须显式记录为 45；control 和 candidate
必须完全相同。若结果使用了增长到完整 200-chunk horizon 的 cache，该结果不具备本任务
headline 资格。

profiler-off server 不打开 layerwise NVTX。单独的 Nsight server 打开
`--enable-layerwise-nvtx-marker`，但 compile graph 没有在 SQLite 留下可用的
`to_q/to_k/to_v` 或 `to_qkv` module range。projection 因此按固定 shape/name、每 rank
150 组调用和同 stream 时间序列归因：control 是连续三个短 GEMM，candidate 是一个三倍 N
GEMM；CUDA 是 kernel duration 和，wall 是首个 projection kernel start 到最后一个 end。
peer-first Triton kernel 按 `_fused_pack_peer_first_qkv_kernel` 归因；A2A 使用 NCCL 自带的
`ncclAlltoAll` NVTX range，而不是用全部 `ncclDevKernel_SendRecv` 冒充。NVTX/SQL 只用于
profile 证据，不进入 headline 路径。

## 实际 A/B

### 质量与兼容性（v03–v05，root-invalid 诊断）

| 检查 | 当前结果 | 资格边界 |
| --- | --- | --- |
| S0 + QKV CUDA/compile unit + Ulysses pack | 36 passed，21 warnings | H200、torch 2.11.0+cu130；无 skip/fail |
| SP1 control vs candidate 最终帧 | 129 帧 bitwise；max_abs=0、RMSE=0、SSIM=1 | eager、固定 seed/backend |
| SP1 candidate vs replay | 129 帧 bitwise；max_abs=0、RMSE=0、SSIM=1 | eager 确定性 |
| SP1 首次 layer probe | 输入/权重/Q/K/V/QK norm/block/output 全部 bitwise | shape `[1,858,3072]`；runner naming repair applied |
| SP1 whole-model compile | 129 帧完成；8 chunk scheduler wall 约 253/198/299/418/553/689/739/886 秒，总计约 67.2 分钟 | 动态 cache/sequence shape 逐块冷编译，不是 headline；v03 后续失败，不能作验收 lane |
| TP2 candidate | QKV 两 rank 均加载 fast lane，首个请求失败 | 未改动的 `MinWMRMSNorm.variance_epsilon` TP 路径；v03 因此 root-invalid |
| v04 TP2 control/candidate | 两边均在同一 RMSNorm 属性精确失败 | 证明 blocker 不由单 GEMM 引入；S4 不改 S3 |
| v04 FP8 fallback | 明确回退三 projection；129 帧完成 | 日志含 quantized fallback 与 parity-fallback mode |
| v04 SP2/SP4 | control/candidate 均 129 帧 bitwise；SP2 replay 也完成 | 后续 compile gate 比较错误使整个 v04 root-invalid，只作诊断 |
| v05 QKV eager vs compile | 实际 contract/request 相等；33 帧；max_abs=192、RMSE=3.09572、SSIM=0.98447169 | 未过 fast-lane 门槛；在 SP/FP8/profiler-off 前止损，需四角归因 |

这里的结果只是在 root-invalid attempt 中保留的可核查诊断，不替代 v08 全量门槛，也不
外推为 SP2/SP4 profiler-off 已通过。v04 的 whole-model compile 缩成 2 chunks：覆盖初始
与增长后的两种 cache/sequence shape，scheduler wall 约 253.0/204.3 秒。但脚本错误地把
该 2-chunk compile 输出与 8-chunk eager 的前 33 帧比较；请求包含 `minwm_total_chunks`，
输入合同不同，得到 max_abs=255/RMSE=16.15/SSIM=0.839 不能归因到 QKV 或 compile。
v05 改为同一 2-chunk case 的 candidate eager-reference 与 candidate compile，生成前比较
seed、total_chunks、KV、prompt/image、SP/TP/precision、feature flag 等请求计划，且只允许
compile 开关不同；生成后实际 client contract/request 完全相同，并记录 33 帧逐值与逐帧
SHA。它正确发现质量门失败，却不能区分 whole-model compile 既有漂移与 QKV 额外漂移，
因此后续 attempt 改跑四角：C eager↔C compile、QKV eager↔QKV compile、C eager↔QKV eager、
C compile↔QKV compile。四边都记录 max_abs/RMSE/SSIM、整体/逐帧 SHA 和每帧首差异；
preflight 对每条边只允许 `compile_enabled` 或 feature flag 这一项预期差异。只有 control
compile 同量级失败、两种 compile 彼此通过 fast-lane 门且 eager QKV bitwise 时，才记为
既有 whole-model compile blocker并继续 compile-off 验收；否则止损并回退 compile 模式。

### v09 正式质量门

v08 在 36 项 H200 测试通过后、任何 client 启动前，因为 runner 没有显式创建 SP1 结果
目录而 `FileNotFoundError`；它按全局质量前置失败写 root-invalid，不能复用。v09 的产品
修复范围只有在 preflight 前 `mkdir -p "${SP1_RESULTS}"`，并新增静态顺序测试，机器证明
结果目录创建发生在第一次写入之前。v09 启动后先复核 v01–v08 八个 marker，才运行四角
和后续矩阵。

四角使用完全相同的 2-chunk 请求合同；每条边分别执行 preflight、actual metadata、逐帧
SHA 与质量判定：

| 边 | max_abs / RMSE / SSIM | 逐值结果 | 判定 |
| --- | --- | --- | --- |
| C eager ↔ C compile | 192 / 3.0957156 / 0.98447169 | 首差异 frame 1 | 未过 fast-lane 门 |
| QKV eager ↔ QKV compile | 192 / 3.0957156 / 0.98447169 | 首差异 frame 1 | 未过 fast-lane 门 |
| C eager ↔ QKV eager | 0 / 0 / 1 | 33 帧及整体 SHA 相同 | bitwise |
| C compile ↔ QKV compile | 0 / 0 / 1 | 33 帧及整体 SHA 相同 | bitwise |

两条 eager→compile 边的 max_abs、RMSE 与 `1-SSIM` 比值都严格为 1.0；eager 两份 NPY
SHA 均为 `149d0e4b20cdb9df0efdb94d97799d3083e2f3c2d58380306d39e3ad007aa991`，compile
两份均为 `331dd00a4caa528855e3a44b78abf73f5684cee7db2e15b6461dd9c17e054278`。因此分类为
`existing_whole_model_compile_blocker`：whole-model compile 本身违反当前质量合同，但 QKV
在 eager 与 compile 各自模式内都没有额外偏差，允许继续 **compile-off** S4 验收；这不把
whole-model compile 标成支持。

四角原始 client/scheduler 与 stage trace 中的两块 timing 如下，单位 ms；斜线前后为
chunk 0/chunk 1：

| lane | client payload | scheduler forward | DiT wall / CUDA | VAE wall / CUDA |
| --- | --- | --- | --- | --- |
| C eager | 13684.076 / 1379.273 | 13483 / 1417 | 5484.647 / 571.663；5484.718 / 578.541 | 6774.090 / 368.044；7109.587 / 709.479 |
| QKV eager | 4050.219 / 1372.703 | 3861 / 1410 | 2503.164 / 564.965；2508.604 / 583.796 | 469.133 / 369.207；804.550 / 712.141 |
| C compile | 254522.551 / 200120.421 | 254333 / 200161 | 252664.834 / 199319.900；252664.328 / 199319.453 | 480.057 / 369.048；815.306 / 704.825 |
| QKV compile | 197247.964 / 189185.063 | 197054 / 189230 | 195555.036 / 188342.171；195554.609 / 188341.734 | 478.604 / 370.758；815.959 / 706.693 |

这里没有伪造“纯 compiler timer”：compile-inclusive 时间定义为同 feature 下 compile lane
减 eager reference，包含 graph compile/recompile、autotune 和 compiled execution。按 client
payload，control 两块分别为 240838.475 / 198741.148 ms，QKV 为
193197.745 / 187812.360 ms；相应 DiT wall 为 247180.187 / 198748.237 ms 与
193051.872 / 187777.206 ms。这些冷编译诊断不用于 headline，也不用于声称 QKV 编译更快。
四角 summary SHA256 为
`ad87d602fc1a568fb879fb255f870d2fcff7c0977f79febb57a3bfe00e90a2c3`；派生 timing JSON
逐一记录不可变原始日志的 path/size/SHA，未改写张量或日志。

通过四角后，v09 完成 H200 36 项 CUDA/source gate；SP1/SP2/SP4 control↔candidate 的
129 帧全部 bitwise，SP1/SP2 candidate replay 也 bitwise，质量汇总含 69 个 layer probe
文件。static FP8 请求明确日志为量化安全 fallback 并走三 GEMM，129 帧完成。TP2 control
和 candidate 均精确命中同一个未改动 S3 `MinWMRMSNorm.variance_epsilon` blocker；两侧
行为一致后 runner 才继续，不把 TP2 写成 QKV 支持。

### v09 profiler-off 同序探索结果

质量门通过后，v09 才按 SP2 control→candidate、SP4 control→candidate 的固定顺序运行；
每个 lane 启动一个新 server，并在该 server 上连续跑两次 20 warmup + 200 measured。八个
JSON 全部通过 b924 schema/validator，Scheduler/DiT/VAE wall 的 `count` 都是 200；四个
lane 的 S0 必选 Client FPS、Scheduler FPS、DiT wall、VAE wall CV 都通过 3% 门。下表
CV 使用两样本 sample standard deviation / mean，收益正数表示 candidate 更快：

| SP / metric | control mean（CV） | candidate mean（CV） | candidate 性能变化 |
| --- | ---: | ---: | ---: |
| SP2 Client FPS | 12.5317（1.383%） | 12.5706（0.589%） | +0.310% |
| SP2 Scheduler FPS | 12.5420（1.380%） | 12.5812（0.587%） | +0.312% |
| SP2 chunk wall ms | 1328.085（3.267%） | 1332.350（1.464%） | −0.321% |
| SP2 DiT wall ms | 754.772（0.142%） | 755.957（0.185%） | −0.157% |
| SP2 VAE wall ms | 424.815（0.119%） | 424.382（0.035%） | +0.102% |
| SP4 Client FPS | 14.5614（0.200%） | 15.2947（0.287%） | +5.036% |
| SP4 Scheduler FPS | 14.5761（0.223%） | 15.3099（0.299%） | +5.034% |
| SP4 chunk wall ms | 1187.098（0.458%） | 1111.658（2.732%） | +6.355% |
| SP4 DiT wall ms | 759.239（0.379%） | 712.209（0.303%） | +6.194% |
| SP4 VAE wall ms | 233.158（0.242%） | 233.512（0.016%） | −0.152% |

SP2 control 的 chunk wall CV 3.267% 超过 3%，虽然 S0 必选集合使用 Scheduler FPS 而不是
chunk-total wall，因此 aggregate 仍通过；这里仍把它作为客户端 payload/write 或 session
顺序噪声证据，不隐藏。profiler-off schema 不提供 DiT/VAE CUDA、kernel/launch、SM 或
Tensor Active；这些字段在本表是 **unavailable，绝不是 0**，必须由后续 d5 exact-window
Nsight 补齐。

稳态窗口内的 1 Hz GPU telemetry 没显示 SP4 candidate 享受更高时钟：control 两次 active
GPU 平均 SM clock 为 1975.1/1974.3 MHz，candidate 为 1973.8/1973.7 MHz；平均 GPU util
从 71.54%/71.02% 升到 73.29%/73.39%，平均每卡功耗从 429.6/432.5 W 升到
442.9/447.9 W，温度从 58.54/60.78°C 升到 59.80/62.13°C。candidate 更晚、更热且时钟
略低仍更快，支持“做了更多有效 GPU 工作”的解释，但不能替代 kernel profile。SP2 的
时钟、功耗与利用率差异很小，和接近 0 的 wall/FPS 变化一致。

不过固定顺序仍有明确混杂：两次 repeat 复用同一 server；每个 server 的第一次
init→first-payload 是约 6–10 秒，第二次约 1.5–1.7 秒。whole-model compile 虽关闭，日志
仍显示 `segment_compile=True`，四个 server 共享同一个容器内
`/root/.cache/sgl_diffusion/torch_compile_cache`，且日志没有可核验的 hit/miss 事件。因此
v09 **只标 exploratory，不作为 PR headline**。独立反序 ABBA 的设计是
candidate→control→control→candidate；每个位置都完整 stop、校验端口/进程退出、再启动
新 server，并在位置前后只读记录 inductor/triton cache 的 file count、size 与 metadata
listing SHA。v10 因证据 PVC 双挂载失败，v11 去掉旧 PVC 后完整执行，结果如下。

### v10 root-invalid 与 v11 profiler-off headline

v10 新 Job/PVC 在 setup 前约 1 秒退出：runner 同时挂载 v09 旧 PVC 到 `/prior-results`，
节点 SELinux label 使 root 也无法遍历 `/prior-results/attempts`。这不是模型、QKV 或 client
失败。v10 root-invalid marker 大小 483 bytes、SHA256
`2729d505ea6cc86fd1a18f78cff0292712541aabae4cd9be59bb6e96a4e941db`；唯一诊断文件
大小 4631 bytes、SHA256
`4b00dc4a4d3fe650ae5a38bcce15bf89988cefeec25f97b13c23c214779baa10`。v09/v10 PVC、
marker 和诊断均原位保留。

v11 使用新 Job/PVC，静态反例和 live spec 都证明没有 `/prior-results` volume/mount。
启动日志打印由外部单-PVC只读 reader 已核验的 15 条 immutable path/size/SHA（含 v09
八个 profiler-off JSON、质量/四角/summary/complete 与 v10 marker/diagnostic），随后 H200
source/CUDA gate 为 39 passed、21 warnings。SP2/SP4 每个位置都重启 server，八个 JSON
全部通过 b924 validator，所有 available wall count 都等于 200；本地只读复制后 56 个
核心文件与 PVC 逐文件 SHA manifest 完全一致。complete SHA256 为
`c246644fa10c2974d08ee76f9173758bfb057570f7769061fd9b27d337636962`，ABBA summary
SHA256 为 `225e4ca40b70db42af7bafe04408eca49e106292ebae5773f5c0db56061e150f`。

| SP / metric | control mean（CV） | candidate mean（CV） | candidate 性能变化 |
| --- | ---: | ---: | ---: |
| SP2 Client FPS | 12.6707（0.008%） | 12.5730（0.177%） | −0.771% |
| SP2 Scheduler FPS | 12.6818（0.013%） | 12.5842（0.176%） | −0.769% |
| SP2 chunk wall ms | 1290.753（0.455%） | 1329.663（1.671%） | −3.015% |
| SP2 DiT wall ms | 756.051（0.085%） | 756.865（0.290%） | −0.108% |
| SP2 VAE wall ms | 423.862（0.038%） | 423.762（0.126%） | +0.024% |
| SP4 Client FPS | 14.6640（2.191%） | 15.4117（2.929%） | +5.099% |
| SP4 Scheduler FPS | 14.6767（2.202%） | 15.4287（2.942%） | +5.124% |
| SP4 chunk wall ms | 1147.753（3.721%） | 1091.733（4.379%） | +4.881% |
| SP4 DiT wall ms | 750.840（0.188%） | 713.468（0.873%） | +4.977% |
| SP4 VAE wall ms | 233.346（0.172%） | 233.462（0.042%） | −0.050% |

SP2 的 control 双点几乎不漂移，candidate 仍稳定回退约 0.77%；这与 DiT stage 仅
0.108% 的差异不完全一致，提示 scheduler 中未被 stage wall 覆盖的 device/launch/通信
开销需要 Nsight 归因。SP4 两侧均随位置变慢，Client/Scheduler CV 尚低于 3%，但
chunk-wall CV 超过 3%，因此 headline 保留噪声警告。不过 ABBA 的两端 candidate 对中间
control 仍给出约 5% 同向收益，且 DiT wall 同向 +4.977%，不是单纯 client write 偏差。

cache snapshot 显示 SP2 position 1 前为空，结束后稳定为 inductor 59 files / triton
130 files；SP4 position 1 后扩展为 127 / 154，之后三个位置的 count、size、listing SHA
完全不变。首个 candidate 承担每个 SP 的 cache 建立却仍是 SP4 最快位置，cache 不能解释
SP4 收益；时间漂移仍需在结论中保留。

按 client steady window 切片的 1 Hz telemetry 显示 SP4 四位置平均 SM clock 都在
1974.0–1975.4 MHz。candidate position 1/4 的平均功耗为 453.5/439.1 W，control
position 2/3 为 437.2/426.9 W；温度约 59–60°C。没有 candidate 时钟优势，功耗升高与
DiT 更快一致。SP2 candidate/control 时钟都约 1938–1939 MHz；candidate 平均 GPU util/
功耗略低（约 87.0–88.3% / 626.5–627.9 W）于 control（88.6–89.0% /
630.4–631.9 W），与没有 device-stage 收益的方向一致。

### Provenance

| 字段 | control | candidate |
| --- | --- | --- |
| SGLang 临时测量 commit | `8ba6408133618ad4ebf8eba04a51803e060ca66b` | 同左 |
| minWM commit | `2efc6485f65e8fcab506665efde79bc41406385e` | 同左 |
| container image | `…/leap-world/minwm-training@sha256:bedc07ea…53ef5f2a` | 同左 |
| GPU active / allocated | SP2: 2/8；SP4: 4/8，NVIDIA H200 | 同左 |
| SP / precision / UTC | SP2、SP4 / BF16 / 2026-08-07 | 同左 |

### Profiler-off headline（20 + 200）

| SP | lane | Client FPS | Scheduler FPS | chunk wall | DiT wall | VAE wall | CV | 状态 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 3 GEMM control | 12.6707 | 12.6818 | 1290.753 ms | 756.051 ms | 423.862 ms | 必选项 ≤0.085% | v11 headline |
| 2 | 1 GEMM candidate | 12.5730 | 12.5842 | 1329.663 ms | 756.865 ms | 423.762 ms | 必选项 ≤0.290% | v11 headline，Client −0.771% |
| 4 | 3 GEMM control | 14.6640 | 14.6767 | 1147.753 ms | 750.840 ms | 233.346 ms | 必选项 ≤2.202%；chunk 3.721% | v11 headline，噪声说明必需 |
| 4 | 1 GEMM candidate | 15.4117 | 15.4287 | 1091.733 ms | 713.468 ms | 233.462 ms | 必选项 ≤2.942%；chunk 4.379% | v11 headline，Client +5.099% |

### Nsight steady state（20 precondition + 1 discard + >=10）

v12 control 与 v13 candidate 都固定临时 runner
`ad22c2556e3f07fb2862b7e0235c9dd9d6d65839`，其 ancestor 为 canonical
`d5b25227d4487d113e62c86a0fb572a62d6bcc5b`。两侧均是 exact stable indices 1..10、discard
index 0；DiT/VAE wall/CUDA count 都是 10。每次采集 8 个 target，SP2 只聚合 active
`pwGpuId=[2,3]`，SP4 只聚合 `[0,1,2,3]`；target mapping、kernel/API/launch boundary、
SM/Tensor/DRAM coverage 均通过 validator，未出现 degradation。

下表的 kernel/launch/API 是 **每 rank、每 stable chunk**；short buckets 是所有 active
device 每 chunk 的 `<10 / 10–50 / 50–100 / >=100 us`。profiler-on 的 wall/FPS 含 Nsight
扰动，只作归因，headline 仍是上面的 v11 profiler-off ABBA。

| SP/lane | Client / Scheduler FPS | chunk wall | DiT wall / CUDA | VAE wall / CUDA | kernel = launch | CUDA API | short buckets | GPU busy | SM / Tensor / DRAM Active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| SP2 control | 12.6111 / 12.6193 | 1299.1 ms | 740.252 / 739.788 ms | 443.591 / 442.967 ms | 17304 | 45495.4 | 18217.2 / 12283.3 / 1188.7 / 2918.8 | 76.844% | 61.788% / 28.352% / 8.302% |
| SP2 candidate | 12.9491 / 12.9576 | 1278.0 ms | 713.950 / 713.533 ms | 443.211 / 442.617 ms | 17304 | 45364.0 | 18553.5 / 11661.2 / 1216.0 / 3177.3 | 77.905% | 63.565% / 29.205% / 8.590% |
| SP4 control | 13.9853 / 13.9860 | 1213.5 ms | 789.431 / 788.943 ms | 254.816 / 254.195 ms | 17300.5 | 46134.225 | 49085.2 / 13415.7 / 1895.5 / 4805.6 | 67.447% | 37.310% / 15.722% / 4.341% |
| SP4 candidate | 14.4198 / 14.4300 | 1141.4 ms | 744.296 / 743.812 ms | 255.124 / 254.479 ms | 17300.5 | 45840.8 | 50144.7 / 11759.5 / 2497.2 / 4800.6 | 66.511% | 38.344% / 16.244% / 4.477% |

projection/layout/communication 继续按每 rank、每 chunk 归一化。V layout 是 candidate
相对 control 多出的、紧跟 Q/K norm 的两个 materialization kernels；A2A wall 是 300 个
`ncclAlltoAll` NVTX range duration 的和，包含 input/output A2A，不是所有 SendRecv device
kernel duration。

| SP/lane | projection GEMM kernels | projection CUDA / trace wall | V layout kernels / CUDA / wall | peer-pack kernels / CUDA | A2A calls / NVTX wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| SP2 control | 450 | 19.303 / 33.317 ms | 0 / 0 / 0 | 150 / 3.221 ms | 300 / 11.849 ms |
| SP2 candidate | 150 | 19.305 / 19.305 ms | 300 / 2.802 / 3.541 ms | 150 / 3.099 ms | 300 / 11.765 ms |
| SP4 control | 450 | 11.363 / 30.063 ms | 0 / 0 / 0 | 150 / 1.660 ms | 300 / 12.469 ms |
| SP4 candidate | 150 | 10.359 / 10.359 ms | 300 / 1.713 / 2.675 ms | 150 / 1.435 ms | 300 / 12.255 ms |

6a 的实际 device 解释是：SP2 projection CUDA 几乎不变（+0.012%），但三次独立 GEMM
之间的 idle/launch gap 被消除，trace wall −42.06%；SP4 更短的 M 让三次小 GEMM gap 更
显著，且 3N GEMM 采用更合适的 `192x176` tile，projection CUDA −8.83%、trace wall
−65.54%。相应地，Tensor Active 在 SP2/SP4 分别提高 3.01%/3.32%（相对值），SM Active
提高 2.88%/2.77%。这解释了 SP4 profiler-off/归因两边都受益；SP2 profiler-on 的正向样本
与 v11 headline 回退相反，说明其收益接近运行顺序/系统噪声，不能据此推荐默认打开。

projection 少掉的每次调用两个 GEMM launch，被 V materialization 的两个 kernel 抵消，
所以 whole-graph kernel/launch 总数完全不变。candidate 的 V layout + peer-pack 合计仅占
DiT wall 约 0.93%（SP2）和 0.55%（SP4）；A2A 数量不变，NVTX wall 仅 −0.71%/−1.72%。
这组绝对量是 6b 的 no-go 证据。

## 与预期不符处

本地环境偏差：macOS torch 2.13.0 的仓库导入会在 compile decorator
初始化时失败，因此本地 compile 被明确跳过并转移到 H200；这不是实现失败。

H200 v01 暴露了另一个 runner 偏差：仓库源码 checkout 存在
`python/sglang/test/ci/ci_register.py`，但安装后的 editable package 没把它放进有效导入
路径。pytest 在收集 `test_ulysses_qkv_pack.py` 时退出 2。该 attempt 使用
`backoffLimit=0`，全局前置失败 marker 保留 setup/test 日志、文件大小与 SHA256；没有
profiler-off/on JSON，不能纳入 A/B。v02 增加 source-tree import machine-check，并使用新
Job、Pod 和 attempt 目录，但把 import 放在依赖安装前，因镜像当时还没有 `orjson` 而
exit 1；它已成功验证 v01 marker，同样没有进入模型 client。v03 把 import gate 移到 setup
依赖安装后、pytest/质量/client 前，并机器验证 v01/v02 marker 后才继续。

v03 质量脚本又暴露一个纯 runner 命名偏差：模型一直输出
`self_q_norm_000.pt` / `self_k_norm_000.pt`，临时脚本末尾却检查旧式
`self_norm_q_output_000.pt` / `self_norm_k_output_000.pt`。发现时前三个 SP1 lane 已完成且
whole-model compile 正在运行，若不处理会在全部昂贵 lane 结束后误报“缺 probe”。本
attempt 标记为 **runner naming repair applied**：原始 tensor 不改写、不移动，只建立四个
相对 symlink；唯一 audit JSON 记录 runner commit、发现/创建/复核时间、期望与原始路径、
原始 size/SHA256、link target 和可恢复性，并复核 symlink 解析后的 device/inode 与内容
SHA 均等于原文件。更严格的“audit 必须先于 link”指令在初次链接后到达；audit 如实保留
该事件顺序及初版记录 SHA，不把时间倒写。后续 runner commit `602894a60c` 已直接改用
真实名字，并加静态测试；当前 v03 仍明确记录其启动的 runner 是 `fe4e87df36`。

v03 随后在 TP2+SP1 full-model smoke 的首个请求失败：两个 rank 都已成功加载 fast lane，
但未改动的 tensor-parallel RMSNorm 分支访问不存在的
`MinWMRMSNorm.variance_epsilon`。这属于 S3/既有 TP 路径，本 PR 不越界修改。v03 因质量
全局前置失败写 root marker，SP1 结果降为诊断，正式 profiler-off 没有启动。v04 用同一
短程 case 分别跑 TP2 control 与 candidate，只有两边都出现该精确既有错误才把它记录为
外部 blocker 并继续 S4 的 SP1/SP2/SP4/FP8 门；若任一边行为不同则立即失败。

v04 的 TP2 control/candidate 确认同错后继续完成 FP8、SP4 与 SP2，但最终 compile 比较使用
不同 `total_chunks` 的请求，属于 runner contract 错误。它以 root marker 保留，不能因为
其他 lane 已观察到 bitwise 就拆出来当正式验收。v05 将 compile plan preflight、实际
contract/request postflight、逐帧 SHA 与 frame metric 都写入结果，并把该门前移；参数一致
但 QKV eager/compile 质量超阈值，故在 TP/SP/FP8 前止损。四角 runner 再把门放在所有
完整 SP1、TP2、FP8、SP2/SP4 之前；最终 quality summary 同时包含 preflight plan、实际
client metadata 与四边证据。

数值方面比保守预期更好：尽管单 GEMM 允许 BF16 bucket 变化，v09 的 eager/compile
同模式 control↔candidate、SP1/SP2/SP4 和 replay 都是 bitwise。性能则分叉：SP4 的
M=214/215 小 GEMM 符合“加宽 N 提高 Tensor Core 占用”的预期，ABBA DiT/端到端都约
+5%；SP2 M=429 没有收益，Client/Scheduler 反而约 −0.77%，与“少 300 launch 必然更快”
的简化预期不符。d5 Nsight 给出的解释是：

- projection 内部严格是 450→150 GEMM/rank/chunk，但 V layout 物化又增加 300 个 kernel，
  所以全图 kernel/launch 总数不降；不能把 module 调用数直接当全图 launch 收益；
- SP2 三个 GEMM 的 CUDA 总和与 fused GEMM 几乎相同，但首末 kernel trace span 缩短
  42.06%；这个细粒度改善未稳定穿过 profiler-off 的顺序/系统噪声；
- SP4 的 fused shape 额外带来 projection CUDA −8.83%，Tensor Active 相对 +3.32%，并且
  projection span −65.54%，与 profiler-off 的约 +5% DiT/端到端方向一致；
- V layout + pack 只占 candidate DiT wall 的 0.93%/0.55%，A2A 数量不变且 wall 基本不变，
  它们不是 SP2 回退或 SP4 收益的主导项；
- 允许 BF16 bucket 改变的风险没有在本 workload 兑现：layer、latent、最终视频和 replay
  都 bitwise；这只是已测 shape 的结果，不外推到未测设备/量化格式。

## 证据与决策过程

1. 先选择“构造/加载时只有一份 packed parameter”，避免 forward cat 和双份权重常驻。
2. 保留环境变量默认关闭，因为 GEMM shape 改变本身就是数值契约变化。
3. 所有量化格式先 fallback；等格式逐一证明 packed scale/metadata 合同后才可能放行。
4. SP fast path 先显式 materialize V，保证继续命中已有 peer-first Triton pack；这让 6a
   的收益/代价可单独归因。
5. 只有 6a 后 pack + V copy 仍占显著 steady kernel/wall，并且可维护实现有正收益时，
   才进入 6b。若收益落入噪声、A2A 主导或实现需要侵入 GEMM backend，6b 结论就是“不做”。
6. v01 证明“源码 checkout 存在”不等于 registered tests 可导入；v02 将显式 PYTHONPATH 和
   `import sglang.test.ci.ci_register` 引入为门，但放在依赖安装前又暴露 `orjson` 缺失；v03
   将它放在 setup 后、pytest/质量/client 前，既验证源码路径又不误判尚未安装的运行依赖。
7. b924 Nsight 的 raw 表含 discarded chunk 却按 stable=10 归一化，无法做 exact-window
   归因；保留 profiler-off 口径，物理移除当前 runner 的 Nsight 路径；正式 Nsight 改 pin
   `d5b25227d4` 并另起 Job。
8. 真机 probe 纠正了纸面 shape 假设：projection 的 SP1 M 是 patch 后的 858，而不是
   patch 前 latent 位置 3432；SP2 为 429，SP4 纸面为 215/215/214/214。正式 trace 又进一步
   证明 SP4 四 rank 都命中现有 peer-first Triton pack，因此以实际 kernel coverage 为准，
   不再把它预判成 varlen fallback。
9. v10 证明“只读旧 PVC + 新结果 PVC”也会因节点 SELinux label 破坏 setup；v11 完全去掉
   旧 PVC，只把外部核验的 path/size/SHA 作为不可变常量打印，既保留证据链又不双挂载。
10. v11 ABBA 证明 v09 的 SP4 +5% 方向可复现，同时推翻 v09 的 SP2 小幅正收益；因此
    profiler-off headline 改用 v11，Nsight 必须同时解释 SP2 回退和 SP4 收益，不能只挑
    有利的 SP4。
11. v12/v13 均通过 d5 exact-window、all-target active mapping 和严格 validator；projection
    的同 stream 序列把 control 三短 GEMM 与 candidate 单 3N GEMM 一一对应到 150 次
    block×forward/rank/chunk，不依赖模糊的全局 kernel name 猜测。
12. NCCL 的 `ncclAlltoAll` NVTX range 给出 300 calls/rank/chunk；使用这个高层 range 而非
    全部 SendRecv device kernel，避免把 unrelated send/recv 或 overlap 时间计入 A2A wall。
13. V materialization 和 peer-pack 的绝对 wall 占比低于 1%，因此 6b 在写任何定制 epilogue
    前就被数据门否决；保留现有清晰接口比追求不足 1% 的理论上限更重要。

## 尝试后放弃或暂缓的方案

- **forward 中 `torch.cat([Wq, Wk, Wv])`**：每次重复分配/复制权重，直接违反目标，未采用。
- **保留三个 module，再在 post-load 复制一份 fused weight**：会常驻双份约 3×3072²
  参数，并给 FSDP/device move/save 制造双源真相，未采用。
- **直接融合所有量化 QKV**：独立 scale/packed metadata 未验证，改为安全 fallback。
- **在 6a 同时写 GEMM epilogue/定制布局**：无法区分 GEMM 聚合与布局优化收益；6b 数据门
  最终不通过，因此未实现。
- **为 profile 默认加入 NVTX**：会污染 profiler-off headline，改成只在独立 Nsight 运行打开。

## 6b go/no-go

6b 最终结论为 **NO-GO / NO IMPLEMENTATION**。进入实现原本要求同时满足：

1. 6a 已通过兼容性和质量门槛；
2. SP2 主验收中 V copy + peer-first pack 仍占可重复的显著 DiT CUDA 或 wall；
3. profiler 证明瓶颈不是 A2A wait 或其他串行阶段；
4. 候选方案能保持 fallback 和清晰接口，不要求维护私有 GEMM backend fork；
5. 6b 独立 A/B 在 profiler-off 也有收益，而非只在 Nsight 下好看。

实际测量没有满足第 2、3、5 条：candidate 的 V layout + peer-pack 合计仅占 DiT wall
0.93%（SP2）/0.55%（SP4）；peer-pack 自身没有回退，A2A 仍是固定 300 calls/rank/chunk，
wall 只变化 −0.71%/−1.72%。即使理想 epilogue 消除全部 V layout，理论上限也只有约
0.50%/0.36% DiT wall，低于 headline 噪声并需要侵入 GEMM 输出布局。故不写 6b、无需
独立 profiler-off A/B；表中的 6a control/candidate layout 证据就是条件门的 A/B 记录。

## 审计、失败产物与资源止损

H200 Job 必须设置 `backoffLimit: 0`，并把结果写入按 Pod/attempt 隔离的目录；controller
不得自动重跑后覆盖或混合 provenance。runner 拒绝复用已存在的结果目录，也不得删除或
覆盖旧 `.nsys-rep`、`.sqlite`、JSON、日志或质量产物。

任何失败、旧契约或 partial attempt 都必须物理保留。marker 至少记录失败原因、UTC
时间、每个已有文件的相对路径、大小、SHA256 和可恢复性；旧数据可原地保留，或在不丢
provenance 的前提下移动到同一 attempt 的 `invalid/`，不能删除后伪装成 clean run。

marker 的作用域必须与失败范围一致：setup、质量、parity 或 source-tree import 等全局
前置失败写 attempt-root marker；某个 profiler-off lane 失败，写该 lane 的
`invalid-marker*.json`；profiler-off 已完整验证后，后续 Nsight 失败只在
`profiler-on/` 写 marker，不得作废已合格 headline，也不得影响 sibling SP/lane。聚合器从
当前 JSON 的 parent 向最近 measurement root 检查 marker；路径本身位于 `invalid/` 时直接
排除，sibling marker 不传播。Job EXIT trap 若已发现 runner 的 scoped marker，只写 marker
路径和 SHA256 引用，不再补 attempt-root marker；只有 runner 尚未初始化、没有 scoped
marker 时才把失败提升为全局。

v01 是 pytest collection 全局前置失败，root marker 合法；PVC
`minwm-s4-qkv-h200-results-20260807` 继续保留。两个最初的只读 evidence-reader 因 EBS
卷的 Pod 安全上下文无法穿过 `/results/attempts`，也没有修改证据。v02 以正常读写 mount
验证 v01 marker 后自身在全局 import 前置失败；v03 已同时验证 v01/v02 marker 的 reason、
recoverability、artifact count 和 marker SHA256，再写自己的独立目录。两份 marker SHA256
分别为 `d3edfa8b8fb9afabb238c61b8d3b613f8f1cefbd42885f0c7fb8564a2e57c91f` 和
`267bc5c51d9271ea7e436a2f7a99f12dd3399147b4a50b3e6780b7bc37df54ae`。

v03 的 root-invalid marker SHA256 为
`a15b1e94c3c577cd58bc1b5c0f121a2a28a220e341247d0ae60871b8eb65afa5`，包含 266
项产物。v04 的 root-invalid marker SHA256 为
`6954b1d87ddfbbe15e1c9fdf74e26ae829b2f37638e63ae04a9e832d510f702b`，包含 314
项产物。v05 的 root-invalid marker SHA256 为
`977c8135f4b5c6c3dba684809a39cf0359327c9c6942a539d0f3edee9292114d`，包含 268
项产物；v06 启动前已机器复核 v01–v05 全部 marker。v03 的 probe naming repair
audit 位于 quality attempt 根，audit SHA256 为
`4096c6411ac97ccf63d6910227bea99acfd5c6da72c6608447b6cee1e40517b2`。四个 link
逐一验证 resolved inode 与原文件相同、resolved SHA256 与原文件相同；这项 runner 修复
不改变任何张量值，也不把链接当作新的独立测量样本。

v06 模板误把实现提交写成短 SHA `6a704fcace`，而启动脚本要求 checkout 后的完整 SHA
与配置逐字相等。发现时只完成旧 marker 核验和源码 clone，尚未进入依赖、模型或 client；
因此精确删除自己的 v06 Job 控制对象止损，PVC attempt 与 TERM/EXIT invalid marker 保留，
不复用。v06 实际先因完整 SHA 比较失败而 exit 1；v07 的审计门误预期 exit 143，读取 v06
marker 后立即失败，同样没有进入依赖、模型或 client。两者 attempt 均保留。v08 使用完整
SHA `6a704fcace9f804513e1f40b0c55ad6749fd03bc`，live spec 已复核 context、region、NodePool
与 `backoffLimit=0`，并在任何模型/client 前要求 v01–v07 七个 marker 全部存在且 reason
匹配。v06/v07 只属于 runner provenance，不能混入质量或 A/B。

v08 在 H200 测试通过后由 runner 的结果目录顺序缺口失败，root-invalid marker SHA256 为
`be99ac0dfaf4c81c126f9580a69ff040c8ed32ebdfbe63f8b3a5095b59605e5c`，包含 17 项
产物；没有启动四角或正式 client。v09 使用新 Job
`minwm-s4-qkv-h200-20260807-09`、Pod/attempt
`minwm-s4-qkv-h200-20260807-09-5zsch`，live spec 机器复核 `backoffLimit=0`，并在任何
模型/client 前重新核验 v01–v08 全部 marker 的 reason、artifact count、recoverability 和
SHA256。v08 marker/PVC 保持原位，不复制、不覆盖、不删除。v09 的 kube context 为
`codex-minwm-test-phx2`，region/zone/NodePool 为
`us-west-2` / `us-west-2-phx-2a` / `minwm-test-phx2-p5e-spot`。

v10 ABBA 使用新 Job `minwm-s4-qkv-abba-h200-20260807-10`、新结果 PVC
`minwm-s4-qkv-h200-results-20260807-v10`，并以
`kubernetes.io/hostname=i-06888dc1ca88547e1` 约束到 v09 的同一现有节点；该不可供
Karpenter 新建节点的 hostname 约束同时阻止扩容。第一次创建时显式
`preemptionPolicy=Never` 被 admission 拒绝：未指定 PriorityClass 时该字段必须等于 admission
计算的 `PreemptLowerPriority`。当时没有 Pod、attempt 或 PVC 数据；为停止 controller 的
FailedCreate 重试，精确删除该 Job 控制对象后原名重建，PVC 保留。修正版不设置
PriorityClass、不改变默认优先级，与已先运行的 S3 同优先级；scheduler 现场证据为
`Insufficient nvidia.com/gpu`、`No preemption victims found`，所以 v10 自然 Pending，不抢占
S3。这个 setup 事件不冒充 workload attempt。

修正版 v10 随后成功创建 Pod，但旧 v09 PVC 与新 v10 PVC 同时挂载时触发 SELinux label
权限错误，在 clone/setup/test/server/client 前 exit 1。attempt-root marker 与唯一 diagnostic
的 SHA 分别为 `2729d505ea6cc86fd1a18f78cff0292712541aabae4cd9be59bb6e96a4e941db`
和 `4b00dc4a4d3fe650ae5a38bcce15bf89988cefeec25f97b13c23c214779baa10`；v10 Job、Pod、
PVC 和证据均保留。v11 新 PVC 完全移除旧 PVC mount，以 manifest 常量打印外部只读 reader
核验的 path/size/SHA，避免证据读取改变运行时安全上下文。

v11 Job `minwm-s4-qkv-abba-h200-20260807-11`、Pod
`minwm-s4-qkv-abba-h200-20260807-11-rctrp` 在固定节点 exit 0，restarts=0、backoffLimit=0；
八位置 JSON、cache snapshot、telemetry、summary 和 complete 全部保留在新 PVC
`minwm-s4-qkv-h200-results-20260807-v11`。审计 reader 首版因 PVC source 与 container
mount 双重 readOnly 无法做 SELinux relabel；改为仅 container mount readOnly 后可读且
`/results` 不可写，核心树远端/本地 SHA manifest 完全一致。reader Pod 已精确删除，PVC
未删除；首次中断的 39 MiB 本地 partial copy 也保留但不作证据。

正式 Nsight control v12 使用新 Job/PVC，manifest SHA256 为
`842514cefb8700b10f0c7a7f5874abbb7fbf90564940b151d3d9601ccb20532f`，runner 固定
`ad22c2556e3f07fb2862b7e0235c9dd9d6d65839`（包含 canonical d5）。client/server dry-run、
嵌入 shell、无旧 PVC、默认 priority、固定 hostname、backoff0、8GPU 均通过。Job
`minwm-s4-qkv-nsys-control-h200-20260807-12` / Pod 后缀 `49c8v` 最终 exit 0、零重启；PVC
`minwm-s4-qkv-nsys-control-results-20260807-v12` 保留。candidate v13 只在 control 成功后
创建一次，manifest SHA256 为
`b70cc201d471a19143edf9705c1ceb6dd6d1640a920f8c2c586a2c1723237f8d`；除 lane、flag=1、
Job/PVC/attempt identity 外与 control 合同相同。Job
`minwm-s4-qkv-nsys-candidate-h200-20260807-13` / Pod 后缀 `sx6tb` 同样 exit 0、零重启；PVC
`minwm-s4-qkv-nsys-candidate-results-20260807-v13` 保留。两边 H200 source/CUDA gate 都是
63 passed、21 warnings，均无 invalid marker。live 审计固定 context
`codex-minwm-test-phx2`、hostname `i-06888dc1ca88547e1`、region/zone
`us-west-2/us-west-2-phx-2a`、NodePool `minwm-test-phx2-p5e-spot`、capacity `spot`、默认
priority、`backoffLimit=0`；没有抢占或扩容。

只读审计 reader 都固定同一节点、无 GPU、单 PVC、`/results` 写 probe 被拒绝；复制小型
JSON/校验信息后只精确删除 reader Pod，未删除 Job/PVC/raw。证据摘要如下：

| lane/SP | measurement JSON SHA256 | `.nsys-rep` size / SHA256 | SQLite size / SHA256 |
| --- | --- | --- | --- |
| control SP2 | `82c9b1f4e3b73092a66aef2c50739ed367955c1560d063083f6ad5336f7d2b25` | 55,298,849 / `7e79e3b0127cc5cabae037264d16d5ba7a0cee9b272b8bab3283c1ed247bfd47` | 1,482,256,384 / `cdc5913bd49e63c672c1b4296ec79aa35b777395555a4ecd04c98bbb7f74dbfa` |
| control SP4 | `6392f8cf2bd6831d99c68a638b3e8d46354246ed4782196ac811bd64d9aac29d` | 91,968,058 / `39f5ec1fb7182ff20e9da70bd3f6ab50161b8e1583dc98e3bcaffc8622332c39` | 1,446,555,648 / `547c91e883992474c280fae94b1be467d48d96c50048149245adc74d4d85efff` |
| candidate SP2 | `82eebb4af2244afb751722eebc6532f1dc47357e1464f0dcb2423435a6d3e3b5` | 54,758,780 / `e6debf37d9d4c22fb52ca7716b0fa169bb32afe60dfd7c9ad8480e17544f340d` | 1,459,085,312 / `47a1deddbd9df3d1c09901d1ad6f76a147f68850ace9efbe3126878e999fa39f` |
| candidate SP4 | `5119b17d9b6cdedb591c8420ad5031299ac22be78cea68eae1937685c5975360` | 91,136,098 / `f2fafb3bf2afdb27d37e02df1b854e13922afffd471c8d39eb8db805078d0a24` | 1,408,634,880 / `fb2756945ef9a47c91e49600a25e0d03fcbc864a4980d8aaea3e56465a09594b` |

小型 control/candidate 审计副本分别位于本机
`/tmp/minwm-s4-v12-control-audit` 和 `/tmp/minwm-s4-v13-candidate-audit`；四个 JSON 都再次
通过本地 d5 validator。raw 的权威副本仍是对应 PVC。

若需要集群止损，只允许删除名称精确匹配本任务 `minwm-s4-qkv-*` 的 Job/Pod 控制对象；
PVC 和其中的诊断证据必须保留。所有 `kubectl` 读取、dry-run、apply、logs 和 delete 都显式
使用 `--context codex-minwm-test-phx2`，并在提交前记录 region、NodePool、zone 与
capacity type。

## 风险、回滚与验收状态

主要风险是 BF16 数值轨迹、FSDP/TP packed load、compile graph、量化 metadata 和 V copy
抵消收益。回滚路径始终是 `MINWM_FUSED_QKV_PROJECTION=0`；保存于任一开关下的 state
dict 都有反向加载路径。若 fast lane 启动日志没有出现 `single-gemm-fast-lane`，该次结果
必须视为 fallback，不得计入 candidate。

当前验收状态：

- 6a 实现/CPU 语义/静态检查：通过；
- H200 v01 pytest collection、v02 setup 前 import：均失败并判 invalid，均未运行实现；
- H200 v03 CUDA/source gate：36 passed；SP1 eager/replay/compile 完成后被既有 TP2 RMSNorm
  路径阻断，root-invalid，未执行 FP8/SP2/SP4/profiler-off；
- H200 v04：TP2 同错、FP8 fallback、SP2/SP4 诊断完成，但 compile 跨合同比较失败，
  root-invalid，未启动 profiler-off；
- H200 v05：同合同 QKV eager/compile 实际请求相等，但质量超阈值，root-invalid，未启动
  TP2/FP8/SP2/SP4/profiler-off；
- H200 v06：短 SHA 配置错误在模型前人工止损，控制对象删除、PVC/marker 保留；
- H200 v07：v06 marker reason 预期错误，在模型前失败并保留；
- H200 v08：36 项测试后因 SP1 结果目录未创建而在 client 前失败，root-invalid/PVC 保留；
- H200 v09 四角：两条同模式 QKV 边 bitwise；两条 eager→compile 边以完全相同幅度失败，
  分类为既有 whole-model compile blocker，只放行 compile-off；
- layer/latent/video/determinism：v09 SP1/SP2/SP4 与 replay 完整矩阵通过；
- profiler-off A/B：v09 同序仅为 exploratory；v10 因旧 PVC 双挂载 SELinux 权限错误
  root-invalid；v11 反序 ABBA 八个位置全部完成并成为 headline，SP2 Client −0.771% / DiT
  wall −0.108%，SP4 Client +5.099% / DiT wall +4.977%，SP4 chunk wall CV 超 3% 已披露；
- Nsight：control v12 / candidate v13 均固定 `d5b25227d4` 语义和临时 runner
  `ad22c2556e`，SP2→SP4 exact-window 全部完成；kernel/API/launch、DiT/VAE CUDA、
  SM/Tensor/DRAM 与 8-target active mapping 均 available；
- 6b：no-go，不实现；V layout + pack 的 wall 占比不足 1%，A2A 数量/时间基本不变；
- 默认开关：保持关闭。

## 给负责人掌握代码的检查题

1. **开关在哪里读，默认是什么？**

   `minwm.py` 顶部 `_MINWM_FUSED_QKV_PROJECTION`，默认 `False`。

2. **为什么说没有 forward cat？**

   `_project_qkv` fast 分支只调用 `to_qkv` 后 `chunk`；cat 只在 loader/pre-hook 合并权重。

3. **原生 `self_attn.q.weight` 如何进入 fused parameter？**

   `_minwm_fused_qkv_param_names_mapping` 先复用现有 `self_attn.q -> to_q` 链，再按
   merge index 0/1/2 合到 `to_qkv.weight`。

4. **fast state_dict 为什么能被默认模型加载？**

   普通 `load_state_dict` 由 `_minwm_qkv_load_state_dict_pre_hook` 拆分；component/FSDP
   loader 由 `preprocess_loaded_state_dict` 在 name mapping 前拆分。

5. **peer-first wire layout 是什么？**

   `_usp_pack_peer_first_qkv` 输出
   `[destination_peer, batch, local_sequence, local_heads, 3 * head_dim]`，最后一维为 Q/K/V。

6. **为什么 6a 仍有一次 V copy？**

   packed GEMM 的 V 是最后一维 strided view；Q/K norm 会物化，V 不会。SP>1 时显式
   contiguous 才能继续命中现有 Triton pack。源码是一次 materialization，d5 compile
   trace 是每次调用两个 kernel；边界在 `minwm.py` 的 `_project_qkv` 后、USP pack 前。

7. **为什么不能把 single GEMM 称作 bitwise parity？**

   输出 N 从 3072 变 9216，BF16 GEMM bucket/algorithm/reduction 可能改变；必须看 layer
   probe、latent、最终帧和 replay，而不是只比较数学公式。

8. **projection kernel 理论上每 chunk 从多少降到多少？**

   30 blocks × 5 forwards 下从 450 降到 150/rank；v12/v13 的 exact-window trace 实测就是
   450→150。全图 launch 不降，因为 V materialization 增加 300 kernels/rank/chunk。

9. **看到 FPS 没提升时先查什么？**

   查 projection CUDA/wall 是否真降、V copy/pack 是否增加、A2A wait 是否主导，以及
   SM/Tensor Active 和 SP 后 GEMM M shape；不能只用 kernel 数解释。

10. **什么条件下 6b 应直接放弃？**

    pack/V copy 不显著、收益落入 repeat CV 噪声、A2A 才是主瓶颈，或方案需要难维护的
    私有 GEMM backend 且 profiler-off 无收益时，保留证据并不实现。本次 0.93%/0.55%
    layout+pack wall 占比已触发该 no-go。
