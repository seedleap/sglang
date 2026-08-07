# MinWM Ulysses A2A 前后融合实现记录

状态：pre-A2A 候选已因真实端到端 parity 失败从产品实现和开关中删除；只在本节文档与
benchmark 证据中保留负结论。post-A2A 候选已通过 H200 CUDA exact gate，以及 SP2/SP4
short 与 KV45 eviction 的逐帧 SHA parity；post-only profiler-off A/B 已完成。SP2 headline
近零；SP4 v06 约 +4.95%，独立 server 的逆序 ABBA 仍为 +3.56%，排除了单纯
baseline-first/cache 顺序偏差。Nsight 使用 S0 exact-window/GPU-target canonical 后另跑。

基线为 `origin/main@9a9dc59cd1`，实现分支为
`codex/ulysses-pre-post-a2a-fusion`。统一测量契约来自 S0
`b9240233b2438829cbd72ee3dfbc1d37ed675560`，v06 profiler-off 结果在该合同下有效；后续
ABBA off-only runner 使用
`900b5f279b65b2afcfbe6cc9b36cfa4496b41bc3`，正式 Nsight 必须使用最新 canonical
`d5b25227d4487d113e62c86a0fb572a62d6bcc5b`。S0 未合并前，它们只叠加到临时 H200
测量分支，不进入本 PR 对 main 的实现 diff。

## 假设与预期

本任务把两个假设独立裁决：

1. A2A 前，把 across-head Q/K RMSNorm 与 peer-first QKV pack 合并，理论上可减少 Q/K
   norm 与 pack 的 launch、读写和短 kernel，但不能改变 Q/K/V projection 的三次 GEMM。
   v05 证明 standalone 随机输入 exact 不代表真实模型调用链 exact；该假设验收失败并删除。
2. A2A 后，在不需要重排历史 cache 的 append/recompute 路径，把 Q RoPE、raw K/V
   写入和 rotated K 生成放进一个 kernel，可减少 cache copy 与 RoPE kernel。窗口淘汰
   或动态选择仍使用原路径。

端到端收益不是先验结论。A2A 是同步依赖：即使两侧 kernel 变少，如果 NCCL 或其前后
idle gap 是关键路径，DiT wall 和 Client FPS 可能不变。保留某段的门槛是数值合同通过，
且目标区域 launch/busy time 明确下降；优先目标为 DiT 至少 3%，但低于该值时仍按
critical path 证据决定是否值得保留。

## 数据布局与 ownership

| 阶段 | 每 rank 输入/输出 | token ownership | head ownership | 允许的融合 |
| --- | --- | --- | --- | --- |
| Q/K/V GEMM 后 | `[B,S_local,H_global,D]` | 本 rank 的连续 sequence shard | 全部 heads | 原 eager Q/K across-head RMSNorm；V 不变 |
| peer-first send | `[P,B,S_local,H_local,3D]` | 本 rank token，按目的 peer 排列 | 每 peer 的 head slice | 保留现有 QKV pack，不再融合 norm |
| NCCL A2A | `all_to_all_single` | 交换 token shard | 交换 head shard | 不允许 kernel 跨越该边界 |
| A2A receive | `[B,S_global,H_local,3D]` | 全局 sequence，含各 source rank 的连续片段 | 本 rank 的 local heads | 只做 post-A2A 工作 |
| causal cache | `[B,S_visible,H_local,D]` | 当前可见的 sink/pin/tail 时间顺序 | 本 rank local heads | raw K/V 写、Q/K RoPE、rotated K 写 |

Q/K RMSNorm 的 reduction 维度是 `H_global*D=3072`。A2A 把 heads 切成
`H_local` 后已不再持有完整 reduction domain，因此 norm 必须在通信前完成。cache 在
sequence 已聚合、heads 已分片后创建，`cache_head_start=0`，所以 RoPE、cache append 和
rotated K 写只能发生在通信后。

## 开关、默认值与 fallback

| 开关 | 默认值 | 支持条件 | fallback |
| --- | --- | --- | --- |
| `MINWM_FUSED_POST_A2A_ROPE_CACHE` | `false` | CUDA、BF16/FP16、local-head contiguous cache、`MINWM_CACHE_ROTATED_K=true`、无需历史重排的 append/recompute | 原 cache planner/apply、Q RoPE、K RoPE 和 cache copy |

产品代码只保留 post 开关，可做 `00/01` 消融；默认关闭，直到 H200 性能与稳定性证据
齐全。旧 `MINWM_FUSED_PRE_A2A_QK_NORM` 不再被产品代码读取，避免误开失败路径。post
位于 A2A 后，只要 cache plan 不要求淘汰重排就不依赖 source shard 是否均匀。错误
prepared metadata 仍抛出原有异常，不以 fallback 隐藏状态错误。

## cache 状态语义

- 首块/普通 append：planner 决定 `write_start` 与 visible window；fast path写 raw K/V，
  并重建整个 visible rotated K，随后才提交 position/token/cursor metadata。
- 同一 active chunk 的四次 DMD 与一次 clean-cache recompute：raw K/V 始终覆盖 active
  range；已有 rotated history 有效时只重算 current range。clean-cache 最终写因此不会被
  跳过。
- growth：planner允许完整历史且逻辑 window 大于物理 buffer 时，先沿用现有
  `_grow_to_fit`，再准备新 buffer 的 fused write。
- eviction、sink/dynamic pin 或其他 `preserves_all_history=false` 路径：不进入 post
  kernel，完整使用原 `_select_kv_with_plan` 和 metadata 提交流程。
- `absolute` 与 `block_relative` position 都由原 planner 生成 query/key position ids；
  kernel只消费对应的 FP32 cos/sin，不重新解释位置。

## 实际本地结果

| 检查 | 环境 | 结果 |
| --- | --- | --- |
| Python syntax、`ruff check`、`git diff --check` | macOS，本地 checkout | 通过 |
| cache 首块/recompute/growth/eviction fallback/错误 metadata | Python 3.11.13、torch 2.13.0，临时环境仅补 `uvicorn` | post/fallback 目标用例通过 |
| MinWM realtime CPU 语义回归 | 同上 | 全文件 `118 passed` |
| v03/v04 CUDA/Triton BF16 bitwise | H200、torch 2.11.0+cu130、Triton 3.6.0 | post 初始 `2 failed/3 passed`；两处均为 1 ULP，已按 eager FP32 指令边界修复 |
| v05 CUDA/Triton BF16 bitwise | 同上，commit `8fee184e47` | 当时注册的 6 项全部 exact；产品修剪与 predicate 加固后改为 7 项 post-only CUDA/回归 gate |
| v05 SP2/SP4 post-only E2E | 1248×704、short 129 frames、KV45 eviction 241 frames | lane01 对 lane00 四项 `.npy` SHA 与逐元素比较均 bitwise exact |
| v05 SP2/SP4 pre-enabled E2E | 同上 | lane10/11 四项均 SHA 偏离；pre 验收失败并从产品代码删除 |
| v06 post-only CUDA gate | H200、runner `52cc450575` | 7/7 exact，含 nonzero position、unsupported stride 与 malformed cache rank fallback |
| v06 post-only E2E | SP2/SP4、short 129 frames、KV45 eviction 241 frames | lane01 对 lane00 四项逐帧 SHA bitwise exact |
| v06 profiler-off | H200、每 lane 2×(20+200)、KV45 | 四 lane JSON 均通过 b924 schema、count=200 与 CV gate；见下表 |
| SP4 off-only ABBA | H200、4 个独立 server | A1/B1/B2/A2 均 count=200、CV pass；逆序配对仍复现收益 |
| Nsight | 等待单独 attempt | b924 capture window 有归一化缺陷；正式采集只允许 d5b |

本地系统默认 Python 3.9.6，无法解析仓库的现代 union type；改用已安装的
`/opt/homebrew/bin/python3.11`。为避免修改仓库或全局环境，测试使用一次性
`--system-site-packages` venv，只安装 `uvicorn`，并设置
`TORCHDYNAMO_DISABLE=1`；这只验证 CPU cache 语义，不代表 compile 或 CUDA 结论。

### v05 pre-A2A 负结论与帧级量化

v05 的 standalone gate 只用单一随机 seed、短 sequence（SP2 `S=17`、SP4 `S=5`）和
随机 norm weight 比较候选 kernel 与 eager 公式；两组输出 exact，但没有覆盖真实 5B
GEMM 产出的 BF16 分布、真实 1248×704 token 数、30 blocks 和迭代反馈。因此它只能证明
样例值未跨舍入边界，不能证明 reduction 在完整调用链中恒等。真实 E2E 中 lane10 与
lane11 总是得到同一份 pre 候选输出，lane00 与 lane01 总是得到同一份原路径输出，排除了
post 候选是偏差来源。

| SP | case | pre 首个不同位置 `[frame,y,x,c]` | baseline/candidate | differing frames | mismatch fraction |
| ---: | --- | --- | --- | ---: | ---: |
| 2 | short | `[1,0,0,2]` | `251/250` | 128/129 | 0.625805835 |
| 2 | KV45 eviction | `[1,0,0,1]` | `249/250` | 240/241 | 0.781033403 |
| 4 | short | `[1,0,9,2]` | `252/251` | 128/129 | 0.647508988 |
| 4 | KV45 eviction | `[1,0,2,1]` | `251/250` | 240/241 | 0.790975091 |

这些是保存的 uint8 decoded frame，不存在可解释的浮点 ULP；v05 未开启中间 latent dump，
不能从视频反推 latent/ULP。真实调用链诊断会用失败实现 commit `8fee184e47` 单独采集
baseline/pre 的第一个 Q/K norm、RoPE、attention 与 block latent，报告首个 mismatch、
max abs/ULP。该诊断只服务负结论，不会把 pre 实现带回产品 diff 或 post-only v06。

## H200 A/B 与 Nsight

每个 profiler-off lane 使用 20 warmup + 200 measured，至少两次重复并由 S0 工具计算
CV；SP2 为主验收，SP4 复验。headline 只取 profiler-off。所有 baseline/candidate
固定 `MINWM_S0_KV_CACHE_NUM_FRAMES=45`（等价 client
`--kv-cache-num-frames 45`），不随 `max_chunks` 扩张；这是 rolling-window steady-state
contract，20 个 warmup chunk 后必须已进入窗口淘汰态。

| SP | lane | Client FPS | Scheduler FPS | chunk wall ms | DiT wall/CUDA ms | VAE wall/CUDA ms | parity | 状态 |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 2 | `00` baseline | 12.8208 | 12.8318 | 1311.595 | 744.737 / 744.314 | 419.754 / 419.204 | exact | 2 repeats，CV pass |
| 2 | `01` post | 12.7889 | 12.7997 | 1311.308 | 720.054 / 719.641 | 420.169 / 419.602 | exact | 2 repeats，CV pass |
| 4 | `00` baseline | 14.9448 | 14.9594 | 1154.905 | 745.494 / 745.011 | 231.196 / 230.636 | exact | 2 repeats，CV pass |
| 4 | `01` post | 15.6844 | 15.7001 | 1095.115 | 705.839 / 705.410 | 231.232 / 230.675 | exact | 2 repeats，CV pass；ABBA 已复验 |

表中 FPS/wall 是两个 canonical JSON 的算术平均；CUDA 是同一 200-chunk stage trace 中
`component=minwm_denoising/vae_decoder` 的 CUDA event 补充汇总。b924 的 profiler-off
schema只把 wall 写入 headline JSON，因此 CUDA 数值来自保留的完整 server trace，不与
Nsight device 汇总混用。SP2 candidate 相对 baseline 为 Client `-0.249%`、Scheduler
`-0.250%`、chunk wall `-0.022%`；满足“不回退超过 1%”，但没有端到端收益。SP4 为
Client `+4.949%`、Scheduler `+4.951%`、chunk wall `-5.177%`；由于 v06 顺序固定
baseline→candidate，该轮单独看只能作为待复验信号；后续 ABBA 的逆序独立 server
设计仍复现同向收益，因此不再归因为顺序/cache 偏差。

真实 session window 审计没有发现 baseline 降频：SP4 四次均全程 P0，active GPU SM
clock 均值为 baseline `1971.67/1971.65 MHz`、candidate `1971.10/1969.99 MHz`，温度最高
`72/73°C` 对 `73/74°C`。平均功耗反而是 candidate 更高（baseline
`438.87/441.27 W`，candidate `450.87/454.92 W`），与低频/热限速造成假收益不符。
repeat1 首 payload 为 baseline `9.831s`、candidate `9.479s`；同 server repeat2 为
`1.538s/1.561s`，首次 segment compile/cache 与热态也基本对称。启动时
`nvidia-smi -q` 的 HW thermal/power braking 计数为零；v06 没有保存每个 session 前后的
限速计数，所以该轮不能单独作为完整历史证明；ABBA 已为每个 position 保存 before/after
快照。

阶段证据显示差异集中在 DiT。SP4 DiT wall/CUDA 均约 `-5.32%`，VAE 约 `+0.02%`；
SP2 DiT 也约 `-3.31%`，但 scheduler 中未归入 DiT/VAE 的部分增加约 `27.4ms`，抵消
`24.7ms` DiT 缩短。SP4 未归类部分反而减少约 `10.8ms`。telemetry 与 stage trace
无法解释这个 SP 依赖和 baseline-first 顺序效应，故新增并完成 SP4 off-only ABBA：
candidate A1→baseline B1→baseline B2→candidate A2，每个 position 独立 server、
20+200/KV45、独立 telemetry/marker/限速快照。Job
`minwm-s3-post-sp4-abba-20260807-07` 固定已有 S0 节点，等待 S0 -08 释放后自然调度，
最终 26 分钟完成；整机分配 8×H200，模型 active 4 卡，约为 3.47 allocated GPU·小时、
1.73 active GPU·小时。

| position | lane | Client FPS | Scheduler FPS | chunk wall ms | DiT wall/CUDA ms | VAE wall/CUDA ms |
| --- | --- | ---: | ---: | ---: | --- | --- |
| A1 | candidate | 15.5930 | 15.6150 | 1087.230 | 710.671 / 710.225 | 232.038 / 231.436 |
| B1 | baseline | 14.8445 | 14.8639 | 1163.125 | 741.391 / 740.894 | 231.784 / 231.216 |
| B2 | baseline | 15.0704 | 15.0891 | 1114.835 | 744.525 / 744.059 | 233.369 / 232.764 |
| A2 | candidate | 15.3867 | 15.4054 | 1081.350 | 713.467 / 713.003 | 233.454 / 232.883 |

ABBA 两个 candidate 对两个 baseline 的均值差为 Client `+3.559%`、Scheduler
`+3.563%`、chunk wall `-4.802%`、DiT wall `-4.158%`、VAE wall `+0.073%`；所有
wall/CUDA count 都是 200，candidate 与 baseline aggregate 均通过 CV gate。四个
position 的 active GPU 都全程 P0，SM clock 均值为 `1973.94/1974.23/1973.95/1973.98
MHz`，最高温度 `72/71/71/72°C`；HW Power Braking、HW Thermal Slowdown、SW Thermal
Slowdown、Sync Boost 的 before/after counter 增量均为 0。candidate 功耗仍略高，不能用
baseline 降频、热限速或先后顺序解释该收益。A1 的首 payload 较慢但不进入 20+200
稳态窗口，A2 仍复现相同方向。

Nsight 稳态至少 10 chunks，单独记录 A2A 后 kernel/launch、短 kernel 分桶、NCCL
SendRecv 时间与次数、两侧 idle gap、GPU kernel busy、SM Active、Tensor Active，以及
可得时的 DRAM。若 GPU metrics 因权限不可用，按 S0 schema 填 `unavailable`、原因和
采集证据，不留空。b924 的 raw capture 实际覆盖 1 discarded + 10 stable，却用 10
归一化，故任何 b924 Nsight 正式结果均无效；v06 与 ABBA runner 物理不含 Nsight
路径。正式 profiler-on 只允许临时 pin d5b：按 API start-time half-open 归属并分别保留
API/launch 边界证据，采集 all 8 target 后只汇总 active `pwGpuId`，并要求 exact 10
ranges、DiT/VAE CUDA、kernel/API/launch、SM/Tensor 与 target coverage 全部通过。

正式 Nsight 使用独立临时 runner `58ed4daf7e4208eedde4f8fc8f0a8c1e20e0007d`，其祖先
包含 d5b，且产品相关代码逐文件与 `61aa8809e6` 一致。四条 lane 顺序为 SP2 baseline、
SP2 candidate、SP4 baseline、SP4 candidate；每条均复用 v06 对应 profiler-off source，
但重新启动独立 server，执行 20 precondition + 1 discarded + exact 10 stable、KV45、
all-8 GPU metrics capture。每条 lane 在完成后先单独验收 stage wall/CUDA count=10、range
1..10、active CUDA device 到 `pwGpuId` 映射；失败只在该 lane 写 marker。比较器另报
post-A2A fused/语义匹配 kernel、所有减少的 kernel name、NCCL SendRecv/AllToAll duration、
其前后 device-visible gap，以及 profiler-off/on 的 scheduler 未归类余量。

正确性另跑两组：短程无淘汰用例覆盖首块、growth、append 与同一 active
chunk 的 DMD/clean-cache recompute；45-frame 固定窗口用例覆盖稳态淘汰与 fallback
parity。当前 post fast lane 在 eviction plan 上主动 fallback，因此若 45-frame 稳态不命中、
wall 无改善，这就是必须记录的 critical-path/覆盖证据，不得用短程数据替代
headline。

## 与预期不符处

1. 本地完整 pytest 最初不是用例失败，而是系统 Python 3.9 语法版本不满足；切换
   Python 3.11 后，macOS torch 的模块级 compile 又需要 Triton。通过禁用 CPU-only
   compile 后，剩余仅是 `uvicorn` 轻依赖。
2. cache growth 不是简单设置 `allow_growth=true` 就会命中。若逻辑
   `attention_window_size` 已等于较小物理 cache，planner会按既有窗口语义先选择淘汰；
   真正 growth case 是逻辑 window 能保留完整历史、但物理 buffer 暂时不足。
3. pre-A2A 的两个 standalone 随机 shape exact，但真实 5B E2E 从 frame 1 起大面积分叉。
   这说明 reduction 等价性必须用真实模型输入和完整迭代链验收，不能以少量随机向量外推。
   按 bitwise 合同不放宽容差，直接删除产品实现和开关。
4. post-A2A 最初只有 1 ULP 漂移，显式分离 eager 的 FP32 mul/add/sub 后，v05 CUDA gate
   和四组 E2E 均 exact；这是可修的操作顺序问题，不需要修改数值合同。

## 证据与决策过程

- 没有把 norm 放到 A2A 后：post rank只持有 local heads，已缺失 across-head reduction
  domain，会改变数值定义。
- 没有让 pre kernel调用 NCCL：collective是跨进程同步与 ownership 迁移，不是单 GPU
  kernel 可跨越的内存边界。
- pre 候选曾增加 peer-first packed 输入接口，但真实 E2E 失败后该接口与 kernel 一并从
  产品 diff 删除，避免给 S4 留下未经验收的依赖；三份 Q/K/V GEMM 始终未改。
- cache planner继续在 Python 层产生 plan：sink/pin/window/position 是控制语义，不在
  Triton 内重复实现。kernel先 launch，成功后再 commit metadata；不支持的 plan 在任何
  metadata mutation 前返回 fallback。
- append 先重建 visible rotated K，而不是复用旧历史：这是保守选择，避免在首版里对
  block-relative/reordered position 做未经证明的增量假设。recompute 才复用已验证历史。

## 尝试后放弃或暂缓的方案

- 跨 `all_to_all_single` 的单 kernel：ownership 与 NCCL 边界不成立，直接放弃。
- 在 post kernel内实现 eviction/dynamic pin：会复制 planner 逻辑并扩大错误面，暂缓；
  现路径安全 fallback。
- 修改 Q/K/V GEMM 数量或合并 projection：属于 S4，明确不做。
- 保留 pre fast lane 并放宽 tolerance：真实 decoded frame 已大面积分叉，与 bitwise
  合同冲突，放弃并删除。
- 默认开启 post fast lane：SP4 ABBA 已确认收益不是单纯顺序偏差，但 SP2 headline 仍近零，
  且正式 Nsight 还未完成；当前证据不足以改变默认关闭。

## 风险、回滚与复现

主要风险是 post RoPE 指令顺序随 Triton/PyTorch 版本变化，或 kernel 减少但同步 A2A
仍主导 wall。设置 `MINWM_FUSED_POST_A2A_ROPE_CACHE=false` 即回滚到原路径；pre 已物理
删除，不存在误开启入口。若 post metadata/cursor 检查失败，应视为 correctness bug，
不应强制 fallback 继续运行。

本地目标测试：

```bash
TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 PYTHONPATH=python \
  <temporary-python-3.11>/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py \
  -k 'fused_post_a2a'
```

v06 profiler-off 临时叠加 S0 `b9240233b2438829cbd72ee3dfbc1d37ed675560`；其结果仍
有效。ABBA off-only runner 临时叠加 S0
`900b5f279b65b2afcfbe6cc9b36cfa4496b41bc3`；后续正式 Nsight 临时叠加
`d5b25227d4487d113e62c86a0fb572a62d6bcc5b`。产物必须通过
`benchmark/minwm_realtime_parity/measurement_tool.py validate`；正式 PR 不包含 S0
基础设施。最终 JSON 记录实际 checkout SHA，`gpu.count` 按 active GPUs、
`gpu.allocated_count` 单列；Nsight/API 统一使用 `raw_total`、`total_per_chunk`、
`per_rank_per_chunk`，其中 per-rank 只在 coverage 检查通过时采用；最终 JSON 再用
该版本的 jsonschema validator 复验。所有 wall/CUDA latency 必须显式带 `count`，且
自定义 validator 要求它等于 `workload.measured_chunks`；S3 runner 另做一次独立 count
断言。profiler-off/on 的 DiT/VAE wall 只有在
`expected_indices=0..N-1` 完整、两者均为 `status=available`，且 profiler-off count=200
时才保留；否则重跑对应 lane。具体
Job 名与产物路径在创建后补入，且只清理带本任务唯一前缀的资源。

首次 `minwm-s3-a2a-h200-20260807-01` 在 setup/staging 阶段收到旧 S0 pin 更新，
尚未进入 kernel/parity/A/B client 即删除该精确 Job，保留本任务 PVC；不保留任何
25cc 有效测量；原始 setup 产物已标记 invalid 并保留。第二次
`minwm-s3-a2a-h200-20260807-02` 已使用 59aa，但在任何
CUDA kernel/client 启动前，registered test 收集因临时 runner 未把 source tree 加入
`PYTHONPATH` 而失败；这不是实现测试失败，也不产生可保留的测量数据。修正后的 runner
commit 为 `88d54943f1d77a0919643fcb35e5961931464bef`，第三次 Job
`minwm-s3-a2a-h200-20260807-03` 继续复用同一结果 PVC，并重新从 correctness gate 开始。
v03 是真实实现失败：pre-A2A 的 SP2/SP4 kernel shape bitwise exact，post-A2A raw K/V
写 exact，但 first/append/recompute 的 RoPE 中两项断言失败；因此没有启动 parity 或
client。v04 使用旧 kernel 加定量断言复现 mismatch count/fraction、max abs、BF16 ULP
与重复调用确定性。实测 first rotated K 仅 1/26112 元素不同
（fraction `0.000038297`、max abs `3.81469727e-06`、max ULP 1），recompute Q
仅 1/3840 元素不同（fraction `0.000260417`、max abs `1.86264515e-09`、max ULP 1）。
后者不经过 fresh/cache 选择，证明根因是 Triton 将 eager 的两次 FP32 multiply 与后续
add/sub 收缩，first Q exact 只是该组值未跨 BF16 舍入边界。修复显式使用逐条
`mul.rn.f32` 与 `add/sub.rn.f32`，并把 fresh/cache 改为 uniform token 标量分支。v05
的 6 项 CUDA gate 全部 exact；随后完整跑完 SP2/SP4 × 00/10/01/11 × short/KV45。
汇总在首个 SP2 short pre mismatch 处失败，未创建 `measurements/`，也没有任何 nsys
文件。post-only lane01 对 lane00 的 SP2/SP4 short/eviction 四项 SHA 和逐元素均 exact；
pre-enabled lane10/11 四项均偏离，故产品 PR 删除 pre kernel/env/runtime/tests，只保留
负证据。v05 根级 invalid marker 为 `runner_exit_nonzero`、exit 1，覆盖 114 files、
7,886,195,088 bytes，逐项 SHA256 长度正确，marker 自身 SHA256 为
`e10f006966fc47f4be4bdfdca7cc73cc6f26471f9ee22c82c6d1c4f466f258cc`。

v05 操作审计还有一处人为偏差：为按新规则在 formal 前停止，曾把
`STOP_BEFORE_FORMAL_AB` 写到 attempt 外层；脚本实际检查的 `RUN_ROOT` 是其下
`${MINWM_RUN_ID}/s3`，所以哨兵层级错误。该文件保留，未伪装为有效 checkpoint。此次未
越界的原因不是错误哨兵，而是 parity 汇总先抛错并触发 `set -e`/EXIT trap；审计确认
`measurements/` 不存在、nsys 路径为空。v06 使用 post-only runner，quality 后直接执行
off-only 20+200，脚本物理不引用/安装/启动 nsys，因此不依赖哨兵阻止 profiler-on。

v06 Job `minwm-s3-a2a-h200-20260807-06` 在 runner
`52cc450575d32e7b839537f2e6761d20dd32b77c` 上成功完成；Pod attempt 为
`minwm-s3-a2a-h200-20260807-06-swtj9`，结果根目录为其下
`${MINWM_RUN_ID}/s3-post-off-only`。四组 quality SHA 分别为 SP2 short
`38e7ef07cffb7e8df2e59323dcbd9dacda92d31ab4a268d1276b554b7f3e833b`、SP2 eviction
`34bd7aff45c51fb0476f5b3b053e9924780b4308e9c85b9cf666ca36971bcf6f`、SP4 short
`14af1068a53d0e4479fcc163fdfd5edc3415242c261852fc844cedf19e3c5a4c`、SP4 eviction
`61805d90b010d3bd918fa5eb5823f44e0f03ae64d12448d85d9e3e22da365d4b`，lane00/01 每组
一致。四个 measurement lane 各两次重复、共 1600 measured chunks，全部
schema/count/CV pass；Job 正常完成，不存在 invalid marker。

ABBA 使用 immutable runner `29c6ada1a514c137c2ca4cf81b58fdc2065b401a`，manifest 证据
commit `f669e8f628`，S0 pin 为 900b，产品实现对应 PR commit `61aa8809e6`（临时 runner
中的等价 cherry-pick 为 `050791ffd3`）。Job 固定 hostname
`i-06888dc1ca88547e1`、backoffLimit=0；创建后因该节点被 S0 -08 占用而 Pending，事件为
`Insufficient nvidia.com/gpu`，随后在原节点自然调度并成功完成。精确 hostname selector
阻止 Karpenter 为该 Job 新建节点；全程不抢占、不扩容、不清理其他任务对象。

ABBA 证据根为
`/results/attempts/minwm-s3-post-sp4-abba-20260807-07-dfvvc/minwm-s3-post-sp4-abba-20260807-07/s3-post-sp4-abba-off-only`，
共 50 个文件、24,255,371 bytes、无 invalid marker。A1/B1/B2/A2 JSON SHA256 分别为
`a2e43d89ba8384d170b2482cd86703836bd0d91c3e50c97bff8a917662da0dbc`、
`44f275648c4122c66c8130406e7ff8897a54b5685795049d8e24a7b60545a003`、
`ff4488613ddc13e6b0936181836d9962dcce28b9a826b8bd8c89f03d2d3eb20e`、
`c4cf80ed5934ed41d8c90b78d856ea8bfab8a132f7c946704b8fcb6fc7794135`；candidate/baseline
aggregate SHA256 为
`9abda8475206f70bc5ea4a34043803d8848c177df25100ae1cea40246cea5424`、
`f24672b7d1a34257c629901cd2f7da022e45cc10bb41d25ef8dbdc88c6ed2714`，总 summary 为
`ee3e01d8d6960962b3b68dfbbc6d4071252092a15ea9b49f7aa8b67f8a6a41f5`。审计用 0-GPU、
只读 PVC reader 已精确删除，PVC 与全部 evidence 保留。

Nsight manifest 证据 commit 为 `1b6a91d695`，Job
`minwm-s3-post-nsys-h200-20260807-08` 使用 `backoffLimit=0`、8×H200、`SYS_ADMIN` 和现有
S3 PVC，固定 hostname `i-06888dc1ca88547e1`，防止 Karpenter 扩容。提交时目标节点被
S4 占满，Pod `minwm-s3-post-nsys-h200-20260807-08-swlsq` 自然 Pending；调度事件为目标
节点 `Insufficient nvidia.com/gpu`、其余四节点不匹配 selector，且没有 preemption
victim。此状态未分配 GPU、未启动 client，也未创建或清理其他任务对象。

所有失败、旧契约和 partial attempt 在 PVC 上物理保留：原路径或对应 scope 的
`invalid/` 下必须有 marker，记录原因、UTC、逐文件路径/大小/SHA256 与可恢复性。
setup/质量/parity 这类全局前置失败使用 attempt measurement root marker；某个已开始的
measurement lane 后段失败只在该 lane 写 marker，不能误伤已验证的 sibling/headline。
聚合器从当前 JSON parent 向上检查到最近 measurement root，排除沿途任一 marker；
sibling lane marker 不影响本 JSON。只删除本任务精确命名的 Job/Pod 控制对象止损，
不删除 PVC 证据。

## 给负责人掌握代码的检查题

1. **为什么 Q/K norm不能移到 A2A 后？** 参考：`MinWMRMSNorm` 对最后的 3072 维做
   reduction，而 A2A 后只有 `H_local*D`；看 `minwm.py::_minwm_qk_norm_op`。
2. **为什么 standalone pre kernel exact 仍被删除？** 参考：它只覆盖一个随机 seed 和
   短 sequence；v05 真实调用链四组均从 decoded frame 1 分叉，bitwise 合同优先。
3. **NCCL 后 cache为什么使用 `cache_head_start=0`？** 参考：每 rank cache创建时已只分配
   `num_heads/ulysses_degree`；看 denoising stage 的 `_num_causal_cache_attention_heads`。
4. **clean-cache 第五次 forward 会不会被 fast path跳过？** 不会；recompute仍覆盖 raw
   K/V 与 current rotated K。看 `prepare_fused_post_a2a_update` 的 recompute 分支。
5. **哪类 cache plan 必须 fallback？** `preserves_all_history=false` 的 selection/eviction，
   以及 cache head slice/grad/unsupported layout；看同一 prepare 方法与 kernel predicate。
6. **pre 删除后如何避免与 S4 冲突？** 现有 Q/K/V GEMM、eager norm、USP pack 接口完全
   恢复 main；S4 不会依赖本任务曾增加但未验收的 packed-input 接口。
7. **为什么 append 当前重转整个 visible K，而 recompute只转 current K？** append采用
   保守 position 合同；recompute已持有同一 visible plan的有效 rotated history。
8. **kernel数下降但 Client FPS 不变时先看什么？** 看 A2A 前后 idle gap、NCCL
   SendRecv wall、DiT critical path与 VAE负对照，而不是只报 launch 百分比。
9. **如何一键回滚？** 设置 `MINWM_FUSED_POST_A2A_ROPE_CACHE=false`；默认本来就是关闭，
   pre 开关已从产品代码删除。
