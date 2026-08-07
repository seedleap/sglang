# MinWM Ulysses 异步 A2A 实施与验收记录

状态：实施中，尚未通过验收。`async_op=True`、可 capture、non-blocking copy、collective
数量减少或单个 NCCL kernel 变短都不单独构成 overlap。最终只在 profiler-off 客户端 FPS
达到硬门槛，且 Nsight 证明 A2A exposed critical-path time 被真实 compute 隐藏时判定通过。

## 版本与环境合同

| 项目 | 当前合同 |
| --- | --- |
| 上游 MinWM 集成 | `9a9dc59cd1d9bf33842378f75fa0f46c41e07b7e` |
| MinWM 逻辑基线 | `0e30671cf8a00622fd138c71af3faa93353b5425`（`codex/ulysses-pre-post-a2a-fusion`；失败的 pre-A2A QK-norm 已删除，post-A2A parity 路径默认关闭） |
| upstream 通信基线 | 待在本分支整合 `754b692afc29`（SP2 IPC）、`a5888c956f90`（packed QKV + reusable staging）、`bfce378e5fbc`（capture-safe PyNCCL A2A）；三项整合与回归通过后的 SHA 才是正式同步 baseline |
| 实现分支 | `codex/minwm-async-a2a-overlap` |
| 候选提交 | 待实现后固定；真机任务只使用已推送的不可变 SHA |
| 主硬件 | B200/B300 同节点 Spot；若容量不可得，H200 只用于诊断，不能冒充最终硬件结论 |
| 软件 | 待真机 provenance 固定；已有 H200 参考栈为 PyTorch `2.11.0+cu130`、CUDA `13.0`、Triton `3.6.0`、Nsight Systems `2026.4.1`，NCCL 版本待采集 |
| checkpoint | 预定沿用 canonical MinWM 5B DMD checkpoint；精确 S3 URI、VersionId、ETag/CRC64/SHA256 在任务 dry-run 与 provenance 中固定 |
| 模型基座 | `Wan2.2-TI2V-5B-from-diffusers` |
| 主输入 | 1248×704（项目 720p）、5s、固定 prompt/首帧/action、固定 seed、SP2 |
| 次输入 | 同一 720p 5s 的 SP4；832×480 仅作边界对照 |
| 稳定性 | 同配置连续至少 10 请求或等价长跑；覆盖 eager 与生产启用时的 CUDA Graph |
| 结果根目录 | 待创建；每次尝试使用不可覆盖的 run-id 子目录，负结果同样保留 |

所有正式命令、环境值和路径按实验追加记录，不以“与上次相同”省略。已有 post-A2A
融合证据只作为基线审计资料，不算本任务的异步 A2A 验收。

## 依赖图与理论 overlap 窗口

### 当前同步关键路径

```text
Q/K/V projection
  -> Q/K norm
  -> peer-first packed-QKV pack
  -> input A2A
  -> split Q/K/V
  -> cache plan + raw K/V ownership update + Q/K RoPE
  -> causal attention
  -> reverse output A2A
  -> output projection
  -> residual / norm / FFN
```

### 绝对不能跨越的依赖

- input A2A 之前，每个 rank 持有本地 sequence shard 和全部 heads；之后才持有全 sequence
  与本地 heads。Q/K across-head norm 和 peer-first pack 必须完成后才能通信；cache、RoPE、
  attention 不能消费尚未收齐的远端 token。
- attention 必须消费 input A2A 的完整 Q/K/V；reverse output A2A 必须消费 attention 输出。
  output projection、residual 更新及其后 FFN 必须等待 reverse A2A 的完整输出。
- cache metadata commit、current/tail ownership、rotated-K validity 与请求/chunk 代际不能提前
  暴露；不得让下一请求复用仍被通信 stream 读取/写入的 buffer。
- 不跨越 causal cache eviction/sink/pin 选择，不改变 attention ownership 或数值顺序来制造
  overlap。

### 待验证的合法窗口

| A2A 阶段 | launch 与 wait 之间可能独立的 GPU 工作 | 当前风险/验证要求 |
| --- | --- | --- |
| packed-QKV input A2A | 同一 block 内仅限不读取 A2A 输出且不修改其 source/storage 的工作；候选包括已准备好的 cache metadata/position bookkeeping，或不同 request/chunk 的独立计算 | 单请求层内窗口预计很小；必须在 trace 中看到通信与具体 compute kernel 时间相交，CPU bookkeeping 不算 GPU overlap |
| reverse output A2A | 同一 block 内 output projection/residual/FFN 都依赖结果，不能提前；候选主要是不同 request/chunk 的独立计算或显式流水 | 若 scheduler 只有一个 in-flight chunk，单纯拆 API 不会产生收益，需评估跨 request/chunk 双缓冲且保持 cache 隔离 |

首轮代码前先量化真实 forward 中可移动的独立 GPU 工作。若 launch→wait 间没有合法 compute，
立即把结果写入“偏离原认知”，不把接口异步化当作完成；再评估跨 request/chunk pipeline。

## 设计约束与回滚

- API 拆成明确的 `begin/launch` 与 `wait/consume`；handle 拥有 output、work/event、stream、
  generation/slot 和 source lifetime。
- communication stream 与 current compute stream 通过 CUDA event 建立单向依赖；consumer
  只等待对应完成 event，不引入 device-wide synchronize。
- 至少双缓冲；slot 在完成 event 被消费前不可复用。对通信所读写 tensor 使用
  `record_stream` 或等价 lifetime 保护。
- ProcessGroupNCCL、capture-safe PyNCCL、SP2 IPC transport 的选择必须显式；不支持、异常或
  开关关闭时回到同步基线。fallback 不得掩盖半完成 collective 或跨 rank 分歧。
- eager 和 CUDA Graph 分开验证。任何 backend 在 capture 中不安全时必须一致回退，不能让
  不同 rank 选择不同路径。
- 预定总开关：`MINWM_ASYNC_A2A=0` 回滚到同步基线；更细的 input/output/backend 开关待最小
  实现后记录最终名称和默认值。

## NVTX 与统计合同

至少区分：`qkv_projection`、`qk_norm`、`qkv_pack`、`input_a2a_launch`、
`input_a2a_wait`、`post_input_a2a_cache_rope`、`attention`、`output_a2a_launch`、
`output_a2a_wait`、`output_projection`、`ffn`。每个 handle 记录 launch、wait-enter、
complete/consume event；统计 input/output A2A 的 launch→wait 距离、wait exposed time、
与 compute kernel 的区间交集及 buffer slot/generation。

## 验收合同

### A. 正确性

- 不降低现有 MinWM 5s parity 阈值；A2A round-trip 优先 bitwise exact。
- 同 seed/prompt/首帧/checkpoint 的 720p 5s：SP2 baseline/candidate 视频与中间 tensor parity；
  SP4 至少完整跑通并比较。
- 连续至少 10 请求或等价长跑，无 hang、buffer 污染或串数据；覆盖 eager/CUDA Graph。
- 新增 async ordering、buffer reuse、event dependency、fallback、SP2/SP4 parity 测试，并运行
  现有相关单测。

### B. profiler-off 端到端性能（最终裁决）

- 同机同进程配置，预热后 ABBA/BAAB 交替，baseline/candidate 各至少 5 个有效样本；报告
  median、p10/p90、CV/变异与原始 JSON。
- 主验收为 B200/B300 Spot 同节点、720p 5s、SP2；SP4 次验收，480p 边界对照。
- 保留门槛：客户端 FPS median 至少 `+3%`，目标 `+5%`；scheduler/chunk 或 DiT wall 同向，
  且 parity/稳定性不回退。低于 `+3%` 明确判定未通过。

### C. Nsight Systems 机制验收

- baseline/candidate 在同机采相同稳态窗口，`torch.profiler=false`；profiler-on wall 不进入
  最终 FPS。
- 每 chunk 报告 NCCL A2A 总时长、A2A exposed critical-path time、launch→wait 距离、
  compute overlap ms/比例、前后 idle gap、CUDA/NCCL kernel 数、SM Active/Tensor Active。
- 至少一个主要 A2A 阶段出现非零、可重复的 compute/communication overlap；A2A exposed
  time 下降至少 `20%`，或 A2A+相邻 idle gap 的关键路径下降至少 `10%`。
- trace 检查新增全局 synchronize、host wait 和串行 stream dependency；发现任一抵消收益则
  不通过。

## 按时间追加的实验日志

### 2026-08-07：基线与历史分支审计

- 假设：最新 MinWM 长历史分支可能已经包含可直接复用的 async A2A。
- 检查：审计 `codex/minwm-realtime-api`、`codex/ulysses-pre-post-a2a-fusion`、
  `codex/minwm-qkv-peer-first-fast-lane`、`codex/minwm-cuda-graph` 及相关测量分支和工作树。
- 证据：通用 `turbo_layer.py` 虽有 `async_op=True`，但调用链在返回前 wait；MinWM 自定义
  causal attention 走 `runtime/models/dits/minwm.py` 与同步 `runtime/layers/usp.py`。
- 决策：从 `0e30671cf8a0` 创建独立分支。只继承已通过 parity 且默认关闭的 post-A2A
  fusion；不恢复失败的 pre-A2A QK-norm，不混入 QKV fast-lane 或 CUDA Graph 分支。随后
  发现 MinWM origin 线落后于 upstream 通信实现，因此正式同步 baseline 还必须最小整合
  `754b692`、`a5888c9`、`bfce378`，通过回归后再固定 SHA。
- 正确性：继承基线证据；本任务候选尚未实现，未运行新 parity。
- wall/FPS：尚未测量。
- Nsight：尚未采集。
- 结论：未验收；下一步绘制精确调用依赖并测出真正可用的 GPU overlap 窗口。

## 偏离原认知

### 2026-08-07：MinWM origin 与 upstream 通信提交不在同一条主线

- 原认知：`codex/ulysses-pre-post-a2a-fusion` 已自然包含题目所述 upstream IPC、packed
  staging 与 capture-safe PyNCCL。
- 实际证据：该分支祖先是 `origin/main@9a9dc59`，其中没有 `ipc_a2a.py`；本地
  `main@3654740` 有三项通信提交，却没有 MinWM 模型文件。两条线共同祖先较早，不能选
  其中任一条直接满足全部前提。
- 根因：MinWM 以独立 squashed integration 进入 `origin/main`，尚未同步同日期 upstream
  的后续通信提交。
- 决策：保留 MinWM/post-parity 逻辑基线，只 cherry-pick 三个已合入 upstream 的目标提交
  并解决局部冲突；不把整个 upstream main 合并进来，避免无关大范围升级。整合后的回归
  与真机 provenance 决定正式同步 baseline SHA。

### 2026-08-07：已有 post-A2A fusion 对 SP2 无客户端收益

- 原认知：减少 post-A2A 后的 pointwise/copy launch，且 DiT wall 下降，可能自然传导为客户端
  FPS 提升。
- 实际证据：已有 H200 profiler-off SP2 20+200 结果中，DiT wall 约下降 `3.31%`，但
  Client FPS `-0.249%`、chunk wall `-0.022%`；SP4 独立 ABBA 才达到约 `+3.56%`。
- 根因：SP2 scheduler 未归类时间增加，抵消 DiT 缩短；Nsight 中 NCCL residence 变短也
  不能证明 exposed A2A 被隐藏。
- 决策：post fusion 可留作默认关闭的 parity-passed 基线能力，但不算本任务异步收益；主验收
  仍以 SP2 B200/B300 profiler-off FPS 与 exposed-time 证据为准。

## 最终 before/after、复现与剩余风险

待完成。最终将列出每个硬件/SP/分辨率的 before/after median、p10/p90、变异、parity、
稳定性、trace 统计、完整命令、结果路径、回滚开关和未解决风险，并给出明确 PASS/FAIL。
