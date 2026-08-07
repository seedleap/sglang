# MinWM Ulysses 异步 A2A 实施与验收记录

状态：实施中，尚未通过验收。`async_op=True`、可 capture、non-blocking copy、collective
数量减少或单个 NCCL kernel 变短都不单独构成 overlap。最终只在 profiler-off 客户端 FPS
达到硬门槛，且 Nsight 证明 A2A exposed critical-path time 被真实 compute 隐藏时判定通过。

## 版本与环境合同

| 项目 | 当前合同 |
| --- | --- |
| 上游 MinWM 集成 | `9a9dc59cd1d9bf33842378f75fa0f46c41e07b7e` |
| MinWM 逻辑基线 | `0e30671cf8a00622fd138c71af3faa93353b5425`（`codex/ulysses-pre-post-a2a-fusion`；失败的 pre-A2A QK-norm 已删除，post-A2A parity 路径默认关闭） |
| upstream 通信基线 | 已 cherry-pick `754b692afc29`（本分支 `4202d2a043`，SP2 IPC）、`bfce378e5fbc`（本分支 `36066c3d44`，capture-safe PyNCCL A2A）、`f829fb30d3d3`（本分支 `35a313e616`，IPC lifecycle/reset）和 `44bde391d0b1`（本分支 `ed152d6c48`，真实 peer CUDA ordinal）；`a5888c956f90` 的 packed QKV + reusable staging 与 MinWM 现有同名但不同返回合同的实现重复，未强行合并 |
| 实现分支 | `codex/minwm-async-a2a-overlap` |
| 候选提交 | 核心实现 `dad5de09f1b644074cb3d977fdad11ffd18772bf`；GPU 测试隔离修复 `0a975bac14e20660e6a31744bf864c8d51d99aff`；精确 Nsight chunk 窗口与 forward 标记整合至 `c6309c3004`；IPC output A2A trace 识别修复为 `3ed01a42a33423a9d86ceccd7aff2c2d95cca02d`。完整模型任务继续使用每次已推送的不可变 SHA，并在对应实验条目固定 |
| 主硬件 | B200/B300 同节点 Spot；若容量不可得，H200 只用于诊断，不能冒充最终硬件结论 |
| 软件 | 首轮 H200 transport 合同实测为 PyTorch `2.12.1+cu130`、CUDA `13.0`、NCCL `2.29.7`；完整模型与正式 B200/B300 各自继续保存 runtime provenance。Nsight Systems 目标版本为 `2026.4.1` |
| checkpoint | 预定沿用 canonical MinWM 5B DMD checkpoint；精确 S3 URI、VersionId、ETag/CRC64/SHA256 在任务 dry-run 与 provenance 中固定 |
| 模型基座 | `Wan2.2-TI2V-5B-from-diffusers` |
| 主输入 | 1248×704（项目 720p）、5s、固定 prompt/首帧/action、固定 seed、SP2 |
| 次输入 | 同一 720p 5s 的 SP4；832×480 仅作边界对照 |
| 稳定性 | 同配置连续至少 10 请求或等价长跑；覆盖 eager 与生产启用时的 CUDA Graph |
| 结果根目录 | H200 transport：PVC `minwm-async-a2a-contract-results-20260807`，成功路径 `/results/minwm-async-a2a-contract-h200-20260807-04`；完整模型与正式性能每次使用不可覆盖的 run-id 子目录，负结果同样保留 |

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

首个候选把 input collective 按最后一维拆成 `QK` 与 `V` 两段：先计算并归一化 Q/K，
launch QK A2A；compute stream 随即执行独立的 V projection，再 launch V A2A，随后才 consume
QK/V。这样增加一次 collective launch，但通信字节数不变，并给 QK A2A 一个真实的 V GEMM
窗口。是否有净收益由真机决定；不得用依赖图推断替代 trace。reverse output 当前只有
begin/consume 观测接口，默认关闭且立即 consume，因为层内没有合法工作可插入。

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
- 总开关：`MINWM_ASYNC_A2A=0`（默认）回滚到同步 packed-QKV 基线；开启后 input 使用
  QK/V split。`MINWM_ASYNC_A2A_OUTPUT=0` 默认保持 reverse output 同步，设为 `1` 只用于
  测量立即 begin/consume。`MINWM_ASYNC_A2A_BACKEND=auto|process_group|pynccl|ipc`，默认
  `auto`：eager 与同步基线同用 ProcessGroupNCCL，capture 强制选择 PyNCCL；显式 IPC 仅限
  SP2 且要求 staging 已 warm。capture 无安全 backend 时 fail-fast，不能静默进入可能挂死的
  ProcessGroupNCCL graph。

## NVTX 与统计合同

至少区分：`qkv_projection`、`qk_norm`、`qk_pack`、`v_pack`、
`input_a2a_launch_qk`、`input_a2a_launch_v`、`input_a2a_wait_qk`、
`input_a2a_wait_v`、`input_a2a_overlap_v_projection`、
`post_input_a2a_cache_rope`、`attention`、`output_a2a_launch`、
`output_a2a_wait`、`output_projection`、`ffn`。每次 sequence-sharded model forward 另发出
`minwm_forward_start_current_<current_start>` 与
`minwm_forward_end_current_<current_start>` NVTX mark，使 SQLite 分析可按 chunk 的
`current_start` 边界聚合，而不以服务器 wall log 猜测 trace 窗口。每个 handle 记录 launch、
wait-enter、complete/consume event；统计 input/output A2A 的 launch→wait 距离、wait exposed
time、与 compute kernel 的区间交集及 buffer slot/generation。

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

### 2026-08-07：H200 transport/order/CUDA Graph 合同

- 假设：在启动完整 checkpoint 前，独立 A2A 合同应先证明 ProcessGroup eager 的事件顺序、
  双缓冲复用、同步 packed-QKV 精确对齐、reverse output A2A，以及 capture-safe PyNCCL 的
  graph capture/replay 均不挂死。
- 命令与资源：Kubernetes context `codex-minwm-test-phx2`，固定 Ready 节点
  `i-06888dc1ca88547e1`，镜像
  `829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-training@sha256:bedc07ea3ba55059a8c1c569c3b177c4d00d41f37d4fa9105375531534ef5f2a`，
  候选 `0a975bac14e20660e6a31744bf864c8d51d99aff`。成功任务
  `minwm-async-a2a-contract-h200-20260807-04` 依次执行
  `MINWM_ASYNC_TEST_WORLD=2 python3 -m pytest -q -s .../test_minwm_async_a2a_gpu.py` 与
  `MINWM_ASYNC_TEST_WORLD=4 ...`。manifest 为
  `benchmark/minwm_realtime_parity/k8s/minwm_async_a2a_contract_h200_20260807.yaml`。
- 硬件与软件：`NVIDIA H200`，4 张可见 GPU，PyTorch `2.12.1+cu130`，CUDA `13.0`，
  NCCL `2.29.7`。
- 正确性：SP2 `1 passed`（约 `77.42s`），SP4 `1 passed`（约 `36.14s`），Job
  `Complete 1/1`。每个 world size 覆盖 12 轮 continuous ordering/buffer reuse、同步
  packed-QKV bitwise round-trip、独立 kernel 位于 begin/wait 之间、reverse output parity，
  以及 PyNCCL graph capture 后 3 次 replay。
- wall/FPS、Nsight：本测试不是完整 MinWM，不产生客户端 FPS，也不作为 overlap/Nsight
  验收证据。
- 结论：transport 合同通过，可以进入 720p 5s 完整模型 parity；本任务整体仍未验收。

### 2026-08-07：完整模型质量与 profiler-off 测量合同

- 假设：历史 MinWM S0/S4 测量工具已有稳定的 stage-trace 完整性、样本数、CV 与失败证据
  合同，可复用其纯 benchmark 部分，而无需带入 QKV fast-lane 产品代码。
- 修改：移植 measurement schema/tool/Nsight 统计基础；新增
  `run_async_a2a_quality.sh`，固定 1248×704、8 chunks/128 生成帧、seed 42，SP2/SP4 各跑
  baseline 与候选，候选同一服务连续 10 请求，并逐 rank 比较视频及中间 tensor bitwise；
  新增 `run_async_a2a_measurement.sh`，每个位置重启服务，默认按 ABBA/BAAB/ABBA 收集
  baseline/candidate 各 6 样本，输出 median、p10/p90、CV，且显式固定 output A2A 关闭、
  input backend 为 ProcessGroupNCCL。客户端最终裁决显式设置
  `SGLANG_REALTIME_TRACE_SYNC_CUDA=0`，不把为 stage 计时注入的全局同步带进 profiler-off
  FPS；scheduler chunk wall 仍是同一客户端可见完成边界，DiT 单独 wall 仅作辅证。
- 正确性：本地 measurement/runner 合同测试 `15 passed`；完整 H200 checkpoint 运行待提交。
- wall/FPS、Nsight：待运行。
- 结论：测量工具不改变产品路径；下一步使用独立 PVC，在固定 H200 节点分配 SP4 所需的
  4 GPU 先跑诊断质量门，不抢占该节点上已有的 1-GPU 任务。

### 2026-08-07：H200 720p 5s 完整模型质量门，attempt 01

- 命令与资源：Job `minwm-async-a2a-quality-h200-20260807-01`，固定 H200 节点
  `i-06888dc1ca88547e1` 的 4 GPU，SGLang `e41b56f626c83854dabc430149a58d41324fb763`，
  minWM `2efc6485f65e8fcab506665efde79bc41406385e`，checkpoint bytes
  `10007171771`、SHA256
  `1dc42d498cad84349987db2015120ce4d77e6b641f7f38c75ec9df3f942a7975`，
  PyTorch `2.11.0+cu130`、CUDA `13.0`、NCCL `2.28.9`、transformers `5.12.1`。
  结果 PVC `minwm-async-a2a-quality-results-20260807`，路径
  `/results/attempts/minwm-async-a2a-quality-h200-20260807-01-grpd9/minwm-async-a2a-quality-h200-20260807-01/async-a2a-quality`。
- 正确性：SP2 与 SP4 均完成同步 baseline 1 次和候选同进程连续 10 次完整请求；每次输入为
  1248×704、8 chunks、128 生成帧、seed 42。两种 SP 的第 10 次候选与 baseline 都是
  129 帧视频 bitwise exact，`max_abs=0`、`RMSE=0`、`SSIM=1`；无 hang 或视频串数据。
- 最终状态：视频与长跑通过后，attempt 01 在 tensor 文件集合检查上报 FAIL：候选多出
  `cross_k/cross_norm_k/cross_v_output_001.pt`，baseline 文件没有缺失，尚未执行逐 tensor
  数值比较完成。因此本条按原始任务状态保留为失败，不把视频成功等同于全部质量门成功。
- wall/FPS、Nsight：质量任务不计入性能；尚未运行。
- 决策：修正 comparator 后只读复用同一批 artifact 做逐 rank tensor 验证；不重写或删除
  attempt 01 的失败 marker。

### 2026-08-07：H200 attempt 01 artifact 只读重验

- 命令：CPU-only Job `minwm-async-a2a-quality-reader-20260807-01` 检出
  `ca3cf1dbfa9787d07ef4b68f50ef041e33f2426c`，在原 PVC 上执行
  `validate_async_a2a_quality.py --root <attempt-01>/async-a2a-quality --case-id
  00_forward_080_pottery_720p --sp-degrees 2 4 --output
  <root>/quality-validation-ca3cf1dbfa.json`；没有重新执行 GPU 推理。
- 正确性：reader Job `Complete 1/1`。SP2 的 rank0/rank1、SP4 的 rank0-rank3 各逐项
  比较 `69` 个 baseline tensor probe，全部 shape/dtype 相同且 bitwise exact；两种 SP 的
  视频 bitwise 状态也均为 `true`。candidate-only 的跨请求 `_001` probes 单列，不作为
  baseline 缺失或数值差异。
- 结论：H200 eager 完整模型的 SP2/SP4 720p 5s parity 与候选连续 10 请求稳定性通过。
  CUDA Graph 当前生产 benchmark 未启用；capture-safe PyNCCL 由前述 H200 standalone
  graph capture + 3 replay 合同覆盖。正确性门通过不代表性能或 overlap 门通过。

### 2026-08-07：H200 SP2 profiler-off 四位置预检

- 假设：若 QK A2A 与 V projection 的窗口足以抵消第二个 collective，短 ABBA 预检应至少
  显示 DiT/chunk wall 同向，不必等待正式 6+6 样本就可排除明显失败方案。
- 命令与资源：Job `minwm-async-a2a-perf-preflight-h200-20260807-01`，固定节点
  `i-06888dc1ca88547e1`、4 张 H200、SP2、1248×704、ProcessGroupNCCL、
  `SGLANG_REALTIME_TRACE_SYNC_CUDA=0`，顺序 `candidate baseline baseline candidate`；每个位置
  重启同配置 server，预热 5 chunks 后测 20 chunks。SGLang
  `59fd1050805e87966e9d3fec65601f6276078a1d`，minWM `2efc6485f65e8fcab506665efde79bc41406385e`，
  PyTorch `2.11.0+cu130`、CUDA `13.0`、NCCL `2.28.9`。结果路径
  `/results/attempts/minwm-async-a2a-perf-preflight-h200-20260807-01-krsh9/minwm-async-a2a-perf-preflight-h200-20260807-01/async-a2a-abba`，PVC
  `minwm-async-a2a-perf-results-20260807`。
- 正确性：该轮只做已过完整质量门候选的 profiler-off 性能预检；每个位置 20/20 chunk
  latency 完整，无 hang。
- wall/FPS：baseline 两位置 Client FPS 为 `13.0718, 12.9135`，median `12.9927`；candidate
  为 `12.6935, 12.8333`，median `12.7634`，即 `-1.765%`。candidate DiT wall median
  `744.332 ms` 对 baseline `712.588 ms`，性能 `-4.455%`；scheduler chunk wall median
  `1303.675 ms` 对 `1258.350 ms`，性能 `-3.602%`。两 lane Client FPS CV 分别
  `0.861%` 与 `0.774%`。
- Nsight：本轮严格关闭 profiler，不能说明是否已有部分 QK/V overlap；下一步使用精确外层
  chunk NVTX 窗口采 baseline/candidate 同机 trace，分解第二个 collective 与实际隐藏量。
- 结论：候选 1 的当前 ProcessGroup 实现明确低于 `+3%` 保留门槛，短预检即为负结果；不把
  它扩成正式 6+6 性能验收。保留代码开关用于机制 trace 和下一轮定向修改，默认继续关闭。

### 2026-08-07：H200 SP2 精确稳态 Nsight 机制验收

- 假设：profiler-off 负收益可能仍包含可保留的真实 QK-A2A/V-GEMM overlap；若其 exposed
  time 或 A2A+idle 下降达到门槛，可继续优化第二个 collective 的固定成本。反之应淘汰整个
  split-collective 形态，而不是只调 API。
- 命令与资源：Job `minwm-async-a2a-nsys-h200-20260807-01`，manifest
  `benchmark/minwm_realtime_parity/k8s/minwm_async_a2a_nsys_h200_20260807.yaml`，固定 H200
  节点 `i-06888dc1ca88547e1` 的 4 GPU（活跃 SP2 rank 为 GPU0/1），SGLang
  `9b8a6ee787883a79b92cab61883c5dd0a2ff10d2`，minWM
  `2efc6485f65e8fcab506665efde79bc41406385e`，PyTorch `2.11.0+cu130`、CUDA `13.0`、
  NCCL `2.28.9`、Nsight Systems `2026.4.1.191`。同一 Job 先 baseline 后 candidate；每 lane
  在 profiler 外完成 20 chunks 预条件，再以 `nsys launch/start/stop` 只捕获 1 个 discard +
  10 个稳态 chunk。固定 1248×704、seed 42、SP2、ProcessGroup input backend、output async
  关闭、`trace=cuda,nvtx`、`trace-fork-before-exec=true`、`cuda-graph-trace=node`、
  `gpu-metrics-devices=all@10kHz`、`SGLANG_REALTIME_TRACE_SYNC_CUDA=0`，无 torch profiler。
  复现入口：

  ```bash
  kubectl --context codex-minwm-test-phx2 apply \
    -f benchmark/minwm_realtime_parity/k8s/minwm_async_a2a_nsys_h200_20260807.yaml
  ```

- 结果：Job `Complete 1/1`。PVC `minwm-async-a2a-perf-results-20260807`，根目录
  `/results/attempts/minwm-async-a2a-nsys-h200-20260807-01-nlkgg/minwm-async-a2a-nsys-h200-20260807-01/async-a2a-nsys`；
  原始 trace 为 `baseline/baseline.nsys-rep` 与 `candidate/candidate.nsys-rep`，SQLite、
  `profile-client.json`、`a2a-metrics.json`、`comparison.json` 均同目录保留。修正 SP2 IPC
  output 识别后，以 `3ed01a42a3` 的 CPU-only Job
  `minwm-async-a2a-nsys-reanalyze-20260807-01` 只读重放两份 SQLite，新证据为
  `baseline/a2a-metrics-ipc-output.json`、`candidate/a2a-metrics-ipc-output.json`、
  `comparison-ipc-output.json`、`ipc-output-summary.json`；没有重跑 GPU。
- input A2A：关键 rank 每 chunk 中位数从同步 packed baseline 的 `68.631 ms`
  （p10/p90 `27.503/122.300`）增到候选总 A2A `172.551 ms`
  （`79.182/329.463`），其中候选 QK/V 分别为 `110.663/58.627 ms`。候选真实
  compute/communication 交集为 `1.630 ms`（`0.645/2.717`），10/10 chunk 非零，但 QK
  overlap ratio 中位仅 `1.646%`；launch-start→wait-start 与 launch-end→wait-start 的
  chunk×rank 中位分别为 `0.713/0.511 ms`，wait CPU range 中位 `0.0418 ms`。最终 input
  exposed kernel `68.631→170.398 ms`，是 **增加 `148.282%`**；input A2A+邻接 idle
  `70.595→178.174 ms`，增加 `152.391%`，远离 `-20%/-10%` 两个通过门槛。
- output reverse A2A：SP2 同步路径走 IPC，而不是 NCCL。每个稳态窗精确识别 3000 个
  output range，含 6000 个 `elementwise_kernel` 数据 copy、3000 个
  `bump_signal_kernel` 和 3000 个 `spin_wait_kernel`。虽然候选没有开启 async output，输入
  阶段造成的 rank skew 仍使 output IPC transport 关键 rank 中位 `56.160→95.966 ms`
  （增加约 `70.88%`），output+依赖边界 `56.998→97.385 ms`（增加约 `70.86%`）；全窗
  spin-wait 总时长 `569.623→903.204 ms`。
- kernel/利用率/同步：每 chunk 两 rank CUDA kernel `34908→35208`；每 rank/chunk launch
  `17454→17604`，NCCL kernel 全窗 `6060→9060`，即拆分后每 rank/chunk恰好多 150 个
  NCCL kernel。SM Active mean `60.822%→56.136%`，Tensor Active mean
  `28.113%→25.689%`，DRAM read throughput mean `8.209%→7.531%`。检测到的
  `cudaStreamSynchronize` 数 `1550→1550`，没有新增全局同步 API；其累计 duration
  `6336.560→6504.648 ms`，候选仍因 stream/rank 等待变慢。
- wall/FPS：Nsight 捕获窗客户端为 `12.505→11.423 FPS`，只用于与 trace 对齐，明确
  `profiler_wall_headline_eligible=false`，不进入最终 FPS。最终 wall 仍以同机 profiler-off
  预检的 `-1.765%` 为准。
- 结论：机制验收 **FAIL**。真实 overlap 已被证明，但隐藏量远小于新增 collective 及其引发
  的跨 rank/反向 IPC wait；不是“API 不够异步”，而是本依赖图下的单层合法窗口不足。停止
  扩测 QK/V split ProcessGroup 候选，默认开关保持关闭；下一候选必须保持单 packed input
  collective，并从跨 request/chunk 的独立 GPU 工作或不增加 collective 的调度方式寻找窗口。

### 2026-08-07：scheduler/cache 依赖审计与候选 2——split IPC

- 假设 A（跨 chunk）：若下一 chunk 的 GPU 工作能在本 chunk A2A 期间提前执行，可保留单次
  packed collective 并获得更大的独立窗口。
- 依赖审计：Realtime WebSocket 入口用 `_ACTIVE_SESSION_IDS` 拒绝第二个并发 session；
  `_generate_loop` 必须 `await process_generation_batch(...)` 完整返回后才能提交下一 chunk。
  GPU worker 的 `RealtimeSessionCache(max_sessions=1)` 和 scheduler 的
  `_can_dynamic_batch=False` 又分别禁止第二份 session cache 与 realtime 动态 batching。
  单个 MinWM chunk 内是 4 次逐步依赖前一 latent 的 DMD transformer forward，随后第 5 次
  clean-cache forward 才把最终 K/V 写为下一 chunk 的历史；`current_chunk_start_frame` 也只在
  clean-cache 完成后推进。因此下一 chunk 的 attention、cache/RoPE 或 block 计算不能合法跨过
  当前 clean-cache commit。当前 scheduler 不存在可直接接入双缓冲的第二份独立 GPU 工作。
- 决策 A：不做只增加 host queue 的“跨 chunk 双缓冲”。真正的跨请求方案必须同时增加
  WebSocket admission、每 session 独立 cache/workspace、scheduler in-flight capacity，并把
  whole-DiT 原子 forward 重构成按 block 交错的两请求 wavefront；只做 request batching 会把
  payload 合进同一 collective，仍不会产生 compute/communication overlap。该方案作为候选 3
  的可执行设计保留，但在验证更小的 transport 候选前不直接扩大改动面。
- 假设 B（split IPC）：SP2 的 IPC 协议用 peer staging copy、单 CTA `spin_wait_kernel` 和
  GPU-side sequence counter，可能比两次 ProcessGroupNCCL 更少占用 SM/固定开销，使 QK IPC
  与 V GEMM 真正并行。同步 input baseline 仍是一次 packed ProcessGroup A2A；因此这是新
  transport 产品候选，不是“同 transport 复测”，最终仍必须直接对生产同步 baseline 判定。
  SP4 不支持 IPC，必须明确 fallback 到已证负收益的 ProcessGroup 路径，故候选即使 SP2 通过，
  SP4 也只能先满足正确性、不能宣称同机制性能通过。
- 修改/正确性：GPU 合同新增 SP2 连续 12 轮 split IPC input/output、双 slot 复用，以及预热后
  IPC CUDA Graph capture + 3 次 replay；每轮和同步 packed-QKV/output 做 bitwise 比较。本地
  `py_compile` 已通过，真 GPU 合同尚待运行；合同通过前不启动整模型 A/B。
- wall/FPS、Nsight、结论：尚未运行，候选 2 **未验收**。下一顺序为 H200 小合同 → 同节点
  profiler-off 四位置预检；只有 Client FPS/DiT/chunk wall 同向且接近门槛，才采精确 Nsight。

### 2026-08-07：候选 1——QK A2A 与 V projection 重叠

- 假设：完整 packed-QKV A2A 前没有可移动工作，但 Q/K/V projection 相互独立；将 wire
  payload 拆成 QK 与 V 后，可以用 V GEMM 隐藏 QK A2A 的 exposed time，并克服新增一次
  collective launch 的代价。
- 修改：新增显式 comm stream、ready/done/wait-enter/wait-exit CUDA events、无
  device-wide synchronize 的 begin/consume handle；workspace 对 input-QK、input-V、output
  分角色使用两个 slot、generation/busy 检查与 `record_stream` lifetime；新增 QK/V Triton
  peer-first pack。ProcessGroup eager launch 使用 `async_op=True`，只在 comm stream 上用
  CUDA Work dependency 建立 completion event，consumer compute stream 仅等待该 event；
  PyNCCL/IPC 直接在 comm stream enqueue。异常只能在 collective 未发出前 fallback；若 QK
  已发出而 V launch 失败，先 retire QK 再报错。
- 正确性：本地执行完整
  `PYTHONPATH=python TORCHDYNAMO_DISABLE=1 /opt/homebrew/bin/python3.11 -m pytest -q python/sglang/multimodal_gen/test/unit/test_ipc_a2a_lifecycle.py python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py`，
  结果 `123 passed`。新增 CPU 合同证明 SP2/SP4 split pack 与同步 packed-QKV bitwise exact，
  并覆盖 capture backend fallback。新增真 GPU 入口
  `test_minwm_async_a2a_gpu.py`，待 B200 运行 process-group 连续 12 轮、双缓冲、output parity
  与 PyNCCL graph 三次 replay。
- wall/FPS、Nsight：尚未运行，因此候选 1 **尚未验收**。
- 结论：进入真机小规模合同测试；只有其通过才跑 checkpoint parity 与 A/B。

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

### 2026-08-07：H200 合同镜像不是可直接运行测试的完整开发环境

- 原认知：固定 MinWM 训练镜像可直接 import SGLang 测试入口。
- 实际证据：前三次不可覆盖任务分别在 GPU 测试前因缺少 `orjson`（`-01`）、缺少
  `IPython`（`-02`），以及测试继承重型 `CustomTestCase` 后触发镜像内不兼容的
  `transformers.PreTrainedConfig` import（`-03`）失败；这些任务与日志均保留。
- 根因：镜像面向训练/推理，不包含 SGLang test collection 的全部可选 Web 依赖；GPU
  transport 测试本身不需要 `CustomTestCase` 的模型/transformers 依赖。
- 决策：只补齐测试 import 所需最小依赖，并把独立 GPU 合同改为标准
  `unittest.TestCase`；未替换镜像的 PyTorch/CUDA/NCCL 栈。第四次运行才进入真实 CUDA
  合同并通过，前三次不能计入 GPU 正确性样本。

### 2026-08-07：10 请求稳定性使候选 probe 集合严格大于单请求 baseline

- 原认知：baseline 单请求与候选 10 请求的 parity dump 文件集合应完全相等。
- 实际证据：SP2/SP4 视频均 bitwise exact；tensor comparator 显示 baseline 的所有文件都在
  候选中，但候选额外包含第二次请求首次触发的
  `cross_k_output_001.pt`、`cross_norm_k_output_001.pt`、`cross_v_output_001.pt`。
- 根因：cross-attention K/V 在单请求内被 cache，baseline 只调用一次；候选连续请求为验证
  跨请求污染，第二个请求产生合法的 `_001` hook 输出。文件数差异不是数值或 buffer 污染。
- 决策：合同改成“candidate 必须覆盖 baseline 全部 probe；逐个比较 baseline probe
  bitwise；candidate-only probe 单独报告”。原始 attempt 01 仍标为失败，并通过新 reader 对
  原 artifact 重验，避免为了修 comparator 重跑 GPU 推理。

### 2026-08-07：B300 有 Ready 节点，但盘点时没有空闲 GPU

- 原认知：`aws03-usw2` 的 8 个 Ready `p6-b300.48xlarge` 节点可能可立即用于正式验收。
- 实际证据：只读盘点显示 64 张 B300 均已被四个双节点训练任务占用：
  `w22-s0-16000-mix-refine-seg-0805-long-b6828eb`、
  `wan22-5b-stage3-dmd-44-0807-df5cf37ac5fe8`、
  `wan22-5b-varlen-pure-product-720p-ct1000-0806-a67b9ae`、
  `wan22-5b-varlen-stage2-cd-23-0807-0933b5d7331`。
- 根因：容量块节点已满载，不是调度器或 manifest 错误。
- 决策：不删除、不抢占他人任务；先用空闲 H200 完成诊断与淘汰明显失败方案，正式
  PASS/FAIL 仍等待 B200/B300 同节点资源，且不会把 H200 结果冒充主验收。

### 2026-08-07：真实 V projection 窗口未抵消 split collective 成本

- 原认知：V projection GEMM 的独立窗口可能足以隐藏 QK A2A，并抵消把一个 packed-QKV
  collective 拆为 QK 与 V 两个 collective 的启动和协议成本。
- 实际证据：H200 SP2 profiler-off 四位置预检中 Client FPS `-1.765%`、DiT wall
  `-4.455%`、scheduler chunk wall `-3.602%`；各 lane 变异低于 `1%`（chunk wall 的
  candidate CV 为 `2.319%`），方向稳定且三项关键指标一致变差。
- 实际证据补充：精确 Nsight 的 10/10 chunk 都有真实 overlap，但关键 rank 中位仅隐藏
  `1.630 ms`、QK overlap ratio 仅 `1.646%`；input exposed `68.631→170.398 ms`，output IPC
  exposed stage `56.998→97.385 ms`，SM/Tensor Active 同时下降。候选没有新增
  synchronize API，却每 rank/chunk 多 150 个 NCCL kernel/launch。
- 根因：V GEMM 的合法窗口存在但 GPU 时间交集极小；把一个 packed collective 拆成 QK/V
  两个后，第二次协议/launch 和 rank 到达时差主导关键路径，并把偏斜传递到 reverse IPC 的
  `spin_wait_kernel`。因此简单减少 host wait、改 event 或声称 `async_op=True` 都不能修复。
- 决策：正式淘汰 QK/V split ProcessGroup 产品形态，不运行其 6+6 主验收，也不宣称 async
  已获益。保留默认关闭的诊断开关与完整负证据；下一设计先验证 scheduler 是否能以隔离 cache
  和双缓冲承载跨 request/chunk 计算，同时保持单 packed collective。

### 2026-08-07：曾误把 MinWM 同步 input packed A2A 识别为 IPC

- 原认知：看到通用 attention 的 `_ipc_input_a2a_qkv` 和 SP2 trace 中大量 IPC copy/signal/spin
  kernel 后，一度判断上一轮同步 input baseline 也命中了 IPC，而 candidate 被单独固定为
  ProcessGroupNCCL。
- 实际证据：沿 MinWM 自定义 `MinWMCausalSelfAttention.forward` 追到
  `_usp_input_all_to_all_qkv`，uniform sequence-shard input 明确调用一次
  `torch.distributed.all_to_all_single`；通用 attention 的 `_ipc_input_a2a_qkv` 不在该调用链。
  IPC trace 来自 `_usp_output_all_to_all` 的 reverse output。上一轮新增的每 rank/chunk 150 个
  NCCL kernel 对应把一次 packed input 拆为 QK/V 两次，而不是 baseline transport 不一致。
- 根因：只按 transport helper 的全仓存在性和 IPC kernel 总量判断，没有先限定 MinWM 自定义
  causal attention 的精确调用链。
- 决策：保留候选 1 的公平 PG A/B 负结论；候选 2 明确标为“split IPC 对生产 packed PG”的
  新 transport 比较。以后 input/output 分开按 NVTX 调用点与 kernel 协议双重归因。

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

### 2026-08-07：upstream packed-QKV 提交不能原样 cherry-pick 到 MinWM

- 原认知：三项 upstream 通信提交可按顺序无语义冲突地 cherry-pick。
- 实际证据：`a5888c9` 定义的 `_usp_input_all_to_all_qkv` 返回 contiguous `(q,k,v)`，而
  MinWM 已有同名 API，返回共享 receive buffer 上的 packed tensor 并由 causal attention
  `chunk(3)`；MinWM 还已有跨 30 blocks 共享的 `qkv_send/qkv_recv/attention_recv` workspace
  和专用 `ulysses_qkv_pack.py`。原样合并会破坏已通过 parity 的调用合同，并复活当前树已
  删除的旧 kernel/Minimax 测试路径。
- 根因：MinWM 独立集成已提前实现了同一优化族，但 API 与 upstream 通用路径演化不同。
- 决策：中止未完成的 `a5888c9` cherry-pick，保留 MinWM 等价实现；只引入实际缺失的 IPC
  与 capture-safe PyNCCL。后续测试显式证明 packed round-trip 与 buffer reuse，不把“提交
  存在”替代行为验证。

### 2026-08-07：完整 packed-QKV 的层内窗口实际为零

- 原认知：也许可在现有 packed-QKV `launch→wait` 之间移动 cache metadata、RoPE 或其他
  GPU 工作，而不改变 collective 结构。
- 实际证据：当前 model forward 已在 block loop 前预计算 cache plan 与 RoPE；block 内
  cache ownership update、Q/K RoPE 和 attention 全部消费 input A2A 结果，reverse A2A 后的
  output projection/residual/FFN 也全部消费其结果。单请求中不存在可合法插入完整 input 或
  output A2A 窗口的 GPU kernel。
- 根因：MinWM 每个 block 串行持有单请求 causal cache，且 Ulysses 在 A2A 前后改变 sequence/
  head ownership。
- 决策：不提交只有 async API 的伪优化；input 改为 QK/V split 以创造 V projection 窗口，
  output 仅提供可量化 begin/consume 并默认关闭。若真机净收益不足，再评估跨 request/chunk
  cache 隔离流水，而不是错误越过 residual/cache 边界。

### 2026-08-07：通信基线移植与本地回归

- 假设：IPC lifecycle/peer-device 修复和 capture-safe PyNCCL 可独立移植，且不会改变
  MinWM packed-QKV 的数值合同。
- 修改：整合上表四个 upstream 通信提交；`_ipc_ready_group()` 在 layout-only、未初始化
  process group 的 CPU 测试中显式回退，而不是因伪造的 SP size 触发断言。
- 正确性：
  `PYTHONPATH=python TORCHDYNAMO_DISABLE=1 /opt/homebrew/bin/python3.11 -m pytest -q python/sglang/multimodal_gen/test/unit/test_ipc_a2a_lifecycle.py python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py -k 'ipc_a2a_lifecycle or ulysses or fused_post_a2a'`
  结果为 `15 passed, 105 deselected`。
- wall/FPS、Nsight：尚未启动真机；此处仅建立同步行为基线，不能用于性能结论。
- 结论：通信依赖可作为 async 候选的共同基线；下一步先实现可观测 begin/wait，再只在
  input A2A 上引入合法的 V projection overlap。

### 2026-08-07：已有 post-A2A fusion 对 SP2 无客户端收益

- 原认知：减少 post-A2A 后的 pointwise/copy launch，且 DiT wall 下降，可能自然传导为客户端
  FPS 提升。
- 实际证据：已有 H200 profiler-off SP2 20+200 结果中，DiT wall 约下降 `3.31%`，但
  Client FPS `-0.249%`、chunk wall `-0.022%`；SP4 独立 ABBA 才达到约 `+3.56%`。
- 根因：SP2 scheduler 未归类时间增加，抵消 DiT 缩短；Nsight 中 NCCL residence 变短也
  不能证明 exposed A2A 被隐藏。
- 决策：post fusion 可留作默认关闭的 parity-passed 基线能力，但不算本任务异步收益；主验收
  仍以 SP2 B200/B300 profiler-off FPS 与 exposed-time 证据为准。

## 当前 before/after、复现与剩余风险

以下为候选 1 的 H200 诊断结论，不是 B200/B300 最终硬件验收：

| 指标（SP2，1248×704） | 同步 baseline | QK/V split candidate | 变化/判定 |
| --- | ---: | ---: | ---: |
| profiler-off Client FPS median | 12.9927 | 12.7634 | `-1.765%`，FAIL（门槛 `+3%`） |
| profiler-off DiT wall median | 712.588 ms | 744.332 ms | 性能 `-4.455%` |
| profiler-off chunk wall median | 1258.350 ms | 1303.675 ms | 性能 `-3.602%` |
| input exposed kernel median | 68.631 ms | 170.398 ms | exposed 增加 `148.282%`，FAIL |
| input A2A+idle median | 70.595 ms | 178.174 ms | 关键路径增加 `152.391%`，FAIL |
| output IPC transport median | 56.160 ms | 95.966 ms | 增加约 `70.88%` |
| output IPC+依赖边界 median | 56.998 ms | 97.385 ms | 增加约 `70.86%` |
| compute/comm overlap | 0 | 1.630 ms，10/10 chunk | overlap 真实但不足 |
| SM Active / Tensor Active mean | 60.822% / 28.113% | 56.136% / 25.689% | 双双下降 |
| 同步 API count | 1550 | 1550 | 无新增，但不构成通过 |
| 720p 5s parity / 连续请求 | baseline | SP2/SP4 bitwise；候选连续 10 请求 | H200 eager 正确性通过 |

回滚开关为 `MINWM_ASYNC_A2A=0`（默认）；`MINWM_ASYNC_A2A_OUTPUT=0` 继续保持。候选 1
不进入 B200/B300 正式性能验收。剩余风险与下一步：

- B300 64 张卡在最近盘点时仍被四个他人训练任务占满；不抢占。最终 PASS 必须在同节点
  B200/B300 上重做 profiler-off ABBA/BAAB 各至少 5 样本与相同 Nsight 稳态窗。
- 单请求 block 内完整 input/output A2A 都没有合法独立工作。下一步必须先审计 realtime
  scheduler 的在途 request/chunk 数、causal cache/session ownership 和输出顺序，证明能安全
  放入另一请求/下一 chunk 的 GPU work，再实现跨请求双缓冲；不能只把现有 begin/wait 拉开。
- SP2 reverse IPC 对 rank skew 很敏感；后续分析器已同时识别 NCCL 与
  copy/signal/spin-wait IPC 协议。任何下一候选都要分别报告 input 与 output，不得再把 IPC
  output 误报为 0。
- 480p 边界对照、SP4 次验收、生产 CUDA Graph 完整模型 10 请求稳定性及最终 trace
  截图/统计仍待下一可行候选与 B200/B300 资源；当前整体状态仍为 **未验收**。
