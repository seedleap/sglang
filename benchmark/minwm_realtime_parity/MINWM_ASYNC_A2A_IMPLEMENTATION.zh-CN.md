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
| 候选提交 | 核心实现 `dad5de09f1b644074cb3d977fdad11ffd18772bf`；GPU 测试隔离修复 `0a975bac14e20660e6a31744bf864c8d51d99aff`。完整模型任务继续使用每次已推送的不可变 SHA，并在对应实验条目固定 |
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

## 最终 before/after、复现与剩余风险

待完成。最终将列出每个硬件/SP/分辨率的 before/after median、p10/p90、变异、parity、
稳定性、trace 统计、完整命令、结果路径、回滚开关和未解决风险，并给出明确 PASS/FAIL。
