# MinWM 5.3 融合小算子 S0：统一测量与 H200 基线

## 状态与范围

本文只覆盖 S0：冻结基线、定义机器可读测量契约、提供 CPU-only 测试，并采集 H200 SP2/SP4 基线。S1–S5 的算子优化不在本 PR 中。

- 仓库：`seedleap/sglang`
- PR：[#19](https://github.com/seedleap/sglang/pull/19)
- 分支：`codex/minwm-fused-ops-s0`
- 测量契约最小 commit：`30cb16708fc768adb063c31c2f1a21eac5a016d2`
- 真机 runner 与口径修正起点：`411d9b9ec40b2fca2a7d85e17a05c11a4723750e`
- latency count canonical：`b9240233b2438829cbd72ee3dfbc1d37ed675560`
- exact-window / trace-drain canonical：`e4d6d67c76`
- 真机 Nsight 2026.4 schema canonical：`401e4ec8a1`
- 跨进程 component CUDA timing relay canonical：`839f312c3b4622c8e04c5c76620d22d6c2497fa0`
- Nsight SQLite export/stats 顺序 canonical：`f1b047942d86715297ca79a9a3c5e7fae1e4a306`
- Nsight active CUDA→PerfWorks target 映射 canonical：`900b5f279b65b2afcfbe6cc9b36cfa4496b41bc3`
- API 边界归属与有界 GPU metrics 聚合 canonical：`d5b25227d4487d113e62c86a0fb572a62d6bcc5b`
- base：`main`。`main` 的 `9a9dc59cd1` 已完整包含 MinWM realtime API、causal Ulysses、Parallel VAE 和历史 benchmark；`codex/minwm-realtime-api` 还叠加了量化与性能实验，不适合作为 S0 的独立 review base。

## 假设与预期

### 固定 workload

| 项 | 固定值 | 说明 |
| --- | --- | --- |
| 模型 | MinWM 5B | 不用 LingBot 14B 代替 |
| checkpoint | `global_step_003200/ema_student/model.pt` | step-3200；记录 VersionId、ETag、CRC64、字节数与 SHA-256 |
| 分辨率 | 1248×704 | 当前 MinWM 有效 720p tier |
| 精度 | BF16 | 不混入 FP8/NVFP4 |
| fast lane | 开启 | packed deterministic causal attention + Parallel VAE |
| 每 chunk | 4 latent / 16 pixel frames | 每个 chunk 执行 4 次 DMD forward 和 1 次 clean-cache forward |
| KV | 45 latent frames | 固定稳态窗口，避免 220 chunk full-history 导致形状和显存随时间增长 |
| profiler-off | 20 warmup + 200 measured | 每个 SP 至少两次重复 |
| profiler-on | 20 个前置 warmup + 1 个 capture 内丢弃 + 10 个稳定 chunk | Nsight 只抓稳定窗口 |
| 主验收 | SP2 | SP4 复验 |

预期：Client FPS、Scheduler FPS、DiT wall、VAE wall 的重复 CV 默认不超过 3%。超过时 runner 自动补第三次，并结合 GPU 时钟、P-state、功耗、温度和利用率解释；不能用旧的 2026-08-03 数据冒充本轮结果。

### 预期性能关系

本轮冻结的是“优化前基线”，不是给 S1–S5 预设收益。根据仓库历史 H200 结果，预期 SP4 的 Parallel VAE 比 SP2 更短，而 DiT 未必继续随 SP degree 明显缩短。该历史关系仅用于发现数量级错误，不作为本轮实测值。

## 测量契约

### 文件和入口

- JSON Schema：`measurement_schema.json`
- Python 构造与校验：`measurement.py`
- profiler-off / profiler-on WebSocket 客户端：`benchmark_realtime_throughput.py`
- Nsight SQLite 解析：`nsys_metrics.py`
- 校验、Nsight merge、重复 CV 汇总：`measurement_tool.py`
- H200 编排：`run_s0_measurement.sh`
- H200 清单：`k8s/minwm_s0_fusedops_h200_20260807.yaml`
- CPU-only 测试：`test_measurement.py`

### 四种时间域不能互换

| 时间域 | 定义 | 用途 |
| --- | --- | --- |
| Client wall | 客户端相邻完整 chunk payload 的单调时钟间隔 | headline Client FPS；包括服务端、传输和接收开销 |
| Scheduler wall | 服务端包住 scheduler request 的单调时钟 | headline Scheduler FPS 和每 chunk wall；不含客户端传输 |
| Stage wall | 服务端 DiT/VAE pipeline stage 的单调时钟；stage 末尾 CUDA event 同步，因此包括该 stage 排队的 GPU 工作 | profiler-off DiT/VAE wall |
| CUDA time | CUDA event 或 Nsight device 时间 | profiler-on DiT/VAE CUDA；不包括 CPU gap、传输和输出写入 |

`mode=profiler_off` 时，`measurement_contract.headline_eligible=true`。`mode=profiler_on` 时该值必须是 `false`，Nsight 下观测到的 wall/FPS 只保存在 `observed_wall_with_profiler_overhead`，不得写进 headline。

所有 `available` 的 wall/CUDA latency summary 都必须显式包含 `value.count`。JSON Schema 保证字段存在且为正整数，自定义 validator 进一步要求 `value.count == workload.measured_chunks`；缺 count 或错 count 的产物即使 FPS 看起来正常，也不能进入 A/B 基线。

### profiler-on 字段

Nsight 记录以下字段：

- DiT/VAE CUDA 时间；
- kernel 总数；
- CUDA API 总数与 kernel/graph launch API 数；
- `<10 us`、`10–<50 us`、`50–<100 us`、`>=100 us` 四个短 kernel 分桶；
- 按设备合并 kernel interval 后的 GPU kernel busy；
- SM Active、Tensor Active，以及硬件暴露时的 DRAM 指标。

计数字段同时保留 `captured_raw_total`、稳定窗口内 `raw_total` 和
`excluded_raw_total`。客户端给请求设置唯一 `trace_id`，服务端在
`process_generation_batch` 外发出包含 trace/request/chunk/role 的 outer NVTX
range。解析器必须恰好看到 discard index 0 一次、measured indices 1–10 各一次，
且 request id 唯一、时间顺序一致、区间不重叠。CUDA runtime API 与 launch
是离散计数，按 event `start_ns` 归属唯一半开 measured range `[start,end)`：
起点在 range 内即只计一次，即使调用跨过 range 末端；起点在所有 range 外就不计，
即使 duration 与 range 有交集。输出保留 `boundary_spanning_count`、按 start
纳入/排除数，以及 raw API name、process、start/end、owner range 和 overlap ranges
证据。Kernel count、短 kernel duration 分桶和 busy 仍要求 kernel 完全落入单个
range；GPU samples 用同一 range union 的 `[start,end)` 过滤。任何 duration/busy
都没有因 API start 规则扩大，range 间隙、discard、outside 和 sibling trace 均不计入。

Kernel 与短 kernel 分桶按稳定窗口内 `deviceId` 展开；GPU kernel busy 的分母
是 10 个 range 时长之和，而不是首尾 kernel span。API/launch 提供
`total_per_chunk`；只有 runtime 进程、kernel `globalPid`、device→process 和
active rank 完全一致时才提供 `per_rank_per_chunk`。2026.4 真机 runtime 表使用
`globalTid`，解析器清除低 24-bit thread 部分后必须命中 `PROCESSES.globalPid`，
再与 kernel `globalPid` 交叉验证。所有正式归一化的
`capture_scope=union of exact measured outer chunk NVTX ranges`。

GPU metrics 不使用固定 metric id。SM Active 只接受规范化后的精确别名
`SM Active` 和真机 `SMs Active [Throughput %]`；`SM Issue`、
`Unallocated Warps in Active SMs`、Tensor Active 等名称不会因宽泛 substring
被误命中。解析器仍保留 raw metric name、全 capture/stable sample count、
nonzero/min/p50/p95/max、每 device/typeId×chunk 覆盖。

GPU_METRICS 解析只流式扫描选中的 SM/Tensor/DRAM metricId，用
metric×typeId×chunk 计数器和 0–100 native value 直方图计算统计量，不再把 8 卡
全表样本装入 Python list。`aggregation_mode` 写入 schema；真实 1.47 GB SQLite
上与旧解析器的 count/min/max/mean/p50/device/chunk/raw name 逐字段相等，新增 p95
也与直接从 SQLite 原始值列表重算相等。

整机隔离 Job 用 `--gpu-metrics-devices=all` 采集 allocated 8 张卡，再按 Nsight
导出的真实映射筛选 active SP 卡：stable-window kernel 的
`CUPTI_ACTIVITY_KIND_KERNEL.deviceId` 是 CUDA logical device；
`TARGET_INFO_GPU.cuDevice -> pwGpuId` 给出 PerfWorks ID；GPU metrics composite
`typeId & 0xFF` 得到同一个 GPU ID。正式值只汇总这 SP2/SP4 个 active typeId，
并记录 `collected_target_count=8`、`active_target_count=2/4`、
`allocated_target_count=8` 和逐卡 bus/UUID 映射。该位域定义及 0–100 整数百分比
见 NVIDIA [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics)。

权限不足、target 覆盖不完整或指标不存在时输出：

```json
{
  "status": "unavailable",
  "reason": "permission_denied",
  "evidence": "Nsight status/start 的原始错误与缺失表信息"
}
```

不允许用空值、0 或推测值代替不可得指标。正式
`--require-complete-stable-nsys` 还要求 SM Active、Tensor Active 都 available，
且每个 active CUDA device/typeId × 10 chunks 均有样本；collected target 必须覆盖
全部 8 张 allocated GPU。若 stable-window CUDA kernel 覆盖所有 active device，
但 SM Active 的所有 active 样本仍为 0，解析器以
`gpu_metric_all_zero_under_kernel_load` fail closed。无 GPU metrics 的 fallback report
只能作为 invalid lane 诊断。DRAM 仅在 Nsight raw metric names 确实未暴露匹配项
时允许 `metric_not_exposed`。

### Nsight 稳态窗口

runner 使用：

```text
nsys launch --trace=cuda,nvtx --trace-fork-before-exec=true --cuda-graph-trace=node
20 chunk 前置 warmup（未 start）
nsys start --gpu-metrics-devices=all --gpu-metrics-frequency=10000 --sample=none
1 chunk 丢弃 + 10 chunk measured
nsys stop
```

如果带 GPU metrics 的 `nsys start` 失败，runner 将错误追加到
`nsys-capture-status.log`，再只用 CUDA/NVTX 重试以保留诊断 report；由于正式
闸门要求 SM/Tensor available，该 lane 随后必须失败并写 lane marker，不能进入
baseline。`SGLANG_DIFFUSION_TORCH_PROFILER_DIR` 必须未设置；Nsight 与
torch.profiler 不同时运行。

## 实际结果

`-04` 只保留已独立验证的 SP2 profiler-off；其 profiler-on lane 已标 invalid。
`-05` 证明 exact-window outer marker 与 GPU metrics 能启动，但暴露出 worker
component timing 不能经进程内 sink 到达 API 的结构性缺口，其 SP2 profiler-on lane
同样只保留 invalid 诊断。`-06` 验证跨进程 relay 生效，但在导出后处理阶段发现
`nsys stats` 会隐式创建 SQLite；严格只读复查又发现旧 runner 采错 PerfWorks target，
该 lane 仍只保留 invalid 诊断。`-07` 尚未启动即按审计要求精确删除控制对象，
没有产生新 artifact。`-08` 首次拿到 exact-window、8-target GPU metrics 和完整
component CUDA trace，但旧严格 containment 门因一条跨 marker 边界的
`cudaEventQuery_v3020` 正确拒绝该 lane；原产物继续标 invalid。`d5b25227d4`
已在该 invalid SQLite 上离线严格通过，正式 profiler-on 与 SP4 改由新名
`minwm-s0-fusedops-h200-20260807-09` 补齐。
`-09` 已以 `d5b25227d4` 在 H200 整机隔离节点成功完成 SP2/SP4：六条正式
measurement record 均重新通过当前 validator，其中两条 profiler-on 使用
`--require-complete-stable-nsys` 严格校验；attempt 内没有 invalid marker。
旧 H200/B300 表以及 `-01/-02/-03` 失败诊断只用于背景与异常证据。

### 运行来源

| 项 | 实际值 |
| --- | --- |
| SGLang | SP2 profiler-off source=`b9240233b2`；正式 profiler-on/SP4 runner=`d5b25227d4487d113e62c86a0fb572a62d6bcc5b`；exact-window schema=`401e4ec8a1`；component relay=`839f312c3b`；active GPU metrics mapping=`900b5f279b` |
| MinWM | `2efc6485f65e8fcab506665efde79bc41406385e` |
| 镜像 | `minwm-training@sha256:bedc07ea...f5f2a` |
| GPU | NVIDIA H200；`gpu.count` 是 active 2/4 卡；`allocated_count=8` 是整机隔离预留 |
| kube context | `codex-minwm-test-phx2`；所有命令显式传 `--context`，未切换全局 current-context |
| region / zone | AWS `us-west-2` / `us-west-2-phx-2a` |
| NodePool | `minwm-test-phx2-p5e-spot`（共享的既有 NodePool，S0 未创建或删除） |
| 实例 | `p5e.48xlarge` Spot；正式 `-09` 节点 `i-01a57ab8567279852`；`-06/-08` 节点 `i-06888dc1ca88547e1` |
| 资源隔离 | Job 请求完整 8 GPU；不与 CUDA Graph 或 S1–S4 Job 共用 GPU 节点 |
| 正式 attempt | Job `minwm-s0-fusedops-h200-20260807-09`；Pod `minwm-s0-fusedops-h200-20260807-09-s9cc4`；`backoffLimit=0`；1/1 Complete |

### profiler-off 重复

所有 FPS 均来自 profiler-off；profiler-on 观测 FPS 不进入本表。

| SP | 重复 | Client FPS | Scheduler FPS | scheduler chunk wall mean | DiT wall mean | VAE wall mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 12.904769 | 12.917758 | 1264.635 ms | 747.187 ms | 419.249 ms |
| 2 | 2 | 12.884662 | 12.896310 | 1272.865 ms | 745.509 ms | 419.569 ms |
| 4 | 1 | 14.706324 | 14.723746 | 1170.100 ms | 737.319 ms | 231.422 ms |
| 4 | 2 | 15.187021 | 15.204717 | 1080.520 ms | 733.717 ms | 231.370 ms |

| SP | Client CV | Scheduler CV | DiT wall CV | VAE wall CV | 验收 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 0.110% | 0.118% | 0.159% | 0.054% | 通过（均 <3%） |
| 4 | 2.274% | 2.273% | 0.346% | 0.016% | 通过（四个主验收指标均 <3%） |

SP4 的非 headline `scheduler chunk wall` CV 为 5.629%（1170.100 vs
1080.520 ms），但 Client/Scheduler FPS 同时只波动约 2.27%，DiT/VAE wall 分别
为 0.346%/0.016%。因此主验收通过，同时保留该 CPU/排队口径噪声，不把它隐藏或
替换成 profiler-on 数据。

### profiler-on 稳态窗口

每个 SP 均恰好观察 discard index 0，并只用 measured indices 1–10 做归一化。
下表 wall 是 Nsight 下的诊断 wall，不能作为 headline；wall/CUDA 的 count 均为 10。

| SP | DiT wall / CUDA mean | VAE wall / CUDA mean | count |
| ---: | ---: | ---: | ---: |
| 2 | 732.160 / 731.669 ms | 441.010 / 440.307 ms | wall=10，CUDA=10 |
| 4 | 777.803 / 777.318 ms | 254.349 / 253.737 ms | wall=10，CUDA=10 |

计数是 exact 10-range union 内的稳定窗口总数；括号内为每 stable chunk。

| SP | kernels | CUDA APIs | launch APIs | <10 us | 10–<50 us | 50–<100 us | >=100 us | kernel busy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 346,080（34,608.0） | 910,753（91,075.3） | 346,080（34,608.0） | 184,276 | 121,169 | 11,946 | 28,689 | 76.539% |
| 4 | 692,020（69,202.0） | 1,849,328（184,932.8） | 692,020（69,202.0） | 495,006 | 130,280 | 18,742 | 47,992 | 68.271% |

| SP | SM Active | Tensor Active | DRAM | 权限/采集证据 |
| ---: | --- | --- | --- | --- |
| 2 | 61.973%（251,754 samples） | 28.607%（251,754） | 8.344%（251,754） | 8 targets collected；active CUDA 0/1 → pwGpuId 2/3；active=2、allocated=8；每 target×10 chunks 完整 |
| 4 | 37.935%（447,617 samples） | 16.074%（447,617） | 4.430%（447,617） | 8 targets collected；active CUDA 0–3 → pwGpuId 0–3；active=4、allocated=8；每 target×10 chunks 完整 |

原始名称分别是 `SMs Active [Throughput %]`、
`Tensor Active [Throughput %]`、`DRAM Read Bandwidth [Throughput %]`。
SP2/SP4 的 kernel、launch、CUDA API `boundary_spanning_count` 均为 0，kernel
`boundary_overlap_count` 也为 0；因此本轮正式计数没有依赖边界归属例外。

下表是 `-08/sp2/profiler-on` 的 **invalid capture 离线诊断**，只用于证明
`d5b25227d4` 的解析/闸门，不是正式 baseline，也不解除 lane marker：

| exact ranges | DiT CUDA | VAE CUDA | kernels | CUDA APIs | launch APIs | SM / Tensor / DRAM |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| indices 1–10 | 745.092 ms（count=10） | 443.259 ms（count=10） | 346,080 | 910,469 | 346,080 | 61.637% / 28.327% / 8.297% |

CUDA API capture 总数为 1,035,461，按 start 纳入 910,469、排除 124,992。
唯一 boundary-spanning event 是 `cudaEventQuery_v3020`：start=`9599599814`、
end=`9599602724`，起点在全部 measured ranges 外，尾部进入 chunk 7，故明确
排除；launch boundary count=0，launch count 与 kernel count 同为 346,080。

### SP2 stage trace 完整性与 S1 -07 只读对照

正式 S0 SP2 的 client contract 需要 1 discard + 10 measured，即每个 selector
的唯一 chunk indices 必须恰好为 0–10。`-09` server.log 的 source-specific 结果：

| selector | S0 `-09` | S1 `-07` 只读证据 |
| --- | --- | --- |
| DiT wall：`scheduler_result_metrics` | 11/11，indices 0–10，11 request IDs | 11/11 |
| VAE wall：`scheduler_result_metrics` | 11/11，indices 0–10，11 request IDs | 11/11 |
| DiT component：`minwm_denoising` | worker raw 11/11；API relay `scheduler_result_component_timing` 11/11，均带 CUDA | worker raw 11/11；API relay 0/11；client 报 missing 0–10 |
| VAE component：`vae_decoder` | worker raw 11/11；API relay `scheduler_result_component_timing` 11/11，均带 CUDA | worker raw 11/11；API relay 0/11；client 报 missing 0–10 |

S0 的 server、runner、client 都来自 `d5b25227d4`：runner 以 `nsys launch`
启动 `sglang serve`，在独立 20-chunk precondition 后执行 `nsys start`，再由同一
checkout 的 `benchmark_realtime_throughput.py` 发送 1+10 chunks。S1 `-07` 则先从
`S1_RUNNER_REF=952af69455c7cdb0f411631def90944f13579ca0` 复制 runner（其中
`MINWM_S0_TOOL_REF=d5b25227d4`），随后把实际 server checkout 切到
`SGLANG_GIT_REF=5f92d276c08086db638f05536a46fa5434ecb169`，再使用同样的
launch/start/client 结构。跨进程 relay commit `839f312c3b` 是 `d5b25227d4` 的祖先，
但不是 `5f92d276` 的祖先。因此 S1 的 payload/stats/wall 完整、worker component
完整而 API component 0/11，最小解释是 server implementation checkout 没有 relay，
不是 Nsight 参数或 trace queue 容量差异。这里只读检查 S1 Job/日志，没有改动 S1
代码、Job、Pod、PVC 或 marker。

### 正式产物与 SHA-256

正式根目录：

```text
/results/attempts/minwm-s0-fusedops-h200-20260807-09-s9cc4/
  minwm-s0-fusedops-h200-20260807-09/s0-measurement/
```

只读 reader 使用 4 MiB streaming chunks 逐文件计算 SHA；SP2 1.476 GB SQLite
耗时 24.070 s，SP4 1.421 GB SQLite 耗时 25.297 s，未超时或 OOM。

| 相对路径 | bytes | SHA-256 |
| --- | ---: | --- |
| `baseline-summary.json` | 119,480 | `3d7dcc8f39f7ac027452577f4ed1fba699ffcd53761b121b84b2bf60de95006c` |
| `sp2/profiler-off-repeat1.json` | 6,730 | `528961ebbe5eedcfe1be4690da1999d46cd11fa0dc51a7414bf012a53e10f570` |
| `sp2/profiler-off-repeat2.json` | 6,730 | `8e846ca5d096d6d08b116cfe99b465d6cf37631236754a6d2043ba91b3abdb28` |
| `sp2/repeat-summary.json` | 1,656 | `0a8eb88c7a23dab6a8877d6000ce090c7eb423d4db6851b533bdafc6618f9c36` |
| `sp2/profiler-on/client.json` | 9,671 | `3f4e7c6db85ba4dfd3295ab5145d2a870d50e3c4cc393d29ab6345b6a354fa34` |
| `sp2/profiler-on/measurement.json` | 45,898 | `bd14c5f171edbc57dba715305d6557e49d5b627fe7a469feead64954b262063d` |
| `sp2/profiler-on/server.log` | 570,055 | `dc66697396377d72d55ff7e58ab2912ed0239640957717eb15dac2ee535d28cf` |
| `sp2/profiler-on/nsys-stats.txt` | 37,982 | `7c4c4403f1ae8fc1339ec67424e2a339caa88d982cec958b4fe07b84992b3f17` |
| `sp2/profiler-on/sp2.nsys-rep` | 55,121,489 | `de8b5c0e380c6310cc009210f8b44bbb372af673ab210b54d58f382528dd425f` |
| `sp2/profiler-on/sp2.sqlite` | 1,476,804,608 | `6ea3a325d2715ed9944b30870b117c8bd34c6e5fe6854d0e0e9f9b6d282131e3` |
| `sp4/profiler-off-repeat1.json` | 6,712 | `75d19895145565f270f25f2196a41c1f3bfa6d4c40e60483fe0da5ad0d66e0ab` |
| `sp4/profiler-off-repeat2.json` | 6,723 | `82226b770ecb581aff7a4c9ad906e0613aaf8bd5fba9faeea3397107683ed826` |
| `sp4/repeat-summary.json` | 1,635 | `775d2edabe517b7e430b269ea670a3e13381cec1c5520c04fce17f79cc7e01bf` |
| `sp4/profiler-on/client.json` | 9,707 | `efc7f73b8996444b7ea987547edc13becb6746702e298d6d09edb17ec6747ec1` |
| `sp4/profiler-on/measurement.json` | 52,834 | `39c8d6b8c4922ebc0b7bc9d5a5d88a0955ffbe1cf4e6df10ed0c00b4d2499292` |
| `sp4/profiler-on/server.log` | 571,939 | `b9dff2e02042e6839c650cd447db55f5743b62017565e468d3d42093689df2d5` |
| `sp4/profiler-on/nsys-stats.txt` | 38,420 | `c01cb36692e0e406cdfecd468cafc9cffb8d0af86a1fe03f4276e51a4e2a2407` |
| `sp4/profiler-on/sp4.nsys-rep` | 91,842,432 | `22edc35043490919b9aa3cfcbcc47a8b2b0ac39e3d23d2685cd8d49dbe873178` |
| `sp4/profiler-on/sp4.sqlite` | 1,421,049,856 | `2c58f556f31a24b8473f3877fa31c6888ba97424e7ac72f29254286e29cbc8ed` |

## 与预期不符合的地方

- SP2 profiler-off 两次重复的四个验收 CV 均小于 0.16%，优于默认 3% 门槛。SP4 四个主验收 CV 也均通过，但 Client/Scheduler FPS 接近 2.27%，明显高于 SP2；非 headline scheduler chunk wall CV 为 5.629%。当前两次重复的 GPU stage wall 很稳定，且 FPS 仍在 3% 门内，因此按契约保留并解释为 CPU/排队环境噪声，不用 Nsight FPS 替换或挑选更好重复。
- `-04` profiler-on 的 GPU metrics start 成功并生成 38,106,433-byte report，但服务端 generation-complete close 早于最后 component trace 发出，客户端收到正常 code1000 且无合格 JSON；不得进入正式表。
- 同一 report 用 Nsight 2026.4 导出得到 397,185,024-byte SQLite：`NVTX_EVENTS` 31,000 行、kernel 394,526 行、runtime 1,036,567 行、GPU metrics 9,176,220 行；outer marker 数为 0，证明旧 report 无法支持 exact-window 归一化。
- `-04/sp2/profiler-on/invalid-marker-20260807T045715060254120Z.json` 原地保留 7 个文件、40,487,575 bytes 的逐文件 SHA；聚合验证该 marker 只排除 profiler-on，两个 sibling profiler-off run 均保留。
- `-05` 的 GPU metrics start 成功，Nsight 2026.4 report 为 37,589,254 bytes；同一正式 trace id 在 server.log 中有 worker DiT/VAE component 各 11 条、API scheduler-result wall 各 11 条，但客户端 component selector 各缺 0–10。原因是 `_notify_realtime_trace_sinks` 只在当前进程有效，并非 256 条队列溢出；该 lane 的 marker 保留 7 个文件及逐文件 SHA。
- 对 `-05` 产物实测 lane-scoped 审计：profiler-on report 为 invalid，两个 sibling profiler-off JSON 均非 invalid；聚合接受两个 run，`excluded=[]`。
- `-06` 客户端首次完整收到 profiler-on 的 1 discard + 10 measured payload/stats/wall/component trace，生成 9,695-byte `client.json`、38,148,535-byte report 和 205,168,640-byte SQLite；GPU metrics start 成功。随后显式 `nsys export` 发现同名 SQLite 已被前置 `nsys stats` 隐式创建而拒绝覆盖，runner 对该 lane 写 marker。
- 对 `-06` invalid lane 使用 canonical `900b5f279b` 只读诊断：merge 成功，exact measured indices 1–10、DiT/VAE CUDA count=10（mean 746.727/440.882 ms）、kernel=346,080、CUDA API=910,877、launch=346,080、kernel busy=76.315%，process/device coverage 全部 available；formal validator 只因三项 GPU metrics 的 target coverage 失败。SQLite 的 CUDA device 0/1 映射到 `pwGpuId` 2/3，但旧 capture 只有 typeId 低位 0/1；全 capture 的 SM/Tensor 两个 target 均 0，证实是采到闲卡而非缩放或窗口错误。
- canonical 诊断保存在 `.../-06/.../sp2/profiler-on/diagnostic-20260807T060700Z/`，含 `measurement.json`、完整 validator log、raw metric×target×chunk summary、输入/输出 SHA。`merge_status=0`、`validate_status=1` 是预期；lane marker 未解除，任何数据均未进入 baseline。
- `-08` 的 55,231,110-byte report 与 1,473,888,256-byte SQLite 覆盖 8 个 GPU metrics target；SM/Tensor/DRAM 对 active PerfWorks 2/3 均为 available，typeId×10 chunks 完整。正式 merge 只因一条 2,910 ns `cudaEventQuery_v3020` 从 measured range 外进入 chunk 7 而失败，runner 在 `sp2/profiler-on` 原地写 lane marker，sibling SP2 profiler-off 仍有效。
- canonical `d5b25227d4` 的离线诊断保存在 `.../-08/.../sp2/profiler-on/diagnostic-20260807T070200Z-canonical-d5b25227d4/`：严格 validator exit=0，measurement 与上一版逐字节相同；解析耗时 17.662 s、峰值 RSS 326,196 KiB、SQLite 1,473,888,256 bytes。旧全量 list 解析现场峰值约 8,909,528 KiB，故流式方案将最低诊断 Pod 内存需求降到远低于当前 16 GiB limit。
- 等价证据在同 lane 的 `diagnostic-20260807T065300Z-boundary-bounded-v4/equivalence-report-v2.json`：除有意更新的 attribution-policy 文案外，window、DiT/VAE CUDA、kernel/bucket/busy、capture coverage 及三项 GPU 的旧字段全相等；p95/每 chunk count 与 raw SQLite 独立重算全相等。原始 v1 false 报告和两次计时失败日志也保留，未覆盖。
- `-09` 正式 SP2/SP4 均完成 exact 10-range、active rank/process/device、DiT/VAE wall+CUDA count、kernel/API/launch、SM/Tensor/DRAM 全部门槛；六个 JSON 的独立 validator 重试全部通过。正式产物继续保留在 PVC，completed Job 未删除。

## 证据与决策过程

1. **base 选择**：比较 `origin/main...origin/codex/minwm-realtime-api` 后，确认 `main` 已含完整 API/benchmark，而旧分支还含大量量化实验，所以选择 `main`。
2. **不改算子实现**：S0 只修改 realtime trace 的可靠 flush、outer NVTX marker、schema、SQLite 解析和 benchmark 编排；没有修改 MinWM DiT、Ulysses、Parallel VAE 或 kernel 计算路径。
3. **KV45**：20+200 的 full-history KV 会持续增大序列和显存，不能作为融合小算子的稳定 A/B。选择仓库现有 throughput profile 已使用的 KV45，并把它写进 `comparison_contract`。
4. **完整节点隔离**：发现另一 CUDA Graph Job 使用同一 H200 pool 后，没有清理或复用其 Job/PVC；S0 请求完整 8 卡，调度器只能等待整节点或拉起新节点。
5. **Nsight 而非 torch.profiler**：Nsight 用 launch/start/stop 抓稳定窗口；torch profiler 目录变量存在即拒绝启动。
6. **动态 GPU metric 名称**：不同 Nsight/GPU 版本的 metric id 不稳定，因此以名称匹配；原始 metric 名称保存在 evidence 中。
7. **失败即停**：最初 `backoffLimit=2` 暴露出旧 runner 失败后 controller 会继续占用整机；`-03/-04` 改为 `backoffLimit=0`。每个 Pod attempt 仍写入独立的 `/results/attempts/$HOSTNAME/`，失败不会覆盖先前诊断，是否重跑必须先修复并使用新 Job 名。
8. **active 与 allocated 分离**：最初 runner 把整机预留的 8 写入 `gpu.count`，会让 SP2/SP4 看起来都用了 8 卡。数据落盘前修正为 active 2/4，并新增 `allocated_count=8`；validator 禁止 allocation 小于 active。
9. **首轮 runner 失败**：`-01` 的两个 attempt 在 setup 完成后均以 exit 141 结束，原因是 `set -o pipefail` 下 `nvidia-smi | head` 让上游收到 SIGPIPE。没有 profiler-off/on JSON 被当成 baseline；修正为完整消费输入的 `sed`/`awk` 后，用新 commit 和 `-02` Job 重跑，`-01` 的 PVC attempt 诊断保留。
10. **集群 context 固定**：桌面默认 context 可能漂移，因此所有 read、dry-run、apply、logs 和 delete 都显式传 `--context codex-minwm-test-phx2`。本任务只删除过自己精确命名、尚未产出合格 baseline 的旧 `-01/-02/-03` Job；PVC 与失败诊断保留，没有删除其他 Job/Pod/PVC/NodeClaim/NodePool，也没有误投其他集群。
11. **最后一条 trace 竞态**：`-02` 的两次 SP2 profiler-off 都有完整 200 个 payload/stats，但 DiT/VAE wall 仅观察到 199/200，因为客户端在最后 payload 后先于最后 stage trace 退出。这两次 JSON 明确标为 `incomplete_trace_metric`，不进入 baseline。`59aa68a382` 增加显式完整 trace 等待，以 `expected_indices=set(range(total_chunks))` 的子集关系判断；timeout 会列出每个 selector 的 missing/unexpected indices 以及 stats/payload 缺口。`-02` PVC 中 repeat1/repeat2 和 Pod 退出诊断保留。
12. **latency count 缺口**：`-03` repeat1 虽观察到完整 200 条 stage trace，但 `benchmark_realtime_throughput.py` 曾有一份不含 `count` 的重复 `latency_summary()`，导致三项 wall 的 `value` 无法机器证明覆盖 200 chunks。该 repeat 只保留诊断，repeat2 启动后立即用精确 Job 名停止 `-03`；`b9240233b2` 删除重复实现、复用 `measurement.latency_summary`，并由 schema + validator 同时约束 count。正式数据改用新名 `-04` 重跑。
13. **失败产物只归档、不删除**：失败、旧契约和 partial attempt 必须原地保留，或先生成 `minwm-realtime-invalid-attempt/v1` marker 后移动到该 attempt 的 `invalid/`。Marker 记录原因、UTC 时间、每个文件的原路径/保留路径/大小/SHA-256/可恢复性；runner 非零退出时原地写 marker，发现同路径旧结果时归档后才继续，并拒绝覆盖 Nsight 文件。聚合 CLI 明确排除路径中含 `invalid/` 的 JSON。Spot `SIGKILL` 无法执行 trap 时，后处理 reader 必须补 marker；PVC 本身不能删除。
14. **generation-complete trace drain**：`trace_queue` 的每次 get/drop 都配对 `task_done()`；正常关闭前先用 event-loop barrier 等待 `call_soon_threadsafe` enqueue，再等待 `queue.join()`。sender 异常与 join 并发等待，避免死锁；异常/断连仍由 finally cancel。
15. **严格稳定窗口**：静态审计确认 precondition 20 在 `nsys start` 前，capture 内实际为 1 discard + 10 measured。旧解析器仍会把 discard 的 SQLite 行除以 10，约有系统性污染；因此新增 outer marker range union，不能证明 exactly 10 就禁止 `per_stable_chunk`。
16. **真实 schema 现场校验**：2026.4 的 runtime 只给 `globalTid`，kernel 给 `globalPid`，并有 `PROCESSES` 映射；GPU metrics 有两个 target `typeId`。fixture 改为相同列布局，覆盖 globalTid 掩码、kernel process/device 交叉验证及每 type/chunk 样本矩阵。
17. **lane 粒度审计与恢复**：marker 从结果文件父目录逐层检查到最近 `s0-measurement`；profiler-on marker 不影响 sibling off。`-05` 从 `-04` source lane 重新校验并复制 SP2 off 到新 attempt，SP4 则正常重测，所有目标路径均拒绝覆盖。
18. **component timing 跨进程传输**：`-05` server.log 证明 worker 侧 component/CUDA trace 和 API 侧 wall trace 都完整，但前者只通知 worker 进程内 sink。没有放大队列或放宽完整性门；`839f312c3b` 将 `event/component/duration_ms/cuda_ms/chunk/request` 的纯标量记录附在 `RequestMetrics` 上，随 `result.metrics/metrics_list` 序列化返回 API，再以 `source=scheduler_result_component_timing` 重发。相同 identity 的相同记录去重，冲突记录 fail-closed；缺 `cuda_ms` 仍可保留 wall 证据，但正式 CUDA metric 会因 count 不足失败。
19. **SQLite 不覆盖**：`-06` 证明 2026.4 的 `nsys stats REPORT` 会在 report 同目录隐式生成 SQLite，导致后续显式 export 与“不覆盖已有 Nsight 产物”冲突。没有加 `--force-overwrite`；`f1b047942d` 改为先显式 export 到确认不存在的目标，再运行 stats 复用该 SQLite，并加静态回归测试锁定顺序。
20. **GPU logical ordinal 不能当物理 metrics ID**：旧 runner 从 `nvidia-smi index` 取 0/1 传给 Nsight，但 `-06` 的 `TARGET_INFO_GPU` 明确显示 CUDA logical 0/1 实际对应 PerfWorks 2/3。没有继续猜 ordinal 或只改成 2/3；新 runner 在整机隔离下采集 all 8，再由 SQLite 位域和 `cuDevice/pwGpuId` 映射筛 active target，兼容 SP4 非连续映射。
21. **SM 全零不是合法“可用值”**：NVIDIA schema 的 `GPU_METRICS.value` 已是 0–100 整数百分比，无需浮点缩放。稳定窗口 kernel busy 76.315% 时 SM Active 全 capture 仍全 0，只能说明 target/采集错误；新增 parser 与 custom validator 双重 all-zero gate，禁止这种记录通过 formal acceptance。
22. **API 边界按 start 唯一归属**：`-08` 的真实 crossing 是正常 generation 中高频轮询的 `cudaEventQuery_v3020`，进程属于已验证 rank，起点在 measured union 外、结束后 360 ns 落入 chunk 7；它不是 teardown、launch 或 marker 乱序。采用半开 start 规则后明确排除且保留 raw evidence；launch 仍与 kernel 346,080 一致。Kernel duration/busy 没有借此放宽。
23. **GPU metrics 有界聚合**：旧实现把全表所有 metric/target 样本装入 list，8-target SQLite 现场峰值约 8.50 GiB。新实现只扫描三个选中 metricId，并用小型 counter/histogram 保留所有验收字段；在相同 invalid SQLite 上做旧/新与 raw reference 三方等价后，才允许 `-09` 使用该 canonical。

## 风险与回滚

- 风险：Spot reclaim。补救：`backoffLimit=0` 先停住并保留 attempt，核查后用新 Job 名安全重跑；不删除其他任务释放容量。
- 风险：Nsight GPU metrics 权限不足。补救：保留 `nsys status -e` 和 fallback report，但正式 lane 失败；SM/Tensor 不允许权限降级后通过，DRAM 只允许 `metric_not_exposed`。
- 风险：容器 CUDA ordinal 与 Nsight/PerfWorks ID 顺序不同。补救：采集 all allocated target，依据 SQLite 的 `deviceId/cuDevice/pwGpuId/typeId&0xFF` 现场映射；collected/active/allocated 数量与每 device×chunk coverage 由 schema 强制。
- 风险：realtime trace 队列或跨进程 relay 丢事件。补救：每个 stage metric 必须恰好覆盖 measured chunk 数；worker component timing 仅传输 pickle-safe 标量，API 对 metrics/metrics_list 去重并拒绝冲突，缺失或缺 CUDA 时字段标记 `incomplete_trace_metric`，验收失败。
- 风险：失败重试覆盖审计证据。补救：runner 不删除旧文件；非零退出写逐文件 checksum marker，同路径重跑先移动到 `invalid/`，聚合排除 invalid；只能删除精确 Job/Pod 控制对象止损，不能删除 PVC。
- 风险：Nsight 开销污染 FPS。补救：schema 从结构上禁止 profiler-on 结果成为 headline。
- 风险：schema 影响现有 summary。补救：新 JSON 仍保留顶层 `profile_name`、`server`、`client`、`warmup_chunks` 等兼容字段。
- 风险：8-target GPU_METRICS 导出很大，离线解析 OOM。补救：流式 selected-metric 聚合；真实 1.47 GB SQLite 的峰值约 318 MiB。执行完整 merge 的 diagnostic Pod 保留 8 GiB request / 16 GiB limit；只读查看 JSON/日志和 streaming SHA 的 reader 为 100m CPU / 256 MiB，不能用来重跑 merge。elapsed/RSS/size 都写入诊断证据。
- 回滚：删除本 PR 新增的 benchmark/schema/doc 文件，并恢复 `benchmark_realtime_throughput.py`；没有模型实现或 checkpoint 格式迁移需要回滚。

## 复现命令与产物路径

### 本地 CPU 测试

```bash
python3 -m pytest -q \
  benchmark/minwm_realtime_parity/test_measurement.py \
  benchmark/minwm_realtime_parity/test_common.py

TORCHDYNAMO_DISABLE=1 PYTHONPATH=python python3.11 -m pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py
```

`d5b25227d4` 加最终文档的复验结果：measurement/common `45 passed`，realtime
runtime `47 passed`；两组都不依赖 GPU。最终文档文件的 pre-commit hooks 也全部通过。

### 校验与汇总

```bash
python benchmark/minwm_realtime_parity/measurement_tool.py validate RESULT.json

python benchmark/minwm_realtime_parity/measurement_tool.py aggregate \
  sp2-repeat1.json sp2-repeat2.json \
  --output sp2-repeat-summary.json

python benchmark/minwm_realtime_parity/measurement_tool.py merge-nsys \
  --result sp2-client.json \
  --sqlite sp2.sqlite \
  --status-log nsys-capture-status.log \
  --output sp2-measurement.json

python benchmark/minwm_realtime_parity/measurement_tool.py validate \
  sp2-measurement.json --require-complete-stable-nsys
```

### H200 清单

```bash
kubectl --context codex-minwm-test-phx2 apply --dry-run=client \
  -f benchmark/minwm_realtime_parity/k8s/minwm_s0_fusedops_h200_20260807.yaml

kubectl --context codex-minwm-test-phx2 apply --dry-run=server \
  -f benchmark/minwm_realtime_parity/k8s/minwm_s0_fusedops_h200_20260807.yaml

kubectl --context codex-minwm-test-phx2 apply \
  -f benchmark/minwm_realtime_parity/k8s/minwm_s0_fusedops_h200_20260807.yaml
```

Job 完成后，使用只挂载 S0 PVC 的 CPU artifact reader 读取产物；它也必须显式指定同一 context：

```bash
kubectl --context codex-minwm-test-phx2 apply \
  -f benchmark/minwm_realtime_parity/k8s/minwm_s0_artifact_reader_20260807.yaml
```

PVC：`minwm-s0-fusedops-h200-results-20260807`。每次尝试的根目录：

```text
/results/attempts/<pod-name>/minwm-s0-fusedops-h200-20260807-09/s0-measurement/
├── baseline-summary.json
├── contract.txt
├── sp2/
│   ├── profiler-off-repeat{1,2[,3]}.json
│   ├── profiler-off-gpu-telemetry.csv
│   ├── repeat-summary.json
│   └── profiler-on/{measurement.json,sp2.nsys-rep,sp2.sqlite,nsys-capture-status.log}
└── sp4/...
```

## 给负责人掌握代码的检查题

1. **为什么 Client FPS 和 Scheduler FPS 不应相等？**
   - 参考：Client window 包含传输与接收；Scheduler 只包 `process_generation_batch`。看 `benchmark_realtime_throughput.py::receive_run` 与 `realtime_video_api.py::_generate_loop`。
2. **4 次 DMD 后的第 5 次 DiT forward 在哪里，为什么不能漏算？**
   - 参考：`causal_denoising.py::_denoise_and_update_causal_block` 先 `_denoise_causal_dmd_chunk`，再 `_update_causal_context_cache` 用 clean latent 写 KV。
3. **DiT wall 和 DiT CUDA 分别从哪条 trace 取，如何证明均值覆盖完整窗口？**
   - 参考：wall 选择 `source=scheduler_result_metrics`；CUDA 同时选择 `source=scheduler_result_component_timing`、`component=minwm_denoising` 的 `cuda_ms`。worker 先把纯标量 timing 放入 `RequestMetrics`，API 从 result 重发；两者的 `value.count` 必须等于 `workload.measured_chunks`，由 schema 和 `validate_measurement` 双重检查。
4. **为什么 profiler-on 的 `observed_wall_with_profiler_overhead` 不能拿去做 headline？**
   - 参考：Nsight 注入 tracing/metrics 开销；`measurement_contract.headline_eligible` 只允许 profiler-off 为真，validator 会检查。
5. **GPU metrics 表不存在时，脚本如何区分权限问题与普通未采集？**
   - 参考：`nsys_metrics.py::_permission_reason` 检查 status/start evidence；三个字段仍必须输出 availability object。
6. **短 kernel 分桶如何处理边界 10、50、100 微秒？**
   - 参考：`<10`、`10<=x<50`、`50<=x<100`、`>=100`，实现和 fixture 在 `nsys_metrics.py`、`test_measurement.py`。
7. **两次重复 CV 超标后 runner 做什么，哪些指标决定验收？**
   - 参考：自动补第三次；Client FPS、Scheduler FPS、DiT wall、VAE wall 四项决定 `passes_cv_target`，scheduler chunk wall 仍报告但不决定该门槛。
8. **为什么 S0 Job 请求 8 卡，却仍分别记录 SP2/SP4？**
   - 参考：8 卡是云端资源隔离；`workload.sp_degree` 与 `provenance.gpu.count` 都是 active 2/4，`provenance.gpu.allocated_count=8` 才是预留数。Nsight 采 all 8，但 `nsys_metrics.py::_gpu_metrics` 只汇总由 stable kernel `deviceId -> TARGET_INFO_GPU.cuDevice/pwGpuId -> typeId&0xFF` 映射出的 active 2/4 卡。
9. **一个 Spot attempt 失败后，为什么不会覆盖下一次的证据？**
   - 参考：manifest 用 `/results/attempts/${HOSTNAME}`；最终 `backoffLimit=0` 会失败即停，诊断后用新 Job 名手动安全重跑，每个 Pod 名与 attempt 目录都不同。
10. **一个 CUDA API 从 measured range 外开始、结束时进入 chunk，应不应该计数？如何审计？**
   - 参考：不计。`nsys_metrics.py::_discrete_event_start_attribution` 只用 `start_ns` 在半开 `[start,end)` 中唯一归属；`boundary_event_examples` 保留 raw name、globalPid、start/end、owner 和 overlap chunks。若 start 在 range 内但跨 end，只计一次；API rank coverage 仍由 `globalTid -> PROCESSES.globalPid` 与 kernel device/process 交叉验证。
