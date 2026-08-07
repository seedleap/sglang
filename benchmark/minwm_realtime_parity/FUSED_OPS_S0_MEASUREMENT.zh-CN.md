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
- 真机 Nsight 2026.4 schema canonical：`401e4ec8a1`（后续 A/B 与 `-05` 均固定此 SHA）
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
且 request id 唯一、时间顺序一致、区间不重叠。Kernel/runtime 行只有完全落入
单个 measured range 才计数；跨边界 event 会让该类 normalized metric
`unavailable`。GPU samples 用同一 range union 的 `[start,end)` 过滤，range
间隙、discard、outside 和 sibling trace 均不计入。

Kernel 与短 kernel 分桶按稳定窗口内 `deviceId` 展开；GPU kernel busy 的分母
是 10 个 range 时长之和，而不是首尾 kernel span。API/launch 提供
`total_per_chunk`；只有 runtime 进程、kernel `globalPid`、device→process 和
active rank 完全一致时才提供 `per_rank_per_chunk`。2026.4 真机 runtime 表使用
`globalTid`，解析器清除低 24-bit thread 部分后必须命中 `PROCESSES.globalPid`，
再与 kernel `globalPid` 交叉验证。所有正式归一化的
`capture_scope=union of exact measured outer chunk NVTX ranges`。

GPU metrics 不使用固定 metric id。解析器读取 `TARGET_INFO_GPU_METRICS.metricName` 动态匹配；权限不足或指标不存在时输出：

```json
{
  "status": "unavailable",
  "reason": "permission_denied",
  "evidence": "Nsight status/start 的原始错误与缺失表信息"
}
```

不允许用空值、0 或推测值代替不可得指标。正式
`--require-complete-stable-nsys` 还要求 SM Active、Tensor Active 都 available，
且每个 active `typeId × 10 chunks` 均有样本；无 GPU metrics 的 fallback report
只能作为 invalid lane 诊断。DRAM 仅在 Nsight raw metric names 确实未暴露匹配项
时允许 `metric_not_exposed`。

### Nsight 稳态窗口

runner 使用：

```text
nsys launch --trace=cuda,nvtx --trace-fork-before-exec=true --cuda-graph-trace=node
20 chunk 前置 warmup（未 start）
nsys start --gpu-metrics-devices=... --gpu-metrics-frequency=10000 --sample=none
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
exact-window profiler-on 与 SP4 由 `minwm-s0-fusedops-h200-20260807-05` 补齐。
旧 H200/B300 表以及 `-01/-02/-03` 失败诊断只用于背景与异常证据。

### 运行来源

| 项 | 实际值 |
| --- | --- |
| SGLang | SP2 profiler-off source=`b9240233b2`；exact-window runner=`401e4ec8a1` |
| MinWM | `2efc6485f65e8fcab506665efde79bc41406385e` |
| 镜像 | `minwm-training@sha256:bedc07ea...f5f2a` |
| GPU | NVIDIA H200；`gpu.count` 是 active 2/4 卡；`allocated_count=8` 是整机隔离预留 |
| kube context | `codex-minwm-test-phx2`；所有命令显式传 `--context`，未切换全局 current-context |
| region / zone | AWS `us-west-2` / `us-west-2-phx-2a` |
| NodePool | `minwm-test-phx2-p5e-spot`（共享的既有 NodePool，S0 未创建或删除） |
| 实例 | `p5e.48xlarge` Spot；`-04` 节点 `i-01a57ab8567279852` |
| 资源隔离 | Job 请求完整 8 GPU；不与 CUDA Graph 或 S1–S4 Job 共用 GPU 节点 |

### profiler-off 重复

采集完成后填写；所有 FPS 均来自 profiler-off。

| SP | 重复 | Client FPS | Scheduler FPS | scheduler chunk wall mean | DiT wall mean | VAE wall mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 12.904769 | 12.917758 | 1264.635 ms | 747.187 ms | 419.249 ms |
| 2 | 2 | 12.884662 | 12.896310 | 1272.865 ms | 745.509 ms | 419.569 ms |
| 4 | 1 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 |
| 4 | 2 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 |

| SP | Client CV | Scheduler CV | DiT wall CV | VAE wall CV | 验收 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 0.110% | 0.118% | 0.159% | 0.054% | 通过（均 <3%） |
| 4 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 |

### profiler-on 稳态窗口

| SP | DiT CUDA mean | VAE CUDA mean | kernels | CUDA APIs | launch APIs | <10 us | 10–<50 us | 50–<100 us | >=100 us | kernel busy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 |
| 4 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 | 采集中 |

| SP | SM Active | Tensor Active | DRAM | 权限/采集证据 |
| ---: | --- | --- | --- | --- |
| 2 | 采集中 | 采集中 | 采集中 | 采集中 |
| 4 | 采集中 | 采集中 | 采集中 | 采集中 |

## 与预期不符合的地方

- SP2 profiler-off 两次重复的四个验收 CV 均小于 0.16%，优于默认 3% 门槛。
- `-04` profiler-on 的 GPU metrics start 成功并生成 38,106,433-byte report，但服务端 generation-complete close 早于最后 component trace 发出，客户端收到正常 code1000 且无合格 JSON；不得进入正式表。
- 同一 report 用 Nsight 2026.4 导出得到 397,185,024-byte SQLite：`NVTX_EVENTS` 31,000 行、kernel 394,526 行、runtime 1,036,567 行、GPU metrics 9,176,220 行；outer marker 数为 0，证明旧 report 无法支持 exact-window 归一化。
- `-04/sp2/profiler-on/invalid-marker-20260807T045715060254120Z.json` 原地保留 7 个文件、40,487,575 bytes 的逐文件 SHA；聚合验证该 marker 只排除 profiler-on，两个 sibling profiler-off run 均保留。
- SP4 与 exact-window profiler-on 仍待 `-05` 真机结果；完成前 PR 保持 draft。

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

## 风险与回滚

- 风险：Spot reclaim。补救：`backoffLimit=0` 先停住并保留 attempt，核查后用新 Job 名安全重跑；不删除其他任务释放容量。
- 风险：Nsight GPU metrics 权限不足。补救：保留 `nsys status -e` 和 fallback report，但正式 lane 失败；SM/Tensor 不允许权限降级后通过，DRAM 只允许 `metric_not_exposed`。
- 风险：realtime trace 队列丢事件。补救：每个 stage metric 必须恰好覆盖 measured chunk 数，否则字段标记 `incomplete_trace_metric`，验收失败。
- 风险：失败重试覆盖审计证据。补救：runner 不删除旧文件；非零退出写逐文件 checksum marker，同路径重跑先移动到 `invalid/`，聚合排除 invalid；只能删除精确 Job/Pod 控制对象止损，不能删除 PVC。
- 风险：Nsight 开销污染 FPS。补救：schema 从结构上禁止 profiler-on 结果成为 headline。
- 风险：schema 影响现有 summary。补救：新 JSON 仍保留顶层 `profile_name`、`server`、`client`、`warmup_chunks` 等兼容字段。
- 回滚：删除本 PR 新增的 benchmark/schema/doc 文件，并恢复 `benchmark_realtime_throughput.py`；没有模型实现或 checkpoint 格式迁移需要回滚。

## 复现命令与产物路径

### 本地 CPU 测试

```bash
python3 -m pytest -q \
  benchmark/minwm_realtime_parity/test_measurement.py \
  benchmark/minwm_realtime_parity/test_common.py
```

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
/results/attempts/<pod-name>/minwm-s0-fusedops-h200-20260807-05/s0-measurement/
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
   - 参考：wall 选择 `source=scheduler_result_metrics`；CUDA 选择 `component=minwm_denoising` 的 `cuda_ms`。看 `stage_trace_values` 调用处；两者的 `value.count` 必须等于 `workload.measured_chunks`，由 schema 和 `validate_measurement` 双重检查。
4. **为什么 profiler-on 的 `observed_wall_with_profiler_overhead` 不能拿去做 headline？**
   - 参考：Nsight 注入 tracing/metrics 开销；`measurement_contract.headline_eligible` 只允许 profiler-off 为真，validator 会检查。
5. **GPU metrics 表不存在时，脚本如何区分权限问题与普通未采集？**
   - 参考：`nsys_metrics.py::_permission_reason` 检查 status/start evidence；三个字段仍必须输出 availability object。
6. **短 kernel 分桶如何处理边界 10、50、100 微秒？**
   - 参考：`<10`、`10<=x<50`、`50<=x<100`、`>=100`，实现和 fixture 在 `nsys_metrics.py`、`test_measurement.py`。
7. **两次重复 CV 超标后 runner 做什么，哪些指标决定验收？**
   - 参考：自动补第三次；Client FPS、Scheduler FPS、DiT wall、VAE wall 四项决定 `passes_cv_target`，scheduler chunk wall 仍报告但不决定该门槛。
8. **为什么 S0 Job 请求 8 卡，却仍分别记录 SP2/SP4？**
   - 参考：8 卡是云端资源隔离；`workload.sp_degree` 与 `provenance.gpu.count` 都是 active 2/4，`provenance.gpu.allocated_count=8` 才是预留数，Nsight 只选实际 lane 使用的设备。
9. **一个 Spot attempt 失败后，为什么不会覆盖下一次的证据？**
   - 参考：manifest 用 `/results/attempts/${HOSTNAME}`；最终 `backoffLimit=0` 会失败即停，诊断后用新 Job 名手动安全重跑，每个 Pod 名与 attempt 目录都不同。
10. **2026.4 runtime 只有 `globalTid` 时，如何证明 API 覆盖了全部 rank？**
   - 参考：`_api_metrics` 清除低 24-bit thread id 后要求命中 `PROCESSES.globalPid`；`_kernel_process_coverage` 再验证每个 `deviceId` 恰有一个 kernel `globalPid`，两组 process id 必须完全相等 active rank 数。
