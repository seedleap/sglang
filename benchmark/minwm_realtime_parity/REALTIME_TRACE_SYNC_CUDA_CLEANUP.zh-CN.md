# MinWM P0：默认 realtime trace CUDA host 同步清理

## 状态、目标与非目标

- 状态：实现与 CPU/fake-CUDA 单测完成；704p SP2/SP4 ABBA Job 已提交并等待新 Spot 节点，独立 Nsight 尚未提交，当前结论为 **no-go（真机证据未完成）**。
- 本任务分支：`codex/minwm-trace-sync-cleanup`。
- 安全复用来源：从 S0 冻结分支 `codex/minwm-fused-ops-s0` 的 `6c79fdfa63263814dc4e698b7bd808c6313b655c` 创建，没有从其他 worktree 复制文件或直接修改其他 worktree。
- 目标：默认 realtime trace 只保留 host wall，可选 CUDA event timing 改为显式 opt-in；默认 span 结束不等待 CUDA；保留 worker→API 的 wall/CUDA 状态可观测性；提供 bitwise 与同机 paired A/B 工具。
- 非目标：删除 realtime tracing、修改 MinWM/TAEHV/VAE 数学、实现其他 kernel/Graph/A2A 优化，或移除传输正确性所需的 event/stream barrier。

统一来源总账为飞书文档 `TFm8dG8nuoDTioxO4HFcuEBInRb` revision 8。本次使用
`--profile kejun --as bot` 重读了“统一验收合同”“当前不符合预期的地方”和“记录要求”。
冻结口径为 MinWM 5B step-3200、1248×704、BF16、每 chunk 4 DMD + 1 clean-cache
forward、KV45，SP2 主验收、SP4 复验；headline 为 profiler-off 20 warmup + 至少
200 measured，每个 control/candidate 至少两次同机 paired run。

## 基线与已有实现痕迹审计

### Git 与负责人

初始 worktree 在 detached `3654740347`，工作区干净；这个本地 `main` 不含 MinWM
realtime API。`origin/main=9a9dc59cd1` 才包含 API，而冻结 S0 已在其上叠加测量修复。
因此本任务没有在错误的本地 `main` 上重新实现，而是安全建分支：

```bash
git switch -c codex/minwm-trace-sync-cleanup codex/minwm-fused-ops-s0
```

S0 冻结提交 `6c79fdfa63` 的 author/committer 为 Jack47；只读检查
`/Users/chenshengdong/.codex/worktrees/b8ad/sglang` 显示 branch 与
`origin/codex/minwm-fused-ops-s0` 为 `+0/-0`，没有 tracked/untracked 修改。

本任务直接依赖的既有实现是 `839f312c3b4622c8e04c5c76620d22d6c2497fa0`
（`fix(minwm): relay worker CUDA timings to realtime clients`，author Jack47）。它把
worker 的纯标量 component timing 经 `RequestMetrics` 送回 API。`78af9cf8ed` /
`52a3a45051` 是 async-VAE trace 的两条历史线索，不是 P0 已完成实现，也没有直接
cherry-pick。

### S0 Job、运行状态与产物

所有现场读取都显式使用 `--context codex-minwm-test-phx2`。桌面全局 current context
实际为 `codex-seed-leap-use1`，验证了不能依赖默认 context。

| 项 | 只读核对结果 |
| --- | --- |
| Job | `minwm-s0-fusedops-h200-20260807-09`，1/1 Succeeded，非运行中 |
| 时间 | `2026-08-07T06:47:27Z` 开始，`07:07:25Z` 完成 |
| owner | `chenshengdong` |
| Pod / node | `minwm-s0-fusedops-h200-20260807-09-s9cc4` / `i-01a57ab8567279852` |
| source | `d5b25227d4487d113e62c86a0fb572a62d6bcc5b` |
| image | `minwm-training@sha256:bedc07ea…f5f2a` |
| PVC | `minwm-s0-fusedops-h200-results-20260807`，Bound 100 GiB |
| reader | `minwm-s0-fusedops-artifact-reader-20260807`，Succeeded |
| 正式产物根 | `/results/attempts/minwm-s0-fusedops-h200-20260807-09-s9cc4/minwm-s0-fusedops-h200-20260807-09/s0-measurement/` |

reader 和 Job 的 `kubectl logs --tail` 本轮返回空 stdout，因此没有把“空日志”当作
成功产物证据；正式 JSON/SQLite 的路径、字节数和 SHA-256 以 S0 冻结文档中的独立
streaming 校验为准。失败的 `-04/-05/-06/-08` Pod 仍保留，未删除或覆盖。

## 代码与数据布局合同

### 默认与 opt-in

| 条件 | host 行为 | trace 字段 | 回退 |
| --- | --- | --- | --- |
| 环境变量未设置 | 不创建/等待 CUDA event | `duration_ms`、`wall_timing_source=perf_counter`、`cuda_timing_status=disabled` | wall-only |
| `SGLANG_REALTIME_TRACE_SYNC_CUDA=1/true/yes/on` | 创建 start/end event；span 结束显式等待 | 上述字段 + `cuda_ms`、`cuda_timing_status=available` | 只用于显式 profiling |
| 显式 opt-in 但无 CUDA/不支持 event/计时失败 | 不伪造 0 | wall + `cuda_timing_status=unavailable`，无 `cuda_ms` | 安全 wall-only |
| 未知环境变量值 | fail closed，不同步 | `cuda_timing_status=disabled` | wall-only |
| 调用点 `measure_cuda=False` | 始终不同步 | wall-only | 用于 load/post-decode CPU span |
| 调用点 `measure_cuda=True` | 显式 opt-in，覆盖环境变量 | CUDA timing 或显式 unavailable | 测试/定向 profiling |

`GenerateSession.__init__` 仍用 session UUID 设置默认 `trace_id`；没有通过“让 trace id
为空”规避同步。trace log、sink、WebSocket 事件、component wall 和跨进程 relay 都保留。
新增 `cuda_timing_status` 让消费者可区分 disabled/unavailable/available，不把缺失值
静默解释成 0。

### bitwise 与 ABBA 产物

`benchmark_realtime_throughput.py` 不保留大帧 payload，但为每个 raw-RGB batch 计算
SHA-256，再按 batch index 生成 chunk digest；20+200 的 220 个 chunk 全部写入
`client.payload_sha256_by_chunk`。`compare_trace_sync_abba.py` 要求 control1、candidate1、
candidate2、control2 四次 digest map 完全相同，同时校验 BF16、20+>=200、CV<=3%、
Client/Scheduler FPS 不回退。

正式结果布局：

```text
/results/attempts/<pod>/<run-id>/trace-sync-abba/
├── contract.txt
├── comparison-summary.json
├── sp2/
│   ├── control-repeat{1,2}.json
│   ├── candidate-repeat{1,2}.json
│   ├── *-server.log
│   ├── *-gpu-telemetry.csv
│   └── *-telemetry-summary.json
└── sp4/...
```

每次 server health 后开始逐秒采 active GPU 显存，sidecar 报 aggregate peak，以及后
20 samples 相对中段 20 samples 的显存变化；不把未采集的地址稳定性或显存增长写成 0。

## 默认生产路径同步点清单

| 位置 | 默认实时路径 | 当前处理 | 原因 |
| --- | --- | --- | --- |
| `RealtimeTraceSpan.__exit__ -> cuda_end.synchronize()` | 是；默认 trace id 总存在，覆盖 MinWM denoise、VAE encode/decode | 默认关闭，显式 opt-in 保留 | 本任务主同步点 |
| `GenerateSession.trace_id -> Req.realtime_trace_id` | 是；未传 query/body trace id 时也回退到 session UUID | 保留 trace，只改变 CUDA timing 默认值 | 证明旧同步不是仅 debug request 才触发 |
| realtime VAE encoder / decoder span | 是 | 默认 wall + `cuda_timing_status=disabled`；opt-in 才 event sync | 两个默认 CUDA span |
| realtime MinWM denoising span | 是 | 同上 | 每 chunk DMD/clean-cache 主 span |
| VAE decoder load / post-decode span | 是 | 原本已 `measure_cuda=False`，保留 | CPU/host wall span |
| pipeline stage `perf_counter` wall trace | 是 | 保留，无新增 CUDA synchronize | wall 可观测性不丢失 |
| `StageProfiler` 的 device synchronize | 仅 `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1` | 不改；默认 0 | 已是显式 profiler |
| `GPUWorker` frame materialize synchronize | 同上 | 不改；默认 0 | 已是显式 profiler |
| RealESRGAN upscaler event synchronize | 非本轮 MinWM pipeline；仅启用 upscaling 时 | 不改 | 独立可选后处理组件 |
| LoRA/offload cache clear 与 sleep/awake device synchronize | 非 chunk steady-state；本 workload 未启用 | 不改 | 权重/显存生命周期 barrier |
| disaggregation `stage_event.synchronize()` | 非本轮默认单体 MinWM 路径 | 不改 | buffer 发送前的正确性 barrier，不是 trace timing |
| disaggregation/transport stream synchronize | 非本轮默认路径 | 不改 | 传输所有权/生命周期 barrier |
| benchmark 脚本内 `torch.cuda.synchronize()` | 否 | 不改 | 离线 benchmark 计时 |
| profiler utility 内 synchronize | 否 | 不改 | 显式 profiler |

本轮没有发现第二个由默认 realtime trace 引入的 stream-wide/device-wide host sync。

## 实际命令、测试与不符合预期

### 仓库与 S0 审计

```bash
pwd
rg --files -g 'AGENTS.md'
git status --short --branch
git worktree list --porcelain
git log --all --oneline --grep='trace.*sync\|sync.*trace\|RealtimeTraceSpan\|REALTIME_TRACE' -i
git grep -n 'SGLANG_REALTIME_TRACE_SYNC_CUDA\|RealtimeTraceSpan' origin/main
git show --format=fuller --stat 839f312c3b
git -C /Users/chenshengdong/.codex/worktrees/b8ad/sglang status --porcelain=v2 --branch --untracked-files=all
```

### 飞书冻结口径

```bash
LARKSUITE_CLI_NO_UPDATE_NOTIFIER=1 LARKSUITE_CLI_NO_SKILLS_NOTIFIER=1 \
  lark-cli docs +fetch --profile kejun --as bot \
  --doc 'https://icnimsatr0zz.feishu.cn/docx/TFm8dG8nuoDTioxO4HFcuEBInRb' \
  --scope keyword --keyword 'S0|SGLANG_REALTIME_TRACE_SYNC_CUDA|RealtimeTraceSpan|CUDA event|host 同步|trace sync' \
  --detail with-ids --format json
```

随后以相同 profile/identity 读取 outline，并按 block id 局部读取“统一验收合同”“当前
不符合预期的地方”“记录要求”；返回 revision 均为 8。

### Kubernetes 只读核对

```bash
kubectl config current-context
kubectl --context codex-minwm-test-phx2 -n default get job \
  minwm-s0-fusedops-h200-20260807-09
kubectl --context codex-minwm-test-phx2 -n default get pods \
  -l job-name=minwm-s0-fusedops-h200-20260807-09 -o wide
kubectl --context codex-minwm-test-phx2 -n default get pvc \
  minwm-s0-fusedops-h200-results-20260807
kubectl --context codex-minwm-test-phx2 -n default logs \
  pod/minwm-s0-fusedops-artifact-reader-20260807 --tail=220
```

最后一条返回空 stdout；未据此声称日志内容完整。没有执行 apply/delete/kill。

### 测试

先写失败合同测试：

```bash
TORCHDYNAMO_DISABLE=1 PYTHONPATH=python python3.11 -m pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_realtime_trace.py
```

实际：`4 failed`，分别证明旧代码默认启用 CUDA、缺 status、unsupported 无显式降级、
未知 env 值会错误开启；这是预期红灯。

实现后：

```bash
TORCHDYNAMO_DISABLE=1 PYTHONPATH=python python3.11 -m pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_realtime_trace.py \
  python/sglang/multimodal_gen/test/unit/realtime/test_realtime_runtime.py
```

实际：`51 passed`。

第一次测量工具回归：

```bash
python3 -m pytest -q \
  benchmark/minwm_realtime_parity/test_measurement.py \
  benchmark/minwm_realtime_parity/test_common.py
```

实际：`46 passed, 1 failed`。原因不是算法，而是本机默认 Python 的 `zip()` 不支持
`strict=True`。前置已严格验证两臂各两次，因此移除该关键字以兼容 runner 镜像；复验
为 `47 passed`。同时执行 `bash -n run_trace_sync_abba.sh` 与 Python `py_compile`，通过。

尝试运行仓库 hook：

```bash
pre-commit run --files <本任务文件>
```

实际失败为 `zsh: command not found: pre-commit`，本机与 worktree 都没有该可执行文件。
没有把缺依赖写成 hook 通过；改用已安装的 `black --check`、`ruff check`、`bash -n`。
Black 首次指出 comparator 与 measurement test 需格式化；运行 Black 后复查：7 个 Python
文件均 unchanged，Ruff all checks passed，shell syntax 通过。格式化后再次执行两组测试，
仍为 `47 passed` 和 `51 passed`，`git diff --check` 通过。

首次暂存后的 `git diff --cached --check` 又定位到本文第 35 行一个行尾空格；同一 shell
没有 `set -e`，所以后续 commit 仍被执行。该 commit 尚未 push；立即删除空格、重新
执行 cached diff check，并 amend 后才允许作为 Job source。这条失败没有从记录中省略。

补充稳定性/布局回归：

```bash
TORCHDYNAMO_DISABLE=1 PYTHONPATH=python python3.11 -m pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py \
  -k 'nonuniform_sp8 or sequence_shard_frame_indices'
python3 -m pytest -q \
  benchmark/minwm_realtime_parity/test_prompt_switch_harness.py
```

实际分别为 `2 passed, 113 deselected` 和 `2 passed`，覆盖非均匀 SP8 output shard、
mid-frame sequence shard boundary 与 prompt-switch harness；这是 CPU/局部回归，不冒充
704p GPU headline 或 200-chunk scene-cut 真机结果。

## 待执行真机矩阵与当前门禁

1. 提交安全代码 commit 并 push；Job 只能 checkout immutable SHA。
2. 对独立 manifest 做 client/server dry-run，核对 image、SHA、checkpoint、PVC、
   H200 `p5e.48xlarge`、8 GPU 整机隔离、SP2/SP4、20+200、KV45、ABBA。
3. profiler-off 同 Pod/同 node 执行 control1,candidate1,candidate2,control2；报告 Client/
   Scheduler FPS、chunk/DiT/VAE wall、peak VRAM、CV、220-chunk bitwise。
4. 独立 Nsight，不与 torch.profiler 并用；报告 DiT CUDA、kernel/launch、短 kernel、
   A2A、GPU busy、SM/Tensor Active、DRAM。Nsight FPS 不作 headline。
5. 832×480 只作 prompt switch/scene cut/eviction 回归；不替代 704p。

当前 P0 Job 已提交但尚未产生 P0 真机 JSON/SQLite，因此相关栏位是“未测”，不是 0。
在 bitwise、两次同机 paired A/B、CV、SP4 复验与 Nsight 归因完成前保持 **no-go**。

## P0 ABBA Job 提交记录

安全实现 commit 为 `39065138b377d0117e2313983020e80666f70c24`，已推送到
`origin/codex/minwm-trace-sync-cleanup`，并用 `git ls-remote` 复核远端 ref 精确命中该
SHA。Job manifest 是
`k8s/minwm_p0_trace_sync_h200_20260807.yaml`，固定该 SHA 和 S0 镜像/checkpoint。

```bash
kubectl --context codex-minwm-test-phx2 apply --dry-run=client \
  -f benchmark/minwm_realtime_parity/k8s/minwm_p0_trace_sync_h200_20260807.yaml
kubectl --context codex-minwm-test-phx2 apply --dry-run=server \
  -f benchmark/minwm_realtime_parity/k8s/minwm_p0_trace_sync_h200_20260807.yaml
kubectl --context codex-minwm-test-phx2 apply \
  -f benchmark/minwm_realtime_parity/k8s/minwm_p0_trace_sync_h200_20260807.yaml
```

两种 dry-run 均显示 PVC/Job `created (dry run)`；正式 apply 创建：

| 项 | 值 |
| --- | --- |
| Job | `minwm-p0-trace-sync-h200-20260807-01` |
| Pod | `minwm-p0-trace-sync-h200-20260807-01-7p95f` |
| PVC | `minwm-p0-trace-sync-h200-results-20260807`，50 GiB |
| 资源 | H200 `p5e.48xlarge` Spot，完整 8 GPU，64/128 CPU，400/800 GiB memory |
| 保护 | `backoffLimit=0`，`activeDeadlineSeconds=21600`，独立结果路径 |
| 初始调度 | `Pending`；现有节点 `Insufficient nvidia.com/gpu`，没有抢占/清理其他任务 |
| 新 NodeClaim | `minwm-test-phx2-p5e-spot-msk8k`，已 Launched，provider `i-0973db0dc2a8448d1`，尚待 Registered/Ready |

新节点注册时出现一次 `s3.csi.aws.com` 尚未注册的 `FailedMount`，随后卷 attach、9.39 GB
固定镜像拉取和容器启动均成功。Pod 已 Running、restart=0；日志证明 checkout 精确为
SGLang `39065138b377…`、minWM `2efc6485…`，环境为 Python 3.12.3、Torch
2.11.0+cu130、H200。setup 后 SP2 control repeat1 已启动，GPU0/1 可见约 96%/93%
utilization，其余 6 卡空闲隔离；尚未产生完整 A/B 数值。

一次只读状态命令没有给 zsh 的 `containerStatuses[0]` 加引号，因 glob 展开失败并打印
`no matches found`；同次日志读取仍成功。后续将 custom-columns 表达式整体单引号包住，
状态命令恢复。该失败不影响 Job，但作为命令失败保留。

## Nsight 两条诊断 lane

准备 Nsight runner 时发现一个 schema/闸门缺口：
`measurement_tool.require_complete_stable_nsys()` 原先强制 exact window、kernel/API/
launch、busy、SM/Tensor/DRAM，却没有把 `dit_cuda_ms`、`vae_cuda_ms` 加入 required。
这与 S0 文档声称“正式 CUDA metric 缺 count 即失败”不一致。没有沿用这个静默缺口：

- 默认 `--require-complete-stable-nsys` 现在同时要求 DiT/VAE component CUDA available；
- 只有生产 no-sync Nsight 诊断可显式加 `--allow-missing-component-cuda`；该模式仍严格要求
  exact window 和全部 Nsight kernel/launch/GPU metrics，只诚实允许 trace CUDA event 字段
  unavailable；
- profiling-optin lane 保持默认严格门，必须有 DiT/VAE CUDA。

对应单测构造同一份完整 Nsight record，验证默认对缺 component CUDA fail closed，而显式
diagnostic override 仅放宽这两个字段。`run_s0_measurement.sh` 的默认行为不变；新增的
skip-profiler-off、result root、trace-sync 和 session-label 控制都必须显式设置。

首次起草 Nsight manifest 时根据短 SHA 手工补成了不存在的
`1e9c11322fb8e6f58f5d0372d1496153160574bc`。在任何 dry-run/apply 前用
`git rev-parse 1e9c11322f` 发现真实 SHA 是
`1e9c11322feb27502a45ec308f3bd30d6d7dc4f8`，立即修正 manifest 三处 pin。错误 SHA
从未提交到集群，也没有产生 Job/PVC；保留这条记录，后续仍需 `git ls-remote` 再校验。

## P0 ABBA 运行中结果（2026-08-07）

同 Pod、同 node 的 SP2 `control1,candidate1,candidate2,control2` 已完成；Pod restart=0，
每臂均是 20 warmup + 200 measured，且 220 个 raw RGB chunk SHA-256 map 四臂完全
一致。control 的 trace 审计均为 `cuda_timing_status=available`、440 个匹配事件；candidate
均为 `cuda_timing_status=disabled`、440 个匹配事件，wall timing 仍保留。因此本轮 BF16
bitwise 和“可观测性没有静默丢失”两项通过。

| SP2 指标 | control1 | control2 | candidate1 | candidate2 | control mean | candidate mean | candidate/control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Client FPS | 12.6904 | 12.5208 | 12.5787 | 12.9868 | 12.6056 | 12.7827 | +1.41% |
| Scheduler FPS | 12.7037 | 12.5338 | 12.5921 | 13.0011 | 12.6187 | 12.7966 | +1.41% |
| scheduler chunk wall (ms) | 1330.29 | 1329.89 | 1327.93 | 1262.92 | 1330.09 | 1295.43 | -2.61% |
| DiT wall (ms) | 744.75 | 747.06 | 728.99 | 725.12 | 745.90 | 727.06 | -2.53% |
| VAE wall (ms) | 419.95 | 420.34 | 435.61 | 436.81 | 420.14 | 436.21 | +3.82% |

Client/Scheduler/DiT/VAE 两次重复 CV 均 <=3%；但 candidate chunk wall CV=3.55%，超过
统一默认门槛。检查发现 Job pin 的旧版 `compare_trace_sync_abba.py` 只把 Client、Scheduler、
DiT、VAE 纳入 `cv_pass`，错误漏掉了同样要求报告的 chunk wall。已新增失败合同测试并将
`METRICS` 五项全部纳入 CV 门；修复后本地测量工具回归为 `48 passed`，Black/Ruff/
`git diff --check`/shell syntax 均通过。由于运行中 Job checkout 是 immutable SHA，没有
热改 Pod；最终将用修正版离线重验完整产物，并把 SP2 当前状态视为 **no-go / 需追加重复**，
不采信旧 Job 可能生成的绿色 `go`。

显存逐秒采样为：control1 peak 97694 MiB、tail-middle 0；candidate1 97686/0；candidate2
97688/0；control2 97686/+262.8 MiB（均为 active ranks aggregate）。最后一个非零 proxy
尚需检查原始时间序列，当前不写成“无增长”。

运行中还发生两条不影响 Job 的本地只读命令失败：一次把 context 名误写成 namespace，
API 返回 `namespaces "minwm-test-phx2" not found`，随后从 manifest 确认 namespace 是
`default`；一次尝试用本机 GNU `timeout` 做有界轮询，macOS 环境没有该命令，后续改用
工具层有界 wait。两者均未对 Pod/PVC/Job 产生写操作。SP4 control1 已在同一 Pod/node
启动；Nsight Job 仍未提交，避免与 profiler-off headline 抢占资源。

同步点复审命令还覆盖了 `multimodal_gen/runtime` 与本 benchmark 下全部
`.synchronize()`、`RealtimeTraceSpan`、trace id 传播。第一次把不存在的
`runtime/entrypoints/realtime*` glob 直接交给 zsh，搜索在执行前以 `no matches found`
失败；改为确定目录的 `rg` 后完成。没有把失败搜索当作审计通过。

SP4 candidate repeat1/2 已完成，220 hashes 与 SP4 control1 完全一致，trace status 均为
`disabled` 且各有 440 个 denoising 事件。两次 candidate 的 chunk wall 分别为 1122.17、
1068.21 ms，CV 约 3.48%，也超过 3%，因此追加范围从 SP2 扩为 SP2+SP4 repeat3/4。

追加 runner 默认仍是 SP2+SP4、repeat1/2；只有 manifest 显式传
`MINWM_TRACE_SYNC_REPEATS="3 4"`。comparator 支持显式 SP lane 与至少两次重复，五个指标
全部进入 CV 门；本地回归增至 `49 passed`。追加源码为
`f4b7d5eaf4d054f9d93bfb1ae26d90a0a9f932f2`，已用远端 ref 精确核对；Job
`minwm-p0-trace-sync-h200-20260807-02` 固定到原 node `i-0973db0dc2a8448d1`、同一 PVC，
整机 8 GPU，因此只能在主 Job 释放资源后开始。

最初创建的追加 Job 是 SP2-only Pending。发现 SP4 CV 也超标后，client dry-run 通过，
但直接 server dry-run 被 Kubernetes 以 Job `spec.template` immutable 拒绝，命令链在删除
前停止。随后将只用于 dry-run 的对象名经 stdin 改为 `-02-dryrun`，server create dry-run
通过；确认旧 Pod 仍为 Pending、未挂卷、无产物后，仅删除并从 manifest 重建该 Pending
Job/Pod。PVC、主 Job、已有结果均未删除；新 Pod `...-55dbf` 当前因原 node GPU 被占满
而 Pending。EKS auto-mode 同时警告 hostname 不能用于新节点 provisioning；默认 scheduler
仍保留该 Pod，这个限制符合“不另起机器”的目的。
