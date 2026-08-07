# MinWM saturated full-DiT CUDA Graph correctness

## 状态

- 负责人：本 Codex task（独立 worktree `/Users/chenshengdong/.codex/worktrees/4b03/sglang`）
- 当前分支：`codex/minwm-cuda-graph-correctness`
- 当前结论：**Graph correctness GO（SP1、1248x704 SP2/SP4 均逐 block/forward 与端到端 bitwise）；默认开启仍 NO-GO，开关继续默认关闭，等待正式性能、200-chunk 稳定性与完整 fallback 矩阵。**
- 工作负载合同：MinWM 5B step-3200、BF16、每 chunk 4 次 DMD forward + 1 次 clean-cache commit forward、固定 45 latent-frame rolling KV。
- 第一阶段：Graph-only H200 单卡 bitwise；只做 block/forward 首差定位和修复。
- 后续门槛：只有单卡 bitwise 恢复后，才允许 1248x704 SP2 主验收与 SP4 复验。

## 目标与非目标

### 目标

1. 在同一 H200、同一输入、同一 eager kernel 选择下比较 eager 与 full-DiT graph。
2. 找到首个不 bitwise 的 DMD step、DiT block/forward 边界，并恢复 bitwise。
3. 明确 graph 的 cache、shape、address 与 collective 不变量以及 eager 回退边界。

### 非目标

- correctness 恢复前不扩展 growing KV、clean graph family、NCCL graph、multi-bucket/LRU graph pool。
- correctness 恢复前不扩大 headline 性能矩阵，不把 profiler 下 FPS 当 headline。
- 832x480 只用于最小复现；不得替代 1248x704 验收。

## 来源审计

- 接管时本 worktree：detached `main` at `3654740347f63edd3e8df78b2282ec79782d4f2c`，`git status` 干净。
- 已有实现来源：`codex/minwm-cuda-graph` / `origin/codex/minwm-cuda-graph` at `dc3c5c40d5774a4019e9935bafb9b7b0e31fb5ed`。
- 来源 worktree：`/Users/chenshengdong/.codex/worktrees/minwm-cuda-graph`；审计时干净，无该路径对应的活跃进程。未改动该 worktree。
- 来源负责人/作者：Graph 提交均为 Jack47 `<scsvip@gmail.com>`。
- 来源基线：MinWM 集成 `9a9dc59cd1`；与本地 `main` 的 merge-base 为 `80decc78ec226ec168977406277fec707c96b718`。为避免把 40 余个分叉提交逐个挑入不兼容的 `main`，本任务从已审计来源 tip 建独立分支，完整保留来源历史。
- Graph 增量（相对 `9a9dc59cd1`）：11 个文件，约 1062 行新增、67 行删除。
- 来源未提供本任务专属过程文档；这是接管时发现的交付缺口。

## 当前代码与数据布局合同

- 开关：`--enable-cuda-graph`，默认 `false`。
- 当前只允许 CUDA（非 HIP）、tensor prompt/action、无 `image_kwargs`、无外部 `attn_metadata`、有 self/cross attention cache。
- KV 必须 bounded、`block_relative` RoPE、已经 saturated；每层 self-attention cache 的 K/V/rotated-K 地址以及 cross-attention K/V 地址进入 graph key。
- 输入 static buffers 保留原 shape/stride/dtype/device；latent/prompt/timestep/action 在每次 replay 前 copy。
- captured attention plan 及 RoPE tensor 必须被 runner 强引用，避免 allocator 回收地址。
- 输出是 graph-owned static allocation，返回前 clone，避免下一次 replay 覆写下游 scheduler 尚在使用的结果。
- 当前仅一个 runner；key 变化会替换 runner。尚无 graph pool 容量/生命周期方案，本阶段也不讨论扩展。
- block/forward 首差开关：`MINWM_CUDA_GRAPH_VERIFY_BLOCKS=N`，默认 `0`；正整数表示最多验证 N 次 graph replay。诊断 capture 把 `block_00..block_29` 边界 snapshot 纳入 graph，每次 replay 后用相同输入和当前 attention plan 跑 matched eager，按 block 到 forward 顺序做 bitwise；首差硬失败。该模式增加显存与一次 eager forward，不用于性能数据。

## 为什么当前跳过第一个 DMD step

代码以 `current_timestep == 0` 作为 eager fallback 条件。DMD 循环从 0 编号，因此每个 chunk 的第一个 DMD forward 被跳过；clean-cache commit forward 也传 `current_timestep=0`，因此同样被跳过。现有编号没有区分“DMD step 0”和“clean commit”，所以当前实现不是只排除 clean commit。

原始动机是 graph 仅覆盖 saturated recompute：第一个 DMD step 负责先建立/刷新本 chunk 的 attention plan，clean commit 会写入并推进 rolling KV；二者都不是同一只读 cache/plan 状态下的纯 replay。只有中间 DMD recompute steps 满足固定 shape、固定地址、固定 saturated plan 的候选合同。后续诊断必须证明这一点，不能把编号复用当作充分理由。

## 已有 H200 作业与结果审计

- kube context：`codex-minwm-test-phx2`（已先核对 context 列表）。
- image：`829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-training@sha256:bedc07ea3ba55059a8c1c569c3b177c4d00d41f37d4fa9105375531534ef5f2a`。
- checkpoint：step-3200，version `wduScksw2f3yPErnG9lBioOuE2AToyAP`，source ETag `5c29d614972e06fd0df859abfd1d6f4d-191`。
- 旧结果 PVC：`minwm-cudagraph-sp12-h200-results-20260807`，Bound 50Gi。
- 旧 704p Job：`minwm-cudagraph-sp12-704p-h200-phx2-20260807-22`，commit `f355e52143`，2026-08-07 14:21-14:32 CST，Complete；pod 已由 TTL 删除，`kubectl logs job/...` 无可读 pod。
- 该 Job 只跑 SP1/SP2、10 warmup + 30 measured、KV=20，不符合本任务 45-frame/20+200 验收合同，只能作为历史线索。
- 旧 832x480 诊断 `2adf8e9f1b`：带 replay 后 eager verification 时 SP1 payload bitwise，但因诊断额外 eager forward 性能回退；SP2 payload 不同，sample PSNR 13.431 dB、max abs 207。此结果要求先分离单卡 graph 首差，再看 collective；不得扩展 NCCL graph。
- 用户提供的严重错误对应 PVC 中较早的 run `minwm-cudagraph-sp12-h200-phx2-20260807-07`（commit `195cd412ae`）：832x480 H200 SP1/SP2 Client FPS 约 17.71->21.84 / 16.99->22.80，但 payload hash 不同，PSNR 10.400/7.599 dB。
- `5318284afe` 的 capture/replay 诊断：capture 当次 output 对 eager `mean_abs=1.08049, max_abs=6.21875`，而 replay 1/2 与 eager exact。这证明 capture 当次结果不可直接作为用户输出。
- `2adf8e9f1b` 的 current-plan eager 诊断：SP1 从 block 4 开始所有已记录 replay exact；SP2 在 block 4/5 exact，block 6 首次出现 `mean_abs=0.67547, max_abs=4.85156`。这是 forward/chunk 级首差，不是 DiT 内部 block 索引；进一步 block 仪器仍需补齐。
- `fe414b1a6a` 增加 graph output snapshot，`f355e52143` 强引用 captured attention plan/RoPE inputs。随后 PVC run `...-20`（短样本）、`...-21`（832x480，122880 samples）以及 `...-22`（1248x704，122880 samples）均记录 SP1/SP2 payload SHA 相等、sample max abs 0。
- 这些后续结果说明已有修复方向成立，但旧验收仍是 KV=20、10+30，且不是本任务当前 tip 的独立复验；因此它们不能单独解除 NO-GO。下节记录了本任务 KV=45 的独立 Graph-only H200 复验。

### 根因与修复判定

- 根因一是 capture 边界语义：stream capture 期间产生的 output 不能当成常规 replay 结果。`068a972078` 在 capture 完成后显式 `graph.replay()`，首个用户可见结果与后续 replay 走同一路径；`5318284afe` 的 `capture != eager`、但 replay 1/2 exact 是直接证据。
- 根因二是输出生命周期：CUDA Graph 每次 replay 复用同一个静态 output allocation；旧路径把该 tensor 直接交给下游 scheduler，下一次 DMD replay 可能在 scheduler 消费完之前覆写它。`fe414b1a6a` 在离开 graph 路径前 clone，切断这条复用地址别名。
- 根因三是 capture 输入生命周期：attention plan 及其中 RoPE tensor 在 capture 前创建，graph 只记录原始设备地址；旧 runner 不持有这些对象，allocator 可在后续 chunk 回收并复用地址。`f355e52143` 把完整 plan 存入 runner 的 `capture_dependencies`，使地址与 runner 同寿命。
- 本任务没有重新发明修复，而是审计并保留来源提交，再增加可硬失败的 30-block + forward 首差仪器，使用 KV=45/current tip 独立复验上述修复是否真实成立。

## 命令与失败记录

以下命令均在本 task worktree 执行；只记录有审计/复现意义的命令，输出结论见相邻章节。

1. `rg --files -g 'AGENTS.md' ...`、`git status --short --branch`、`git branch --all --list '*minwm-cuda-graph*'`：只找到 `docs/AGENTS.md`（仅约束 docs 子树）；本 worktree 初始干净 detached main。
2. `git worktree list --porcelain`、`git log HEAD...codex/minwm-cuda-graph`、`git diff 9a9dc59cd1..codex/minwm-cuda-graph`：确认来源 worktree、提交链与 11 文件 Graph 增量。
3. `git -C .../minwm-cuda-graph status --short --branch`、进程列表检查：来源 worktree 干净，未发现该路径活跃进程。
4. `git switch -c codex/minwm-cuda-graph-correctness codex/minwm-cuda-graph`：从审计后的 `dc3c5c40d5` 安全建立独立分支。
5. `kubectl config get-contexts -o name` 后以 `--context codex-minwm-test-phx2` 查询旧 jobs/pods/PVC：context 正确；旧 704p Job Complete，pod 已删除，PVC 仍 Bound。
6. `kubectl logs` 批量读取旧 pod：第一次循环因无匹配输出/非零状态提前停止，随后改为逐 pod 查询；记录为一次命令编排失败，没有改动集群。
7. 旧 pod 日志确认多次 `BackoffLimitExceeded`，以及 `2adf8e9f1b` 的 SP1 exact / SP2 drift 诊断。旧日志含 pip 依赖冲突警告（open-clip/timm、wandb/protobuf），但安装和目标 unit tests 当次完成。
8. artifact reader 的 client/server dry-run 已生成；随后用于展示字段的 `yq` 不存在，命令链在 `kubectl apply` 前停止，因此没有创建 Pod 或改动集群。修复：改用已安装的 `jq` 读取 server-rendered JSON。
9. reader 首次 apply 后保持 Pending；事件为未容忍 `seedleap.ai/capacity-pool=minwm-test-phx2-karpenter:NoSchedule`，并非 PVC/镜像错误。修复：只删除本任务创建的 Pending reader，补齐旧 H200 Job 相同的 nodepool selector 与 toleration 后重提。
10. reader 第二次成功调度到现有节点 `i-06888dc1ca88547e1`，0 GPU；读取 `/results` 报 `Permission denied`。修复：补上仓库既有 artifact helper 使用的 SELinux `spc_t` pod security context，再只重建本 reader。
11. reader 第三次成功只读列出 PVC；第四次改为汇总全部 `summary.json` 与 verification log，确认严重 drift 的旧 run、capture/replay 首差，以及 `f355e52143` 后的 bitwise run。reader 仅占 CPU/内存，不请求 GPU；读取完成后删除。
12. `python3 -m compileall -q ...` 通过。首次本地 pytest 在 conftest 导入阶段因未设置 `PYTHONPATH` 报 `ModuleNotFoundError: sglang`，0 条测试执行；修复为 `PYTHONPATH=python` 后重跑。
13. macOS 默认 Python 3.9 不满足仓库 `>=3.10`；改用本机 Python 3.11 后，conftest 导入触发本地 torch/Inductor 的 Triton union-type 冲突，`TORCH_COMPILE_DISABLE=1` 也未规避，仍为 0 条测试执行。该失败只影响本地环境；目标用例将作为 Linux/CUDA 镜像作业的前置 pytest，不能在完成前记为通过。
14. `ruff check`、`ruff format --check`、`git diff --check` 通过；`compileall` 通过。
15. 安全检查点提交并推送：`60672ea51f6fead2bb21f0e7ad3fe0c3f9a534e0`（`debug(minwm): localize CUDA graph block parity`），远端分支 `origin/codex/minwm-cuda-graph-correctness` 已核对同 SHA。
16. `kubectl ... apply --dry-run=client/server -f ...minwm_cudagraph_correctness_sp1...yaml` 通过；server-rendered 字段确认 1 GPU、H200 Spot nodepool、固定 image/SHA、SP1、16+4、KV=45、block verify=12、结果 PVC。
17. 提交 Job `minwm-cg-correct-sp1-h200-20260807-01`；立即调度到 `i-06888dc1ca88547e1`，pod `...-lsp2c`。日志确认 checkout sglang `60672ea51f`、minWM `2efc6485f6`、Python 3.12.3、初始 torch 2.12.1+cu130、NVIDIA H200。依赖安装最终使用 torch 2.11.0+cu130；pip 报 open-clip/timm 与 wandb/protobuf 既有冲突警告，但安装未失败。
18. 目标镜像前置 pytest：Graph/cache/fallback 选择集 `14 passed, 113 deselected`；server-args 选择集 `1 passed, 117 deselected`。两组各有 20 个既有 deprecation warning，无失败。
19. checkpoint staging：14 files、34,201,345,192 bytes、46 秒；聚合 SHA-256 `1dc42d498cad84349987db2015120ce4d77e6b641f7f38c75ec9df3f942a7975`。模型转换确认 30 blocks、5,003,467,456 parameters、45-frame local/sliding window、step-3200。
20. Job 于 `2026-08-07T09:48:05Z` Succeeded。外层 eager/graph measured payload SHA 均为 `8dec25dcb7ef498776e50b0b199e4cd9de6e29a460889b91cfc32636157b2773`；16,384 个采样像素 exact fraction 1、max abs 0。
21. 完成后以只读 reader 重挂结果 PVC。reader 第一次 grep 只包含旧 verification 关键字，漏掉新 `block verification` 行；修复 reader 正则并只重建该 reader。最终取得 chunk 10-13、step 1-3 共 12 次证据：每次 31 个边界（30 blocks + forward）均 `exact=True`，无 first-difference。读取完删除 reader；不再占 CPU，Job pod 已 Completed、不占 GPU。
22. SP1 通过后检查 H200 Spot nodepool：两台 8-GPU 节点中，`i-06888dc1ca88547e1` 当时有一个 Running 4-GPU 作业，另有 4 GPU 可用；未删除或抢占其他任务。以已提交 SP1 Job 为不可变模板，`kubectl get job -o json | jq ... | kubectl create --dry-run=server -f -` 只修改 run/name、case 文件、case id、SP degrees 与 GPU request/limit；server dry-run 确认 1248x704 case、SP `2 4`、4 GPU、16+4、verify=12、KV/window=45、代码 `60672ea51f` 和固定 image。随后用相同 jq 变换创建 `minwm-cg-correct-sp24-704p-h200-20260807-01`；立即调度到同一 H200 节点，未发生 Pending 或资源争用。
23. Job 启动后复核 `aws_b200_entrypoint.sh` 发现 harness schema 缺口：`CG_DEGREES` 可配置，但合法值正则和 Python summary 循环只硬编码 SP1/SP2。因此该 Job 会先产出有效 SP2 eager/graph，再在 SP4 前以 `unsupported CUDA graph SP degree: 4` 明确失败；这不是模型 correctness 失败，也不能记成 SP4 结果。保留该 Job 取得 SP2 证据，不中途删除；本分支只把 harness allowlist/summary 扩到 SP4，待提交新 SHA 后单独复验 SP4。
24. 目标镜像前置 pytest 再次为 `14 passed, 113 deselected` 与 `1 passed, 117 deselected`。SP2 在 chunk 10-13 的 step 1-3 共 12 次 replay 上，30 blocks + forward 共 31 个边界全部 `exact=True`；外层 eager/graph measured payload SHA 均为 `7913b3cdf498ae6eda44e72e060974d62aee9497bdb0d277cee264bc1fc6f477`。短测 Client FPS 9.3632 / 9.6028，Scheduler FPS 9.3952 / 9.6473；仍不是 headline。
25. 组合 Job 最终状态 Failed，唯一末尾错误与预审一致：`unsupported CUDA graph SP degree: 4`；SP4 server 未启动、无 SP4 结果。完成 pod 内尝试用 `jq` 读 SP2 throughput，因镜像没有 jq 而失败；改用 Python 成功读取。随后增强只读 reader 同时输出无 summary 的逐 profile throughput，复核两个完整 payload hash 相同并取得 12 条内部 bitwise 日志；reader 读取后删除。
26. SP4 harness 修复通过 `bash -n` 与 `git diff --check`，提交并推送 `a67b2c9e76ecb2f16ccadc57e7e045eec57ba564`。以 server dry-run 确认该 SHA、SP4、4 GPU、1248x704、16+4、verify=12、KV/window=45 后，创建 `minwm-cg-correct-sp4-704p-h200-20260807-02`。首次调度事件为两节点 GPU 不足，随后 nominated 并启动新 Spot node `i-06678115fccf0b8d7`；未抢占其他任务。
27. 新节点首次 S3 PVC mount 报 `driver name s3.csi.aws.com not found`，当时节点上的 S3 CSI DaemonSet 尚未注册；driver pod 进入 3/3 Running 后 kubelet 自动重试成功，无需绕过存储。一次 `custom-columns=...conditions[-1]...` 状态查询被本地 zsh 当作 glob 而失败；改用 `-o json | jq`，无集群改动。pod 在创建约 3 分 11 秒后 Running，日志确认 checkout 精确 SHA `a67b2c9e76...`。
28. SP4 harness 实际启动 `--sp-degree 4 --ulysses-degree 4` 的 eager/graph 服务，不再被 allowlist 阻断。目标 pytest 仍为 `14 passed, 113 deselected` 与 `1 passed, 117 deselected`。graph lane 在 chunk 10-13、step 1-3 的 12 次 replay 上，每次 31 个 block/forward 边界均 `exact=True`；四 rank 无 first-difference、异常退出或 hang。
29. SP4 Job 于 `2026-08-07T10:12:50Z` Succeeded。eager/graph measured payload SHA 均为 `7913b3cdf498ae6eda44e72e060974d62aee9497bdb0d277cee264bc1fc6f477`；16,384 samples exact fraction 1、MAE/RMSE/max abs 均 0。短测 Client FPS 9.7776 / 10.4449，Scheduler FPS 9.7353 / 10.3226，DiT p50 817.339 / 745.4865 ms；只有一次 16+4，不能用于 headline/CV。
30. 最后一次只读 reader 复核 SP4 summary、逐 profile hash 与 12 条内部 bitwise 日志后删除。最终资源检查：本任务 SP1/SP4 pod Succeeded，组合 pod 仅因已知旧 harness 缺口 Failed；没有 Running/Pending 的本任务 reader 或 GPU pod。
31. 最终本地静态检查：相关 Python 文件 `ruff check`、`ruff format --check`、`compileall` 通过；benchmark harness `bash -n` 通过；`git diff --check` 通过。Linux/H200 目标 pytest 结果以三次作业日志中的 14+1 passed 为准。

## 当前 H200 Graph-only 复现

- Job：`minwm-cg-correct-sp1-h200-20260807-01`
- Pod/Node：`minwm-cg-correct-sp1-h200-20260807-01-lsp2c` / `i-06888dc1ca88547e1`
- 代码：`60672ea51f6fead2bb21f0e7ad3fe0c3f9a534e0`
- image：`sha256:bedc07ea3ba55059a8c1c569c3b177c4d00d41f37d4fa9105375531534ef5f2a`
- 资源：1x H200，SP1/TP1，无 distributed collective；32 CPU/200Gi request。
- 请求：832x480 BF16、step-3200、4 DMD + 1 clean commit、45-frame bounded rolling KV、block-relative RoPE。
- correctness window：16 warmup + 4 measured；`MINWM_CUDA_GRAPH_VERIFY_BLOCKS=12`。实际在 chunk 10-13 验证 4 个 chunk x 3 次 graphed DMD recompute；每次 31 个边界全部 bitwise。
- 产物：`/results/minwm-cg-correct-sp1-h200-20260807-01/cuda-graph-matrix/`。
- 结束状态：Succeeded，完成时间 `2026-08-07T09:48:05Z`；pod 不再占 GPU。
- 外层 correctness：payload SHA equal；sample count 16,384、exact fraction 1、MAE/RMSE/max abs 均 0。
- 内层 correctness：rank 0，12/12 replay；`block_00..block_29 + forward` 共 31 边界逐次 exact，首差不存在。
- 短测观测（**不是 headline 性能验收**）：Client FPS 17.3248 eager / 17.7672 graph；Scheduler FPS 17.4008 / 17.8273；DiT p50 551.977 / 517.243 ms。只有一次 16+4 run，不能报告 CV，也不能替代 20+200 paired x2。

## 待执行验证

- 本 correctness 阶段已完成：旧 PVC 审计、opt-in block/forward 首差诊断、目标镜像 unit、H200 SP1/SP2/SP4 逐层与端到端 bitwise。
- 尚未执行 20 warmup + 200 measured、同机 paired x2、CV、独立 Nsight、峰值显存、冷启动统计和至少 200 chunks 地址/显存/rank-hang 长跑；因此本文短测 FPS 不作 headline。
- 尚未覆盖完整 growing、prompt switch、scene cut、非均匀 shard、cache eviction、all-rank unsupported hit/miss/fallback 矩阵；当前代码只对 bounded saturated contract 给出实机 correctness 结论。
- growing KV、clean graph family、multi-bucket/LRU graph pool 仍禁止；graph pool 仍无容量上限/session 清理设计。unsupported shape/dtype/backend 的完整 eager 回退验收未完成前不得默认开启。

## 当前 1248x704 SP2/SP4 correctness 复验

- Job：`minwm-cg-correct-sp24-704p-h200-20260807-01`
- 初始 pod/node：`minwm-cg-correct-sp24-704p-h200-20260807-01-2qjz2` / `i-06888dc1ca88547e1`
- 代码/image/checkpoint 与 SP1 相同；4x H200，顺序运行 SP2 eager/graph 与 SP4 eager/graph。
- 请求：1248x704 BF16、step-3200、固定 45-frame rolling KV、16+4、每个 SP graph lane 前 12 次 replay 做 31-boundary matched-eager bitwise。
- 产物：`/results/minwm-cg-correct-sp24-704p-h200-20260807-01/cuda-graph-matrix/`。
- 最终状态：Failed，原因仅为旧 harness 在 SP2 成功后拒绝 SP4。SP2 外层 payload SHA equal，内部 12/12 x 31 boundaries exact；SP4 未启动、没有结果。

## 当前 SP4 独立复验

- Job：`minwm-cg-correct-sp4-704p-h200-20260807-02`
- 代码：`a67b2c9e76ecb2f16ccadc57e7e045eec57ba564`；仅比 graph runtime SHA 多 harness/文档变更。
- 请求：4x H200、SP4、1248x704、16+4、KV45、block verify=12；其余 image/checkpoint 合同不变。
- 产物：`/results/minwm-cg-correct-sp4-704p-h200-20260807-02/cuda-graph-matrix/`。
- Pod/node：`minwm-cg-correct-sp4-704p-h200-20260807-02-6pjmf` / `i-06678115fccf0b8d7`。
- 最终状态：Succeeded，完成时间 `2026-08-07T10:12:50Z`；pod 不再占 GPU。
- 外层 correctness：eager/graph payload SHA equal；sample count 16,384、exact fraction 1、MAE/RMSE/max abs 均 0。
- 内层 correctness：12/12 replay x 31 boundaries exact；四 rank 无 first-difference、异常退出或 hang。
- 短测观测（**不是 headline 性能验收**）：Client FPS 9.7776 / 10.4449；Scheduler FPS 9.7353 / 10.3226；DiT p50 817.339 / 745.4865 ms。

## 最终 go/no-go

当前：**固定 45-frame saturated Graph correctness GO；默认开启 NO-GO**。SP1、1248x704 SP2/SP4 已越过 correctness 门槛，可以进入正式性能与长跑验收；但在 20+200 paired x2、CV、Nsight、峰值显存、200-chunk 稳定性和完整 fallback 场景完成前，开关必须默认关闭，不得用本文 16+4 短测宣称 headline 性能收益，也不得讨论 graph pool 扩展。
