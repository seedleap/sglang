# MinWM saturated full-DiT CUDA Graph correctness

## 状态

- 负责人：本 Codex task（独立 worktree `/Users/chenshengdong/.codex/worktrees/4b03/sglang`）
- 当前分支：`codex/minwm-cuda-graph-correctness`
- 当前结论：**NO-GO（correctness 未恢复，默认开关必须保持关闭）**
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

原始动机是 graph 仅覆盖 saturated recompute：第一个 DMD step负责先建立/刷新本 chunk 的 attention plan，clean commit 会写入并推进 rolling KV；二者都不是同一只读 cache/plan 状态下的纯 replay。只有中间 DMD recompute steps 满足固定 shape、固定地址、固定 saturated plan 的候选合同。后续诊断必须证明这一点，不能把编号复用当作充分理由。

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
- 这些后续结果说明已有修复方向成立，但旧验收仍是 KV=20、10+30，且不是本任务当前 tip 的独立复验；因此当前结论仍保持 NO-GO，直到 KV=45 Graph-only H200 复验完成。

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

## 待执行验证

- 挂载旧结果 PVC，读取并归档 summary/server log，补齐失败作业的真实原因。
- 增加 opt-in Graph-only block/forward 首差诊断，默认关闭，不改变正常 eager/graph 输出。
- CPU/unit：开关默认关闭、unsupported eager fallback、首差比较器行为。
- H200 单卡 832x480 最小复现；同进程 matched eager vs graph，逐 DMD step、逐 block/forward bitwise。
- 只有恢复后才运行 1248x704 SP2/SP4、45-frame KV、20+200 paired 两次及长跑。

## 最终 go/no-go

当前：**NO-GO**。correctness 未恢复；不得默认开启，不得宣称性能收益。
