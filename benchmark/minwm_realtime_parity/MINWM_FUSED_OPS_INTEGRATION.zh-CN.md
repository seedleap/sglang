# MinWM 5.3 fused ops 组合集成

## 状态与范围

本文记录 S5 组合集成的代码、正确性、性能和归因证据。产品基线为
`origin/main@9a9dc59cd19661ac2cac649a009983c3f54d2a19`，产品分支为
`codex/minwm-fusedops-integration`。组合只包含：

- S1 timestep modulation hoist，来源
  `c5d7af2269f8c622a6da2dedbe3407ca9a478427`；
- S3 post-A2A RoPE/cache fusion，来源
  `0e30671cf8a00622fd138c71af3faa93353b5425`；
- S4 fused QKV projection，来源
  `f1c9082bb12ee58d610e6e83bb4db192d9ccf96b`。

S2 `9c8d325513daf0d3c64ba07910e04cb12a949b1f` 是负结论：本集成不恢复其
postprocess runtime 候选。S0 产品/契约 SHA 为
`6c79fdfa63263814dc4e698b7bd808c6313b655c`；canonical exact-window 工具 SHA 为
`d5b25227d4487d113e62c86a0fb572a62d6bcc5b`。S0 测量代码只进入临时 runner，
不进入产品 diff。

当前状态：产品代码已叠加，本地 gate 已通过；H200 attempt-04 使用固定 runner
`49863b90512a09d1d85c4434e918e6b9418c12f0` 完成且退出码为 0。SP2/SP4 严格 bitwise、
profiler-off headline、每个 SP 8 组（共 16 组）factorial Nsight 与产物审计均已通过。最终建议是：S1 继续默认开；
S3/S4 继续默认关，但在 SP4 上同时 opt-in 为 `111`；SP2 暂不默认启用 `111`。

## 预期

1. S1 保持默认开启，并保持严格 bitwise；收益主要来自减少每层重复的 timestep
   modulation 物化。
2. S3 保持默认关闭；其单项结果提示 SP4 可能受益，SP2 不应预设有收益。
3. S4 保持默认关闭；单项结论为 SP2 不推荐、SP4 推荐，且不实现 6b output-layout 候选。
4. 组合收益应先由 `100/010/001` 的单项变化解释。若 `111` 的 wall 残差绝对值超过
   0.5pp，或 kernel/launch 变化不能由单项加和解释，必须补 `110/101/011`。
5. profiler-off 的 Client/Scheduler FPS 与 DiT/VAE wall 是 headline。profiler-on FPS
   只用于诊断，不能替代 headline。

## 产品实现与冲突审计

三个产品头相对同一 base 叠加。任意两两的只读 `merge-tree` 审计都只在
`python/sglang/multimodal_gen/runtime/models/dits/minwm.py` 出现 3 个文本冲突：

1. 模块级环境开关声明；
2. execution-profile 日志格式字符串；
3. execution-profile 日志参数。

解决只保留三方字段，没有改动邻近逻辑：

| 位 | 开关 | 默认 | `0` 回滚行为 |
| --- | --- | --- | --- |
| 1xx | `MINWM_HOIST_TIMESTEP_MODULATION` | `true` | 恢复逐 block modulation 物化 |
| x1x | `MINWM_FUSED_POST_A2A_ROPE_CACHE` | `false` | 恢复 post-A2A eager RoPE/cache 路径 |
| xx1 | `MINWM_FUSED_QKV_PROJECTION` | `false` | 恢复独立 Q/K/V projection |

日志同时打印 `hoist_timestep_modulation`、`fused_post_a2a_rope_cache` 和
`fused_qkv_requested`。S1/S3 的 realtime 单测由 Git 自动合并；核心实现路径没有语义
冲突。最终运行仍要从每个 server 日志机器核验实际开关，不能只相信 Job 名称。

## 决策与否决路径

| 路径 | 决策 | 原因与验证门槛 |
| --- | --- | --- |
| S1 hoist | 纳入，默认开 | 既有严格 bitwise 合同；组合态仍重验 |
| S3 post-A2A | 纳入，默认关 | 保留 SP4 opt-in 候选，不从单项结果外推到 SP2 |
| S3 pre-A2A | 否决 | 既有候选已失败并从 S3 产品路径删除，不恢复 |
| S4 fused QKV | 纳入，默认关 | SP2 单项不推荐、SP4 单项推荐；组合态按 SP 决策 |
| S4 6b output layout | 否决 | 明确 no-go，不实现、不测为可发布候选 |
| S2 postprocess fusion | 否决 | 只保留负结论，无 runtime 改动 |
| 把 S0 runner 合入产品 | 否决 | 测量基础设施放在临时 runner 分支，保持产品 diff 可审计 |
| 放宽数值阈值 | 否决 | latent 与最终视频必须保持既有严格 bitwise |
| profiler-on FPS 作 headline | 否决 | Nsight 会扰动延迟；只使用 profiler-off headline |
| SP2/SP4 共用 server | 否决 | 每个 lane 独立 server，隔离 compile/cache/顺序影响 |

## 验收矩阵

### 本地

| Gate | 实际 |
| --- | --- |
| `git diff --check` | 通过 |
| Python 3.11 `compileall`（7 个变更 Python 文件） | 通过 |
| ruff format/check（同一集合） | 通过 |
| MinWM realtime + QKV CPU 单测 | `138 passed, 2 skipped` |

两个 skip 分别是本机无 CUDA，以及真实 `torch.compile` 用例只在 H200 镜像执行。

### H200 profiler-off headline

SP2、SP4 各跑一轮位置平衡 ABBA。每 lane 使用独立 server、KV45、20 warmup +
200 measured；A/B 分别是 `111/000`，顺序可为 A-B-B-A 或反向。每个 variant 因此已有
两次 repeat。只有必要 headline 指标 CV 不过门，或位置对称诊断显示明显漂移时，才追加
最少 lane，并记录触发原因。初始合计 8 个 server lane，不预设 16 个。

每 lane 必须同时满足：200 个 Client latency、200 个 Scheduler latency、200 个 DiT wall、
200 个 VAE wall，且记录 Client/Scheduler FPS、峰值显存、时钟、功耗、温度和运行顺序。

### H200 exact-window Nsight

在同一 stacked binary 上，对 SP2/SP4 的 `000/100/010/001/111` 各做：20 个外部
precondition、capture 首块 discard、exact 10 stable。capture 时关闭 torch profiler，
使用 S0 `d5b25227...` 的 validator。每个结果必须记录：

- DiT/VAE wall 与 CUDA；
- kernel time/count、launch count、CUDA API time/count；
- SM Active、Tensor Active、DRAM read/write；
- 短 kernel 计数；
- all-8 target 与 active GPU 覆盖（SP2=2、SP4=4）。

交互残差定义为
`R = Δ111 - (Δ100 + Δ010 + Δ001)`。wall `|R| > 0.5pp`，或 kernel/launch 不能由
单项加和解释时，补 `110/101/011` 来定位交互对。

### H200 正确性

同一产品 SHA、正式 BF16 causal workload、4 DMD + clean-cache/recompute，对 stacked
binary 的 `000 vs 111` 比较 latent 和最终视频。要求 shape/dtype 一致、逐 bit 相等、
最大绝对差为 0，并记录两侧 SHA256。任何 capability/layout/quantization fallback 都必须
从日志或断言显式识别；不允许用更宽阈值替代 bitwise。

## 实际结果

### 正确性

`000 vs 111` 在正式 BF16 causal workload 下全部逐 bit 相等：

| SP | final video shape | array_equal | max abs diff | 两侧 SHA256 | latent 覆盖 |
| --- | --- | --- | --- | --- | --- |
| 2 | `129×704×1248×3` | `true` | `0` | `38e7ef07cffb7e8df2e59323dcbd9dacda92d31ab4a268d1276b554b7f3e833b` | rank 0/1、chunk 0..7 全部 BF16 bitwise |
| 4 | `129×704×1248×3` | `true` | `0` | `14af1068a53d0e4479fcc163fdfd5edc3415242c261852fc844cedf19e3c5a4c` | rank 0..3、chunk 0..7 全部 BF16 bitwise |

post-A2A fast lane 在 SP2/SP4 的 exact-window raw launch 分别为 `3000/6000`，即
`300/600` 次每 stable chunk，完全等于预期；explicit fallback launch slot 均为 0。
QKV candidate 在 correctness server 中实际命中，不存在 requested=true 但 runtime fallback。

### profiler-off headline

1248×704、KV45、20 warmup + 200 measured、每个 position 独立 server；正收益表示 `111`
更快：

| SP | variant | Client FPS | Scheduler FPS | chunk wall ms | DiT wall ms | VAE wall ms | 必选项 CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | `000` | 12.8177 | 12.8310 | 1279.013 | 745.405 | 420.964 | ≤0.1245% |
| 2 | `111` | 12.9468 | 12.9606 | 1299.248 | 716.927 | 421.502 | ≤0.2908% |
| 2 | improvement | **+1.007%** | **+1.010%** | **−1.582%** | **+3.820%** | **−0.128%** | 两侧通过 |
| 4 | `000` | 14.8892 | 14.9050 | 1117.003 | 746.198 | 230.723 | ≤1.1848% |
| 4 | `111` | 16.0246 | 16.0436 | 1040.562 | 670.444 | 230.996 | ≤0.4655% |
| 4 | improvement | **+7.626%** | **+7.639%** | **+6.843%** | **+10.152%** | **−0.118%** | 两侧通过 |

SP2 使用两次 repeat，无自适应追加。SP4 的首次 ABBA 在 chunk wall 出现
`000=5.309%`、`111=4.075%` 的位置漂移，故各追加一个独立 server；追加后四个必选 headline
指标全部通过 3% CV 门。`000` 的非必选 chunk-wall CV 仍为 `3.221%`，已保留为环境噪声警告，
不能写成“所有指标 CV 均通过”。

真实 session telemetry 中所有活跃 GPU 100% 为 P0。SP4 两侧 clock p50/p95 均为
`1980/1980 MHz`；`000` 三次 mean clock 为 `1973.84/1974.25/1973.88 MHz`，`111` 为
`1973.30/1973.35/1973.18 MHz`。`111` 功耗略高（约 `425.8–432.0 W` 对
`412.7–418.2 W`），最高温度 `76°C`，未享受更高频率，因此 +7.6% 不是降频或热态假象。
SP2 的 clock p50/p95 为 `1965/1980 MHz`，两侧 mean 相差不超过 6.1 MHz。

### Nsight 与交互残差

SP2/SP4 的 `000/100/010/001/111` 都触发 wall interaction；因此按契约补齐
`110/101/011`。最终共 16 个 Nsight lane，每 lane 均为 20 precondition + chunk 0 discard +
chunk 1..10 exact stable；SP2 捕获 rank/device 0..1，SP4 捕获 0..3，同时保留 all-8 target
mapping。以下数值均为每 stable chunk，wall/CUDA 单位为 ms：

| SP/config | chunk wall | DiT wall / CUDA | VAE wall / CUDA | kernel = launch | CUDA API | GPU busy | SM / Tensor / DRAM Active |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 / `000` | 1279.0 | 730.574 / 730.070 | 440.896 / 440.241 | 34608 | 91016.7 | 76.792% | 62.593% / 28.971% / 8.417% |
| 2 / `111` | 1173.1 | 626.662 / 626.150 | 440.839 / 440.098 | 25338 | 71575.1 | 78.910% | 64.821% / 31.728% / 8.195% |
| 2 / improvement | +8.280% | +14.223% / +14.234% | +0.013% / +0.033% | **−9270（−26.786%）** | **−19441.6（−21.360%）** | +2.118pp | +2.228 / +2.757 / −0.222pp |
| 4 / `000` | 1233.5 | 788.989 / 788.444 | 254.692 / 253.959 | 69202 | 184826.3 | 67.599% | 36.772% / 15.555% / 4.297% |
| 4 / `111` | 1090.9 | 658.671 / 658.173 | 251.423 / 250.776 | 50662 | 144892.0 | 66.147% | 39.291% / 17.732% / 4.475% |
| 4 / improvement | +11.561% | +16.517% / +16.523% | +1.283% / +1.253% | **−18540（−26.791%）** | **−39934.3（−21.606%）** | −1.452pp | +2.519 / +2.177 / +0.178pp |

短 kernel 也随 launch 消除而下降：SP2 `<10µs` 为 `18406.3→13087.8`、`10–50µs`
为 `12141.9→7994.6`；SP4 分别为 `49483.8→32483.1`、`13019.7→10958.2`。
kernel/launch 的 `111` 变化严格等于三个单项之和，残差为 0；CUDA API 残差只为
SP2 `−217.1`、SP4 `−304.5` 每 chunk。设备时间则不是简单相加：

| SP | all-on DiT wall 非加和残差 | `110` | `101` | `011` | 解释 |
| --- | --- | --- | --- | --- | --- |
| 2 | `−25.390 ms`（`−3.475pp` of `000`） | −2.318pp | −1.196pp | −3.739pp | 三对都协同；launch 数虽可加，关键路径/等待不可加 |
| 4 | `+2.561 ms`（`+0.325pp`） | −1.081pp | +0.154pp | +1.297pp | all-on DiT 近似可加；S3+S4 有小幅互相抵消 |

SP2/SP4 的 scheduler chunk-wall 非加和残差分别为 `−207.6 ms` 与 `+125.8 ms`。它比
DiT 残差大得多且方向随 SP 翻转，结合 profiler-off 的 SP2 chunk wall 回退，说明未归入
DiT/VAE 的同步、客户端节拍和 server 调度仍是主要噪声/瓶颈；不能只拿 profiler-on 的
`+8.28%` 宣称 SP2 端到端收益。SP4 在 profiler-off、DiT wall/CUDA 和 Client/Scheduler
三层均同向，才构成可发布的 opt-in 证据。

## 与预期不符之处

正式测量前的 runner 审计发现并修复了三项与预期不符：

1. 初版把 correctness latent dump 环境带入 200-chunk headline，会把严格 8-chunk 目录扩成
   220 个文件；修复为 bitwise-only server 与 headline server 完全隔离。
2. 隔离后的中间版仍在 A1/B1 前各跑 correctness server，可能通过进程外 compile/cache/热态
   预热污染位置平衡；修复为先跑纯 A-B-B-A，再在独立目录运行 correctness。
3. 初版二次汇总会覆盖自适应触发原因，且追加一个 lane 后没有最终 CV hard gate；现已保留
   原因，并在最小追加后仍不通过时把对应分析 lane 标为 invalid。

这些 attempt 都在正式 H200 测量前停止，不能用于性能结论。attempt-04 还出现四项与纸面
预期不同，但不构成正确性失败：

1. SP4 首轮 ABBA 两侧 chunk wall 位置漂移都超过 3%，因此实际不是最初预计的 8 个
   headline server，而是为 SP4 两侧各补一个，共 10 个；补测后必选项通过，但 `000`
   chunk-wall CV 仍为 3.221%，作为非必选噪声明确披露。
2. 三个单项 SP4 profiler-off Client 收益粗略相加为 `0.504% + 3.559% + 5.099% = 9.162%`，
   实际 `111` 只有 `+7.626%`，少 `1.536pp`。跨 PR 的单项 run 不是同一时间窗，不能把差值
   全当实现交互；同 Job factorial Nsight 则显示 DiT wall all-on 残差仅 `+0.325pp`，主差异
   更可能来自非 DiT 调度段和 run-to-run 环境漂移。
3. SP2 单项 Client 粗略相加约 `−1.048%`（S1 `−0.028%`、S3 `−0.249%`、S4
   `−0.771%`），实际组合却为 `+1.007%`。同 Job Nsight 观测到 DiT wall 额外
   `−3.475pp` 协同，但 profiler-off chunk wall 反而回退 1.582%，因此只判定“无明显 Client
   回退且存在 device 协同”，不把它升级为默认开启依据。
4. VAE 不属于三个候选的目标路径。profiler-off 中 SP2/SP4 VAE wall 分别回退
   `0.128%/0.118%`，均远低于 1%；Nsight SP4 VAE 反而改善 1.283%。两种 instrumentation
   下方向不同，结论是环境/采样扰动而非 VAE 优化。

attempt-04 无 invalid marker，严格 bitwise、fallback、all-target coverage、stage count 和
exact-window 均通过。环境/调度失败只标记对应 lane invalid，不改写成实现结论。

## H200 隔离、失败与产物保留

- kube context 固定为 `codex-minwm-test-phx2`；一次占用完整 8-GPU H200 节点；
- active GPU 为 SP2=2、SP4=4；`backoffLimit=0`；
- 每个 attempt/variant/lane 使用独立目录；
- 不抢占、删除或修改其他任务资源；
- 失败只写 lane-scoped invalid marker，保留 Pod、PVC、raw `.nsys-rep`、SQLite、日志与
  telemetry；
- 对每个产物记录路径、字节数、SHA256 和 recoverability；大型 Nsight 在 PVC 内解析，
  不依赖不完整的本地复制。

## 复现

配置位顺序固定为 hoist/post-A2A/QKV：

```text
000: MINWM_HOIST_TIMESTEP_MODULATION=0, MINWM_FUSED_POST_A2A_ROPE_CACHE=0, MINWM_FUSED_QKV_PROJECTION=0
100: MINWM_HOIST_TIMESTEP_MODULATION=1, MINWM_FUSED_POST_A2A_ROPE_CACHE=0, MINWM_FUSED_QKV_PROJECTION=0
010: MINWM_HOIST_TIMESTEP_MODULATION=0, MINWM_FUSED_POST_A2A_ROPE_CACHE=1, MINWM_FUSED_QKV_PROJECTION=0
001: MINWM_HOIST_TIMESTEP_MODULATION=0, MINWM_FUSED_POST_A2A_ROPE_CACHE=0, MINWM_FUSED_QKV_PROJECTION=1
111: MINWM_HOIST_TIMESTEP_MODULATION=1, MINWM_FUSED_POST_A2A_ROPE_CACHE=1, MINWM_FUSED_QKV_PROJECTION=1
```

本地复现：

```bash
ruff format --check \
  python/sglang/jit_kernel/diffusion/triton/minwm_ulysses.py \
  python/sglang/multimodal_gen/runtime/models/dits/{causal_wanvideo,minwm,minwm_kv_cache}.py \
  python/sglang/multimodal_gen/test/unit/{realtime/test_minwm_realtime,test_minwm_qkv_projection}.py \
  test/registered/jit/diffusion/test_minwm_ulysses_fused.py
ruff check <同一文件集合>
PYTHONPATH=python TORCHDYNAMO_DISABLE=1 python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py \
  python/sglang/multimodal_gen/test/unit/test_minwm_qkv_projection.py
```

H200 Job 与 validator 固定在 runner-only 分支；所有集群命令必须显式带
`--context codex-minwm-test-phx2`。

固定 runner 的提交为 `49863b90512a09d1d85c4434e918e6b9418c12f0`，Job manifest 位于
runner-only 分支的
`benchmark/minwm_realtime_parity/k8s/minwm_s5_fusedops_h200_20260807_attempt04.yaml`：

```bash
kubectl --context codex-minwm-test-phx2 apply --dry-run=server \
  -f benchmark/minwm_realtime_parity/k8s/minwm_s5_fusedops_h200_20260807_attempt04.yaml
kubectl --context codex-minwm-test-phx2 apply \
  -f benchmark/minwm_realtime_parity/k8s/minwm_s5_fusedops_h200_20260807_attempt04.yaml
```

## 产物与 SHA

| 类别 | SHA / 路径 | 状态 |
| --- | --- | --- |
| 产品 base | `9a9dc59cd19661ac2cac649a009983c3f54d2a19` | 固定 |
| S1 产品头 | `c5d7af2269f8c622a6da2dedbe3407ca9a478427` | 已叠加 |
| S3 产品头 | `0e30671cf8a00622fd138c71af3faa93353b5425` | 已叠加 |
| S4 产品头 | `f1c9082bb12ee58d610e6e83bb4db192d9ccf96b` | 已叠加 |
| canonical exact-window | `d5b25227d4487d113e62c86a0fb572a62d6bcc5b` | runner-only |
| H200 验证的产品树 | `3d159d20fc6cfac9bfe09fdc06dee99cd8713011` | 固定；后续只补文档 |
| 固定测量 runner | `49863b90512a09d1d85c4434e918e6b9418c12f0` | runner-only |
| runner 分支当前头 | `c65c3b09fb` | 含 attempt-04 manifest；Job 固定上行测量 SHA |
| H200 PVC | `minwm-s5-fusedops-h200-results-20260807`（200Gi） | 保留 |
| H200 attempt-04 root | `/results/attempts/minwm-s5-fusedops-h200-20260807-04-rz9q6/minwm-s5-fusedops-h200-20260807-04` | 完成；exit 0；无 invalid marker |
| profiler-off SP2 | `s5-summary/headline-sp2.json` / `6d38aa3f75f13efebc9bfde3d919c74290af5f8ef734c082efea380265e13dc0` | 5,374 bytes |
| profiler-off SP4 | `s5-summary/headline-sp4.json` / `e44226efa724840e1240e0f06da32df509dc1f930c6ac8a8b732b25e03a044ef` | 6,190 bytes |
| factorial Nsight SP2 | `s5-summary/nsys-sp2.json` / `2d3b1fed8b851bddcb1d3d0e35d2f33881c28627635a841281651f139db84b56` | 296,951 bytes |
| factorial Nsight SP4 | `s5-summary/nsys-sp4.json` / `93f1dba572937d4094a611edd79b0d5953ded59644cbe7a3b9e4cc94ed079df5` | 346,990 bytes |
| correctness SP2 | `correctness/sp2/correctness-summary.json` / `b7899f40aaac9fdb00a873d63e6860d81eba0f17565e76a1c6417c05dde1678a` | 8,397 bytes |
| correctness SP4 | `correctness/sp4/correctness-summary.json` / `a8962ffe0348a391aafba1a02e4c462e1009dd555f8d9fc41ebde2dc63ac95e8` | 14,957 bytes |
| artifact manifest | `s5-summary/artifact-manifest.json` / `cd7f8f171b01d1a97ef7e6b6de6a87a247087535c47da30e5ef2d0dc48eb9de0` | 2,479 files / 32,620,946,396 bytes；PVC 内可恢复 |
| PR | `seedleap/sglang#26` | H200 gate 完成；文档提交后转 ready |

正式测量前的 attempts 均在同一隔离节点、不同 Pod/host-scoped root 中保留：

| Attempt | 固定 runner | 终止阶段 | invalid 原因 | 可恢复性 |
| --- | --- | --- | --- | --- |
| `01` | `dfd0d4383f` | preflight/setup | correctness dump 会混入 headline | Pod/PVC/raw 保留 |
| `02` | `8e4f9a9d88` | preflight/setup | correctness server 会预热 A1/B1 | Pod/PVC/raw 保留 |
| `03` | `0e03a900a4` | preflight/setup | 自适应原因和最终 CV/fallback gate 不完整 | Pod/PVC/raw 保留 |
| `04` | `49863b9051` | 完成 | 无；exit 0；无 invalid marker | Pod 按 TTL 清理，PVC/raw 全部保留 |

## 收益大小解释框架

按 1248×704 的正式 profiler-off Client FPS 排序：S4 SP4 `+5.099%` > S3 SP4
`+3.559%` > S1 SP4 `+0.504%`；S2 无可发布 runtime 候选。三项可以叠加，但实际 `111`
为 `+7.626%`，不是简单相加的 `+9.162%`。SP2 三个单项都在近零到负向区间，组合则为
`+1.007%`，仍不足以覆盖 chunk-wall 回退和跨运行噪声，所以不做全局默认。

device 侧原因很清楚：S1/S3/S4 的 kernel/launch 变化可严格相加，SP2/SP4 分别少
`9270/18540` 次每 chunk；CUDA API 少 `21.36%/21.61%`，短 kernel 大幅减少，SM/Tensor
Active 上升。S3 的 fusion 虽新增每 chunk `300/600` 个 fused post kernel，却替代更多
RoPE/cache 小算子；S4 的 QKV module 调用从 3 GEMM 变 1 GEMM，但配套 layout kernel 会抵消
全图 count 下降，所以其主要收益是缩短 projection span；S1 则稳定减少 timestep gather/fill。
这些优化都作用在 DiT，VAE 基本不变。

为什么 SP4 收益大：分片后 GEMM 与 elementwise 更小，固定 launch/同步开销占比更高，融合后
DiT profiler-off 缩短 10.152%，Client 仍能兑现 7.626%。为什么 SP2 只兑现 1.007%：虽然
Nsight 中 DiT CUDA 缩短 14.234%，但 profiler-off 的非 DiT 调度段抵消了大部分收益，chunk
wall 反而增加 1.582%；因此设备微基准和端到端吞吐必须同时看。

## 最终建议

- **默认配置**：保持 S1 开、S3/S4 关，即现有 `100`；S1 是严格 bitwise 且无明显回退的
  低风险默认收益。
- **SP4 推荐配置**：显式开启 S3 与 S4，形成 `111`。正式 1248×704 Client/Scheduler 为
  `+7.626%/+7.639%`，DiT wall `+10.152%`，且 latent/video bitwise。
- **SP2 配置**：暂不默认开启 S3/S4。组合 Client 虽 `+1.007%`，但 chunk wall
  `−1.582%` 且单项 S3/S4 都未给出稳定端到端正收益；保留 opt-in 供特定部署复测。
- **不可直接相加**：未来容量规划采用实测 SP4 `+7.6%`、SP2 `+1.0%`，不采用单项和
  `+9.2%/−1.0%`。若模型、分辨率、KV 窗口或硬件变化，重新跑 ABBA + exact-window，而不是
  沿用本轮交互系数。

## 让我掌握代码改动：检查题

1. 三位配置中的每一位分别控制什么，为什么位序必须固定？
2. 为什么默认进程实际是 `100`，而正式 control 必须显式设置成 `000`？
3. 三方合并的 3 个冲突块各在哪里，为什么不能只选任一分支版本？
4. execution-profile 日志必须同时出现哪三个值？它与 Job 名称相比为何更可信？
5. S1 把 timestep modulation 的生命周期从哪里移到哪里？它为什么仍需检查峰值显存？
6. S3 为什么只保留 post-A2A 路径？pre-A2A 候选为什么不得在组合集成中复活？
7. post-A2A fast lane 对 cache layout/metadata 的前置条件是什么？不满足时应该怎样记录？
8. S4 为什么必须创建一个物理 QKV parameter，而不能在 forward 中临时拼接三个权重？
9. 为什么非空 `quant_config` 必须走安全 fallback？怎样证明 fallback 确实发生？
10. 为什么 6b output-layout 不在本轮实现或补测矩阵中？
11. 一轮 A-B-B-A 为什么已经给每个 variant 两次 repeat？什么条件才允许追加 lane？
12. 为什么 profiler-on FPS 不能作为 headline？Nsight capture 时还必须关闭什么 profiler？
13. exact 10 stable 的 `20 + 1 + 10` 各自表示什么？为什么 discard 不能混进统计？
14. 如何从 `100/010/001/111` 计算交互残差？哪两个条件会触发三组 pair 补测？
15. SP2 只用 2 张、SP4 只用 4 张 GPU 时，为什么仍要检查 all-8 Nsight target coverage？
16. `000 vs 111` 正确性为什么要同时比较 latent 和最终视频，并覆盖 clean-cache/recompute？
17. 如果只有一个 BF16 元素相差 1 ULP，是否可以放宽阈值？正确处置是什么？
18. 如果 kernel/launch 明显下降但 DiT wall 不降，应优先检查哪些同步、通信或硬件指标？
19. 为什么每个 lane 必须独立 server？compile cache 和运行顺序会怎样伪造收益？
20. 最终为什么可以给 SP2 和 SP4 不同建议，但不能预设 `111` 是全局最优？
21. 为什么 SP4 三个单项 Client 收益相加是 9.162%，组合却只能按 7.626% 做容量规划？
22. SP2 Nsight DiT CUDA 改善 14.234%，为什么最终仍不建议默认开启 S3/S4？
23. kernel/launch 残差为 0，但 DiT wall 残差非零，说明哪些时间不能由 launch 数线性预测？
24. SP4 `000` chunk-wall CV 为 3.221% 时，为什么 headline 仍可用，又必须附带什么警告？
