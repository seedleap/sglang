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

当前状态：产品代码已叠加，本地 gate 已通过；H200 正确性、headline、Nsight 与最终建议
待完成。这里的“待完成”不是通过结论，后续只用真实产物替换。

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

待 H200 产物。

### profiler-off headline

待 H200 产物。最终表将按 SP、position、variant 列出 Client/Scheduler FPS、DiT/VAE
wall、repeat CV、时钟/功耗/温度与 compile-cache 状态。

### Nsight 与交互残差

待 H200 产物。最终表将同时列绝对值、相对 `000` 的变化和交互残差，避免只报百分比。

## 与预期不符之处

尚无可判定的 H200 组合数据。本节将在出现下列任一情况时逐项记录：方向与单项结论相反、
严格 bitwise 失败、fallback、CV 未过、位置漂移、all-target coverage 不完整、指标缺数、
或交互残差触发补测。环境/调度失败只标记对应 lane invalid，不改写成实现结论。

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

H200 的完整 Job、validator 和产物读取命令将在 runner SHA 固定后补入；所有命令必须显式
带 `--context codex-minwm-test-phx2`。

## 产物与 SHA

| 类别 | SHA / 路径 | 状态 |
| --- | --- | --- |
| 产品 base | `9a9dc59cd19661ac2cac649a009983c3f54d2a19` | 固定 |
| S1 产品头 | `c5d7af2269f8c622a6da2dedbe3407ca9a478427` | 已叠加 |
| S3 产品头 | `0e30671cf8a00622fd138c71af3faa93353b5425` | 已叠加 |
| S4 产品头 | `f1c9082bb12ee58d610e6e83bb4db192d9ccf96b` | 已叠加 |
| canonical exact-window | `d5b25227d4487d113e62c86a0fb572a62d6bcc5b` | runner-only |
| 产品 HEAD | 待最终提交后填写 | 待定 |
| runner HEAD | 待创建并推送 | 待定 |
| H200 root | 待分配独立 PVC/path | 待定 |
| PR | 待创建 draft | 待定 |

## 收益大小解释框架

最终解释不会只报 FPS 百分比：Client 与 Scheduler 用来区分端到端节拍和服务端调度；
DiT/VAE wall 用来定位收益是否真的落在目标阶段；kernel/API/launch 与短 kernel 解释 launch
削减；SM/Tensor/DRAM 解释计算、Tensor Core 或带宽瓶颈；时钟/功耗/温度排除降频；
compile cache、独立 server 与 ABBA 顺序用于排除编译和位置偏差。若 wall 改善而 FPS 不变，
要检查 VAE/客户端/调度瓶颈；若 launch 减少而 wall 不变，要检查 collective、同步或低利用率；
若 SP2 与 SP4 方向不同，要结合分片尺寸和通信占比解释，不能强行给一个全局开关组合。

## 最终建议

待组合数据。允许按 SP 分层：默认安全配置与 SP4 opt-in 配置由 `000/111` headline、
单项 Nsight、交互残差和严格 bitwise 共同决定，不预设 `111` 适合 SP2。

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
