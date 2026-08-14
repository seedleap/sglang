# MinWM QKV projection 与 peer-first A2A fast lane

日期：2026-08-07；续跑：2026-08-14

任务：S4 / 点子 6

状态：**6a 的 H200 质量、确定性、v11 profiler-off ABBA 与 v12/v13 Nsight 已完成：SP2
Client −0.771%，SP4 +5.099%，因此 6a 单独启用为 NO-GO。2026-08-14 按用户要求继续推进
6b：让 peer-first Triton pack 直接读取 fused projection 的 strided V，消除每 chunk 300 个
V materialization kernel；新候选仍保持独立开关默认关闭，H200/Nsight Compute/20+200
复验进行中。**

## 2026-08-14 续跑：预估、证据与当前候选

已审计的 v12/v13 exact-window Nsight（H200、BF16、1248×704、KV45）给出以下每 rank、
每 stable chunk 结果：

| SP | lane | projection GEMM | projection CUDA / trace wall | V layout CUDA / wall | peer pack CUDA | A2A wall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | control | 450 | 19.303 / 33.317 ms | 0 / 0 | 3.221 ms | 11.849 ms |
| 2 | 6a | 150 | 19.305 / 19.305 ms | 2.802 / 3.541 ms | 3.099 ms | 11.765 ms |
| 4 | control | 450 | 11.363 / 30.063 ms | 0 / 0 | 1.660 ms | 12.469 ms |
| 4 | 6a | 150 | 10.359 / 10.359 ms | 1.713 / 2.675 ms | 1.435 ms | 12.255 ms |

这证明 6a 少掉的 300 个 GEMM launch 被 300 个 V materialization kernel 完全补回；SP2
projection CUDA 本身没有下降，收益只来自约 14 ms 的 launch span。若 6b 只消除 V copy，
理论上限约为 SP2 DiT 0.50% / SP4 0.36%。提交前先验预估为 SP2 Client +0.2%～+0.6%；
若 strided pack 同时减少同步/launch gap，目标区间为 DiT +1%～2%、Client +0.5%～+1.2%。
这些是预估，不是实测。

当前 6b 候选只做三件事：Triton pack 接受 Q/K/V 的四维 stride；USP CUDA 路径在
`MINWM_STRIDED_QKV_PACK=1` 时允许 non-contiguous 输入；fused QKV 不再执行
`value.contiguous()`。6b 开关独立且默认 `0`；只有同时启用 6a/6b 才进入目标路径。
它不修改 GEMM backend、collective、wire layout、Q/K norm、RoPE 或 cache。CPU 逻辑测试已覆盖真实 strided QKV
view，15 passed、1 CUDA skipped；CUDA kernel、质量和性能仍必须在 H200 上验证。

基础设施失败也保留：第一次 reader 复用旧固定 hostname `i-06888dc1ca88547e1`，但该节点
已被回收，Pod 90 秒 Ready wait 超时后被精确删除。随后提交的 Phoenix Local Zone Spot
2-GPU Pod `minwm-s4-qkv-ncu-h200-20260814-01` 被 Karpenter nominate 后因当时无 Spot
capacity 保持 Pending；未删除或提高其他任务优先级。新结果 PVC 为
`minwm-s4-qkv-ncu-results-20260814-v01`，旧 v12/v13 PVC 未改动。

## 结论与开关

6a 是显式 fast lane，不是 bitwise parity 修复。设置
`MINWM_FUSED_QKV_PROJECTION=1` 后，每个 MinWM self-attention block 用一个
`to_qkv` 线性层取代 `to_q`、`to_k`、`to_v` 三个线性层。权重只在模型构造/加载时
稳定打包；forward 只执行一次 linear 后用 view/chunk 切分，绝不在 forward 中拼权重。

默认值是 `0`。回滚只需取消该环境变量或设为 `0`，无需转换 checkpoint。量化权重、
未知 linear 子类和不安全的 column-parallel gather 布局会告警并自动走原三 GEMM 路径。

单 GEMM 可能因为 BF16 GEMM shape、reduction/bucket 或 cuBLASLt 算法选择变化而产生
数值差异。因此在完成 layer probe、latent、最终视频质量和确定性 A/B 前，本开关不会
作为默认路径。现阶段也不把 H200 尚未执行的项目写成已通过。

## 范围和非目标

本改动只触碰 self-attention QKV projection 及其到既有 peer-first pack 的布局边界：

```text
norm hidden states
  -> [6a] one to_qkv GEMM
  -> q/k/v views
  -> existing Q/K norm and RoPE/cache logic
  -> V contiguous for 6a；6b opt-in 直接保留 strided view
  -> existing peer-first Triton pack
  -> existing input A2A / attention / output A2A
```

以下不在 6a 中修改：RMSNorm、RoPE、KV cache 的数值/所有权语义，attention backend，
A2A collective，cross-attention QKV 和 FFN。6b 的 GEMM epilogue、定制输出布局或支持
strided V 的 pack kernel 必须等 6a profile 后再决定。

## 假设与预期

MinWM 5B 的 hidden size 是 3072。原路径的 self-attention projection 是三个
`M x 3072 @ 3072 x 3072` GEMM；6a 变为一个
`M x 3072 @ 3072 x 9216` GEMM。1248x704、4 latent frames/chunk 的 nominal token
数为 3432，因此 sequence shard 后大致是：

| SP | 每 rank 的 nominal M | 原输出 N | 6a 输出 N |
| ---: | ---: | ---: | ---: |
| 1 | 3432 | 3 次 3072 | 1 次 9216 |
| 2 | 1716 | 3 次 3072 | 1 次 9216 |
| 4 | 858 | 3 次 3072 | 1 次 9216 |

每个 chunk 有 30 blocks ×（4 DMD + 1 clean-cache）= 150 组 self QKV。忽略启动、
图捕获和后端融合时，projection GEMM kernel 的理论计数从 450 降到 150。预期收益来自：

- 每 chunk 少 300 次 GEMM launch；
- SP2/SP4 较短 M 下，用更宽的 N 聚合工作，可能提高 Tensor Core 利用率；
- 权重总元素数和数学 FLOPs 基本不变，所以如果原 GEMM 已经足够大且 Tensor Active
  已饱和，收益可能很小；
- A2A 数量和 payload 不变，所以通信主导时 headline FPS 不会按 kernel 数同比提升。

一个已知抵消项是 V：packed GEMM 的 Q/K/V 是最后一维的三个 view。Q/K norm 会产生
连续输出，但 V 仍是 strided view。SP2/SP4 的既有 Triton peer-first pack 要求三者连续，
所以 6a 在该边界显式做一次 `value.contiguous()`。这会增加一个 copy kernel；它是 6b
要用数据判断是否值得消除的主要候选，不应隐藏在“单 GEMM”收益里。

## 实现与兼容性

实现位于
`python/sglang/multimodal_gen/runtime/models/dits/minwm.py`：

- `MinWMCausalTransformerBlock.__init__` 在开关打开且布局安全时构造一份物理
  `ReplicatedLinear(3072, 9216, output_sizes=[3072] * 3)`；安全的非 gather
  column-parallel 路径使用 `MergedColumnParallelLinear`。
- 创建 `to_qkv` 后立即删除三个旧 module，因此不会同时常驻两份权重。
- `_project_qkv` 的 fast 分支只有一次 `self.to_qkv(hidden_states)` 和
  `qkv.chunk(3, dim=-1)`；权重 merge 只发生在 loader/pre-hook。
- 原生 checkpoint 的 `blocks.N.self_attn.{q,k,v}.*` 先沿用现有映射到
  `blocks.N.to_{q,k,v}.*`，再按固定 q/k/v 顺序在 load 时合到
  `blocks.N.to_qkv.*`。
- 普通 `load_state_dict` 的 block pre-hook 支持 split state dict -> fused model，也支持
  fused state dict -> fallback model。
- component/FSDP loader 的 `preprocess_loaded_state_dict` 支持把保存的 fused state dict
  拆回 fallback keys。设备和 dtype 移动由注册的单一 parameter 自然继承。
- parity dump 打开时，fused hook 仍按原文件名导出 Q/K/V 输出和 Q 权重，便于做同层对照。

state_dict 的 key 合同是“跨开关可加载”，不是“开关两边 key 文本相同”：

| 模式 | self-attention key |
| --- | --- |
| 默认/fallback | `blocks.N.to_q.*`、`to_k.*`、`to_v.*` |
| 6a fast lane | `blocks.N.to_qkv.*` |

所有非空 `quant_config` 目前都保留原格式感知的三个量化 module。这是有意的安全 fallback：
独立 Q/K/V 可能各自带 scale、zero point 或 packed metadata，未经对应格式的真实 checkpoint
验证，不把三个量化 parameter 强行拼成一份。H200 上会至少验证 BF16 和 static FP8；
H200 不支持的 NVFP4/设备组合只验“明确拒绝或安全 fallback”，不伪装成已运行的快路。

当前 MinWM causal Ulysses 本来就拒绝 TP>1 与 SP>1 组合，也拒绝 SP>1 + whole-DiT
`torch.compile`。本 PR 不扩大这些既有并行边界：SP2/SP4 在 TP1 验证，TP2 在 SP1 做
兼容 smoke；SP1 运行 compile off/on，SP2/SP4 的 compile-on 继续验证为明确拒绝。

## 本地验证边界

本地结果只证明 CPU 语义和静态正确性，不代表 CUDA、Triton、BF16 或真实 compile：

| 环境/检查 | 结果 | 边界 |
| --- | --- | --- |
| Codex Python 3.12.13 `compileall` + AST | 通过 | 无 torch/pytest |
| ruff format/check、`git diff --check` | 通过 | 静态检查 |
| Python 3.11.13、torch 2.13.0、pytest 9.1.1 | 12 passed，1 skipped | CPU 语义；CUDA compile 用例按设计跳过 |

macOS 本地 torch 在导入仓库的 eager `torch.compile` decorator 时会触发其自带
Inductor/Triton typing 错误。CPU 回归使用 `/tmp/codex-minwm-s4-cpu-site` 中的临时
`sitecustomize` 将 `torch.compile` 替换为 identity，并只在该临时目录补了 `uvicorn`；
仓库环境没有改变。真实 `torch.compile` 用例必须在 H200 镜像重新运行。

本地测试覆盖：真实 block 只有一个物理 QKV parameter、forward projection 无 cat、
多维/多 sequence shape、SP1/SP2/SP4 peer-first 线布局、原生与内部 checkpoint key、
跨开关严格 load、保存后反向加载、dtype move、量化 fallback。GPU 侧还需覆盖 BF16
layer probe、Triton kernel 是否命中、不同 head/sequence shape 和真实 compile。

## H200 验收矩阵

统一 workload：MinWM 5B step-3200，1248x704，BF16，4 DMD + 1 clean-cache，
16 frames/chunk。headline 必须是 profiler-off 的 20 warmup + 200 measured；Nsight 在
外部 20 chunks precondition 后丢 capture session 首 chunk，保留至少 10 个 steady chunks，
且不与 torch.profiler 同跑。

所有 control/candidate 的正式稳态 run 固定
`MINWM_S0_KV_CACHE_NUM_FRAMES=45`（client 等价参数
`--kv-cache-num-frames 45`），不得随 `max_chunks` 扩张。这是 rolling-window
steady-state contract。首块、短程 append/recompute、cache growth 和尚未发生淘汰的
数值行为另跑短程质量检查，不与 20+200 headline 混合归因。

| 项目 | control | candidate | 状态 |
| --- | --- | --- | --- |
| SP2 主验收 | `MINWM_FUSED_QKV_PROJECTION=0` | `=1` | 待执行 |
| SP4 复验 | `=0` | `=1` | 待执行 |
| SP1 eager / compile | `=0` | `=1` | 待执行 |
| TP2 + SP1 smoke | `=0` | `=1` | 待执行 |
| static FP8 | 原量化三 projection | 请求 6a 后安全 fallback | 待执行 |
| NVFP4/不支持设备 | 原设备合同 | 同样拒绝或 fallback | 待执行 |
| layer probe | Q/K/V、norm 后、block output | 同输入/权重/seed | 待执行 |
| latent/最终视频 | lossless latent 与 frame metrics | 同 case/seed/backend | 待执行 |
| 确定性 | candidate 重复运行 | candidate 重复运行 | 待执行 |

质量先遵循现有 contract：parity lane 要求 bitwise；本 BF16 fast lane 至少满足
`max_abs <= 8`、`RMSE <= 1.0`、`SSIM >= 0.995`。是否接受即便门槛内但有可见时序漂移，
仍需结合 latent、最终视频和 deterministic replay 审阅，不只看 FPS。

## S0 测量契约与复现

H200 临时测量分支只允许临时引入：

- S0 branch：`origin/codex/minwm-fused-ops-s0`
- S0 commit：`e75e9e24b5`（包含 `411d9b9ec4`）
- draft PR：#19

旧的 `30cb16708f` / `8e06ab2fc3` / `411d9b9ec4` 不再作为 clean runner 的最终 pin。
S0 未合并前，测量分支可以在 S4 实现 commit 上 cherry-pick `e75e9e24b5`；S4 PR 对
main 的最终 diff 必须移除 S0 基础设施。
入口使用 `benchmark/minwm_realtime_parity/run_s0_measurement.sh`，结果再经同一 commit 的
`measurement_tool.py` validate/merge-nsys/aggregate。

若 raw capture 是由 `411d9b9ec4` 启动，可以保留 `.sqlite`，但最终 JSON 必须用
`e75e9e24b5` 的工具重新 merge，并记录实际 checkout SHA，不能沿用旧 schema 的 JSON。

每个 JSON 必须记录实际 SGLang SHA、minWM SHA、镜像、GPU、SP、精度和 UTC 时间。
`provenance.gpu.count` 是 active GPU（SP2=2、SP4=4），整机隔离的 8 卡写入
`allocated_count=8`。Nsight kernel/短 kernel 保留 raw total、per-device、
per-stable-chunk；CUDA API/launch 的精确字段是 `raw_total`、`total_per_chunk`、
`per_rank_per_chunk`。只有 SQLite 能证明覆盖全部 rank 时，最后一项才 available；否则
写 unavailable 和 evidence。

S0 runner 的 `MINWM_S0_KV_CACHE_NUM_FRAMES` 必须显式记录为 45；control 和 candidate
必须完全相同。若结果使用了增长到完整 200-chunk horizon 的 cache，该结果不具备本任务
headline 资格。

profiler-off server 不打开 layerwise NVTX。单独的 Nsight server 打开
`--enable-layerwise-nvtx-marker`，用 `to_q/to_k/to_v` 或 `to_qkv` range 归因 projection；
peer-first Triton kernel 和 NCCL A2A 按 kernel/API 名归因。NVTX 只用于 profile 证据，
不进入 headline 路径。

## 实际 A/B

### Provenance

| 字段 | control | candidate |
| --- | --- | --- |
| SGLang commit | 待填 | 待填 |
| minWM commit | 待填 | 待填 |
| container image | 待填 | 待填 |
| GPU active / allocated | 待填 | 待填 |
| SP / precision / UTC | 待填 | 待填 |

### Profiler-off headline（20 + 200）

| SP | lane | Client FPS | Scheduler FPS | chunk wall | DiT wall | VAE wall | CV | 状态 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 3 GEMM control | — | — | — | — | — | — | 未执行 |
| 2 | 1 GEMM candidate | — | — | — | — | — | — | 未执行 |
| 4 | 3 GEMM control | — | — | — | — | — | — | 未执行 |
| 4 | 1 GEMM candidate | — | — | — | — | — | — | 未执行 |

### Nsight steady state（20 precondition + 1 discard + >=10）

| SP/lane | DiT CUDA | VAE CUDA | projection CUDA / wall | projection GEMM kernels | V copy | pack kernels / time | A2A time | total kernels / launches | short-kernel buckets | GPU busy | SM Active | Tensor Active | DRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| SP2 control | — | — | — | — | — | — | — | — | — | — | — | — | — |
| SP2 candidate | — | — | — | — | — | — | — | — | — | — | — | — | — |
| SP4 control | — | — | — | — | — | — | — | — | — | — | — | — | — |
| SP4 candidate | — | — | — | — | — | — | — | — | — | — | — | — | — |

空值表示尚未测量，不表示 0。若 SM/Tensor/DRAM 指标不可得，必须保留 S0 的
`unavailable + reason + evidence`，不能用 nvidia-smi utilization 冒充。

## 与预期不符处

当前只有一个本地环境偏差：macOS torch 2.13.0 的仓库导入会在 compile decorator
初始化时失败，因此本地 compile 被明确跳过并转移到 H200；这不是实现失败。

性能与数值是否符合预期尚无真机数据。完成 A/B 后，本节必须逐项解释：

- GEMM kernel 是否接近理论 3:1；若不是，是否被 compile/cuBLASLt 或 capture 合并；
- projection CUDA 下降为何没有或有转化为 DiT/chunk wall；
- 新增 V copy 与 pack、A2A、CPU launch gap 是否抵消收益；
- SP2/SP4 的 M/N shape 与 SM/Tensor Active 如何解释差异；
- 单 GEMM BF16 bucket 变化从哪一层开始影响 latent/视频。

## 证据与决策过程

1. 先选择“构造/加载时只有一份 packed parameter”，避免 forward cat 和双份权重常驻。
2. 保留环境变量默认关闭，因为 GEMM shape 改变本身就是数值契约变化。
3. 所有量化格式先 fallback；等格式逐一证明 packed scale/metadata 合同后才可能放行。
4. SP fast path 先显式 materialize V，保证继续命中已有 peer-first Triton pack；这让 6a
   的收益/代价可单独归因。
5. 只有 6a 后 pack + V copy 仍占显著 steady kernel/wall，并且可维护实现有正收益时，
   才进入 6b。若收益落入噪声、A2A 主导或实现需要侵入 GEMM backend，6b 结论就是“不做”。

## 尝试后放弃或暂缓的方案

- **forward 中 `torch.cat([Wq, Wk, Wv])`**：每次重复分配/复制权重，直接违反目标，未采用。
- **保留三个 module，再在 post-load 复制一份 fused weight**：会常驻双份约 3×3072²
  参数，并给 FSDP/device move/save 制造双源真相，未采用。
- **直接融合所有量化 QKV**：独立 scale/packed metadata 未验证，改为安全 fallback。
- **在 6a 同时写 GEMM epilogue/定制布局**：无法区分 GEMM 聚合与布局优化收益，暂缓到 6b。
- **为 profile 默认加入 NVTX**：会污染 profiler-off headline，改成只在独立 Nsight 运行打开。

## 6b go/no-go

6b 当前为 **MINIMAL IMPLEMENTATION / H200 PENDING**。用户在 2026-08-14 明确要求继续推进；
实现仍必须满足以下条件才允许保留并进入 PR：

1. 6a 已通过兼容性和质量门槛；
2. SP2 主验收中 V copy + peer-first pack 仍占可重复的显著 DiT CUDA 或 wall；
3. profiler 证明瓶颈不是 A2A wait 或其他串行阶段；
4. 候选方案能保持 fallback 和清晰接口，不要求维护私有 GEMM backend fork；
5. 6b 独立 A/B 在 profiler-off 也有收益，而非只在 Nsight 下好看。

不满足时不实现或回滚 6b，并在本节保留测量证据。

## 风险、回滚与验收状态

主要风险是 BF16 数值轨迹、FSDP/TP packed load、compile graph、量化 metadata 和 V copy
抵消收益。回滚路径始终是 `MINWM_FUSED_QKV_PROJECTION=0`；保存于任一开关下的 state
dict 都有反向加载路径。若 fast lane 启动日志没有出现 `single-gemm-fast-lane`，该次结果
必须视为 fallback，不得计入 candidate。

当前验收状态：

- 6a 实现/CPU 语义/静态检查：通过；
- H200 BF16、TP/SP、compile、量化 fallback：待执行；
- layer/latent/video/determinism：待执行；
- profiler-off / Nsight A/B：待执行；
- 6b：独立默认关闭的 strided pack 已实现，本地逻辑通过，H200/收益待执行；
- 默认开关：保持关闭。

## 给负责人掌握代码的检查题

1. **开关在哪里读，默认是什么？**

   `minwm.py` 顶部 `_MINWM_FUSED_QKV_PROJECTION`，默认 `False`。

2. **为什么说没有 forward cat？**

   `_project_qkv` fast 分支只调用 `to_qkv` 后 `chunk`；cat 只在 loader/pre-hook 合并权重。

3. **原生 `self_attn.q.weight` 如何进入 fused parameter？**

   `_minwm_fused_qkv_param_names_mapping` 先复用现有 `self_attn.q -> to_q` 链，再按
   merge index 0/1/2 合到 `to_qkv.weight`。

4. **fast state_dict 为什么能被默认模型加载？**

   普通 `load_state_dict` 由 `_minwm_qkv_load_state_dict_pre_hook` 拆分；component/FSDP
   loader 由 `preprocess_loaded_state_dict` 在 name mapping 前拆分。

5. **peer-first wire layout 是什么？**

   `_usp_pack_peer_first_qkv` 输出
   `[destination_peer, batch, local_sequence, local_heads, 3 * head_dim]`，最后一维为 Q/K/V。

6. **为什么 6a 仍有一次 V copy？**

   packed GEMM 的 V 是最后一维 strided view；Q/K norm 会物化，V 不会。SP>1 时显式
   contiguous 才能继续命中现有 Triton pack。这是 6b 的独立候选，不是隐藏融合。

7. **为什么不能把 single GEMM 称作 bitwise parity？**

   输出 N 从 3072 变 9216，BF16 GEMM bucket/algorithm/reduction 可能改变；必须看 layer
   probe、latent、最终帧和 replay，而不是只比较数学公式。

8. **projection kernel 理论上每 chunk 从多少降到多少？**

   30 blocks × 5 forwards 下从 450 降到 150；实际数以 Nsight stable-chunk 归一化为准。

9. **看到 FPS 没提升时先查什么？**

   查 projection CUDA/wall 是否真降、V copy/pack 是否增加、A2A wait 是否主导，以及
   SM/Tensor Active 和 SP 后 GEMM M shape；不能只用 kernel 数解释。

10. **什么条件下 6b 应直接放弃？**

    pack/V copy 不显著、收益落入 repeat CV 噪声、A2A 才是主瓶颈，或方案需要难维护的
    私有 GEMM backend 且 profiler-off 无收益时，保留证据并不实现。
