# MinWM Ulysses A2A 前后融合实现记录

状态：两段实现与本地 CPU 语义回归已完成；H200 CUDA parity、A/B 与 Nsight 尚未完成。

基线为 `origin/main@9a9dc59cd1`，实现分支为
`codex/ulysses-pre-post-a2a-fusion`。统一测量契约来自 S0
`411d9b9ec40b2fca2a7d85e17a05c11a4723750e`；S0 未合并前，它只叠加到临时 H200 测量分支，不进入本 PR 对 main 的
实现 diff。

## 假设与预期

本任务验证两个相互独立的假设：

1. A2A 前，把 across-head Q/K RMSNorm 与 peer-first QKV pack 合并，可减少 Q/K norm
   与 pack 的 launch、读写和短 kernel，但不能改变 Q/K/V projection 的三次 GEMM。
2. A2A 后，在不需要重排历史 cache 的 append/recompute 路径，把 Q RoPE、raw K/V
   写入和 rotated K 生成放进一个 kernel，可减少 cache copy 与 RoPE kernel。窗口淘汰
   或动态选择仍使用原路径。

端到端收益不是先验结论。A2A 是同步依赖：即使两侧 kernel 变少，如果 NCCL 或其前后
idle gap 是关键路径，DiT wall 和 Client FPS 可能不变。保留某段的门槛是数值合同通过，
且目标区域 launch/busy time 明确下降；优先目标为 DiT 至少 3%，但低于该值时仍按
critical path 证据决定是否值得保留。

## 数据布局与 ownership

| 阶段 | 每 rank 输入/输出 | token ownership | head ownership | 允许的融合 |
| --- | --- | --- | --- | --- |
| Q/K/V GEMM 后 | `[B,S_local,H_global,D]` | 本 rank 的连续 sequence shard | 全部 heads | Q/K across-head RMSNorm；V 不变 |
| peer-first send | `[P,B,S_local,H_local,3D]` | 本 rank token，按目的 peer 排列 | 每 peer 的 head slice | normalized Q、normalized K、V 直接写 send buffer |
| NCCL A2A | `all_to_all_single` | 交换 token shard | 交换 head shard | 不允许 kernel 跨越该边界 |
| A2A receive | `[B,S_global,H_local,3D]` | 全局 sequence，含各 source rank 的连续片段 | 本 rank 的 local heads | 只做 post-A2A 工作 |
| causal cache | `[B,S_visible,H_local,D]` | 当前可见的 sink/pin/tail 时间顺序 | 本 rank local heads | raw K/V 写、Q/K RoPE、rotated K 写 |

Q/K RMSNorm 的 reduction 维度是 `H_global*D=3072`。A2A 把 heads 切成
`H_local` 后已不再持有完整 reduction domain，因此 norm 必须在通信前完成。cache 在
sequence 已聚合、heads 已分片后创建，`cache_head_start=0`，所以 RoPE、cache append 和
rotated K 写只能发生在通信后。

## 开关、默认值与 fallback

| 开关 | 默认值 | 支持条件 | fallback |
| --- | --- | --- | --- |
| `MINWM_FUSED_PRE_A2A_QK_NORM` | `false` | CUDA、BF16/FP16、contiguous Q/K/V、uniform SP split、heads 可整除 Ulysses degree、norm weight 同 dtype | 原 eager Q/K norm，再走现有 uniform 或 varlen QKV pack/A2A |
| `MINWM_FUSED_POST_A2A_ROPE_CACHE` | `false` | CUDA、BF16/FP16、local-head contiguous cache、`MINWM_CACHE_ROTATED_K=true`、无需历史重排的 append/recompute | 原 cache planner/apply、Q RoPE、K RoPE 和 cache copy |

两个开关可分别做 `00/10/01/11` 消融。默认关闭，直到 H200 上 bitwise、性能和稳定性证据
齐全。非均匀 SP shard 的 pre 段明确走 varlen fallback；post 段位于 A2A 后，只要 cache
plan 不要求淘汰重排就不依赖 source shard 是否均匀。错误 prepared metadata 仍抛出原有
异常，不以 fallback 隐藏状态错误。

## cache 状态语义

- 首块/普通 append：planner 决定 `write_start` 与 visible window；fast path写 raw K/V，
  并重建整个 visible rotated K，随后才提交 position/token/cursor metadata。
- 同一 active chunk 的四次 DMD 与一次 clean-cache recompute：raw K/V 始终覆盖 active
  range；已有 rotated history 有效时只重算 current range。clean-cache 最终写因此不会被
  跳过。
- growth：planner允许完整历史且逻辑 window 大于物理 buffer 时，先沿用现有
  `_grow_to_fit`，再准备新 buffer 的 fused write。
- eviction、sink/dynamic pin 或其他 `preserves_all_history=false` 路径：不进入 post
  kernel，完整使用原 `_select_kv_with_plan` 和 metadata 提交流程。
- `absolute` 与 `block_relative` position 都由原 planner 生成 query/key position ids；
  kernel只消费对应的 FP32 cos/sin，不重新解释位置。

## 实际本地结果

| 检查 | 环境 | 结果 |
| --- | --- | --- |
| Python syntax、`ruff check`、`git diff --check` | macOS，本地 checkout | 通过 |
| cache 首块/recompute/growth/eviction fallback/错误 metadata | Python 3.11.13、torch 2.13.0，临时环境仅补 `uvicorn` | fused/pre-fallback 目标用例 `9 passed` |
| MinWM realtime CPU 语义回归 | 同上 | 全文件 `118 passed` |
| CUDA/Triton BF16 bitwise | 本机无 NVIDIA GPU/Triton | 未测，必须在 H200 完成 |
| profiler-off / Nsight | 本机无 NVIDIA GPU/Nsight | 未测，不能用历史结果代替 |

本地系统默认 Python 3.9.6，无法解析仓库的现代 union type；改用已安装的
`/opt/homebrew/bin/python3.11`。为避免修改仓库或全局环境，测试使用一次性
`--system-site-packages` venv，只安装 `uvicorn`，并设置
`TORCHDYNAMO_DISABLE=1`；这只验证 CPU cache 语义，不代表 compile 或 CUDA 结论。

## H200 A/B 与 Nsight 待填表

每个 profiler-off lane 使用 20 warmup + 200 measured，至少两次重复并由 S0 工具计算
CV；SP2 为主验收，SP4 复验。headline 只取 profiler-off。

| SP | lane | Client FPS | Scheduler FPS | chunk wall | DiT wall/CUDA | VAE wall/CUDA | parity | 状态 |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 2 | `00` baseline | — | — | — | — | — | — | 待测 |
| 2 | `10` pre | — | — | — | — | — | — | 待测 |
| 2 | `01` post | — | — | — | — | — | — | 待测 |
| 2 | `11` combined | — | — | — | — | — | — | 待测 |
| 4 | `00/10/01/11` | — | — | — | — | — | — | 待测 |

Nsight 稳态至少 10 chunks，单独记录 A2A 前后 kernel/launch、短 kernel 分桶、NCCL
SendRecv 时间与次数、两侧 idle gap、GPU kernel busy、SM Active、Tensor Active，以及
可得时的 DRAM。若 GPU metrics 因权限不可用，按 S0 schema 填 `unavailable`、原因和
采集证据，不留空。

## 与预期不符处

1. 本地完整 pytest 最初不是用例失败，而是系统 Python 3.9 语法版本不满足；切换
   Python 3.11 后，macOS torch 的模块级 compile 又需要 Triton。通过禁用 CPU-only
   compile 后，剩余仅是 `uvicorn` 轻依赖。
2. cache growth 不是简单设置 `allow_growth=true` 就会命中。若逻辑
   `attention_window_size` 已等于较小物理 cache，planner会按既有窗口语义先选择淘汰；
   真正 growth case 是逻辑 window 能保留完整历史、但物理 buffer 暂时不足。

## 证据与决策过程

- 没有把 norm 放到 A2A 后：post rank只持有 local heads，已缺失 across-head reduction
  domain，会改变数值定义。
- 没有让 pre kernel调用 NCCL：collective是跨进程同步与 ownership 迁移，不是单 GPU
  kernel 可跨越的内存边界。
- 新增 `_usp_input_all_to_all_packed_qkv` 窄接口：现有 normalized-Q/K pack 和未来 S4
  直接产出 peer-first buffer 都可复用；本任务不修改三份 GEMM 或权重结构。
- cache planner继续在 Python 层产生 plan：sink/pin/window/position 是控制语义，不在
  Triton 内重复实现。kernel先 launch，成功后再 commit metadata；不支持的 plan 在任何
  metadata mutation 前返回 fallback。
- append 先重建 visible rotated K，而不是复用旧历史：这是保守选择，避免在首版里对
  block-relative/reordered position 做未经证明的增量假设。recompute 才复用已验证历史。

## 尝试后放弃或暂缓的方案

- 跨 `all_to_all_single` 的单 kernel：ownership 与 NCCL 边界不成立，直接放弃。
- 在 post kernel内实现 eviction/dynamic pin：会复制 planner 逻辑并扩大错误面，暂缓；
  现路径安全 fallback。
- 修改 Q/K/V GEMM 数量或合并 projection：属于 S4，明确不做。
- 默认开启 fast lane：H200 bitwise 与 A/B 尚未完成，当前不具备证据。

## 风险、回滚与复现

主要风险是 Triton reduction/RoPE 指令顺序与当前 PyTorch BF16 rounding 不 bitwise，或
kernel减少但同步 A2A 仍主导 wall。任一段可以单独设为 `false` 回滚；两个开关均关闭时
执行原路径。若 post metadata/cursor 检查失败，应视为 correctness bug，不应强制 fallback
继续运行。

本地目标测试：

```bash
TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 PYTHONPATH=python \
  <temporary-python-3.11>/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py \
  -k 'fused_post_a2a'
```

H200 测量临时叠加 S0 `411d9b9ec40b2fca2a7d85e17a05c11a4723750e`，产物必须通过
`benchmark/minwm_realtime_parity/measurement_tool.py validate`；正式 PR 不包含 S0
基础设施。最终 JSON 记录实际 checkout SHA，`gpu.count` 按 active GPUs、
`gpu.allocated_count` 单列，Nsight kernel/launch 按 per-chunk/per-device 归一化。具体
Job 名与产物路径在创建后补入，且只清理带本任务唯一前缀的资源。

## 给负责人掌握代码的检查题

1. **为什么 Q/K norm不能移到 A2A 后？** 参考：`MinWMRMSNorm` 对最后的 3072 维做
   reduction，而 A2A 后只有 `H_local*D`；看 `minwm.py::_minwm_qk_norm_op`。
2. **peer-first send buffer的第一维代表什么？** 参考：目的 Ulysses peer；看
   `minwm_ulysses.py::_fused_qk_rmsnorm_pack_peer_first_kernel` 的 `peer` 计算。
3. **NCCL 后 cache为什么使用 `cache_head_start=0`？** 参考：每 rank cache创建时已只分配
   `num_heads/ulysses_degree`；看 denoising stage 的 `_num_causal_cache_attention_heads`。
4. **clean-cache 第五次 forward 会不会被 fast path跳过？** 不会；recompute仍覆盖 raw
   K/V 与 current rotated K。看 `prepare_fused_post_a2a_update` 的 recompute 分支。
5. **哪类 cache plan 必须 fallback？** `preserves_all_history=false` 的 selection/eviction，
   以及 cache head slice/grad/unsupported layout；看同一 prepare 方法与 kernel predicate。
6. **pre 与 S4 的接口边界在哪里？** S3只负责 norm+pack；S4 若改变 GEMM，可直接产出
   packed buffer，再调用 `_usp_input_all_to_all_packed_qkv`。
7. **为什么 append 当前重转整个 visible K，而 recompute只转 current K？** append采用
   保守 position 合同；recompute已持有同一 visible plan的有效 rotated history。
8. **kernel数下降但 Client FPS 不变时先看什么？** 看 A2A 前后 idle gap、NCCL
   SendRecv wall、DiT critical path与 VAE负对照，而不是只报 launch 百分比。
9. **如何一键回滚？** 两个 `MINWM_FUSED_*` 开关设为 `false`；默认本来就是关闭。
