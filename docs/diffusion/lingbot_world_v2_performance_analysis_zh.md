# LingBot-World 2.0 在 SGLang-Diffusion 中的性能分析与提速路线

> 调研日期：2026-07-12。本文基于当前工作树、LingBot-World 2.0 论文 v1、官方开源推理代码和模型仓库。本文没有把尚未取得的 GPU profile 当成结论；涉及瓶颈占比的判断均标注为“强推断”或“待验证”。

## 1. 结论摘要

当前观测是：同一套 SGLang LingBot-World 2.0 服务在 H100 上约 20 generated FPS、B300 上约 25 generated FPS。若每个 causal chunk 输出 9 个像素帧，则两者对应约 450 ms/chunk 和 360 ms/chunk；达到 60 generated FPS 需要降到约 150 ms/chunk，分别需要约 3.0x 和 2.4x 提速。

但论文的“720p/60 FPS”目前不能当作 SGLang 的同条件基线：

1. 论文确认 60 FPS 来自专门蒸馏的 few-step 实时模型，而非基础 14B 模型直接推理。SGLang 当前确实加载了这类 14B causal-fast 权重并使用 4 个去噪时间步，因此“用了错误权重/40-step teacher”不是当前 20–25 FPS 的主要解释。
2. 论文只笼统公开了 compiler、efficient attention、hybrid parallel、异步 latent/VAE pipeline、专用并行 VAE worker 和增量传输，没有公开 GPU 型号、GPU 数量、精度、chunk 大小、首帧是否计入、是否包含 VAE/网络/客户端，也没有发布 deployment code。官方 README 明确说明不会开源部署代码。
3. 当前 SGLang 文档示例实际是 `832x480`（约 0.40 MP），不是 720p（约 0.92 MP）。在更低分辨率上仍只有 20–25 generated FPS，说明确有明显的执行栈优化空间；但在口径对齐前，不能直接宣称“SGLang 比官方慢 2.4–3 倍”。

最可能的核心原因不是单一 kernel，而是下列三项叠加：

- **DiT 热路径没有 compile/cudagraph 化**：当前部署文档显式使用 `--enable-torch-compile false`。每个 chunk 有 4 次 DiT forward，每次经过 40 个 block，Python/dispatcher/kernel launch 成本会被重复 160 次。
- **8 卡 Ulysses 对小 chunk 的通信占比过高**：每个 causal self-attention block 都要做一次 packed QKV all-to-all 和一次 output all-to-all；4 steps × 40 layers 即每 chunk 至少 320 次关键 all-to-all。每次只生成 3 个 latent frames，小消息通信很容易压过 B300 的额外算力。B300 只比 H100 快约 25% 也是“非纯算力瓶颈”的信号，但仍需 NCCL trace 验证。
- **DiT、VAE、输出在请求关键路径上串行**：当前 composed pipeline 先完成 denoise，再同步执行 causal VAE decode，然后才物化和发送帧。论文的 60 FPS 系统则明确让 latent generation 与独立 VAE workers 异步流水，并边解码边传输。

因此，最现实的路线不是先做低收益微优化，而是：**先统一 FPS 口径并采集分段 profile → 找到最佳并行度而非默认 8-way Ulysses → compile 静态热路径 → 将 VAE/传输移出下一 chunk 的关键路径 → 再做 FP8/NVFP4 和 kernel fusion。**

## 2. 60 FPS 到底指什么

性能讨论必须区分三种 FPS：

| 指标 | 定义 | 是否可用插帧“提高” |
| --- | --- | --- |
| Generated FPS | 模型真实生成并完成 VAE decode 的帧数 / 服务端时间 | 否 |
| Delivered FPS | 客户端实际收到的帧数 / 墙钟时间 | 会受编码与网络影响 |
| Display FPS | 浏览器实际绘制帧率 | 可以被 RIFE/重复帧提高，但不代表模型更快 |

当前 WebUI 同时存在 target FPS、source/theoretical FPS 和 render FPS。验收 60 FPS 时必须关闭 frame interpolation/upscaling，以稳定状态下的 generated FPS 为主，并另报 delivered/display FPS。首 chunk 还包含图像/T5/VAE encode、cache 初始化和 warmup，应与 steady-state chunk 分开统计。

建议统一报告：

```text
model/checkpoint, precision, GPU type/count, NVLink/NVSwitch topology
resolution, pixel frames/chunk, latent frames/chunk, denoising steps
KV sink/recent window, attention backend, compile mode
warmup chunks, measured chunks, P50/P95 chunk latency
DiT ms, VAE ms, materialize/encode ms, websocket ms
generated FPS, delivered FPS, first-visible-frame latency
```

若每 chunk 为 9 帧，则：

```text
generated_fps = 9 / steady_state_chunk_seconds
60 FPS budget = 150 ms/chunk
25 FPS         = 360 ms/chunk
20 FPS         = 450 ms/chunk
```

## 3. 当前 SGLang 实现现状

### 3.1 已经具备的正确优化

当前实现并不是朴素的全序列 diffusion：

- `LingBotWorldV2CausalDMDConfig` 使用 `flow_shift=5.0` 和 `[1000, 750, 500, 250]` 四个 DMD 时间步。
- 每个 chunk 只生成 3 个 latent frames；模型是 40 层、40 heads、hidden size 5120 的 14B Wan-style DiT。
- self-attention 维护跨 chunk causal KV cache；cross-attention K/V 也在 prompt 不变时复用。
- camera conditioner 有跨 denoising step 的复用逻辑，避免同一 chunk 重算完全相同的控制特征。
- 动态 KV window 可在移动时采样 12 帧历史、静止一段时间后采样 3 帧历史；sink 默认为 3 帧。
- Wan VAE 有跨 chunk causal feature cache，并支持 spatial parallel decode。
- WebSocket 路径已有 chunk timing、增量帧传输和 WebP/JPEG/raw 等输出方式。

这些机制说明下一阶段应优化执行方式，而不是重新实现基本 causal cache。

### 3.2 当前关键路径

```text
控制信号/Prompt 采样
  → 当前 chunk latent 准备
  → 4 × [40-layer DiT forward + Ulysses collectives]
  → final context cache update
  → causal VAE decode
  → tensor → CPU/图像物化 → 编码/压缩
  → WebSocket 发送
  → 下一 chunk
```

论文部署栈的目标形态则是：

```text
GPU group A: DiT chunk N+1 ───────────────┐
GPU group B: VAE decode chunk N ──────────┼→ incremental streaming
CPU/NIC:    encode/send frames from N-1 ──┘
```

两者最重要的差异是吞吐关键路径是否包含 `T_vae + T_transport`：

```text
当前串行：T_chunk ≈ 4*T_DiT + T_context_update + T_VAE + T_output
理想流水：T_chunk ≈ max(4*T_DiT + T_context_update, T_VAE, T_output)
```

## 4. 原因分析

### 4.1 基准口径不透明（已证实，最高优先级先解决）

论文没有给出 60 FPS 的硬件和测量表，也未公开部署代码。其系统优化描述更像一个生产集群，而不是官方 GitHub `generate.py`：compiler + efficient attention + hybrid parallel + dedicated parallel VAE workers + asynchronous incremental streaming。官方开源脚本本身仍是 8 卡、480×832、4-step、同步离线保存路径。

因此需要向作者确认或通过官方 demo/服务元数据补齐：GPU 数量与型号、14B/1.3B、精度、原生生成还是插帧、分辨率、chunk size，以及 60 FPS 是否包含 VAE 和传输。在这些信息缺失时，只能把 60 FPS 当作系统目标，不能当作框架回归阈值。

### 4.2 8-way Ulysses 很可能是小 chunk 下的首要 DiT 瓶颈（强推断）

LingBot causal attention 在 sequence shard 模式下，每层执行：

1. 将 Q/K/V 拼接后做 input all-to-all；
2. 更新/采样本 rank 的 KV cache并执行 attention；
3. 做 output all-to-all。

每 chunk 4 个去噪 step、每 step 40 层，即至少 320 次关键 all-to-all。3 个 latent frames 在 480×832、patch 2×2 下约有 15,600 个 query tokens；8-way 切分后单卡计算减少，但 collective 次数不变。尤其 FFN/GEMM 随 B300 算力提高，而通信、同步和 Python launch 不按相同比例缩短，所以 H100 20 → B300 25 FPS 的弱 scaling 与通信/调度占比较高相符。

需要直接 sweep `ulysses_degree={1,2,4,8}`（其余卡可用于 decoder 或多 replica），不要假设“14B 必须 8-way SP”。最佳点取决于显存、NVLink 拓扑、KV window 和 compile 后的计算/通信比。

### 4.3 compile 当前被关闭，重复 launch 开销大（已证实 + 收益待测）

部署文档显式关闭 torch.compile，而热路径形状实际上相当稳定：固定 batch=1、分辨率、3 latent frames、4 steps、40 blocks。主要动态量是 cache position/window 和 prompt 更新。这很适合把稳定的 block/DiT forward 编译成少量固定图，把 cache 管理留在图外。

风险是动态 KV view、collective 和 Python session state导致 graph break；因此不应只看“compile 成功”，而应记录 graph breaks、编译后 kernel 数量和 steady-state latency。建议先编译单 block/单 step，再扩大到完整 DiT，避免一次性编译整个服务状态机。

### 4.4 VAE 与输出没有和下一 chunk 重叠（已证实）

当前 pipeline 将 `CausalVaeDecodingStage` 放在 denoising 后，`forward()` 内同步调用 `causal_decode()`。虽然 VAE 有 temporal feature cache 和 spatial parallel，仍然占用同一 chunk 的完成时间。论文明确使用 dedicated workers 将 latent generation 与 frame reconstruction 异步流水，并在整 chunk 解码完成前增量发送。

如果 profile 显示 VAE+输出占 80–150 ms，将其流水化可能直接决定能否进入 60 FPS；若只占 20 ms，优先级则应低于 DiT 通信/compile。需要用 timeline 决定，不应凭经验定比例。

### 4.5 动态 KV window 已存在，但默认策略仍需实测调优（已证实）

SGLang 已能按控制状态使用 moving=12、still=3 的 sampled history，并保留 sink/current chunk；这与论文的 dynamic KV cache 思路一致。但还存在三个问题：

- 文档仅用环境变量开启，缺少按场景的质量/速度曲线；
- 移动状态长期使用 12 帧是否必要尚无 ablation；
- cache 容量仍至少按基础 45 帧配置分配，虽然 attention view 被采样缩短，但 cache rolling/copy 和显存占用仍可能有成本。

建议测试 moving window `{6,9,12,18}`、still `{0,3,6}`、sink `{3,6}`，同时测长期 identity、回看一致性和动作响应。该优化可降低 attention FLOPs，但不会减少 40-layer FFN，也不会减少 collective 次数，因此单独不太可能提供 2.4–3x。

### 4.6 低精度和融合尚未构成完整的 B300 路径（待验证/中长期）

当前公开部署以 BF16 为主。B300 的优势要通过 FP8/NVFP4 GEMM、量化权重/激活、合适 attention kernel 和更大的局部工作量才能充分释放。可考虑：

- FP8 weight/activation 或 weight-only 量化，优先覆盖 FFN 和线性层；
- fused QKV、AdaLN/modulation、RMSNorm、RoPE、residual+norm；
- cache update 的 ring buffer，避免 eviction 时 `clone`/搬移；
- 固定 shape 的 CUDA Graph；
- 针对 B300 的 tile/autotune，而不是沿用 H100 配置。

但量化属于数值有损路径，必须通过短期画质、动作遵循和 10–60 分钟 rollout 一起验收。论文没有公开其生产精度，因此不能假设官方 60 FPS 是 BF16。

### 4.7 1.3B 不是当前可直接替换的答案（已证实）

论文提到 1.3B 单 GPU版本，但截至调研日，官方 README 仍将 1.3B causal-fast 和 causal-pretrained 标为 TODO。SGLang 当前注册的是 14B causal-fast diffusers checkpoint。未来 1.3B 发布后，它会是达到单机高 FPS 的最直接模型级方案，但画质/容量不同，不能用其结果声称 14B 执行栈达到同样性能。

## 5. 提速路线与优先级

### P0：先建立可信 profile（1–2 天）

目标：回答每个 360/450 ms chunk 到底花在哪里。

1. 固定 832×480、9 pixel frames/chunk、4 steps、CFG=1、关闭插帧/超分，warmup 10 chunks，统计后续至少 100 chunks。
2. 记录现有 `chunk_stats`，补齐或启用 `text/condition`、每个 DiT step、context update、VAE、D2H/materialize、encode/compress、WebSocket 等分段 CUDA event。
3. 使用 Nsight Systems 抓 3–5 个 steady chunks，打开 NCCL trace；使用 PyTorch profiler 抓算子、shape、kernel launch 和 graph break。
4. 同配置比较 H100/B300，报告单卡利用率、HBM 带宽、NCCL 时间、collective message size。

通过条件：分段时间加和与 wall-clock 误差 <5%，能明确 DiT compute、NCCL、VAE、输出各占比。

### P1：并行度 sweep + compile（最可能的近期主收益）

按下列矩阵做实验：

| 变量 | 候选 |
| --- | --- |
| Ulysses | 1 / 2 / 4 / 8（受显存约束） |
| KV moving window | 6 / 9 / 12 / 18 |
| attention backend | FA / 可用的硬件专用 backend |
| compile | off / default / reduce-overhead / max-autotune |
| VAE parallelism | 1 / 2 / 4，或独立 decoder worker |

先找到不 compile 时的最佳 Ulysses，再打开 compile，因为 compile 会改变 compute/communication 比。目标是让 P50 DiT+context update 接近或低于 150 ms；否则即使完全隐藏 VAE 也无法达到 60 FPS。

### P2：拆分异步 decoder pipeline（系统级必要条件）

实现 bounded latent queue：denoiser 将 chunk N latent 写入队列后立即开始 N+1；decoder worker 保持独立的 causal VAE session state；output worker 将 frame/timestamp/event id 按 chunk 顺序重排并增量发送。

关键约束：

- queue 必须有 backpressure，不能靠无限缓存伪造吞吐；
- VAE causal cache 必须按 session 串行，不能让同一 session 的 chunk 乱序；
- prompt/control event 的生效 chunk 和输出 metadata 必须保持一致；
- 分别报告 pipeline fill latency、steady throughput 和 user-input-to-first-visible-frame latency。

若 DiT=140 ms、VAE=90 ms、output=30 ms，串行为 260 ms（34.6 FPS），理想流水可接近 140 ms（64.3 FPS）；这说明异步的价值取决于 DiT 先被压到 150 ms 左右。

### P3：B300 专用低精度与融合

1. 从 FFN/linear 的 FP8 开始，逐层比较误差和 rollout；再评估 attention/cache 的 FP8。
2. 融合 QKV、norm/modulation/residual；减少每层小 kernel 数量。
3. 将固定 shape 的 4-step forward 捕获为 CUDA Graph，动态 cache index 通过 tensor 参数传入。
4. 使用真正的 ring KV cache，避免窗口滑动时全量搬移。

验收不能只看单 chunk PSNR：至少包含固定 seed parity、动作响应、prompt 切换、10 分钟 rollout 和代表性 60 分钟压力测试。

### P4：模型级路线

- 等官方 1.3B causal-fast 发布后增加独立模型支持和质量/速度档位；
- 若允许重新蒸馏，研究 2-step/1-step student，但这是训练项目，不应与 serving-only 优化混为一谈；
- 若最终产品允许插帧，可用 30 generated FPS + 2x RIFE 达到 60 display FPS，但指标必须明确标成 display FPS，且交互响应仍由 30 FPS source 限制。

## 6. 推荐的目标拆解

| 阶段 | 目标 | 说明 |
| --- | --- | --- |
| Baseline | 复现 H100 20 / B300 25 generated FPS | 同 checkpoint、同请求、同 steady-state 口径 |
| Milestone A | B300 32–40 generated FPS | 并行度、KV、attention backend、compile |
| Milestone B | B300 45–55 generated FPS | DiT compile/fusion + VAE/output overlap |
| Milestone C | B300 ≥60 generated FPS | 可能需要 FP8/NVFP4、CUDA Graph、专用 decoder GPU 或更佳模型 |

这些数字是工程目标而非已测收益。每一阶段都应保留质量 gate，不允许通过缩短上下文、减少真实生成帧、开启插帧或排除 VAE 时间来“达标”。

## 7. 建议立即执行的实验顺序

1. **口径复现**：保存完整启动命令、checkpoint revision、请求 payload、GPU topology 和 100-chunk timing CSV。
2. **分段 profile**：首先判断 `DiT/NCCL` 是否 >70%；若是，优先 Ulysses/compile，若 VAE/output >30%，并行启动 decoder 拆分。
3. **Ulysses sweep**：特别测试 4-way 是否优于 8-way；剩余 GPU 可做独立 VAE 或双 replica。
4. **compile spike**：固定 KV window 和 shape，确认 graph break 后再正式接入。
5. **异步 VAE spike**：先用双 CUDA stream/双进程验证理论 overlap，再设计正式 session queue。
6. **低精度 spike**：只在 profile 证明 GEMM 占主导后进行；否则量化不会解决 NCCL/launch/VAE。

## 8. 证据与代码锚点

仓库内关键位置：

- V2 4-step 配置：`python/sglang/multimodal_gen/configs/pipeline_configs/lingbot_world.py`
- 40-layer/3-latent-frame 模型配置：`python/sglang/multimodal_gen/configs/models/dits/lingbot_world.py`
- 4-step loop 和 final context update：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/causal_denoising.py`
- LingBot 动态 KV window：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/lingbot_world/lingbot_world_causal_denoising.py`
- Ulysses input/output all-to-all：`python/sglang/multimodal_gen/runtime/models/dits/lingbot_world.py`
- 同步 causal VAE decode：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/realtime/vae.py`
- composed pipeline 顺序：`python/sglang/multimodal_gen/runtime/pipelines/lingbot_world_causal_dmd_pipeline.py`
- 当前部署示例与 API timing 字段：`docs_new/cookbook/diffusion/LingBot-World/LingBot-World-2.0.mdx`

外部一手资料：

- [LingBot-World 2.0 论文（arXiv 2607.07534）](https://arxiv.org/abs/2607.07534)：few-step consistency+DMD、compiler/attention/hybrid parallel、异步 VAE workers、增量 streaming、动态 KV cache，以及 720p/60 FPS 声明。
- [官方 LingBot-World 2.0 仓库](https://github.com/robbyant/lingbot-world-v2)：公开 14B causal-fast、4-step/8-GPU/480×832 推理脚本；1.3B 尚未发布；明确不计划开源 deployment code。
- [官方 14B causal-fast checkpoint](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast)：用于确认当前比较应绑定到具体权重 revision。

## 9. 最终判断

当前 SGLang 的 20–25 FPS 不像是“模型不支持实时”，而是“已经采用正确的 4-step causal-fast 模型，但尚未复现论文生产部署栈的执行方式”。最值得投入的不是继续堆零散 cache trick，而是先用 profile 将 150 ms/chunk 的预算拆开，并围绕三个结构性问题推进：

1. 小 chunk 下 8-way Ulysses 的通信/同步成本；
2. 未编译的 4×40-layer 热路径；
3. DiT、VAE、物化/传输的串行关键路径。

在没有官方硬件口径和当前 Nsight trace 前，不能承诺单靠 serving 优化一定让 14B BF16 在相同 GPU 数上达到原生 60 FPS；但上述路线能够明确判断 60 FPS 是可通过执行栈达到，还是必须依赖低精度、额外 decoder GPU、1.3B 模型或重新蒸馏。
