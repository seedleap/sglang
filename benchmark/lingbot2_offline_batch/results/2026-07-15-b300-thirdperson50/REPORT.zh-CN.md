# LingBot-World 2.0：B300 离线吞吐实测

## 结论

- 同拓扑 `4 服务 × 2 GPU` 下，B300 节点吞吐为 **2253.02 视频/小时**，是 H100 的 **1.93 倍**（+93.0%）；单条 p50 从 11.82 秒降到 6.16 秒。
- B300 的最大稳定吞吐拓扑是本轮新增的 `8 服务 × 1 GPU`：**2683.06 视频/小时**，是 H100 最优实测的 **2.30 倍**（+129.9%），也是 B300 `4×2` 的 1.19 倍。
- `8×1` 追求总产量，代价是单条 p50 为 9.59 秒；若优先单条延迟，选 B300 `4×2`。
- H100 和 B300 的服务日志都显示总 backend 名为 `fa`，但实际 kernel 不同：H100/Hopper 使用 **FlashAttention 3**；B300/Blackwell 使用 **FlashAttention 4**。

## 同口径结果

| GPU / 拓扑 | 并发请求 | 成功率 | 墙钟时间 | p50 / p95 延迟 | 节点吞吐 | 每 GPU 吞吐 | 聚合实时因子 | 相对 H100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8×H100 80GB，4×2 | 4 | 50/50 | 154.23s | 11.82 / 12.68s | 1167.11 视频/h | 145.89 视频/GPU·h | 1.641× | 1.00× |
| 8×B300 275040MiB，4×2 | 4 | 50/50 | 79.89s | **6.16 / 6.35s** | 2253.02 视频/h | 281.63 视频/GPU·h | 3.168× | **1.93×** |
| 8×B300 275040MiB，8×1 | 8 | 50/50 | **67.09s** | 9.59 / 10.39s | **2683.06 视频/h** | **335.38 视频/GPU·h** | **3.773×** | **2.30×** |

每条视频为 81 帧、832×480、16 FPS，即 5.0625 秒。吞吐包含 realtime WebSocket、raw RGB 返回、客户端 H.264 MP4 编码及落盘；不包含模型加载和每服务 3 个 chunk 的 warmup。

## FlashAttention 实际版本

| 环境 | GPU 架构 | 日志 backend | 实际 dispatch | 实现/版本 |
|---|---|---|---|---|
| H100 | Hopper SM90 | `fa` | `fa_ver=3` | `kernels-community/sgl-flash-attn3` revision `v1`，由 `kernels==0.14.1` 加载；失败时回退 `sglang-kernel==0.4.4` 的 FA3 |
| B300 | Blackwell SM103 | `fa` | `fa_ver=4`（运行时实测 `is_blackwell=True`，prepare 后 3→4） | `flash-attn-4==4.0.0b15` 的 `flash_attn.cute`，`nvidia-cutlass-dsl==4.5.2` |

两边使用相同 SGLang commit 和相同容器镜像，所以 Python 依赖中都安装了 `flash-attn-4==4.0.0b15`；这不代表 H100 会执行 FA4。代码默认 `fa_ver=3`，只有 Blackwell 平台准备阶段才调用 `set_fa_ver(4)`。

B300 实机证据：`NVIDIA B300 SXM6 AC`，compute capability `10.3`，驱动 `595.71.05`，PyTorch `2.11.0+cu130`、CUDA runtime `13.0`。服务日志明确输出 `Attention backends for transformer: fa` 和 `Using fa attention backend`。

## B300 资源利用

| 拓扑 | 活跃期平均 SM | 活跃期平均单卡功耗 | 峰值显存 | 峰值单卡功耗 |
|---|---:|---:|---:|---:|
| 4×2 | 87.9% | 886W | 80,954MiB | 1,052W |
| 8×1 | 93.6% | 977W | 99,352MiB | 1,092W |

`8×1` 能胜出主要不是单请求更快，而是 B300 的大显存允许每卡放一个完整服务，把请求并发从 4 提升到 8；单条请求在 `8×1` 上反而比 `4×2` 慢。

## 测试契约

- AWS `p6-b300.48xlarge`，Atlanta Local Zone `us-east-1-atl-2a`，8×B300；启动时 Spot 报价为 $26.4716/h。
- 对照 H100 为 `p5.48xlarge`，8×H100 80GB。
- SGLang commit：`d9a7e0e6630ea8aea135191115a13e6451618a6f`。
- 镜像：`lmsysorg/sglang:dev@sha256:8f78575e03ab59a39191a4a6f718bbbe1726fa940f72a86a187a3f1628ada9a7`。
- 模型：`robbyant/lingbot-world-v2-14b-causal-fast-diffusers`，revision `59cccf49f2d2dd27418ae7a04b82b10868d455c2`，BF16、4 个 DMD step。
- 数据：同一批 50 个第三人称图片；每 0.5 秒一个 action，每条严格 60% 单键、40% `wasd + ijkl` 组合键。
- 服务内 `batching_max_size=1`；离线并发来自独立服务进程，不是一个服务内部的 tensor batch。

## 冷启动问题记录

- 并发首次下载 Hugging Face Xet 大分片出现 `416 Range Not Satisfiable`；改为单进程普通 HTTP 预取后，26/26 文件完整。
- `8×1` 第一次冷 warmup 在一个服务上出现 `Missing realtime session state ... block_idx=1`，未进入计时；FA4/CuTe 缓存生成后做全新服务重试，50/50 成功。生产部署应先做独立 warmup，再加入任务队列。
- 4×2 首次 FA4/CuTe warmup 为 162.19 秒，8×1 重试 warmup 为 79.32 秒，均从稳态统计中排除。

原始结果见 `4x2gpu/summary.json`、`8x1gpu/summary.json`；运行时版本与 GPU inventory 见 `evidence/`。
