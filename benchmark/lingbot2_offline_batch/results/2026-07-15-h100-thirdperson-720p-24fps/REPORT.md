# LingBot World 2 720p / 24 fps 第三人称离线批推理实测

- 测试时间：2026-07-15（UTC）
- 硬件：1 台 `p5.48xlarge`，8 x H100 80 GB
- 代码：`d9a7e0e6630ea8aea135191115a13e6451618a6f`
- 模型：`robbyant/lingbot-world-v2-14b-causal-fast-diffusers@59cccf49f2d2dd27418ae7a04b82b10868d455c2`

## 测试口径

- 分辨率：1280 x 720
- 帧率：24 fps
- 每条视频：10 chunks，117 帧，即 4.875 秒（模型分块约束下最接近 5 秒）
- 第三人称提示词：火星表面的宇航员与载人漫游车，外部跟踪镜头
- 每种可行拓扑：3 chunks/实例 warmup 后，测量 8 条视频
- 推理参数：4 steps，guidance scale 1.0，不启用超分或插帧
- 输出：WebSocket 返回 raw RGB；客户端同步写入 ffmpeg，保存 H.264 MP4
- 延迟和吞吐包含网络协议、本机 raw RGB 传输及 MP4 持久化，不包含服务器冷启动和 warmup

## 拓扑与吞吐

| 拓扑 | 请求级并发 | GPU/请求 | 成功率 | p50 延迟 | p95 延迟 | 节点吞吐 | 单 GPU 吞吐 | 聚合帧吞吐 | 聚合实时因子 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 x 4 GPU | 2 | 4 | 8/8 | 23.447 s | 23.792 s | **304.81 视频/小时** | **38.10 视频/GPU·小时** | **9.91 帧/秒** | **0.413x** |
| 1 x 8 GPU | 1 | 8 | 8/8 | **13.865 s** | **14.014 s** | 259.70 视频/小时 | 32.46 视频/GPU·小时 | 8.44 帧/秒 | 0.352x |
| 4 x 2 GPU | 4 | 2 | warmup OOM | - | - | 不可行 | 不可行 | 不可行 | 不可行 |

结论：若目标是吞吐，最优可行配置是 2 个独立 4 GPU 实例、请求级并发 2。相比单个 8 GPU 实例，节点吞吐提高 17.4%。若目标是单条最低延迟，则使用 1 个 8 GPU 实例。

2 GPU 实例在 720p warmup 的 VAE encode 阶段 OOM：GPU 总容量 79.18 GiB，仅剩 384.06 MiB 时仍需申请 422.00 MiB，因此当前配置不能把单节点并发扩到 4。错误同时报告 1.76 GiB reserved-but-unallocated；值得单独做 `expandable_segments` 和 1–2 GiB 峰值显存削减的 spike，但只有让 4 x 2 GPU 完整跑完后才能判断它是否会成为新的吞吐最优。

## 当前实际请求协议

这不是专用的离线 Batch API，而是离线队列客户端复用实时 WebSocket 接口：

```text
ws://<server>/v1/realtime_video/generate
Content encoding: MessagePack
```

初始化消息的关键字段：

```json
{
  "type": "init",
  "prompt": "A cinematic third-person wide tracking shot ...",
  "first_frame": "https://www.nasa.gov/wp-content/uploads/2023/03/107427main_image_feature_261_ajhfull.jpg?w=1041",
  "size": "1280x720",
  "fps": 24,
  "num_frames": 9,
  "num_inference_steps": 4,
  "guidance_scale": 1.0,
  "seed": 1000,
  "max_chunks": 10,
  "realtime_output_format": "raw",
  "output_compression": 95,
  "realtime_output_pacing": false,
  "enable_upscaling": false,
  "enable_frame_interpolation": false,
  "profile": false,
  "profile_all_stages": false,
  "condition_inputs": {
    "camera_actions": [
      ["w"], ["w"], ["w"], ["w"], ["w"], ["w"],
      ["w"], ["w"], ["w"], ["w"], ["w"], ["w"]
    ]
  }
}
```

随后发送一条 `camera_actions` event；服务端逐 chunk 返回 `chunk_stats`、`frame_batch_header` 和 raw RGB bytes。客户端为每个 URL 启动一个 worker，因此这里的“batch”是多个模型副本的请求级并发，不是单模型调用内的 tensor batch；每个实例的 `batching_max_size` 仍为 1。

## Nsight Systems 稳态 Profile

采样对象：单个 4 GPU 实例，先完成一条完整 117 帧 warmup，再只采集第二条稳态请求。profile 请求耗时 22.90 秒，与批测 p50 23.45 秒一致。

| 指标 | 实测值 |
|---|---:|
| Trace 窗口 | 22.613 s |
| GPU kernel busy | 94.5% |
| SM Active（硬件采样，4 GPU 平均） | 84.5% |
| Tensor Active（硬件采样，4 GPU 平均） | 57.2% |
| DRAM Read / Write | 14.4% / 7.7% |
| DiT 去噪阶段 | 16.094 s（75.5%） |
| VAE decode 阶段 | 5.230 s（24.5%） |

GPU 已经较充分利用，性能不是被 CPU、磁盘或显存带宽主导。按 GPU kernel 时间，最大的单类热点是 FlashAttention forward（43.8%）；主要 GEMM kernel 合计约 21.7%，NCCL SendRecv 约 5.4%。在 4 GPU/请求形态内，短期无损 kernel 优化的现实空间更像 10%–20%，而不是 2 倍。优先级应是：

1. 先做低成本显存/allocator spike，看能否让 2 GPU 请求稳定跑通；它可能解锁请求级并发 4，但目前尚无吞吐数据。
2. 优化 DiT attention / KV 路径及其 kernel 配置；当前去噪占总阶段时间 75.5%。
3. 优化或流水化 VAE decode；即使把 decode 加速 2 倍，Amdahl 上限也只有约 14%。
4. 做 4 GPU 配置内的 attention split / kernel sweep；8 GPU Ulysses 已显示过度切分，2 GPU 又受显存限制。
5. 若允许模型或质量变化，再评估量化、减少 steps、缩小 KV window；这些不是同质量的纯工程优化。

## 产物

- `comparison.json`：拓扑排名
- `1x8gpu/summary.json`、`2x4gpu/summary.json`：逐请求延迟、帧数和媒体校验
- `4x2gpu/server-2.log`：2 GPU OOM 完整证据
- `profile-4gpu/analysis.json`：硬件采样指标
- `profile-4gpu/cuda-gpu-kern-sum.txt`：CUDA kernel 时间分布
- `2x4gpu/videos/video-00-third_person.mp4`：最优吞吐配置的代表视频，已完整解码验证为 117 帧、1280 x 720、24 fps、4.875 秒
