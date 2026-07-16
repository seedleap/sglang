# LingBot World 2.0：5 秒第三人称离线批推理实测

## 结论

- 实测日期：2026-07-15 UTC
- GPU：1 台 `p5.48xlarge`，8×H100 80 GB，Spot
- 最优拓扑：4 个并行服务，每个服务使用 2 GPU（`4×2 GPU`）
- 请求：8 条纯第三人称视频；每个服务连续处理 2 条
- 输出：832×480、25 FPS、129 帧，即 5.16 秒；4 steps；不启用超分和插帧
- 成功率：8/8（100%）
- 测量窗口：32.539 秒，包含请求、原始 RGB 帧传输以及 H.264 MP4 落盘
- 节点吞吐：885.10 videos/hour
- 单 GPU 吞吐：110.64 videos/GPU-hour
- 单条端到端延迟：p50 16.046 秒，p95/max 16.374 秒
- 聚合实时因子：1.269×，即整台 8×H100 每秒产出约 1.269 秒成片

冷启动 186 秒和每服务 3 chunks 的预热 24.427 秒未计入上述稳态吞吐。

## 第三人称语义验证

首帧使用 NASA 的 [Humans on Mars](https://www.nasa.gov/image-article/humans-mars/) 构想图，画面内可见宇航员与载具。生成提示词要求第三人称宽景跟拍，并在全程保留宇航员与载具。

本地逐帧解码验证 `video-00` 为 129 帧、5.16 秒、832×480、25 FPS。0%、25%、50%、75%、100% 五个时间点均保持外部镜头中的宇航员与载具；镜头持续向前推进，不是第一人称视角。

## GPU 观测

在 `nvidia-dmon` 捕获的主推理活跃窗口中，72 个 GPU 采样的平均 SM 利用率为 97.9%，91.7% 的采样不低于 95%；平均显存占用约 79.3 GiB/GPU，平均功耗约 678 W/GPU。

## 文件

- `4x2gpu/summary.json`：逐请求延迟、帧数、字节数和汇总指标
- `4x2gpu/benchmark.log`：benchmark 输出
- `4x2gpu/server-{0..3}.log`：4 个双卡服务日志
- `4x2gpu/nvidia-dmon.log`：GPU 采样
- `4x2gpu/videos/video-{00..07}_third_person.mp4`：8 条完整样片；每条本地字节数均与服务端记录完全一致
- `video-00-contact-sheet.jpg`：5 个时间点的视觉 QA
- `conditioning-frame-nasa-humans-on-mars.jpg`：条件首帧

8 条 MP4 均已在本地完整解码验证为 129 帧、5.16 秒；逐文件字节数也与服务端完成落盘后生成的 `summary.json` 一致。
