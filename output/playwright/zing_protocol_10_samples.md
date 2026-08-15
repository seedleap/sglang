# Zing WebRTC / WebSocket 10 轮实时采样

采样场景：Spring Valley，左右两路使用同一套 Zing 模型配置和相同按键事件。按键序列为 `W, A, D, Left, Right, S, Up, Down, W, D`，每次按下 90 ms。每轮只有在 WebRTC 与 WebSocket 都呈现带有同一 `event_id` 的视频帧后才结束；10 轮均成功，没有超时或事件错配。

## 聚合结果

| 指标 | WebRTC 平均 | WebRTC P95 | WebSocket 平均 | WebSocket P95 |
|---|---:|---:|---:|---:|
| 单轮媒体数据量 | 0.837 MiB | 0.970 MiB | 5.599 MiB | 5.743 MiB |
| 接收速率 | 4.01 Mb/s | 4.40 Mb/s | 133.26 Mb/s | 160.79 Mb/s |
| 按键上行 | 159.4 ms | 163.9 ms | 171.3 ms | 200.4 ms |
| 模型 chunk | 538.9 ms | 540.0 ms | 543.2 ms | 544.5 ms |
| scheduler 总耗时 | 548.6 ms | 549.5 ms | 553.1 ms | 554.5 ms |
| VAE 排队 | 0.11 ms | 0.15 ms | 0.08 ms | 0.10 ms |
| VAE 解码 | 92.8 ms | 97.4 ms | 73.9 ms | 78.1 ms |
| 编码/编码器投喂 | 3.0 ms | 3.9 ms | 399.7 ms | 422.0 ms |
| WebRTC bridge 节奏队列 | 82.8 ms | 287.7 ms | — | — |
| 下行 | 315.7 ms | 377.1 ms | 174.3 ms | 187.6 ms |
| 浏览器播放/渲染 | 160.1 ms | 222.8 ms | 286.5 ms | 419.5 ms |
| 按键到目标帧实际呈现 | 1680.3 ms | 1815 ms | 1608.9 ms | 1850 ms |

## 结论

- 模型端不是两路差异来源：chunk 平均只差 4.3 ms。
- WebRTC 将媒体量降到 WebSocket 的约 15%，但当前测试码率只有约 4 Mb/s；WebSocket/WebP 实际约 133 Mb/s，因此清晰度差异主要来自压缩预算，而不是模型输出改变。
- WebRTC 平均按键到目标帧呈现比 WebSocket 慢 71.4 ms。主要额外开销位于 bridge 节奏队列和 H.264/RTP 到浏览器实际呈现，而不是 H.264 投喂本身。
- 第 4 轮 WebRTC bridge 队列出现 287.7 ms 长尾，是“突然闪现/卡一下”的直接证据。WebRTC 10 轮中 RTP `packetsLost` 均为 0，但累计 NACK 从 18 增至 23，说明链路仍发生少量恢复请求。
- WebSocket 的 `dropped_frames` 是累计计数，不是每轮增量；它反映前端主动丢弃过期 WebP 帧以追赶实时进度，这也解释了其主观操作响应可能更快。

## 指标口径

- `network_bytes`：从按键前到两路均呈现目标事件帧期间，浏览器收到的媒体字节增量；WebRTC 取 RTP inbound bytes，WebSocket 取 WebP payload bytes。
- `uplink_ms`：浏览器按键到 worker 接收的单向估算，使用四时间戳校正浏览器与服务器时钟偏差。
- `chunk_ms`：模型 denoise；`scheduler_total_ms` 包括调度封装。
- `encode_feed_ms`：WebSocket 是完整 WebP 编码；WebRTC 是帧打包并投喂 FFmpeg/libx264 的同步耗时。H.264 编码器内部异步耗时会落入后续下行/呈现段，不能直接把 3 ms 与 WebP 的 400 ms 当作同口径编码性能比较。
- `key_to_presented_frame_ms`：同一个浏览器单调时钟上的 keydown 到携带对应事件 ID 的视频帧真正呈现，是最可靠的端到端交互指标。

逐轮完整数值见 `zing_protocol_10_samples.csv`。
