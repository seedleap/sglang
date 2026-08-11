# 天鹏模型直连 WebSocket 接口

## 1. 服务范围

该接口只服务天鹏模型，不接入 Zing/LingBot2 对比 WebUI。部署固定使用 1 个双卡 H100 worker（总计 2 张 H100，SP2），由独立 gateway 经 coordinator 分配会话，并通过独立 L4 上的 TAEHV 解码。为控制启动显存峰值，该模型使用 batch size 1 且不启用 CUDA Graph；接口不设置会话超时。模型版本固定为：

```text
wan22-5b-varlen-product-ws1-720p-0810-5bfc5d2-gs1500-direct
```

服务端不设置会话 idle timeout 或 hard lifetime。客户端可以持续保持连接，但必须在不用时主动关闭 WebSocket。Coordinator 仍执行容量分配、同用户单活、Worker 粘性和故障回收。

## 2. 连接地址

当前 `tianpeng-direct-public` 的 NLB 地址为：

```text
ws://k8s-minwmrea-tianpeng-4594f9e987-15648f6d6036903f.elb.us-east-2.amazonaws.com/v1/realtime_video/generate?user_id=PRODUCT_USER_ID&trace_id=TRACE_ID
```

要求：

- `user_id` 必须是产品侧稳定且唯一的用户或会话标识，不能让所有用户共用一个常量。
- 同一个 `user_id` 同时只能有一个活动连接；重复连接会收到 `USER_SESSION_LIMIT`。
- `trace_id` 建议每次 Generate 新建，长度 1 到 128，只使用字母、数字、`_ . : -`。
- 当前 NLB 是明文 WebSocket；若产品页面通过 HTTPS 提供，需要在上游配置 TLS 后使用 `wss://`。

## 3. 协议与初始化

所有控制消息均为 MessagePack 二进制。连接成功后必须先发送一条 `init`。

### I2V 初始化

```js
import { encode } from "@msgpack/msgpack";

const traceId = crypto.randomUUID();
const userId = "product-user-123";
const socket = new WebSocket(
  `ws://k8s-minwmrea-tianpeng-4594f9e987-15648f6d6036903f.elb.us-east-2.amazonaws.com/v1/realtime_video/generate?user_id=${encodeURIComponent(userId)}&trace_id=${encodeURIComponent(traceId)}`,
);
socket.binaryType = "arraybuffer";

socket.addEventListener("open", async () => {
  const firstFrame = new Uint8Array(await imageFile.arrayBuffer());
  socket.send(encode({
    type: "init",
    generation_mode: "i2v",
    model: "wan22-5b-varlen-product-ws1-720p-0810-5bfc5d2-gs1500-direct",
    prompt: "A smooth forward camera move through a mountain valley",
    size: "1280x704",
    fps: 24,
    seed: 42,
    num_inference_steps: 4,
    guidance_scale: 1,
    first_frame: firstFrame,
    realtime_output_format: "webp",
    output_compression: 55,
    realtime_causal_sink_size: 8,
    realtime_causal_kv_cache_num_frames: 32,
    trace_id: traceId,
  }));
});
```

### 连续 T2V 初始化

T2V 不传 `first_frame`。连续生成也不传 `num_frames` 和 `max_chunks`：

```js
socket.send(encode({
  type: "init",
  generation_mode: "t2v",
  model: "wan22-5b-varlen-product-ws1-720p-0810-5bfc5d2-gs1500-direct",
  prompt: "A smooth forward camera move through a mountain valley",
  size: "1280x704",
  fps: 24,
  seed: 42,
  num_inference_steps: 4,
  guidance_scale: 1,
  realtime_output_format: "webp",
  output_compression: 55,
  realtime_causal_sink_size: 8,
  realtime_causal_kv_cache_num_frames: 32,
  trace_id: traceId,
}));
```

有限 T2V 可增加 `num_frames: 121`；帧数必须满足 `1 + N * 4`，完成后服务端用 code `1000` 正常关闭。

## 4. 在线事件

`event_id` 在单条连接内严格递增。按键使用状态模式，状态会一直保持到下一次更新，无需在长按期间反复发送。

```js
// 按住 W + A
socket.send(encode({
  type: "event",
  kind: "camera_actions",
  event_id: 1,
  trace_id: traceId,
  client_sent_epoch_ms: Date.now(),
  payload: {
    mode: "state",
    transitions: [{ actions: ["w", "a"], client_ts_ms: Date.now() }],
  },
}));

// 松开全部按键
socket.send(encode({
  type: "event",
  kind: "camera_actions",
  event_id: 2,
  trace_id: traceId,
  client_sent_epoch_ms: Date.now(),
  payload: {
    mode: "state",
    transitions: [{ actions: [], client_ts_ms: Date.now() }],
  },
}));

// 动态更新 prompt，在下一个可用 chunk 边界生效
socket.send(encode({
  type: "event",
  kind: "prompt",
  event_id: 3,
  trace_id: traceId,
  payload: "Continue into a bright snowy forest",
}));
```

有效按键为 `w/a/s/d/i/j/k/l`，也接受方向键别名 `up/left/down/right`。页面失焦、隐藏或用户结束控制时，应发送空状态，避免粘键。

## 5. 视频响应

常见响应为 MessagePack `frame_batch`：

```js
{
  type: "frame_batch",
  trace_id: "...",
  chunk_index: 12,
  event_id: 3,
  content_type: "image/webp",
  num_frames: 8,
  width: 1280,
  height: 704,
  payload_lengths: [52000, 51000],
  payload: Uint8Array,
}
```

`payload` 是多张独立 WebP 图片的字节拼接。必须按 `payload_lengths` 逐张切片和解码，不能把整段 payload 当成一张图片。大消息会先收到不含 payload 的 `frame_batch_header`，紧接着下一条 WebSocket 二进制消息才是原始 payload。

客户端应按 `event_id` 判断模型实际采用了哪次按键或 prompt。播放平滑由客户端缓冲实现；后端不会按照 `fps` sleep 限速。

## 6. 关闭与错误处理

- 客户端结束时调用 `socket.close(1000, "client complete")`。
- `1013 / CAPACITY_EXHAUSTED`：当前 2 卡 Worker 的 4 个并发槽已满，读取 `retry_after_s` 后重试。
- `1008 / USER_SESSION_LIMIT`：同一 `user_id` 已有活动连接，应复用旧连接或先关闭旧连接。
- `1011`：Gateway/Worker 内部错误，使用新的 `trace_id` 重连并保留原 trace 供排障。
- 直连接口没有 45 秒限制；45 秒单人体验限制只属于 WebUI 的 Zing/LingBot2 固定体验用户。

完整字段、Trace HTTP API 和二进制拆包约定也收录在同目录的 `BACKEND_API_ZH.md`，本文件已包含产品直连所需的最小完整协议。

## 7. 当前部署与验证记录

2026-08-11 的运行态配置如下：

- 天鹏直连 Denoiser：同一台 8 卡 H100 Spot 节点中的 2 张卡，`SP=2`、`Ulysses=2`、`TP=1`。
- 天鹏直连 VAE：独立 1 张 L4，TAEHV 异步解码。
- 会话容量：单个 Denoiser Worker 最多 4 个活动会话。
- 会话时限：`idle_timeout=0`、`max_lifetime=0`，不做 45 秒自动断开。
- 模型路径：`/model-cache/wan22-5b-varlen-product-ws1-720p-0810-5bfc5d2/gs1500-dmd47-step3200-full-v1/model`。

已使用真实参考图片完成 1280x704 I2V 冒烟测试：1 个并发、1 个 warmup chunk、1 个测量 chunk，成功率 100%，客户端观测 action 到首帧约 1.45 秒。该数字用于确认完整链路可用，不等同于正式多轮性能压测结果。
