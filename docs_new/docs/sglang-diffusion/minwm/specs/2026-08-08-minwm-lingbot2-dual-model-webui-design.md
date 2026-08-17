# MinWM 与 LingBot2 双模型实时对比 WebUI 设计

## 1. 文档状态与已确认决策

本文档定义基于 `main@2b801149edd6eff64254a0498626c03750dee8d0` 的双模型实时对比能力。
用户已经确认以下产品与部署决策：

| 主题 | 决策 |
| --- | --- |
| 页面布局 | 保留现有 Realtime Studio 的左侧输入、参考图、预设、参数、录制、Trace 和底部按键；仅把单个视频播放器替换成左右两个播放器 |
| 播放器顺序 | 左侧 `MinWM`，右侧 `LingBot2` |
| 可见标题 | 每个播放器只显示模型名，不显示 SP、CUDA Graph、GPU 数量或 profile 文案 |
| 用户控制 | Generate、Stop、按键和 Prompt 更新都是一套共享输入，同时发送给两个模型 |
| 同步语义 | 两个模型收到相同的用户事件与相同的前端事件编号；媒体独立解码、独立缓冲、独立播放，不做逐帧锁步 |
| MinWM GPU | 2 个副本，每个副本 2 张 H100，`SP=2`，启用 CUDA Graph |
| LingBot2 GPU | 2 个副本，每个副本 2 张 H100，`SP=2` |
| 主机布局 | 使用当前一台 8×H100 Spot 节点，MinWM 与 LingBot2 各占 4 张卡 |
| MinWM VAE | 复用现有 L4 异步 TAEHV Worker 与 Coordinator/Gateway 控制面 |
| LingBot2 模型 | 使用仓库默认的 `robbyant/lingbot-world-v2-14b-causal-fast-diffusers` 固定 revision |
| MinWM 模型 | 使用本文第 5 节固定的 S3 checkpoint，并校验 VersionId、字节数与 SHA256 |
| 交付验证 | 完成浏览器端到端验证，并分别对 MinWM、LingBot2 和双模型模式执行 720p 并发压测 |

## 2. 目标与非目标

### 2.1 目标

- 用户只配置一次 I2V/T2V、Prompt、参考图与推理参数，就能并排比较两个模型。
- 用户的一次按键状态变化必须以同一 `event_id` 广播给两个会话，便于比较动作响应。
- 任一模型慢、断开或容量不足时，另一个模型继续播放，并在对应播放器内显示独立状态。
- 两个播放器沿用现有 Smooth Timeline、Adaptive、Low Latency 和 Full Timeline 策略，但队列、
  解码器、画布、统计数据与丢帧计数彼此隔离。
- 部署遵循飞书文档中的不可变 profile 原则：代码、依赖、checkpoint、TAEHV 资产、配置和
  Kubernetes manifest 都可追溯，不允许 Pod 启动时执行 `git clone` 或联网 `pip install`。
- 在不新增 GPU 主机的前提下完成 4+4 切分，并保留现有 MinWM L4 VAE 与 CPU 控制面。

### 2.2 非目标

- 不在浏览器或网关中对两个模型做逐帧对齐，也不因一个模型落后而阻塞另一个模型。
- 不把两个模型合并到一个 GPU 进程或一个 SGLang Session。
- 不改变 MinWM 或 LingBot2 的模型数学逻辑、Action 映射和 Prompt 语义。
- 不在本次工作中实现跨 Session KV 迁移、跨节点状态复制或 Spot 中断后的无感续播。
- 不恢复旧的可变运行时安装方式，也不复用与当前 checkpoint 不匹配的旧 MinWM profile。

## 3. 浏览器架构

### 3.1 页面结构

现有 `index.html` 的 Controls、Preview/Trace tab、Presets、History 和一套 Camera Controls
保持原有交互。`stage` 内把单个 `canvas#viewport` 替换为双播放器容器：

```text
+----------------------------- Realtime Studio ------------------------------+
| Controls |                         Preview / Trace                           |
|          |  +--------------------------+  +--------------------------+        |
| prompt   |  | MinWM                    |  | LingBot2                 |        |
| image    |  | independent canvas       |  | independent canvas      |        |
| params   |  | independent status       |  | independent status      |        |
| presets  |  +--------------------------+  +--------------------------+        |
|          |        one shared Move / Look keyboard and button controls         |
+----------------------------------------------------------------------------+
```

宽屏使用两列；窄屏按 MinWM、LingBot2 顺序纵向排列。播放器标题只渲染 `MinWM` 和
`LingBot2`。现有顶部全局录制、Preview Scale 与共享参数继续保留，模型级状态放在各自
播放器的紧凑状态行中。

### 3.2 前端会话边界

新增独立的 `RealtimeModelSession`，每个实例只负责一个模型连接：

- WebSocket 生命周期与错误状态；
- 初始化请求和模型专属 endpoint；
- 二进制媒体接收、decoder worker、playback controller 与 canvas；
- chunk、source FPS、render FPS、buffer、display lag 和 sampled event ID；
- 独立停止与资源释放。

页面级 `DualRealtimeController` 负责共享职责：

- 从现有表单构造一份基础 init payload；
- 为两个模型复制 payload，并只替换模型名或 endpoint；
- 分配单调递增的前端 `event_id`；
- 把同一 Action、Prompt、Heartbeat 和 Stop 广播给两个活动会话；
- 聚合录制与 Trace 入口，但不混用两边媒体队列。

这两个边界使后续增加第三个对比模型时不需要复制整个 `app.js`，同时避免在本次工作中
重构与双会话无关的现有录制、回放和播放策略。

## 4. 数据流与一致性

### 4.1 Generate

1. 用户点击一次 Generate。
2. 浏览器验证现有表单；I2V 只读取一次参考图，T2V 不要求参考图。
3. `DualRealtimeController` 创建一个 `comparison_id` 和两份关联 init payload。
4. 浏览器并行连接 MinWM 与 LingBot2 endpoint，不串行等待。
5. 两边分别进入 Connecting、Initializing、Live 或 Error 状态。
6. 只要至少一个会话成功，页面保留可交互状态；两个都失败时才显示全局失败。

### 4.2 Action 与 Prompt

- 浏览器维护唯一的 active key set 和唯一的用户事件序列。
- 一次 key down、key up、held-state refresh 或 Prompt 更新只生成一次 `event_id`。
- 同一个 JSON 事件对象经过序列化后分别发送给两个 WebSocket。
- 每个播放器展示自己的 `last_sent_event_id` 与服务端 `last_sampled_event_id`，从而直接看出
  两个模型是否采用了相同输入，以及各自晚了多少 chunk。
- 一个连接暂时不可写时，不阻塞另一个连接；失败侧记录 missed event，恢复时发送最新完整
  Action state，而不是重放无限历史。

### 4.3 媒体播放

- 两个会话使用独立 decoder worker 和 playback controller。
- 用户选择的 Playback、FPS、Transport、Quality 和 Size 同时应用到两个会话。
- 不做逐帧 barrier，不要求 chunk index 相同，也不把快模型人为降速到慢模型。
- 浏览器为每边单独执行有界缓冲；达到现有字节上限时按各自模式处理，不互相争用队列。
- 页面显示两个模型各自的 source FPS、render FPS、buffer 和 display lag，压测时从浏览器
  probe 同时采集。

## 5. 模型与不可变制品

### 5.1 MinWM

原始 checkpoint：

```text
s3://leap-world-us-east-2/world-model/minwm/checkpoints/dmd-merged/Wan21/Action2V/bidirectional/wan22-5B-varlen-pure-product-720p-ct1000-0806-a67b9ae/global_step_002000/dmd31-step2800-full/model.pt
```

部署前必须验证并写入 profile：

```text
VersionId: V4htNV_NC8LefJqn9.bGXElAoqfbCnFD
ContentLength: 10007165995
ETag: 3e800d48227335cc7cd413b606eb5891-1193
SHA256: 36de945826273583a8cfdfbfa1d0e6eff726c092a7e0b071e92d055028d941ca
```

MinWM 运行配置固定为 2 个 Pod，每个 Pod 2 张 H100、`SP=2`，启用
`--enable-cuda-graph`，使用 segment compile 兼容路径，并接入现有异步 L4 TAEHV。

### 5.2 LingBot2

LingBot2 使用仓库默认模型：

```text
robbyant/lingbot-world-v2-14b-causal-fast-diffusers
revision: 59cccf49f2d2dd27418ae7a04b82b10868d455c2
```

LingBot2 运行配置固定为 2 个 Pod，每个 Pod 2 张 H100、`SP=2`。镜像包含运行依赖、
TAEHV 权重和 WebUI 静态资源。部署前用 smoke case 验证当前默认模型对页面保留的 I2V/T2V
选项；不支持的 generation mode 必须在 runtime config 中显式禁用，不能让用户请求后才断开。

### 5.3 镜像与模型装载

- 代码镜像从本分支已提交 SHA 构建并推送 ECR，Kubernetes 使用 digest，不使用浮动 tag。
- 模型对象使用 VersionId 读取，先做大小与 SHA256 校验，再写入内容寻址 staging 目录。
- 启动脚本只做本地 profile 校验、模型转换/链接和进程启动；禁止修改源码和在线安装依赖。
- TAEHV 资产在镜像构建阶段下载并校验；缺失时 readiness 失败，不静默切换旧 decoder。

## 6. Kubernetes 4+4 部署

目标 H100 Spot 节点是当前 `p5.48xlarge` 8-GPU 节点。最终布局：

```text
H100 0-1: MinWM replica 0, SP=2, CUDA Graph
H100 2-3: MinWM replica 1, SP=2, CUDA Graph
H100 4-5: LingBot2 replica 0, SP=2
H100 6-7: LingBot2 replica 1, SP=2
L4 node:   existing async TAEHV worker for MinWM
CPU:       existing Coordinator/Gateway plus dual-model comparison gateway
```

Kubernetes 不依赖固定 GPU ordinal，而是依靠每个 Pod 的 `nvidia.com/gpu: 2` 请求和同一
Spot NodePool 的 scheduling constraints 实现 2+2+2+2 装箱。每个模型各有 ClusterIP
Service；比较 WebUI 的 CPU gateway 提供同源静态页面，并分别代理 MinWM 与 LingBot2
WebSocket 路由。

部署顺序：

1. 构建并验证镜像，准备 MinWM checkpoint 与两个 profile。
2. 部署 CPU comparison gateway、Services 和 replicas=0 的 GPU workloads。
3. 预检镜像 digest、模型校验值、CLI 参数、Service selectors 和节点可用性。
4. 将当前旧 MinWM `8×SP1` workload 缩容到 0，确认 reservation 已清理。
5. 同时拉起四个 2-GPU Pod，避免逐个滚动更新造成长期混合 profile。
6. 等待四个 Pod readiness，再开放 comparison gateway。
7. 若新 profile 失败，保持公网入口 fail closed；回滚到已记录的旧不可变 manifest，而不是
   在运行 Pod 中修补。

## 7. 错误处理与容量语义

- **单模型容量不足：** 对应播放器显示 `CAPACITY_EXHAUSTED`，另一模型继续；全局 Generate
  不被失败侧强制关闭。
- **单模型 WebSocket 断开：** 只关闭该 `RealtimeModelSession`，停止其 decoder worker 并
  释放 URL；另一侧连接保持活动。
- **双模型都失败：** 全局状态切换为 Error，Generate 恢复可用，保留两边独立错误原因。
- **事件发送失败：** 保留最新完整 Action state；不建立无界重试队列，不在重连后重放全部
  历史按键。
- **慢模型：** 浏览器只约束慢模型自己的 buffer；快模型不中断、不等待。
- **Stop：** 并行向两个连接发送 close，设置超时兜底，并保证多次调用幂等。
- **录制：** 双画面及共享按键区域作为一个 comparison recording；轨迹中同时记录两边
  sampled event ID，便于离线比较。
- **Trace：** Trace 页按 `comparison_id` 查询两条独立 trace，未打开 Trace 页时不把 trace
  明细塞入媒体 WebSocket。

## 8. 测试设计

### 8.1 前端单元与契约测试

先增加失败测试，再实现：

- 页面恰好有两个 canvas、两个模型标题和一套表单/按键区。
- 模型标题只包含 `MinWM`、`LingBot2`，不含 SP、GPU、CUDA Graph 文案。
- Generate 并行创建两个 `RealtimeModelSession`。
- 相同 Action/Prompt 使用相同 `event_id` 广播到两边。
- 一边媒体或错误事件不修改另一边的 queue、canvas、stats 与 lifecycle。
- Stop 并行、幂等关闭两个连接。
- I2V 参考图、T2V 无图、Presets、Recording、Trace 和现有 playback defaults 不回归。

### 8.2 部署契约测试

- MinWM 和 LingBot2 workload 都是 2 replicas、每 Pod 2 GPU、`SP=2`。
- MinWM 参数包含 CUDA Graph，LingBot2 profile 不冒充 MinWM 参数。
- checkpoint VersionId、字节数和 SHA256 与本文一致。
- 所有镜像使用 digest；Pod spec 不包含 clone、editable install 或联网 pip。
- comparison gateway 的两个 WebSocket route 指向不同 Service，静态页面同源可访问。

### 8.3 端到端验收

1. `/health`、`/v1/models`、WebUI 静态资源与两个 WebSocket route 全部通过。
2. I2V：同一参考图与 Prompt 同时启动两个模型，两边均收到首帧并持续输出。
3. T2V：不上传图片启动，按 profile 支持范围验证两边行为。
4. 持续按住 W，再切换到 A/W，验证两个模型收到相同 event ID，页面分别显示 sampled ID。
5. 动态 Prompt 更新，验证两个模型各自产生新 chunk 且不重建整个页面。
6. 人为关闭一个后端 Pod，验证对应播放器失败、另一播放器不中断。
7. Stop 后 Coordinator reservation、GPU session 和浏览器 worker 全部释放。

## 9. 720p 压测方法与验收输出

固定输入为 `1280x704`、24 FPS、9 frames/chunk、4 steps、相同 Prompt/Seed/Action 脚本，
分别执行：

1. 只连接 MinWM；
2. 只连接 LingBot2；
3. 同一浏览器会话同时连接两个模型。

每组并发依次为 `1、2、4、6、8`，每档至少包含 warmup 和 60 秒稳态样本。若出现以下任一
条件，记录容量拐点并停止继续升压，避免无意义地消耗 Spot 资源：

- 成功率低于 99%；
- chunk latency P95 连续两档恶化超过 50%；
- source FPS P50 低于 16；
- GPU OOM、Pod 重启或 admission queue 持续耗尽。

最终报告必须分别给出两个模型和双模型模式的：

- 成功并发数与首次容量失败点；
- 首帧延迟 P50/P95；
- chunk 总耗时 P50/P95；
- denoise、VAE encode/decode、transport 和 browser display lag P50/P95；
- source/render FPS、丢帧、buffer lead、WebSocket bytes；
- 每个 Pod 的 GPU 利用率、显存峰值、CPU/内存、重启次数；
- event sent 到 sampled、sampled 到 first displayed frame 的 P50/P95；
- MinWM/LingBot2 在相同输入下的并排截图与 30 秒录屏。

## 10. 完成标准

- 页面与现有 WebUI 功能等价，仅将单播放器替换为左 MinWM、右 LingBot2 双播放器。
- 两边可独立生成、播放、报错和停止，共享输入事件具有一致的 event ID。
- MinWM 和 LingBot2 均以 `2 replicas × 2 H100 × SP2` 运行；MinWM CUDA Graph 已生效。
- MinWM checkpoint 身份与本文完全匹配，运行镜像和部署 profile 可追溯。
- 完成 I2V/T2V、按键、Prompt、故障隔离和资源释放的端到端测试。
- 产出可复现的 720p 并发压测原始数据与中文报告，不用主观观感代替指标。
