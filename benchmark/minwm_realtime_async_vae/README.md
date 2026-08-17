# MinWM Realtime Async VAE Benchmark

该目录提供同步 VAE 与异步 VAE 的同合同 A/B 压测。每个并发档位会建立独立 WebSocket Session，先运行 warmup chunk，再交替发送完整 Action 状态并记录 action 到首批帧到达、chunk 总耗时、阶段 Trace、FPS 和错误率。

生产验收不允许直连 Denoiser。完整入口必须是
`NLB -> Gateway -> Coordinator -> H100 Denoiser -> L4/L40S TAEHV -> Gateway -> Browser`。
AWS 控制面、不可变镜像、版本化模型制品、Kubernetes 拓扑、真实浏览器探针和显式清理
脚本都在本目录中，详细发布顺序见 `k8s/README.md`。

- 自定义 UI 接入协议：[`BACKEND_API_ZH.md`](BACKEND_API_ZH.md)
- 8 x H100 + 1 x L4 生产链路最终压测报告：
  [`results/20260806T1315Z-production-v22/压测报告.md`](results/20260806T1315Z-production-v22/压测报告.md)

```bash
python benchmark/minwm_realtime_async_vae/load_test.py \
  --ws-url ws://HOST/v1/realtime_video/generate \
  --profile async \
  --concurrency 1,2,4,8 \
  --output artifacts/async.json

# 远程 VAE RIFE 3x 验证：会严格校验协商回执、权重摘要、时间轴、
# 每个 chunk 的 source/output 计数，以及实际接收帧数是否完整。
python benchmark/minwm_realtime_async_vae/load_test.py \
  --ws-url ws://HOST/v1/realtime_video/generate \
  --profile async \
  --realtime-media-profile rife3x_v1 \
  --expected-media-weights-sha256 8f6fb9105ba9e946762ee7190acbca3ca1cf14193eb81ca0955d492fb8558692 \
  --concurrency 1,2 \
  --output artifacts/rife3x.json

python benchmark/minwm_realtime_async_vae/summarize.py \
  --baseline artifacts/sync.json \
  --async-profile artifacts/async.json \
  --min-output-wall-fps 24 \
  --output-json artifacts/report.json \
  --output-md artifacts/report.zh-CN.md
```

`rife2x_v1` 仍保留兼容；新增验证应显式使用 `rife3x_v1`。source timeline
协商为 24 FPS 时得到的 `output_timeline_fps=72` 只是媒体时间轴，不代表服务在一秒墙钟时间内实际
交付 72 帧，更不代表浏览器呈现了 72 帧。压测结果因此分别记录 source/output wall
FPS；独立 output UX 门槛默认检查每会话实测 output/wall `>= 24 FPS`，不会读取
timeline 值来判定通过。3x 的 `output_realtime_factor` 定义为
`output_wall_fps / 72`；它小于 1 只说明没有按 72 FPS 时间轴实时交付，并不等于
24 FPS 显示门槛失败，也不能替代该门槛。

最高稳定并发的默认门槛为：P95 action 到首批帧 `< 1000 ms`、每会话生成速度 `>= 16 FPS`、错误率 `0`。最终人工浏览器验证另行记录 action 到 canvas 首帧的真实耗时。
output/wall 门槛只证明服务端到客户端的完整帧流交付速度；Chrome 的实际 presented
FPS rolling p5、解码积压、掉帧与 finite-session 尾帧 drain 仍是上线硬门禁，必须
由真实浏览器探针验证。

生产链路门禁命令：

```bash
node benchmark/minwm_realtime_async_vae/browser_probe.cjs \
  --url http://NLB_HOST/?mode=t2v \
  --output artifacts/browser.json \
  --screenshot artifacts/browser.png

python benchmark/minwm_realtime_async_vae/e2e_production_chain.py \
  --ws-url ws://NLB_HOST/v1/realtime_video/generate \
  --hardware-json artifacts/hardware.json \
  --browser-metrics-json artifacts/browser.json \
  --output-dir artifacts/production-run
```
