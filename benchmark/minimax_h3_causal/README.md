# MiniMax H3 causal 实验记录

## 目标与边界

本实验只对公开 MiniMax H3 权重改变 attention 可见性，用于验证运行和速度。它不证明现有权重在 causal mask 下具有可接受的生成质量，也不接入 WebSocket 实时接口。

主验收点是实际 NFE=3；基准同时保留 NFE=2/3/5。5 秒请求按 H3 对齐后是 124 帧、24 fps，即 5.167 秒播放时长。单请求预热后，端到端 latency 不超过 5.167 秒时记为 `RTF <= 1`。

## 已确认的设计

- Causal 维度：视频时间维 block-causal；音频与视频按时间映射到同一 block。
- Prefix：text 和 reference/keyframe token 相互双向可见，并对所有 target block 可见；prefix 不反向读取 target。
- Block：每 block 3 个 **DiT latent frames**，block 内 video/audio token 双向可见。
- KV 可见范围：sink=4 frames，sliding window=20 frames，可配置。
- 任务：T2VA、仅首帧条件的 FL2VA；不测试尾帧条件。
- 初始精度：BF16；先做 dense reference 与 FlexAttention parity，再考虑其他优化。

## 与初始认知不一致的地方

1. `causal=True` 不是所需语义。它是在打平后的 token 序列上做逐 token 下三角 mask，会让同一视频帧的空间 token 也单向可见；H3 又把 audio 放在 video 前面，普通下三角还会造成错误的跨模态方向。
2. “3 帧”必须先解释为 DiT latent frames。H3 VAE 的输出帧与 latent time 不是 1:1；当前时间映射遵循 `17n+5` 输出帧边界以及非均匀 latent temporal positions。
3. sink/window 不能切开 block，否则 block 内不再双向。因此 sink=4 向上对齐为 2 blocks=6 latent frames，window=20 向上对齐为 7 blocks=21 latent frames。
4. 当前 H3 API 的 `num_inference_steps=N` 生成 N 个 sigma 点，但 denoise loop 实际执行 `N-1` 次 DiT forward。实际 NFE=3 的请求需发送 `num_inference_steps=4`。
5. 本分支实现的是完整 packed sequence 上与 sliding-KV 等价的稀疏可见性，还没有把 H3 pipeline 改成逐 block 生成并跨 block 持久化 KV。它足以测 attention mask 对整段生成的速度影响，但不等价于已经完成流式 causal pipeline。
6. 官方模型卡说明 H3 在训练末期使用了 native sparse attention，但初始开源只提供 full-attention inference，稀疏实现将另行发布。本实验的 block-causal mask 是给定规格的独立实现，不能声称复原了官方训练 mask。
7. 官方发布材料没有定义本实验的 `sink=4/window=20` 单位。当前暂按 latent frames 解释并对齐为 6/21；若单位实际是 block，应改为 4/20 blocks，即 12/60 latent frames，再做 8 卡性能结论。

## 重大决策

### D1：使用显式 block id，而不是 token 位置下三角

每个 packed row 标记为：prefix、padding 或 target block id。目标 block `q` 可以读取：

- 全部 prefix；
- sink blocks；
- 最近 window blocks；
- 当前 block。

同一 block 的 audio/video 使用同一个 block id，所以保留联合双向 attention。

### D2：音频按 H3 temporal position 映射

视频每 3 个 latent frames 产生一个 block 边界；audio target row 根据已有的 H3 temporal position 落入对应视频 block，而不是按 packed row 顺序判断先后。

### D3：FlexAttention 是性能路径，dense SDPA 只作为小规模参考

完整 1344×768 packed sequence 的 dense attention mask 会占用不可接受的显存。`reference` 模式限制为 4096 rows，仅用于 BF16 parity probe；端到端生成使用 `flex`。

### D4：8 卡拓扑固定总卡数，改变 TP/Ulysses 分解

按以下顺序测试，优先低 Ulysses：

| 标签 | TP | Ulysses | GPU |
| --- | ---: | ---: | ---: |
| `tp8-u1` | 8 | 1 | 8 |
| `tp4-u2` | 4 | 2 | 8 |
| `tp2-u4` | 2 | 4 | 8 |
| `tp1-u8` | 1 | 8 | 8 |

H3 有 56 个 attention heads；以上组合均满足 TP-local heads 可继续被 Ulysses 整除。

### D5：结构相同的 FlexAttention block mask 可跨请求复用

block mask 只依赖 packed row 的 block id、sink/window 配置、长度和设备，不依赖 Q/K/V 数值。默认保留最近 8 种结构的进程内缓存，避免每个预热后请求重复构建 15 万行量级的稀疏元数据；可用 `minimax_h3_causal_cache_block_mask=false` 关闭，以单独记录这项优化的收益。

## 启动与测试

先在单卡上运行 BF16 attention parity probe：

```bash
PYTHONPATH=python python benchmark/minimax_h3_causal/attention_probe.py
```

AWS B200/B300 使用固定镜像和不可变 commit。提交器默认只输出 Job manifest；先对输出运行 server-side dry-run，确认后才加 `--apply`。例如单卡 B300 probe：

```bash
python benchmark/minimax_h3_causal/submit_spot_job.py \
  --phase attention-probe \
  --hardware b300 \
  --git-ref <40-char-commit> \
  > /tmp/minimax-h3-probe.json
kubectl apply --dry-run=server -f /tmp/minimax-h3-probe.json
```

8 卡主测示例；其余拓扑替换 TP/Ulysses，并保持乘积为 8：

```bash
python benchmark/minimax_h3_causal/submit_spot_job.py \
  --phase e2e \
  --hardware b300 \
  --git-ref <40-char-commit> \
  --tp-size 8 \
  --ulysses-degree 1 \
  --causal-mode flex \
  --nfe 3
```

以下为 TP8 + Ulysses1 示例。其他拓扑只替换 `--tp-size` 和 `--ulysses-degree`：

```bash
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --revision bfc8ed0353f5a9733be73e6b2c98ec0948195b86 \
  --model-variant fl2va \
  --num-gpus 8 \
  --tp-size 8 \
  --ulysses-degree 1 \
  --performance-mode speed \
  --dit-precision bf16 \
  --enable-torch-compile false \
  --attention-backend-config 'minimax_h3_causal_mode=flex,minimax_h3_causal_block_frames=3,minimax_h3_causal_sink_frames=4,minimax_h3_causal_window_frames=20,minimax_h3_causal_cache_block_mask=true' \
  --port 30010
```

运行 T2VA/FL2VA 与 NFE 2/3/5：

```bash
python benchmark/minimax_h3_causal/run_matrix.py \
  --topology tp8-u1-b200 \
  --variant causal-flex-mask-cache \
  --model-revision bfc8ed0353f5a9733be73e6b2c98ec0948195b86 \
  --first-frame-uri file:///data/minimax-h3/first-frame.png \
  --output benchmark/minimax_h3_causal/results/tp8-u1-b200.jsonl \
  --video-dir benchmark/minimax_h3_causal/results/videos
```

## 正确性检查

- Mask 单元测试：prefix、same-block、future block、sink、window、padding。
- FL2VA：首帧 condition 必须是 prefix，不能读取 target。
- BF16 parity：相同 Q/K/V 下 FlexAttention 对 dense reference，记录 max/mean absolute error。
- 输出有效性：无 NaN/Inf，MP4 含 24 fps H.264 video 与 AAC stereo audio。
- 长序列稳定性：至少连续生成 10 个请求，检查显存是否持续增长。

## 性能结果

待 GPU 运行后填写：

| GPU | Topology | Task | NFE | p50 latency | p95 latency | RTF | Peak/GPU | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| — | — | — | — | — | — | — | — | 待测 |

## 理解检查问题

1. 为什么把 H3 attention backend 的 `causal` 参数直接改成 `True`，不能得到这里定义的 block-causal 语义？
2. 为什么配置的 sink=4 最终保留了 6 个 latent frames？如果严格只保留 4 帧，会破坏什么性质？
3. H3 packed layout 中 audio rows 排在 video rows 前面，为什么仍然能让同一时间 block 的 video/audio 双向交互？
4. 为什么实际 NFE=3 的 API 参数是 `num_inference_steps=4`？这个结论从哪段循环语义得到？
5. dense-reference parity 通过后，能证明哪些事情？为什么它不能证明 causal 后的视频质量正确？
6. `tp8-u1` 与 `tp1-u8` 都使用 8 张 GPU，它们分别主要增加了哪类 collective？为什么低 Ulysses 不一定最快？
7. 当前完整序列的 block-sparse attention 与真正逐 block 持久化 KV cache，在计算图和可声称的“实时流式生成”能力上有什么区别？
8. 为什么 5 秒请求的实时阈值暂定为 5.167 秒，而不是恰好 5.000 秒？
9. 如果 `sink=4/window=20` 的单位从 latent frames 改成 blocks，mask 稀疏率、缓存体积和实时结论会怎样变化？
