# MinWM 720p 单卡 Attention / FFN Spot 实测

日期：2026-08-18

## 结论

本轮在一张 H200 和一张 B200 上完成了同口径 Spot 真机矩阵。所谓 720p 是 `1248x704`，使用 SP1、KV45、同一 checkpoint、同一镜像和同一代码；每条正式吞吐 lane 先 warmup 20 chunks，再测 100 chunks。吞吐排名采用未开启 profiler 时的 server scheduler FPS，client FPS 只用于交叉验证。

观测到的纯速度冠军是：

- H200：`dense FA3 + online FP8(all transformer linears)`，`9.824 FPS`，相对 packed deterministic BF16 的 `7.763 FPS` 提升 `26.55%`，峰值 `57069 MiB`。
- B200：`dense FA4 + static FP8(FFN-only)`，`14.841 FPS`，相对 packed deterministic BF16 的 `13.994 FPS` 提升 `6.06%`，峰值 `59410 MiB`。

如果希望缩小量化影响面，H 系列更实用的候选是 `dense FA3 + static FP8(FFN-only)`：`9.732 FPS`，只比本轮 raw winner 慢 `0.94%`，但 attention 投影继续保持 BF16。B200 上这一组合本身就是观测冠军。

如果暂时不接受任何量化，则用 `dense FA + BF16`：H200 为 `9.272 FPS`，B200 为 `14.452 FPS`，相对 deterministic reference 分别提升 `19.44%` 和 `3.28%`。

这里的“冠军”是单次 100-chunk run 的观测排名。H/B 的第一、第二名都只差不到 1%，没有跨 run 置信区间，不能宣称名次已经统计稳固。更重要的是，所有非 reference lane 都没有通过 sampled output-frame byte PSNR corruption screen；这个 screen 只比较采样输出帧的字节，不是感知质量指标，因此低 PSNR 既不能直接判定画质损坏，也不能当作画质通过。上线前仍需多 case、固定 seed 的感知质量评估。

## 工作负载与真实 shape

| 项目 | 值 |
|---|---:|
| 输出分辨率 | `1248x704` |
| VAE latent grid | `78x44` |
| DiT patch 后 token / latent frame | `39x22 = 858` |
| latent frames / chunk | `4` |
| 单卡 Q token 数 M | `3432` |
| self-attention Q | `[1, 3432, 24, 128]` |
| self-attention KV45 上限 | `[1, 38610, 24, 128]` |
| cross-attention KV | 约 `[1, 512, 24, 128]`，K/V 已缓存 |
| DiT | 30 blocks，hidden `3072`，24 heads，head dim `128` |
| FFN | `3072 -> 14336 -> 3072`，GELU-tanh |
| FFN GEMM | `(3432,14336,3072)` 与 `(3432,3072,14336)` |

这个 shape 纠正了旧 microbenchmark 中把 SP1 的 M 当成 `13728` 的假设。B200 static FP8 的 `M>=8192` row-wise fast path 在当前 M=`3432` 上不会命中，因此 FFN 仍有针对真实 shape 做 quantize+GEMM picker/fusion 的空间。

## SGLang 当前实际支持范围

### Attention

MinWM 的 cached realtime attention 只有 `MINWM_ATTENTION_IMPL={packed,dense}` 两种实现。它不是通用 DiT backend 列表的完整子集。

| 实现 | self | cross | Hopper（H100/H200） | Blackwell（B200/B300） | 本轮结论 |
|---|---|---|---|---|---|
| `packed` | 专用 varlen | 专用 varlen | 有 `flash_attn_interface` 才走 FA3，否则 FA2 | 有 `flash_attn.cute` 走 FA4，否则 FA2 | 会绕过 `--attention-backend`。当前镜像 H 实际是 FA2，B 实际是 FA4 |
| `dense + fa` | 支持 | 支持 | FA3 | FA4 | H/B 的首选精确 attention 路径 |
| `dense + torch_sdpa` | 支持 | 支持 | 支持 | 支持 | 兼容基线；H200 约等于 packed FA2，B200 显著更慢 |
| `dense + sage_attn` | self 回退 FA | Sage2 | 目标是 SM90 | 不推荐 SM100 | 实际只替换 cross；本镜像未能安装所需 Sage2 版本 |
| `dense + sage_attn_3` | self 仍是 FA4 | Sage3 | 不适用 | 支持的设计目标 | 实际只替换 cross；本轮 CUDA 12.8/torch cu130 编译环境不一致，未形成有效 lane |
| STA/VSA/SVG2/VMOBA/SLA 等 sparse | 不可达 | 被 MinWM selector 过滤 | 不可用 | 不可用 | 通用配置虽枚举，但当前 MinWM cached self/cross 调用链无法选中 |

H200 trace 中，packed 的主 attention kernel 是 FA2 风格的 `flash_fwd_kernel`；dense 的主 kernel 是 SM90 `FlashAttnFwdSm90`。因此 H 上的主要收益不是抽象的 packed/dense layout 差异，而是当前镜像里 `packed FA2 -> dense FA3` 的真实 backend 升级。B200 的 packed 和 dense 都已走 FA4，所以两者只差约 `0.39%`。

单请求 B=1 时 packed 并没有 ragged batch 收益。与其只补齐 packed FA3 依赖，当前更直接的部署选择是显式设置：

```text
MINWM_ATTENTION_IMPL=dense
--attention-backend fa
```

未缓存的 self-attention 另走 compiled FlexAttention，不是本轮 realtime cached 路径的比较对象。`fa3`/`fa4` CLI 在当前 resolver 中只是 `fa` 的别名，不能靠别名强制版本；最终版本由平台和安装依赖决定。

### FFN

MinWM 是普通 dense MLP，不是 MoE，也没有可直接切换的通用 fused-FFN backend。

| 实现 | H 系列 | B 系列 | 当前含义 |
|---|---|---|---|
| BF16 | 支持 | 支持 | 两次 `F.linear/cuBLAS` + GELU，可靠基线 |
| online FP8 | 支持 | 支持 | 动态 activation；当前会量化整个 transformer 的 linear，不是 FFN-only |
| calibrated static FP8 | 支持 | 支持，SM100 有专路 | 本轮修复 prefix 路由后可精确做 FFN-only；共量化 60 个 FFN 权重，attention 保持 BF16 |
| ModelOpt NVFP4 | 不支持 | 支持 | 仅预量化；当前 MinWM builder 量化全部 300 个 block linears，未做 FFN-only，本轮未列为速度候选 |
| MXFP4 / Nunchaku | 当前 MinWM 不支持 | 当前 MinWM 不支持 | 无可用 MinWM 接入 |

本轮 static FP8 checkpoint 使用 `module_scope=ffn`、activation margin `1.05`，忽略 `to_q/to_k/to_v/to_out/attn2`，输出 `7,364,518,752` bytes。H/B 分别独立校准，calibration SHA 不同；跨卡结果不能假定使用了完全相同的 calibration artifact。

## 正式吞吐矩阵

所有 uplift 都相对各卡自己的 `packed deterministic BF16`。client FPS 与 scheduler FPS 的差异不超过约 `0.02 FPS`。

| Lane | H200 scheduler FPS | H uplift | H peak MiB | B200 scheduler FPS | B uplift | B peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| packed deterministic BF16 | 7.763 | reference | 61809 | 13.994 | reference | 61968 |
| packed fast BF16 | 7.897 | +1.73% | 61801 | 14.396 | +2.87% | 61962 |
| dense FA BF16 | 9.272 | +19.44% | 61803 | 14.452 | +3.28% | 61964 |
| dense SDPA BF16 | 7.754 | -0.11% | 61801 | 9.955 | -28.86% | 61964 |
| dense FA online FP8（all-linear） | **9.824** | **+26.55%** | **57069** | 14.762 | +5.49% | **57118** |
| packed fast static FP8（FFN-only） | 8.100 | +4.35% | 59251 | 14.602 | +4.35% | 59410 |
| dense FA static FP8（FFN-only） | 9.732 | +25.36% | 59251 | **14.841** | **+6.06%** | 59410 |

拆成同族增量后：

- H200：dense BF16 比 packed-fast BF16 快 `17.42%`；static FFN 比 dense BF16 再快 `4.95%`；online all-linear 比 dense BF16 快 `5.95%`。
- B200：dense BF16 比 packed-fast BF16 只快 `0.39%`；static FFN 比 dense BF16 再快 `2.69%`；online all-linear 比 dense BF16 快 `2.14%`。

这说明 H200 的第一杠杆是 attention backend，B200 则已没有明显的 packed/dense attention 格式红利，FFN 优化更值得优先做。

## self / cross / FFN 耗时占比

下表来自独立 Nsys run 的显式 NVTX 边界，分母是完整 MinWM DiT forward 的 CUDA kernel 累计时间，不是整个 streaming request 的端到端时间：

- self 包含 AdaLN、Q/K/V、QK norm、RoPE、cache、attention core、out projection 和 residual。
- cross 包含 pre-norm、Q projection、cached K/V attention、out projection 和 residual/AdaLN。
- FFN 包含两次 GEMM、GELU 和 gated residual。

| GPU / lane | 完整 DiT kernel ms / forward | self | cross | FFN | 三阶段外 |
|---|---:|---:|---:|---:|---:|
| H200 dense FA BF16 | 159.115 | **74.96%** | **7.03%** | **16.94%** | 1.07% |
| H200 dense FA online FP8 | 146.788 | **78.72%** | **6.69%** | **13.43%** | 1.15% |
| H200 packed deterministic BF16 | 228.589 | **82.36%** | **5.54%** | **11.35%** | 0.75% |
| B200 dense FA BF16 | 98.208 | **77.99%** | **6.95%** | **13.66%** | 1.39% |
| B200 dense FA static FFN FP8 | 91.697 | **81.65%** | **7.12%** | **9.81%** | 1.43% |
| B200 packed deterministic BF16 | 104.520 | **77.94%** | **7.87%** | **12.83%** | 1.36% |

在 dense BF16 基线上，真正的 self attention core 分别占完整 DiT kernel 时间的 `42.54%`（H200）和 `41.53%`（B200）；cross attention core 只有 `0.87%` 和 `0.78%`。cross phase 的其余时间主要是投影、norm、copy/residual。因此即便 Sage 把 cross core 理想化为零，端到端上限也不到约 1%，不值得优先承担近似误差和部署复杂度。

B200 的 static FFN-only 把 FFN kernel 从 `0.4473 ms/block` 降到 `0.2999 ms/block`，减少 `32.96%`；30 层合计每个 DiT forward 少 `4.42 ms` 的 FFN kernel work。正式 profiler-off 吞吐相对 dense BF16 提升 `2.69%`。static 路径的 FFN kernel calls 从 4 增到 6，CPU launch 开销也增大，因此进一步优化应瞄准 M=`3432` 的 activation quantization、GEMM picker 和融合，而不是只看矩阵乘吞吐。

H200 没有 static FFN 的独立 phase trace，不能把 B200 的 `-32.96%` 直接外推到 H。H200 online FP8 会同时改变 self/cross 投影和 FFN，也不能把其 phase 变化解释成 FFN-only 收益。

## 单卡可行性与推荐组合

两台真机都只向 PyTorch 暴露一张 GPU；H200 Job 为避免同节点干扰而预留整台 p5e，但实际计算仍是 `CUDA_VISIBLE_DEVICES=0` 的单卡 SP1。NVML 记录的 BF16 峰值为 `61801~61968 MiB`（约 `60.4~60.5 GiB`），static FFN 为 `59251~59410 MiB`（约 `57.9~58.0 GiB`），online FP8 为 `57069~57118 MiB`（约 `55.7~55.8 GiB`）。

在本轮 KV45 合约下：

1. 追求本轮观测最高速度：H 用 `dense FA3 + online FP8(all-linear)`；B 用 `dense FA4 + static FP8(FFN-only)`。
2. 希望尽量缩小量化范围：H/B 都先选 `dense FA + static FP8(FFN-only)`，但必须完成多 case 感知质量门禁后才能称为 production-safe。
3. 暂不接受量化：H/B 都用 `dense FA + BF16`。
4. 需要 bitwise reference：保留 `packed deterministic BF16`，接受对应性能损失。

按 H200 的 KV45 已测峰值做容量估算，80 GiB H100 SKU 看起来仍有约 19.6 GiB 余量，具备单卡可行性，但当前精确 workload 仍需在 H100 上短测确认。KV128 的 self K/V/rotated-K 理论缓存约 `56.6 GiB`，叠加模型和运行时后不应假定 H100 安全。B200/B300 的名义显存容量更充足，但同样不能替代目标 SKU 的短测。

本轮只实际测了 H200 和 B200。H100/H200 同属 SM90、B200/B300 走 Blackwell 分支，可以外推 backend 选择方向，但不能外推绝对 FPS、显存精确峰值或不到 1% 的 lane 排名；H100 和 B300 应至少对最终候选各做一次短复测。

## 下一步优化优先级

1. 默认/部署配置显式切到 `dense + fa`，尤其避免当前 Hopper packed 路径静默落 FA2；同时把最终选中的 FA 版本写入结构化日志。
2. 给 FFN-only static FP8 做多 prompt / 多 checkpoint calibration 和感知质量门禁，再针对真实 M=`3432` 优化 quantize+GEMM 与 GELU/quant 融合。
3. 继续拆 self phase：在 H/B dense BF16 中 self 总计约 75%~78%，其中 attention core约 42%，其余大量是 cache/RoPE/pointwise/projection；这比 cross-only Sage 更有价值。
4. whole-DiT compile 需要先处理动态 KV shape 的重复编译。本轮首个新 shape 约 293 秒，后续 shape 继续编译，因此相关 lane 被有界停止，不能拿 exit 结果排名。
5. QKV fusion 是次级项；现有局部 micro 信号显示其 chunk 级上限远小于 attention backend 与 FFN-only FP8。

## 复现与产物

- SGLang workload SHA：`8ae07a622e9fe4c53f85a3b12bdd147bf51e4782`
- minWM SHA：`2efc6485f65e8fcab506665efde79bc41406385e`
- 镜像：`sha256:bedc07ea3ba55059a8c1c569c3b177c4d00d41f37d4fa9105375531534ef5f2a`
- Checkpoint VersionId：`9lr1pX__59kUFlsv6Q.NKFQO3EoRQY9C`
- Checkpoint CRC64：`GLaCsoJZbwM=`
- H200 contract：`results/h200/contract.json`
- B200 contract：`results/b200/contract.json`
- 吞吐摘要：`results/h200/summary.md`、`results/b200/summary.md`
- Phase 摘要：`results/{h200,b200}/profiles/*/phase-summary.json`
- 原始 H/B trace 和完整 19 MiB summary 保留在各自 100 GiB 结果 PVC；为避免把帧 payload/base64 大文件提交进 Git，仓库只保留 compact 摘要。

Spot Job、benchmark Pod 和临时 reader Pod 已清理；结果 PVC 保留，NodeClaim 由 Karpenter 自动回收。
