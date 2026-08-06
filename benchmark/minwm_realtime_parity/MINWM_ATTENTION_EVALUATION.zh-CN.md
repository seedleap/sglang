# MinWM attention 选型与 Spot 实测（2026-08-05/06）

本文记录 issue #15 的实现边界、真实硬件验证和可复现实验合同。2026-08-05 的历史
矩阵使用 KV45/KV128；从 2026-08-06 起的新实验统一使用 `sink=4、window=20`，
与天鹏 serving 配置对齐。不能把不同分辨率、硬件或 KV 合同的 FPS 直接横比。

## 结论

- MinWM 的 source-shaped `packed` 路径是独立的变长 FlashAttention 路径，不读取
  通用 `--attention-backend`。因此 SageAttention / SageAttention3 必须配合
  `MINWM_ATTENTION_IMPL=dense`；现在会对非法组合 fail fast，并在启动日志打印实际
  self、Ulysses 和 cross-attention 后端，避免静默跑错 kernel。
- B200（SM100）与 B300（SM103）继续使用 packed/dense FlashAttention 路径。
  上游 SageAttention 2.2 没有这两个 compute capability；SageAttention3 虽允许
  SM100 编译，但其运行入口只接受 SM120/SM121。把入口实验性放开后，B200 第一轮
  kernel 仍触发 `CUDA unspecified launch failure`，因此产品代码明确拒绝 SM100/SM103。
- RTX 5090 没有 AWS Spot SKU。本轮只把 Spot `g7e.4xlarge` 的 RTX PRO 6000
  Blackwell Server Edition（同为 SM120）当作兼容性与趋势代理，不把其 FPS 冒充
  RTX 5090 实测。SageAttention 2.2 与 SageAttention3 均完成原生编译，GPU 单测通过。
  代理机是 96 GiB 卡，而 RTX 5090 标准显存是 32 GB；本轮 BF16 speed mode 的
  Sage lane PyTorch peak 约 55.9 GiB、整卡占用约 61.8 GiB，因此真实单卡 5090
  还必须配合量化或 offload，不能仅靠更换 attention backend 装下模型。
- SM120 的 704 短 KV 中 Sage2 只比 dense FA/SDPA 快 4.75%，Sage3 反而慢 2.9%；
  到 480 档时，Sage2 在 KV45/KV128 分别慢 1.55%/3.40%，Sage3 分别慢
  10.57%/11.43%。更重要的是两者相对 dense FA 的长 rollout mean SSIM：704
  只有 0.729/0.653，480 只有 0.762/0.598。因此当前不能把 Sage2/Sage3 设成
  质量或速度默认值；Sage2 只适合作为显式 opt-in 实验 lane。
- SageAttention3 上游会原地中心化 K。MinWM 的 self/cross KV cache 是跨 chunk 的
  持久状态，本实现只在 SageAttention3 调用边界复制 K，并由会主动修改输入的 mock
  单测和真实 SM120 GPU 单测验证原 cache bitwise 不变。

## 固定实验合同

| 项目 | 值 |
| --- | --- |
| SGLang base | `codex/minwm-realtime-api@51c3dced66` |
| MinWM | `2efc6485f65e8fcab506665efde79bc41406385e` |
| checkpoint | `global_step_003200/ema_student/model.pt`，S3 version `wduScksw2f3yPErnG9lBioOuE2AToyAP` |
| SageAttention | `thu-ml/SageAttention@d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` |
| 704 档 | `1248x704`；用户所称 702 按现有 VAE 对齐后的 720-class case 验收 |
| 480 档 | `832x480` |
| 历史 KV 合同 | 完整历史，分别在 45 和 128 latent frames 取样 |
| 当前 KV 合同 | `sink=4、window=20`；后续新测试的固定默认值 |
| 设备数 | 每个 Job 只申请 1 GPU |
| 容量 | Spot only；不借用现有 Capacity Block / On-Demand 节点 |

比较 lane 分开运行，避免一次 profile 同时改变多个变量：

1. deterministic packed FA；
2. optimized-components packed FA；
3. nondeterministic packed FA；
4. optimized-components dense FA；
5. optimized-components dense SageAttention 2.2；
6. optimized-components dense SageAttention3。

B200 主矩阵与长 KV 补跑均预热 20 chunks、统计 200 chunks；SM120 的最终补跑预热
10 chunks、统计 100 chunks。画质 lane 使用相同 case、seed、action 与 checkpoint，
保存 `.npy` / `.mp4`，并以 dense FA 为基准计算 RMSE、MAE、PSNR、SSIM。

## 硬件与结果

### B200 Spot

- Job：主矩阵 `ray/minwm-attention-b200-20260805-05`；dense KV128 补跑
  `ray/minwm-attention-b200-20260805-06`
- 节点：`p6-b200.48xlarge`，NVIDIA B200，`us-east-2a/2b`，Spot
- SGLang：主矩阵 `0c03b62e8c13722446d1ccc11e59b04de3e816fd`；补跑
  `450a9a796ab51c1201e62b74dce7a57bb97a6af7`
- GPU 回归：7 passed、1 skipped（Sage3 real-kernel 用例只允许 SM120/121）
- 结果根目录：
  `s3://leap-world-us-east-2/world-model/evals/minwm/attention/20260805/results/`

同一张卡、相同 optimized components 下，短 KV 的 packed/dense 差距很小：704 档
dense FA 快 `0.53%`，480 档 packed FA 快 `1.95%`。packed 路径同时保留现有严格
parity 语义，因此 B200 默认仍推荐 packed FA。

| 分辨率 | profile | KV45 client FPS | KV128 client FPS |
| --- | --- | ---: | ---: |
| 1248x704 | exact packed deterministic | 13.523 | 9.306 |
| 1248x704 | optimized packed | 14.020 | 9.467 |
| 1248x704 | optimized packed nondeterministic | 13.608 | 9.218 |
| 1248x704 | optimized dense FA | 14.095 | 9.550 |
| 832x480 | exact packed deterministic | 22.669 | 20.501 |
| 832x480 | optimized packed | 23.675 | 21.048 |
| 832x480 | optimized packed nondeterministic | 22.685 | 20.415 |
| 832x480 | optimized dense FA | 23.221 | 21.260 |

704 档 deterministic packed 从 KV45 增长到 KV128 后下降 `31.18%`；480 档下降
`9.56%`。关闭 deterministic 没有稳定收益：两档 KV45 只差 `+0.63%/+0.07%`，
KV128 反而是 `-0.94%/-0.42%`，均属于单轮噪声范围。

长 KV 下 dense FA 相对 optimized packed 在 704/480 分别快 `0.88%/1.01%`；结合
短 KV 的 `+0.53%/-1.95%`，差距都很小且方向不完全一致。旧版总结把
optimized-components 相对 exact packed 的差值全部归因给 native components 并不严谨，
因为两条 lane 同时改变了 deterministic 等变量。使用只隔离 native components 的
packed-nondeterministic 对照，704 的 KV45/KV128 分别提升 `3.03%/2.71%`，480
分别提升 `4.36%/3.10%`；仍缺一条 deterministic+optimized lane 来做完全正交归因。

画质 harness 以 dense FA 为数值基准，但这不代表 dense 是产品 ground truth。packed
与 dense 的 129/65 帧 rollout 并不等价：704 的 PSNR 为 `21.94 dB`、mean/min
SSIM 为 `0.8523/0.6662`；480 为 `23.87 dB`、`0.7562/0.5489`。现有 MinWM
parity 合同属于 packed 路径，因此不能仅凭不到 1% 的 704 速度差把 dense 当作无损
替换。

### B300 Spot

- Job：`default/minwm-attention-b300-20260805-03`
- 固定请求：`p6-b300.48xlarge`、`us-west-2d`、Spot、1 GPU
- 等待 6 小时后由 `activeDeadlineSeconds` 以 `DeadlineExceeded` 结束；期间调度事件
  持续报告没有满足约束的 Spot offering，没有创建 GPU 节点，也未切换 On-Demand。
  历史同 checkpoint、`1248x704`、Local Zone Spot B300 SP1 packed FA 结果为
  `15.891 client FPS`，这里只作为已存在的硬件基线，不冒充本 Job 的新结果。

### SM120 Spot（RTX 5090 代理）

- Job：704 主矩阵 `ray/minwm-attention-rtx6000-20260805-04/05`；480 矩阵
  `ray/minwm-attention-rtx6000-20260805-06`
- 节点：`g7e.4xlarge`，NVIDIA RTX PRO 6000 Blackwell Server Edition，Spot
- SGLang：最终矩阵 `450a9a796ab51c1201e62b74dce7a57bb97a6af7`
- CUDA / PyTorch：CUDA 13.0，PyTorch `2.12.1+cu130`
- Sage GPU 回归：8 passed；SageAttention 2.2 / SageAttention3 均由固定源码 SHA
  在目标机原生编译。
- 结果根目录：
  `s3://leap-world-us-east-2/world-model/evals/minwm/attention/20260805/results/`

该机器上的 packed FA4 在 ragged output layout 的 CuTe 构造阶段报
`expects coord and shape of view are weakly congruent`。这发生在 MinWM 独立 packed
路径，不能作为 Sage lane 的结果；dense lanes 仍按各自实际后端独立测量。

首次 704 Job 的有效结果如下。日志同时打印
`MinWM resolved dense attention backends=sage_attn`，证明 Sage2 不是静默回退：

| profile | KV | client FPS | 结果 |
| --- | ---: | ---: | --- |
| dense FA（实际 Torch SDPA） | 45 | 5.556 | 完成 200 measured chunks |
| dense SageAttention 2.2 | 45 | 5.820 | 完成 200 measured chunks；比 SDPA 快 4.75% |
| dense FA | 128 | - | 95 GiB GPU OOM |
| dense SageAttention 2.2 | 128 | - | 95 GiB GPU OOM |
| dense SageAttention3 | 45 | - | 完成约一半测量后 Spot 节点被回收，不接纳为最终 FPS |

v4 被回收前 Sage3 已持续执行真实 MinWM chunk，并保持约 55.9 GiB peak PyTorch
memory；它不是启动失败。v5 使用独立 S3 run id 完成 Sage3 704/KV45：`5.397`
client FPS，比 SDPA 慢 `2.9%`，比 Sage2 慢 `7.3%`。Sage3/KV128 也 OOM。

704 画质相对 dense FA：

| profile | PSNR | mean SSIM | min SSIM |
| --- | ---: | ---: | ---: |
| SageAttention 2.2 | 16.93 dB | 0.7291 | 0.4334 |
| SageAttention3 | 16.86 dB | 0.6533 | 0.4274 |

480 最终矩阵如下。该分辨率三种 dense backend 都能装下 KV128，但 FA/SDPA 在短长
KV 都最快：

| profile | KV45 client FPS | KV128 client FPS | 相对 FA（KV45 / KV128） |
| --- | ---: | ---: | ---: |
| dense FA（实际 Torch SDPA） | 14.212 | 10.071 | 基准 |
| dense SageAttention 2.2 | 13.992 | 9.729 | -1.55% / -3.40% |
| dense SageAttention3 | 12.709 | 8.920 | -10.57% / -11.43% |

480 画质相对 dense FA：

| profile | PSNR | mean SSIM | min SSIM |
| --- | ---: | ---: | ---: |
| SageAttention 2.2 | 23.17 dB | 0.7616 | 0.5244 |
| SageAttention3 | 20.81 dB | 0.5981 | 0.4627 |

这些是 129/65 帧自回归 rollout，不是单次 attention output；数值误差会跨 chunk
放大，但这正是产品实际消费的路径。两种 Sage 都不能按“近似无损”接纳。

### sink=4/window=20 的最小 kernel 与 Nsight

Job `ray/minwm-attention-kernel-sm120-20260806-02` 在同一 SM120 Spot 代理机上固定
`batch=1、heads=40、head_dim=128、chunk=4`，直接调用当前 SGLang wrapper。计时
包含 Sage2 的 smooth/量化和 Sage3 为保护持久 KV 所需的 contiguous K clone、布局转换
与量化，不是只截取低精度 MMA kernel：

| shape | Q / K tokens | BF16 SDPA | Sage2 API | Sage3 API |
| --- | ---: | ---: | ---: | ---: |
| smoke | 1,024 / 5,120 | 0.408 ms | 0.447 ms（慢 9.5%） | 0.567 ms（慢 38.8%） |
| 832x480 | 6,240 / 31,200 | 11.571 ms | 8.213 ms（快 29.0%） | 9.749 ms（快 15.7%） |
| 1248x704 | 13,728 / 68,640 | 52.544 ms | 32.189 ms（快 38.7%） | 34.868 ms（快 33.6%） |

因此“低位宽核心在生产尺寸上不应全面慢于 BF16”的判断成立；旧端到端表格不能被
解释为 Sage kernel 本体更慢。小 shape 会被固定启动、量化和布局成本反噬，生产尺寸
才进入 Sage 的收益区间。

Nsight 用包含 2 次 warmup 和 3 次 measured call 的 backend NVTX range 关联 GPU
kernel。704 每次 Sage2 GPU 总计 `32.20 ms`，其中 INT8-QK/FP8-PV 主 kernel
`28.51 ms`、wrapper `3.70 ms`；Sage3 总计 `34.83 ms`，其中 NVFP4 主 kernel
只需 `25.16 ms`，但 wrapper 达 `9.68 ms`。480 也相同：Sage2 主 kernel/wrapper
为 `6.61/1.53 ms`，Sage3 为 `5.47/3.97 ms`。也就是说 Sage3 的 NVFP4 kernel
确实最快，但 K 隔离、中心化、布局和 block-wise 量化吞掉了相对 Sage2 的优势。
704 单次 API 的 incremental peak allocated 为 SDPA/Sage2/Sage3
`0.14/1.62/4.16 GB`，480 为 `0.065/0.74/1.60 GB`；因此 Sage3 的核心速度
收益同时伴随更高临时显存峰值，不能从 96 GB 代理卡外推 32 GB RTX 5090。

当前镜像里的 packed FA4 在三个 shape 都于 ragged output epilogue 的 CuTe MLIR
构造阶段失败，尚未启动 GPU kernel，错误为
`expects coord and shape of view are weakly congruent`。这是一条独立兼容性问题，不能
用失败 lane 推导 FA4 与 Sage 的性能比例。

原始 `.nsys-rep`、SQLite、CUDA kernel/API/NVTX CSV、benchmark JSON、运行时信息、
`nvidia-smi` 与按 backend range 拆分的 `nsys-analysis.json` 位于：
`s3://leap-world-us-east-2/world-model/evals/minwm/attention/20260806/profiles/minwm-attention-kernel-sm120-20260806-02/`。

线上部署还必须做到：按真实 compute capability fail fast；记录 requested/resolved
backend 和 fallback reason；预构建固定 Torch cu130/NVCC 13.0/Sage SHA 的 wheel，
不在 serving Pod 冷启动现场编译；Sage3 调用后校验 KV bitwise 不变；把 sink/window
写进请求、服务与结果合同；分别监控 API P50/P95、显存瞬时峰值、OOM/CUDA fault、
fallback 比例和长 rollout 质量。发生异常时只在 session 边界回退，不能在已被消费的
KV 上热切换 backend。

## 上游兼容性依据

- [SageAttention 固定源码](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5)
  的通用 Sage dispatch 不包含 SM100/SM103。
- SageAttention3 的
  [构建配置](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention3_blackwell/setup.py#L62-L71)
  虽列出 SM100，但
  [运行入口](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention3_blackwell/sageattn3/blackwell/api.cu#L220-L222)
  只接受 SM120/121；上游也有对应的
  [B200 支持问题](https://github.com/thu-ml/SageAttention/issues/291)。
- NVIDIA 官方规格列出 [RTX 5090 为 32 GB 显存](https://www.nvidia.com/en-eu/geforce/graphics-cards/50-series/rtx-5090/)，
  因此本文 96 GiB SM120 代理机的可运行结果不能直接推导为 5090 可部署。

## 验证与复现

三个 Job manifest 位于 `benchmark/minwm_realtime_parity/k8s/`，都固定镜像 digest、
源码 SHA、checkpoint version、节点池、Spot 类型和 1-GPU 上限。入口脚本会：

- 固定 CUDA 13.0 编译工具链，避免 PyTorch cu130 与系统 CUDA 12.8 / 自动解析出的
  CUDA 13.2 混用；
- 固定并记录 SageAttention 源码 SHA；
- 记录实际 GPU、Torch/CUDA、依赖是否安装、S3 对象身份和完整 profile 合同；
- 单独汇总失败 profile，不用成功 lane 掩盖不兼容 lane。

本地静态检查包括 Ruff、`bash -n`、Python bytecode compile 和 `git diff --check`。
本机 macOS 的完整 pytest collection 被现有 Torch 2.13 / Triton import 冲突阻断；
相关目标用例已在真实 B200 与 SM120 Spot GPU 上执行。
