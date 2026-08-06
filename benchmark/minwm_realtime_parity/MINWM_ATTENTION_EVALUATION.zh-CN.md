# MinWM attention 选型与 Spot 实测（2026-08-05/06）

本文记录 issue #15 的实现边界、真实硬件验证和可复现实验合同。结论只适用于
MinWM 5B、单卡、完整历史 KV、本文固定 checkpoint 与输入；不能把不同分辨率、
硬件或 KV 长度的 FPS 直接横比。

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
| KV | 完整历史，分别在 45 和 128 latent frames 取样 |
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
短 KV 的 `+0.53%/-1.95%`，差距都很小且方向不完全一致。真正稳定的收益来自启用
现有 optimized components：相对 exact packed，704 的 KV45/KV128 分别提升
`3.68%/1.74%`，480 分别提升 `4.44%/2.67%`。

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
