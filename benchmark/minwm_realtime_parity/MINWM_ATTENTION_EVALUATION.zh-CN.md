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
- 当前 `sink=4/window=20` 合同下，704 端到端 Sage2 比 BF16 Torch SDPA 快
  `3.34%`，Sage3 慢 `2.32%`；480 则分别慢 `0.98%/5.07%`。历史 KV45/KV128
  结果的方向也并不支持全面替换。更重要的是两者相对 dense FA 的长 rollout mean
  SSIM：704 只有 0.729/0.653，480 只有 0.762/0.598。因此当前不能把 Sage2/Sage3
  设成质量或速度默认值；Sage2 只适合作为 704 的显式 opt-in 实验 lane。
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

`ray/minwm-attention-kernel-sm120-20260806-02` 完成了第一轮 Nsight 工具链与
wrapper 趋势预跑，但合同复核发现它误用了 Wan 通用配置的 40 heads；当前 MinWM 5B
转换配置是 `24 heads x 128 dim`。因此 run02 原始文件保留作排障证据，但不进入最终
性能结论。最终 run 使用 24 heads，并同时补测 fixed-length dense FA4 与产品当前的
packed varlen FA4。

第一次提交 H=24 Job 时又暴露了留痕系统本身的风险：run03 使用了一个由短 SHA
错误补全出的 40 位字符串，GitHub 中不存在该对象，因此 checkout 即 fail fast，未执行
任何 benchmark。失败 Job 与日志保留，最终有效 run04 固定使用 `git rev-parse HEAD`
得到的完整对象 `876118db8419c2ead4bd84b5f0f6c5c10784da86`。

有效 profile Job `ray/minwm-attention-kernel-sm120-20260806-04` 使用同一张 SM120
Spot 卡、batch=1、H=24、D=128、10 次稳态 CUDA event 计时。self-attention 的
Q/K 来自 `chunk=4、sink=4、window=20`；cross-attention 保持相同 Q，但固定
文本 K=512。以下是完整 API 路径的中位延迟（含量化、layout 和 Sage3 的 K 保护复制）：

| shape | BF16 SDPA | Sage2 | Sage3 | Sage2 / Sage3 相对 SDPA | peak allocated SDPA / S2 / S3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke self，Q=1,024 / K=5,120 | 0.403 ms | 0.339 ms | 0.398 ms | +15.85% / +1.16% | 0.006 / 0.117 / 0.146 GiB |
| 480 self，Q=6,240 / K=31,200 | 7.802 ms | 5.486 ms | 5.727 ms | +29.68% / +26.59% | 0.036 / 0.413 / 0.893 GiB |
| 480 cross，Q=6,240 / K=512 | 0.174 ms | 0.187 ms | 0.259 ms | -7.61% / -48.76% | 0.036 / 0.059 / 0.127 GiB |
| 704 self，Q=13,728 / K=68,640 | 31.926 ms | 18.628 ms | 21.335 ms | +41.65% / +33.17% | 0.080 / 0.906 / 2.325 GiB |
| 704 cross，Q=13,728 / K=512 | 0.334 ms | 0.322 ms | 0.599 ms | +3.85% / -79.09% | 0.080 / 0.124 / 0.273 GiB |

这组最小实验确认：在 MinWM 的长 K self-attention 上，低位宽核心路径确实明显快于
BF16，并非“SDPA 全面领先”。此前看起来矛盾，是把端到端结果、短 K cross-attention
和 attention kernel 本体混在了一起。Sage2 在 SM120 上实际是 INT8-QK + FP8-PV，
不是 NVFP4；只有 Sage3 使用 block-wise scale 的 NVFP4。

Nsight 在每个 backend 的 NVTX range 内采了 5 次调用。704 self 的 BF16
`flash_fwd_kernel` 合计 `161.135 ms`；Sage2 的
`qk_int_sv_f8_attn_kernel` 合计 `83.428 ms`，全部 GPU kernels 合计
`94.421 ms`；Sage3 的 NVFP4 `compute_attn_ws` 合计 `77.317 ms`，但 85 个
GPU kernel 加总为 `106.304 ms`。也就是说 Sage3 主 attention kernel 比 Sage2
主 kernel 快约 7.3%，但 K clone/中心化、量化、layout 和 elementwise 额外消耗约
`28.99 ms/5 calls`，最终 API 仍慢于 Sage2。480 self 的 Sage3 wrapper 占比更高：
主 kernel `15.872 ms/5 calls`，全部 kernels `27.516 ms`。在 K=512 的 cross 路径，
固定前后处理更无法摊薄，因此 Sage3 明显倒挂。

FA4 不能给出诚实的同机数字：当前 `flash-attn 2.8.3.post1` 在 SM120 上，
fixed-length dense 入口在 TMA partition 阶段触发
`AttributeError: 'NoneType' object has no attribute '_trait'`；packed varlen 入口在
ragged output 的 CuTe layout 构造阶段触发 `weakly congruent` 错误。两个错误均发生在
GPU attention kernel 启动前。部署前应把 fixed/packed FA4 各自作为独立兼容性门禁，
不能拿失败 range 的几百毫秒 Python/CuTe 建图时间当作 FA4 性能。

run04 原始 `.nsys-rep`、SQLite、CUDA API/kernel/NVTX CSV、benchmark JSON、运行时、
完整日志与 SHA256 清单归档于
`s3://leap-world-us-east-2/world-model/evals/minwm/attention/20260806/profiles/minwm-attention-kernel-sm120-20260806-04/`。
run02（错误 H=40）和 run03（无效完整 SHA）均保留用于审计，但明确拒绝进入结论。

### sink=4/window=20 的端到端结果

`ray/minwm-attention-rtx6000-kv20-704p-20260806-01` 使用真实 MinWM 模型配置，固定
10 个 warmup chunks 与 100 个 measured chunks。三条 lane 的 checkpoint、case、
seed、请求、组件配置和 SM120 Spot GPU 完全一致：

| 1248x704 backend | client FPS | scheduler P50 | 相对 BF16 SDPA |
| --- | ---: | ---: | ---: |
| BF16 Torch SDPA | 6.913 | 2,314 ms/chunk | 基准 |
| SageAttention 2.2 | 7.144 | 2,239 ms/chunk | +3.34% |
| SageAttention3 | 6.753 | 2,368 ms/chunk | -2.32% |

这证明 Sage2 在天鹏 KV 合同下能给 704 端到端带来小幅但稳定的收益；同时也证明
不能把单个 attention API 的 speedup 直接当作 client FPS。完整 30 层 DiT 还有 QKV
projection、FFN、归一化、RoPE、cross-attention、VAE 和传输；尤其短 K 的 text
cross-attention 会承受 Sage 固定量化/布局成本，可能抵消长 K self-attention 的收益。
Sage3 仍不应成为默认：它端到端更慢，且既有 129 帧 rollout 的 mean SSIM 仅
`0.6533`。

`ray/minwm-attention-rtx6000-kv20-480p-20260806-01` 使用相同 10 warmup + 100
measured 合同，结果再次说明不能按位宽推断端到端速度：

| 832x480 backend | client FPS | scheduler P50 | 相对 BF16 SDPA |
| --- | ---: | ---: | ---: |
| BF16 Torch SDPA | 16.671 | 958 ms/chunk | 基准 |
| SageAttention 2.2 | 16.508 | 965 ms/chunk | -0.98% |
| SageAttention3 | 15.827 | 1,010 ms/chunk | -5.07% |

三条 lane 各统计 1,600 个 pixel frames，均无失败 profile，并在结果 JSON 中同时记录
`sink_size=4` 与 `window_size=20`。480 的 self-attention 序列比 704 短，而每个 DiT
block 仍有固定 K=512 的 text cross-attention；量化、layout 与 kernel launch 的固定
成本占比更高，所以 Sage2 在 704 的小幅收益没有迁移到 480。

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
本机 macOS 的系统 Python 3.9 无法解析仓库要求的 `type | None` 类型语法，因此完整
pytest collection 在加载 SGLang `conftest.py` 时被解释器版本阻断；独立的 attention
shape harness 为 5 passed，相关后端目标用例已在真实 B200 与 SM120 Spot GPU 上执行。
