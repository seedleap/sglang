# MinWM 融合单卡 Nsight Compute 计划

目的：解释融合收益小或不稳定的设备侧原因，并为下一轮优化给出可行动的方向；不把单卡
微基准外推为 SP2/SP4 或端到端结论。

## 取舍

| 项目 | 是否采集 | 原因 |
| --- | --- | --- |
| S4 QKV | 是，必选 | SP2 单项 DiT wall `-0.108%`、SP4 `+4.977%`，而全图 launch 数近乎不变；必须分辨 GEMM 形状、寄存器/occupancy、Tensor/DRAM 与 layout 成本。 |
| S3 post-A2A | 暂不采 | 已有强 DiT CUDA 收益；SP2 端到端近零的直接证据是 scheduler unclassified 抵消。单卡无法解释 A2A/critical path，先不浪费卡。 |
| S1 hoist | 否 | gather/fill 被稳定消除，因果已清楚，预期的进一步 NCU 信息价值低。 |
| S2 postprocess | 否 | 没有可发布 runtime candidate，不能对不存在的路径做性能归因。 |

## 单卡合同

H200、BF16、真实 720p SP1 layer-probe 形状 `[1,858,3072]`：baseline 为三次
`3072→3072` linear，candidate 为一次 `3072→9216` packed linear。每张卡串行运行
baseline/candidate，NCU 采集 kernel duration、Tensor/SM/DRAM throughput、active warps、
registers 与 local-memory spill。采集只包含 `minwm_qkv_baseline` / `minwm_qkv_fused`
NVTX range 内的一次投影，避免 `--set full` 对 warmup/归约做无意义 replay。必须保留 `.ncu-rep`、CSV、环境与 GPU clock；NCU 不与
torch.profiler/Nsight Systems 同跑。

Spot 节点获得后，GPU0 与 GPU1 跑独立同构 A/B pair（交叉顺序），GPU2 可作为重试槽；其余卡
不启动 workload。任务请求整机仅为获得同一 H200 Spot 节点，完成后 Job 自动退出。

首个 Job `...-01` 在 Pod 被 TTL 回收前失败；其日志未保留在 Kubernetes。PVC 原位保留，
`...-02` 启动时必须先列出旧 attempt 文件并记录大小，之后才允许采集。首版缺少 NVTX 过滤且
对多次迭代做 full-set replay，不能把失败解释成 kernel 结论。

`...-02` 复现并定位了 runner provenance 缺口：shallow clone 可变分支后断言旧 SHA，分支在
manifest 提交后已前进，因此在 NCU 预检前失败。`...-03` 改为完整 SHA 的 detached checkout；
前两次均只属于 runner-invalid，不得进入性能结论。

## 验收与决策

1. candidate projection duration 至少不差于 baseline 总 duration，且 Tensor throughput 不降；
2. 若 candidate register/spill 或 active-warps 明显更差，下一步试 tile/algo/packing-layout，而
   不再仅减少 Python module 数；
3. 若 GEMM 本身更快但总 span 不变，优先 profile/simplify chunk-contiguous layout；
4. 若两者都接近带宽上限，停止 QKV fusion 扩展，转向消除 copy/layout；
5. 所有结论只能指导下一轮实现，最终开关仍由 200-chunk profiler-off 与 bitwise 验收决定。

## 重新审视后的任务选择

| 子任务 | 是否需要 NCU | 重新审视后的判断 |
| --- | --- | --- |
| S1 timestep hoist | 只需一次上界核验，已完成 | 剩余 gather 为 11.392 us/次；5 pass/chunk 的 kernel-only 上限约 0.057 ms，不值得继续写复杂 kernel。 |
| S2 postprocess | 否 | 没有通过正确性/发布边界的 runtime candidate；profile 错误候选没有决策价值。 |
| S3 post-A2A | 否（解释当前端到端时） | 收益/抵消发生在 A2A、scheduler 与关键路径；单卡 NCU 无法解释通信等待。后续若单独调 Triton pack，NCU 才有价值。 |
| S4 QKV | 是，唯一主目标 | 三 GEMM 变一 GEMM后产生 stride-9216 Q/K/V view，并在 SP fast path 增加 V materialize；必须按本地 shard shape 分解 GEMM、Q/K norm 与 copy。 |
| S5 组合集成 | 否 | 已有 factorial Nsight Systems；组合问题是 wall 交互与调度兑现率，不是某个未知单 kernel。 |
| S0 测量契约 | 否 | 它是测量基础设施，不是 GPU kernel 优化。 |

早先只计划 profile `M=858` 是不足的：它只能代表 SP1 layer-probe，不能解释正式 SP2 的
`M=429` 或 SP4 的 `M=214/215`。本轮不启动分布式 SP，而是在单卡上复现这两个本地 shape，
同时纳入真实 eager Q/K RMSNorm 和 fused 路径独有的 `value.contiguous()`。

## H200 实测结果

### 资源与 provenance

- context / NodePool：`codex-minwm-test-phx2` / `minwm-test-phx2-p5e-spot`；
- 节点：`i-0a29920a9951065c6`，`p5e.48xlarge` Spot H200；
- Job / Pod：`minwm-qkv-ncu-single-gpu-h200-20260814-04` /
  `minwm-qkv-ncu-single-gpu-h200-20260814-04-n8dgn`；
- 只申请 4/8 GPU，四个 lane 各自单卡；复用已存在节点，没有申请第二台 Spot；
- runner：`de30b59fb3d22d5a329041ce4f5afbbd8b9043da`；manifest：
  `e6c9524eb426572a7f29acae381a7fc51b8a9fbb`；
- image：`minwm-training@sha256:bedc07ea...53ef5f2a`；PyTorch 2.12.1+cu130；
- Nsight Compute：2025.1.1；Job `Complete 1/1`，exit 0，restart 0，用时 6m46s；
- PVC：`minwm-qkv-ncu-h200-results-20260814`，attempt：
  `/results/attempts/minwm-qkv-ncu-single-gpu-h200-20260814-04-n8dgn/minwm-qkv-ncu-single-gpu-h200-20260814-04`；
- 8 份 raw CSV、8 份 timing JSON 与 8 份 `.ncu-rep` 原位保留；下载的小型产物逐文件匹配
  PVC `SHA256SUMS`。本地重算 summary SHA256 为
  `0fe9ae45ba7853426917e3e52e3d22701dfb610d4f15de6ac3c865d64c8f7752`。

每个 M 用两张卡，baseline/fused 反序执行；每次 NCU 进程内又对两个模式做交叉 CUDA-event
timing。NCU kernel 总数和 duration 是主要根因证据；由于 profiler 会影响 host dispatch，
event timing 只使用同进程的相对方向，不使用跨进程绝对值。

### 分解结果

| 本地 M | 3 GEMM | 1 GEMM | GEMM 收益 | Q/K norm：control→fused | V copy | 完整边界：control→fused | kernels | 边界收益 | paired event 收益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 429 | 60.560 us | 50.800 us | 16.116% | 105.984→107.440 us | 7.328 us | 166.544→165.568 us | 19→18 | **0.586%** | 8.062% mean |
| 215 | 44.944 us | 29.152 us | 35.137% | 86.096→88.656 us | 5.136 us | 131.040→122.944 us | 19→18 | **6.178%** | 7.044% mean |

准确的抵消关系是：

```text
M=429: 9.760 us GEMM saving - 1.456 us Q/K stride penalty - 7.328 us V copy
       = 0.976 us/block net saving
M=215: 15.792 us GEMM saving - 2.560 us Q/K stride penalty - 5.136 us V copy
       = 8.096 us/block net saving
```

因此“融合不可能负收益”这个前提不成立：融合减少了两个 GEMM launch，但改变了 GEMM tile，
又把连续的三个输出变成 stride-9216 views；当前 SP 路径还必须为 V 新增一次 copy。`M=429`
的 GEMM 收益几乎被新增/放大的消费者成本完全抵消，任何编译边界、host dispatch 或环境波动
都足以把 0.976 us/block 翻成端到端小负值。`M=215` 的原三 GEMM Tensor Active 只有
39.01%，合并后升到 70.68%，所以小 M 更受益；`M=429` 只从 58.98% 升到 67.89%。

两种 shape 的 fused GEMM 都保持 168 registers/thread，没有 local load/store spill；active
warps 还略升（M429 13.85%→14.12%，M215 13.79%→14.23%）。所以根因不是寄存器、occupancy
或错误 GEMM 算法退化，而是“GEMM 节省随 M 变化”与“固定的 norm/copy 成本”之间的摊销。

`M=858` 的独立首轮证据也与此一致：单 GEMM kernel 比三 GEMM 总和快 11.1%，但加入下游
Q/K consumer 后 kernel-only 净收益只约 3.47%；显式 `Q/K.contiguous()` 或
`Q/K/V.contiguous()` 分别回退 2.61% / 7.45%，所以不能靠多加 contiguous 修复。

### 对正式 SP2/SP4 结果的解释边界

- SP2 正式 DiT wall `-0.108%` 与这里 `M=429` 仅 0.976 us/block 的净 saving 都属于“几乎
  无硬件余量”。按 30 blocks × 5 passes 估算，kernel-only 只有约 0.146 ms/chunk，约为
  756 ms DiT 的 0.02%；出现小负值并不表示 fused GEMM 变慢。
- SP4 的 `M=215` kernel-only 约省 8.096 us/block，即约 1.21 ms/chunk、约 0.17% DiT。
  它支持“SP4 更容易正收益”的方向，却不能单独证明旧 ABBA 的 +4.98% 全来自 QKV；后者
  大于设备侧上界且 chunk CV 超门，仍含 runtime/调度/位置漂移。
- paired event 在两个 M 都约快 7--8%，反映减少一个 kernel 与 Python/CUDA dispatch 的方向，
  但 NCU 下绝对 event 时间跨进程有 20% 以上漂移，不能把它乘 150 后冒充端到端收益。

## 下一步优化方向

优先级 1：实现 MinWM 专用的 dual-Q/K RMSNorm kernel，直接读取 stride-9216 Q/K，写出
head-layout 连续 Q/K，把现在两次 norm 的 16 个 eager kernel 合成 1 个 launch。它必须逐元素
保留 `FP32 normalize -> BF16 round -> BF16 weight -> BF16 output`；generic Triton RMSNorm
把 weight 乘法放在 FP32 中，现有 diffusion QKNorm+RoPE 又按 128 head-dim 归一化，都不能直接
替换 MinWM 的 3072-dim 语义。golden gate 必须是 bitwise，而不是宽容差。

优先级 2：让 peer-first/varlen pack 直接接受 V 的 row stride 9216，删除
`value.contiguous()`。这可直接收回 M429 的 7.328 us、M215 的 5.136 us 与一个 launch；不应
修改 GEMM epilogue，也不应为 Q/K/V 增加显式 contiguous。单独收益仍小，适合与优先级 1
组合成新的 S4b candidate。

否决：`QK + V` 两 GEMM首轮为 97.056 us，慢于单 GEMM的 91.296 us；不进入实现。继续只调
GEMM tile 的价值也低，因为 NCU 已证明 fused GEMM 本身健康，真正的大头是 86--107 us 的
Q/K norm consumer。

新 candidate 的验收顺序：单卡 M858/429/215 bitwise + NCU；再做 720p SP1 profiler-off；
只有 SP1 有稳定正收益才进入 SP2/SP4 20+200。这样避免再次用分布式噪声掩盖本地 kernel 退化。

## 与预期不符和过程决策

1. 最初只测 M858，重新审视后发现不能解释 SP2/SP4；补做 M429/M215，而不是重跑分布式。
2. `-01/-02` 在 NCU 前因可变分支 checkout/provenance 失败；`-03` 因镜像 PATH 中没有 `ncu`
   失败。镜像实际工具在 `/opt/nvidia/nsight-compute/2025.1.1/ncu`；`-04` 使用 find fallback。
   三次失败均不进入性能结论，原诊断留在 PVC，Job/Pod 已精确删除。
3. profiler 的 event 绝对时间跨进程差异大；决策改为 NCU kernel duration 主证据、同进程
   paired event 只看方向。
4. CPU-only reader 因 EBS SELinux/MCS 对 read-only mount 返回 permission denied；改为同节点
   UID0+SYS_ADMIN、RW source mount，只执行 find/tar/stat/sha256sum，未写 PVC。导出完成后精确删除。
5. 原假设重点查 registers/occupancy；实测两边 168 registers/thread、零 local spill，真正问题
   是 M-dependent GEMM 摊销与 Q/K norm、V copy。

资源清理时只删除本轮精确 Job/Pod 与临时 reader，保留 50 GiB PVC 和全部 raw 产物。本轮对象
删除后，节点上仍有另一会话的 1-GPU S2 profile/已完成 issue175 Pod；没有抢占、删除或修改它们，
因此节点是否回收由剩余任务及其 owner 决定，而不是把共享节点误报为“本轮仍占 4 卡”。

## 代码掌握问题

1. 为什么三次 3072→3072 GEMM 合成一次 3072→9216 后，端到端不保证单调变快？
2. `qkv.chunk(3, dim=-1)` 的 Q/K/V row stride 为什么是 9216；哪个消费者会物化，哪个不会？
3. 用 M429 的实测数写出 9.760 us GEMM saving 最终只剩 0.976 us 的完整公式。
4. 为什么 M215 的 Tensor Active 提升比 M429 大，它如何影响 SP4/SP2 的收益方向？
5. 为什么不能直接调用现有 128-dim fused QKNorm+RoPE 替换 MinWM 的 RMSNorm？
6. `normalized.type_as(hidden_states) * weight` 中间的 BF16 round 为什么属于 bitwise 合同？
7. 为什么单卡 NCU 适合解释 S4，却不适合解释 S3 的 A2A/scheduler 抵消？
8. 新 S4b 为什么应先做 SP1 bitwise/NCU，再进入 SP2/SP4 20+200？
