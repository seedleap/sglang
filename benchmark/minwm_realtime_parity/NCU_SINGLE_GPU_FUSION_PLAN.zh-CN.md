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

## 验收与决策

1. candidate projection duration 至少不差于 baseline 总 duration，且 Tensor throughput 不降；
2. 若 candidate register/spill 或 active-warps 明显更差，下一步试 tile/algo/packing-layout，而
   不再仅减少 Python module 数；
3. 若 GEMM 本身更快但总 span 不变，优先 profile/simplify chunk-contiguous layout；
4. 若两者都接近带宽上限，停止 QKV fusion 扩展，转向消除 copy/layout；
5. 所有结论只能指导下一轮实现，最终开关仍由 200-chunk profiler-off 与 bitwise 验收决定。
