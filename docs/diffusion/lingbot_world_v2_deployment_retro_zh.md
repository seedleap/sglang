# LingBot-World 2.0 Spot 部署复盘与交接记录

> 记录范围：2026-07-08 ~ 2026-07-10 在 `leap-world`（wms 账号，us-east-2）及 `leap-world-aws03-usw2`/`leap-world-aws03-use2`（aws03 账号，us-west-2/us-east-2）上多次部署 LingBot-World 2.0 realtime 服务的完整过程。当前 GPU 已全部释放，本文档是给后续接手者（含 Codex）看的交接材料，也是给团队下次部署时的时间预估依据。

## 1. 结论摘要：下次部署大概要多久

分两种情况，差异极大，不能只给一个数字：

| 场景 | 预估耗时 | 说明 |
| --- | --- | --- |
| **顺利情况**（spot 容量充足） | **15-20 分钟** | NodePool 建好到节点 Ready 约 2-4 分钟；镜像拉取+`pip install diffusion` 约 2-3 分钟；模型下载（RunAI Streamer，实测 2-2.5 GB/s）约 2-5 分钟；NCCL/server 启动+GPU 注册约 1-2 分钟，偶尔需要 1-2 次重试（见第 5 节坑点）。 |
| **容量紧张情况** | **不可预测，可能数小时** | 我们在 2026-07-10 实测遇到：H100/B200 的 spot **和** on-demand，在 us-east-2（2 个账号）+ us-west-2（1 个账号），一共 6 个"账号×区域×机型×计费方式"组合**同时全部 `InsufficientCapacity`**，持续了一个多小时。这不是配置问题，是当天 Blackwell（B200）+ Hopper 大机型（H100）在多个区域的真实短缺。 |

**给下次部署的建议**：不要像我们这次一样，撞到容量不足才临时现建 NodePool/账号——应该**一开始就并行**在已知可用的几个池子上同时发起请求（见第 3 节），把"发现容量不足→现找备用池子"这个反应式流程省掉，能显著压低最坏情况的等待时间。

## 2. 已验证可用的部署配方

官方 cookbook 命令本身是对的，跑起来没有问题：

```bash
export SGLANG_LINGBOT_LAZY_VAE_ENCODE_BLACK_FRAMES=60
export SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW=true
sglang serve \
  --model-path robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --pipeline-class-name LingBotWorldCausalDMDPipeline \
  --num-gpus 8 --ulysses-degree 8 \
  --dit-cpu-offload false --text-encoder-cpu-offload false \
  --vae-config.use-parallel-decode true \
  --vae-config.parallel-decode-mode spatial \
  --enable-torch-compile false \
  --host 0.0.0.0 --port 30000
```

镜像：不需要自建，容器启动时内联 `pip install -e "python[diffusion]"` 即可（跟官方 Docker 安装文档 Method 3 一致）。`--shm-size` 用 `emptyDir{medium: Memory, sizeLimit: 32Gi}` 挂 `/dev/shm`。

**⚠ 但不要直接写 `lmsysorg/sglang:dev`**——这是一个浮动 tag（跟着上游 `dev` 分支持续重新构建），不是固定版本。这次部署图省事直接用了 `:dev`，本次会话实测拉到的 digest 是：

```
lmsysorg/sglang@sha256:44abe1937f1f55f38fc175399a885d0db3a16adac9fe903f8643491b30e40b09
```

这个坑不只是"下次可能拉到不一样的镜像"这么简单，它还污染了这次的排查方法论：**本次会话全程是靠读本地 checkout 的仓库源码（`d9a7e0e663` 这个 commit）来推断远端容器里实际在跑的行为**（比如 `camera_actions` Script/State 模式消费逻辑、`interactive_kv_window` 自适应窗口、`output_pace` 限速机制），但从未确认过 `:dev` 镜像里打包的代码版本跟本地 checkout 是不是同一个 commit——如果不是，读代码得出的所有结论都可能对不上容器里实际跑的东西。**下次部署务必固定成一个具体 tag 或 digest**（比如对应某次 release，或者干脆自建镜像固定到某个 commit），部署时把用到的 digest 记下来，方便排查时对照。

## 3. 各区域/账号的 spot 可得性实测数据

这是这次花了不少时间才拿到的一手数据，比 AWS Spot Placement Score 更直接（那个 API 在真正缺货的时候也只会显示 1 分，区分度不够）：

| 账号 | 区域 | 机型 | 结果 |
| --- | --- | --- | --- |
| wms (829115578968) | us-east-2 | H100 (p5.48xlarge) spot | 最初实测能拿到（EC2NodeClass `minwm-test-h100-ec2` 已验证），后来容量紧张时段拿不到 |
| wms | us-east-2 | H100 on-demand | 容量紧张时段拿不到 |
| wms | us-east-2 | B200 (p6-b200.48xlarge) spot | 能拿到，跑了 ~8 小时被 spot 回收；容量紧张时段拿不到 |
| wms | us-east-2 | B200 on-demand | 容量紧张时段拿不到 |
| aws03 (107014413969) | us-west-2 | B200 spot/on-demand | 都拿不到（`InsufficientInstanceCapacity`，us-west-2a/b/d 轮流报，是循环话术不是真的换区能解决） |
| aws03 | us-east-2 | B200 spot | 同上拿不到 |

**结论**：H100/B200 这两种 8 卡大机型，在"俄亥俄/俄勒冈比弗吉尼亚容易拿到"这个经验法则之外，**真正紧张的时候是全区域、全账号、全计费方式一起紧张**，不是换个区域就能绕开的。us-west-2（aws03 账号）现成集群 `leap-world-aws03-usw2` 已经有 `minwm-spot-p6-b200-0703` 等节点组模板，下次要在这个账号试更多机型，照抄这个 EKS 托管节点组模式（不是 Karpenter）。

## 4. 这次修的两个真实 bug（已验证，已修复并重新生成过对照结果）

跑 minWM 的 `testset100_v2` eval set 时发现生成的视频"整体速度快一倍""后半段花掉"，定位到两个独立问题：

1. **`camera_actions` 不能按 4x（`temporal_downsample_factor`）展开**。它是按 **latent 帧粒度**逐帧消费的（`sample_camera_actions(chunk_size)` 的 `chunk_size` = `num_frames_per_block`=3，不是解码帧数），minWM 原始 trajectory（如 `"i*42,k*37"`）应该原样喂给 `camera_actions`，只有目标解码帧数（`len(actions)*4`）需要乘 4。
2. **`fps` 传参用错了**。cookbook 示例的 `fps=25` 是给交互式 realtime demo 用的，这类脚本化长镜头内容应该用 `fps=16`（`LingBotWorldSamplingParams` 自己的默认值，也是 minWM 参考视频用的值）。`fps` 本身不影响生成内容（已读代码确认：只在 `_output_pacing_fps`/服务端限速发送时用到，不影响 DiT/VAE 计算），只影响本地编码 mp4 时贴的"这批帧该按多快播放"这个标签，标签贴错了看起来就像加速。

两个都修完之后，8 个样本重新生成，时长跟 minWM 参考视频对得上（19.75s vs 19.81s），已经上传到 `s3://leap-world-us-east-2/world-model/eval/platform/eval_outputs/testset100_v2_lingbot_world_v2_sglang_switch8/`。

## 5. 悬而未决的问题（接手者请注意）

**WebUI 手动操作时画面运动看起来正常，但通过我们的 eval 脚本（Script 模式，一次性预设 `condition_inputs.camera_actions` 列表）跑同一个样本，运动量比 minWM 参考视频高约 3 倍——这个差异还没有定位到根因。**

已确认的事实：
- WebUI 走的是完全不同的代码路径：**State 模式**（`camera_state_queue`，发 `{"mode":"state","transitions":[...]}` 事件，语义是"当前按住哪些键"），而不是我们脚本用的 **Script 模式**（`camera_script_queue`，一次性推入 `list[list[str]]`，FIFO 逐帧消费）。
- 读了 `control_signals.py` 里两条路径各自的消费逻辑（`ControlScriptQueue.sample_script` vs `ControlStateQueue.sample_chunk`），机械层面没有看出明显 bug（Script 模式就是严格按顺序 `chunk_size` 个一取，没有重复/跳过）。
- 一次"零 camera_actions"对照测试（`condition_inputs: {}`，落到 State 模式的"从未收到任何按键"默认分支）测出来运动量还是偏高（307.8/秒 vs minWM 参考 100.8/秒）——但这个测试本身有设计缺陷：没有触发 `SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW` 的自适应窗口（因为 `condition_inputs` 里根本没有 `camera_actions` 这个 key，`_uses_interactive_kv_window` 直接返回 False），跟真实 WebUI 空闲时的行为不是同一条代码路径，对照组不干净。
- 后续设计了一个用 **State 模式**精确复现 minWM trajectory（`"i*42,k*37"` → 先发 hold-i 事件，42 latent 帧后再发 hold-k 事件）的诊断脚本（`diag_state_mode.py`，逻辑见下），想直接对比 State 模式 vs Script 模式在同一个样本上的运动量差异——**但因为本机 port-forward 反复断线，5 次+20 次重试都没能跑完一整轮（316 帧），没有拿到最终数据就被要求释放 GPU 了**。

如果要继续查，建议：
1. 用稳定的连接方式（内网/公网 NLB，不要用 `kubectl port-forward`，见第 6 节）重跑 State 模式对照测试。
2. 如果 State 模式测出来运动量正常（接近 minWM 参考的 100.8/秒），基本可以确认是 Script 模式这条路径本身有问题（可能在更深的 DiT 条件编码层，不一定在 `control_signals.py` 这层机械逻辑里，需要往 `LingBotWorldCamConditioner` 或 `sample_chunk_inputs` 消费后的下游找）。
3. 如果 State 模式也偏高，说明问题更可能在别的地方（比如具体的 `guidance_scale`/`num_inference_steps` 组合，或者 DMD 蒸馏本身在当前配置下就是比 minWM 参考用的配置更"抖"）。

诊断脚本思路（原脚本在本次会话的临时 scratchpad 里，不在仓库中，**Codex 接手后需要重新写**，逻辑很简单）：
- 用同一张参考图 + 同一个 prompt，不在 `init` 消息里放 `condition_inputs.camera_actions`。
- 连接建立后立即发一个 `event, kind=camera_actions, payload={"mode":"state","transitions":[{"actions":["i"],...}]}`（模拟按住 i）。
- 收满 42×4=168 帧解码帧后，再发一个 state 事件切到 `["k"]`。
- 收满 79×4=316 帧后统计相邻帧灰度差均值，跟 Script 模式的结果（mean_diff≈28.34，motion_per_second≈453.4）和 minWM 参考（mean_diff≈6.30，motion_per_second≈100.8）对比。

## 6. 基础设施踩坑记录

- **自定义 taint 会让 GPU 注册不上**：`nvidia-device-plugin` DaemonSet 只信任预先写死在自己 `tolerations` 里的 taint 白名单（如 `seedleap.ai/workload=wan22-ti2v`），新建 NodePool 时如果打了一个白名单之外的自定义 taint，GPU 节点会一直卡在 `nvidia.com/gpu` 不注册，报错也不直接。**解决办法：不打自定义 taint，只用 label + nodeSelector 做隔离**，跟集群里其他不带 taint 的池子（如 `minwm-test-h100-spot`）保持一致模式。
- **PVC 会把节点锁死在一个可用区**：EBS 卷是分区域的，`WaitForFirstConsumer` 模式的 PVC 一旦第一次绑定到某个 AZ，后续所有重新调度都会被这个 PVC 的 `nodeAffinity` 锁死在同一个 AZ，即使那个 AZ 恰好缺货、别的 AZ 有容量也没用。真出现容量紧张需要跨 AZ 重试时，要先删掉旧 PVC 重建一个全新的（代价是要重新下载一次模型权重，14B 权重用 RunAI Streamer 下载不算慢，划算）。
- **同一 NodePool 不能混用不同机型的 EC2NodeClass**：H100 的 EC2NodeClass 里有针对 p5.48xlarge 硬件布局写死的 EFA 网卡配置（15 个 EFA 接口），直接套给 B200（p6-b200.48xlarge）大概率会因为网卡数量不匹配而失败。不同机型要用各自独立的 NodePool + EC2NodeClass。
- **`kubectl port-forward` 不适合承载真实流量**：这次专门开了一个 session 排查（结论见下），根因是**本机网络环境本身不稳**（不是 EKS token 过期——20小时审计日志实测 145 条 port-forward 记录 0 个 401，连接存活最长到过 3622 秒，完全证伪了 token 15 分钟 TTL 杀连接的猜测），叠加当天本机 kubectl 版本落后 apiserver 4 个大版本（已升级到 1.36.2 通过 Homebrew 覆盖 Docker Desktop 自带的旧版本，`/opt/homebrew/bin/kubectl` 排在 PATH 最前）。**正式方案应该用内网/公网 NLB**（`type: LoadBalancer`，纯 AWS 四层网络，不经过 apiserver），我们已经建了 `lingbot-world-v2-internal-lb` 和一个公网+办公网限制的 NLB 并验证 target health 全绿；更彻底的方案是 SSM Session Manager port-forward（完全绕开 K8s apiserver），但账号目前缺 `ssm:StartSession` 权限，需要有 IAM 权限的人补上。
- **单进程只允许一个活跃 realtime 会话**：`realtime_video_api.py` 里 `_ACTIVE_SESSION_IDS` 硬限制，第二个 WebSocket 连接会被拒（close code 1008）。这个应用层限制跟传输层怎么选无关，多人同时用必须要么排队，要么起多个副本（8 卡拆成 2×4 卡等，会牺牲单会话延迟，未验证过 v2 在 4 卡上是否还能维持实时性）。

## 7. 已释放的资源

以下内容截至本文档写作时已全部删除/终止，不再产生费用：

- wms/us-east-2：Deployment×2、Service×4、PVC×2、NodePool×5、B200 spot 节点（`i-0c9a68c6c4af8d1f6`，2026-07-11 00:49:44 UTC 终止）
- aws03/us-west-2：Deployment、Service、PVC、节点组×2
- aws03/us-east-2：Deployment、Service、PVC、节点组×1

下次部署需要从第 2 节的配方重新开始，本文档第 3 节的容量数据仅供参考、不代表实时可用性，部署前建议先用 `aws ec2 get-spot-placement-scores` 或直接并行尝试确认当前容量状况。
