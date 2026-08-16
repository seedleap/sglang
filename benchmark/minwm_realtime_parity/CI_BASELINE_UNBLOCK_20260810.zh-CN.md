# MinWM CI 基线解阻记录（2026-08-10）

## 目的

这不是新的融合优化，也不改变任何 H200 性能结论。它只修复 `main` 已有的 CI 基线问题，
使 MinWM 5.3 的六个独立 PR 能重新把“真实代码回归”与“仓库质量债”分开验收。

## 根因与修复

| 现象 | 根因 | 修复 | 验证 |
| --- | --- | --- | --- |
| CPU/Arm64 在收集阶段全部失败 | `test_ulysses_qkv_pack.py` 是 pytest 风格测试，却没有直接执行入口；CI 的 registered-test collector 因而 fail-closed | 添加 `sys.exit(pytest.main([__file__, "-v"]))` 入口 | `collect_tests(...)` 成功返回 1 个测试 |
| 全仓 YAML hook 失败 | CloudFormation 模板使用 `!Ref`，并不是通用 YAML loader 可构造的 tag | 仅排除该 CloudFormation 模板；保留所有普通 YAML 检查 | `check-yaml` 通过 |
| shebang hook 失败 | 已提交的可执行脚本缺少 executable mode | 仅对含 shebang 且 mode 为 `100644` 的现有脚本改为 `100755` | shebang hook 通过 |
| isort/ruff/Black 失败 | `main` 累积的机械格式差异 | 直接采用仓库锁定的 pre-commit 版本格式化 | 改动文件的完整 pre-commit 通过 |

## 预期差异与决策

- 格式化覆盖当前 hook 实际报告的文件；没有 API、算法、模型权重、测量口径或 H200 runner
  改动。
- CloudFormation 模板不改写成伪通用 YAML，也不为全仓开启不安全 YAML loader；精确排除该
  模板，后续仍应由 CloudFormation/基础设施校验器验证。
- 这个 PR 的作用是解除 CI 基线阻塞，而不是替代 MinWM 各 PR 的单测、bitwise、SP2/SP4
  profiler-off 和 Nsight 验收。

## 本地验收

- 完整 pre-commit 首轮已完成；对全部 89 个变更文件复验通过。
- `python -m compileall` 通过。
- `scripts/ci/check_registered_tests.py` 通过。
- 与 CI 相同的 `collect_tests` 调用成功收集 Ulysses QKV 测试。

## 让我掌握代码改动：检查题

1. 为什么 registered-test collector 对 pytest 文件缺少 `__main__` 入口要 fail-closed，
   而不是允许它静默跳过？
2. 为什么 CloudFormation 的 `!Ref` 不应通过全局放宽 YAML loader 来解决？
3. 为什么改为 `100755` 是 shebang 问题的正确修复，而不是删除 shebang？
4. 为什么 CI 解阻 PR 必须与算子融合 PR 分开，且不能把格式化后的 FPS 变化归因于本 PR？
