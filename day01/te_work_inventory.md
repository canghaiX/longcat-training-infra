# TE 昇腾 CI/CD 适配工作盘点

## 公开记录边界

不要写入：

- 内部仓库地址、CI 平台地址、账号、Token、机器 IP。
- 未公开芯片型号、数量、拓扑、客户信息。
- 内部代码片段、内部错误日志全文、内部 issue/PR 链接。

可以写入：

- 抽象后的适配层级、问题类型、验证方法。
- 公开版本号、公开 API 名称、公开文档链接。
- 去标识化后的失败分类和改进动作。

## 个人开发环境记录

| 项目 | 当前记录 | 待确认 |
| --- | --- | --- |
| 操作系统 | Windows 10 `10.0.19045` | 是否另有 Linux/NPU 开发机 |
| Python 版本 | Python 3.9.6 | NPU 机器上的 Python 版本 |
| PyTorch 版本 | 当前活动 Python 未安装 `torch` | NPU 环境实际版本 |
| `torch_npu` 版本 | 当前活动 Python 未安装 `torch_npu` | NPU 环境实际版本 |
| CANN 版本 | 当前 shell 未发现 `ASCEND/CANN` 相关环境变量 | NPU 环境实际版本 |
| TE-FL 版本 | `v0.2.0-rc2.post1-25-g963ddd88b` | 是否对应公开基线或内部补丁集 |
| TE-FL 分支 | `feature/ascend-unit-ci` | 是否需要保留为公开描述 |
| 昇腾卡型号与数量 | 当前机器未检测到 `npu-smi` | 在真实 NPU 机器上补充公开可写信息 |
| 当前通信后端 | 当前 shell 未运行分布式任务 | Ascend 多卡通常关注 HCCL，需以实际任务为准 |
| CI/CD 执行方式 | 待盘点 | 本地脚本 / GitHub Actions / 内部 CI 的公开抽象描述 |
| 已有单卡测试情况 | 待盘点 | 通过哪些公开可描述 test category |
| 已有多卡测试情况 | 待盘点 | rank 数、后端、失败类型，不写内部机器信息 |

## 适配层级

| 层级 | 名称 | 达成标准 | 当前状态 | 证据 |
| --- | --- | --- | --- | --- |
| L0 | 仓库与环境可解析 | 能安装依赖、导入包、收集版本信息 | 待确认 | 本机只完成 Git/文档层记录 |
| L1 | CPU/模拟路径测试 | 基础单测可运行，能排除语法和纯 Python 问题 | 待确认 | 待补命令和结果摘要 |
| L2 | NPU 单卡测试 | 单卡 forward/backward/核心 op 单测通过 | 待确认 | 待补公开测试类别 |
| L3 | NPU 多卡通信 | HCCL 初始化、rank/world size、collective 或分布式单测通过 | 待确认 | 待补公开测试类别 |
| L4 | CI 自动触发 | push/PR/定时任务能自动跑目标测试集合 | 待确认 | 不记录内部 CI URL |
| L5 | 回归门禁 | 失败能阻断合入，日志足够定位，flaky 有治理策略 | 待确认 | 待补问题分类 |

## 工作盘点模板

| 事项 | 背景问题 | 我的动作 | 结果 | 还能往下做什么 |
| --- | --- | --- | --- | --- |
| CI 环境变量整理 | 训练/测试脚本依赖设备和 rank 信息 | 抽象必要变量，不记录内部值 | 待补 | 加入 preflight 检查 |
| 单卡测试入口 | 需要先证明核心 op 在单 NPU 可跑 | 待补 | 待补 | 固化 smoke test |
| 多卡测试入口 | 需要验证 HCCL 初始化和 collective | 待补 | 待补 | 增加超时、日志归档、失败分类 |
| 依赖安装 | CANN、torch_npu、TE 版本需要匹配 | 待补 | 待补 | 写版本矩阵 |
| 日志与失败定位 | CI 失败要能快速区分环境/代码/通信问题 | 待补 | 待补 | 标准化错误分类 |

## 下一步核验命令

以下命令只记录“版本/类别/是否通过”，不要提交内部路径或完整日志：

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torch_npu; print(torch_npu.__version__)"
npu-smi info
env | grep -E 'ASCEND|CANN|HCCL|RANK|WORLD|MASTER'
pytest -q <public-or-sanitized-test-path>
```

