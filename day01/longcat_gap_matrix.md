# LongCat 目标岗位能力差距矩阵

> 说明：这里先按“训练基础设施 / TE 昇腾适配 / CI/CD / 分布式训练调试”方向建立能力假设。后续拿到真实 JD 后，再把“岗位需要”列改成 JD 原文抽象，不直接复制内部信息。

## 能力矩阵

| 能力域 | 岗位可能需要 | 当前证据 | 差距判断 | Day 01 动作 |
| --- | --- | --- | --- | --- |
| Linux 与 Python 工程 | 能读写训练脚本、测试脚本、CI 脚本，定位环境问题 | 本机当前 Python 3.9.6；Day 01 新建学习仓库 | 待补充真实项目证据 | 建立环境记录和脚本目录 |
| PyTorch 基础 | 理解 Module、autograd、optimizer、AMP、distributed 基础 API | 当前活动 Python 未安装 `torch` | 本机环境未就绪，认知需系统化 | 写 `training_flow.md` |
| Transformer 训练流程 | 能解释 forward、loss、backward、optimizer、显存构成 | 已开始整理训练流程 | 需要形成可面试表达 | 完成 `training_flow.md` 和 `memory_estimator.py` |
| TransformerEngine 技术栈 | 知道 TE 在模型层、autograd、kernel/backend 的位置 | 当前 TE-FL 仓库版本：`v0.2.0-rc2.post1-25-g963ddd88b` | 需要把调用链讲清楚 | 完成 `te_call_chain.md` |
| 昇腾软件栈 | 理解 `torch_npu`、CANN、HCCL、NPU runtime 的基本分层 | 当前 shell 未检测到 `torch_npu`、CANN 环境变量或 `npu-smi` | 本机不是可直接跑 NPU 的环境，需在真实机器补记录 | 在 `te_work_inventory.md` 留出核验项 |
| CI/CD 适配 | 能说明流水线触发、构建、单测、多卡测试、产物、失败定位 | 当前 TE-FL 分支：`feature/ascend-unit-ci` | 需要盘点做到哪一层 | 完成分层 inventory |
| 分布式通信 | 理解 HCCL/NCCL/Gloo 的使用场景，知道 rank/world size 基础 | 当前 shell 未设置分布式环境变量 | 需要真实单机多卡运行证据 | 暂只记录问题清单 |
| 性能与稳定性 | 能看懂耗时、显存、精度误差、失败重试和 flaky test | 尚未形成公开记录 | 需要补可公开案例 | Day 01 只列问题，不深入性能优化 |
| 面试表达 | 能把自己的 CI/CD 适配说成“问题-动作-结果-边界” | 尚未复盘 | 需要准备问答 | 完成 `interview_questions.md` |

## 今日结论

- Day 01 重点不是扩展到更多技术名词，而是把“我做过什么、做到哪一层、下一层是什么”讲清楚。
- TE 昇腾 CI/CD 适配要按层级盘点：环境可用、单卡测试、多卡测试、CI 触发、失败诊断、回归门禁。
- Transformer 训练流程要能从一个 batch 讲到参数、激活、梯度、optimizer state 和通信。

