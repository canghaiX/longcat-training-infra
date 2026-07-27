# 面试问题准备

## LongCat / 训练基础设施

1. 你怎么理解训练基础设施岗位的核心价值？
2. 如果一个训练任务在单卡能跑、多卡失败，你会怎么定位？
3. 如何区分环境问题、框架问题、算子问题和通信问题？
4. 你做 CI/CD 适配时，如何设计 smoke test、unit test 和 regression test？
5. 一个好的训练 CI 日志应该包含哪些信息？

## Transformer 训练流程

1. 请从一个 batch 开始，讲完整个 Transformer 训练 step。
2. Transformer 训练显存主要由哪些部分组成？
3. Adam optimizer 为什么显存开销大？
4. activation checkpointing 解决什么问题，代价是什么？
5. 混合精度训练为什么可能不稳定？

## TransformerEngine

1. TE 在 PyTorch 训练栈里处在哪一层？
2. TE 的 module 和普通 PyTorch module 有什么关系？
3. 如果 TE 的 forward 通过但 backward 失败，你会看哪些层？
4. FP8 或低精度路径需要额外维护哪些 metadata？
5. TE 适配新后端时，Python API、autograd、backend、kernel 分别要验证什么？

## 昇腾 / CANN / torch_npu / HCCL

1. `torch_npu` 在 PyTorch 和 CANN 之间起什么作用？
2. CANN 环境问题通常会表现成哪些错误类型？
3. HCCL 初始化失败时，你会检查哪些配置？
4. 单卡通过但多卡挂住，可能有哪些原因？
5. 如何在不暴露内部机器信息的情况下描述一个多卡 CI 问题？

## 项目经历表达

1. 你在 TE 昇腾 CI/CD 适配中具体负责哪一层？
2. 这个适配开始前是什么状态，你做完后达到什么状态？
3. 遇到过最难定位的问题是什么？你怎么缩小范围？
4. 有没有把一次失败沉淀成自动化检查？
5. 如果重新做一次，你会先补哪类测试或日志？

## STAR 模板

| 项目 | 内容 |
| --- | --- |
| Situation | 背景是什么，为什么要做 |
| Task | 你的明确目标是什么 |
| Action | 你具体做了哪些动作 |
| Result | 结果如何，最好能量化或分层描述 |
| Boundary | 哪些信息不能公开，如何抽象表达 |

