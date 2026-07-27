# longcat-training-infra

个人训练基础设施学习仓库，用于记录 LongCat 目标岗位能力差距、Transformer 训练流程、TransformerEngine/昇腾适配认知和面试复盘。

## Day 01 目标

今天只完成三件事：

1. 明确 LongCat 目标岗位需要什么能力。
2. 盘点目前 TE 昇腾 CI/CD 适配到底做到了哪一层。
3. 建立 Transformer 训练流程和 TE 技术栈认知。

今天暂时不深入 Megatron、MoE、vLLM 和 Triton。

## 目录

```text
longcat-training-infra/
└── day01/
    ├── longcat_gap_matrix.md
    ├── te_work_inventory.md
    ├── training_flow.md
    ├── te_call_chain.md
    ├── memory_estimator.py
    ├── interview_questions.md
    └── day01_review.md
```

## 信息安全边界

不要提交以下内容：

- 公司内部 Git 地址、CI 地址、制品库地址、账号、Token、Cookie、SSH key。
- 未公开芯片型号、板卡规格、机器 IP、集群拓扑、内部错误日志。
- 公司内部代码、补丁、测试数据、截图和不可公开的 issue/PR 链接。
- 能反推出内部系统名称、项目代号、客户名称或发布时间的信息。

可以提交以下内容：

- 已公开版本号、公开文档链接、个人总结、抽象后的流程图。
- 去标识化后的问题分类、能力矩阵、学习笔记。
- 自己写的公开可分享脚本，例如 `memory_estimator.py`。

