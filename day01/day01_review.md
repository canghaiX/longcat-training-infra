# Day 01 Review

## 今天完成了什么

- [x] 建立 `longcat-training-infra/day01` 目录。
- [x] 建立 LongCat 能力差距矩阵。
- [x] 建立 TE 昇腾 CI/CD 适配盘点表。
- [x] 建立 Transformer 训练流程笔记。
- [x] 建立 TE 调用链笔记。
- [x] 写出可运行的显存估算脚本。
- [x] 准备面试问题清单。

## 今天刻意不做什么

- 不深入 Megatron。
- 不深入 MoE。
- 不深入 vLLM。
- 不深入 Triton。
- 不提交内部地址、账号、未公开芯片信息或内部代码。

## 当前关键结论

1. 目标岗位能力可以拆成：训练流程认知、TE 技术栈、昇腾软件栈、分布式通信、CI/CD 工程化、调试表达。
2. 目前本机只能确认文档和仓库层信息，NPU、CANN、torch_npu、HCCL 需要在真实 NPU 环境补充。
3. TE 昇腾 CI/CD 适配要按 L0-L5 分层盘点，不能只说“做了 CI”。
4. 面试表达要围绕“问题、动作、结果、边界”，避免暴露内部实现细节。

## 明天建议

- 用真实 NPU 环境补齐 `te_work_inventory.md` 的版本和测试状态。
- 选一个公开可描述的 CI 失败案例，整理成 STAR。
- 跑一次 `memory_estimator.py`，用结果解释显存组成。
- 从 TE 的一个 module 入口开始，画出更具体的源码阅读路径。

