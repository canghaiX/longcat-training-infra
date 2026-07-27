# TransformerEngine 调用链认知

## TE 在训练栈里的位置

```text
Training script
  -> PyTorch Module / optimizer / autograd
    -> TransformerEngine Python API
      -> TE module wrapper, FP8/mixed precision metadata, custom autograd
        -> backend extension / dispatcher
          -> device kernel and runtime library
            -> CUDA/NCCL or Ascend/CANN/HCCL 等设备栈
```

## 从一层 Linear 看调用链

1. 模型代码调用 TE 提供的 Linear、LayerNormLinear 或 fused module。
2. Python 层检查输入、参数 dtype、低精度配置和 runtime metadata。
3. forward 进入自定义 autograd function 或 extension binding。
4. backend 选择具体 kernel 或算子路径。
5. device runtime 执行矩阵乘、归一化、cast、transpose、scale 等操作。
6. backward 根据保存的 tensor 和 metadata 计算 input gradient、weight gradient、bias gradient。
7. 分布式场景下，梯度同步或张量并行通信会调用对应 collective backend。

## 昇腾适配需要关注的问题

| 层 | 关注点 | 常见问题 |
| --- | --- | --- |
| Python API | 是否和上层训练代码兼容 | 参数名、默认 dtype、上下文管理差异 |
| Autograd | forward/backward 保存内容是否正确 | backward 缺 kernel、shape/dtype 不一致 |
| Backend binding | extension 是否能编译和加载 | ABI、依赖库、符号、设备检查 |
| CANN/ACL | 算子是否存在，layout 是否匹配 | 算子不支持、workspace、stream 同步 |
| HCCL | 多卡通信是否初始化和收敛 | rank 配置、超时、网卡/拓扑问题 |
| CI/CD | 是否能稳定重现并分类失败 | 环境漂移、日志不足、flaky test |

## 和 CI/CD 的关系

TE 昇腾适配不是只看“代码能不能编译”，而是要分层验证：

- import 层：包能导入，extension 能加载。
- op 层：核心算子 forward/backward 单测通过。
- module 层：TE module 和 PyTorch module 行为对齐。
- dtype 层：FP32/FP16/BF16/FP8 或相关路径的数值误差可解释。
- device 层：NPU 单卡能跑。
- distributed 层：HCCL 多卡能初始化、通信、退出。
- CI 层：自动触发、日志归档、失败阻断、版本矩阵清晰。

## Day 01 不展开的边界

- 不深入 Megatron 的 tensor/pipeline parallel 具体实现。
- 不深入 MoE router、expert parallel 或负载均衡。
- 不深入 vLLM 推理调度。
- 不深入 Triton kernel 编写。

