# Transformer 训练流程

## 一句话主线

Transformer 训练就是：从数据集中取一个 batch，执行 forward 得到 loss，再通过 backward 得到梯度，optimizer 根据梯度更新参数；在大模型训练里，还要同时管理显存、混合精度、分布式通信和 checkpoint。

## 单步训练流程

1. 数据读取
   - tokenizer 把文本转成 token id。
   - dataloader 组 batch，通常得到 `input_ids`、`labels`、`attention_mask`。
   - 训练语言模型时，labels 常常是右移后的 token。

2. Forward
   - token id 进入 embedding，得到 `[batch, seq, hidden]`。
   - 每个 Transformer block 通常包含 attention、MLP、residual、normalization。
   - 最后一层 hidden states 经过 lm head 得到 vocab logits。
   - logits 和 labels 计算 cross entropy loss。

3. Backward
   - autograd 从 loss 反向传播。
   - 每个参数得到 `.grad`。
   - activation 是否保存、是否重算，决定显存和速度取舍。

4. Optimizer step
   - optimizer 根据梯度更新参数。
   - Adam 类 optimizer 通常保存一阶矩和二阶矩，因此 optimizer state 显存很大。
   - 使用 gradient accumulation 时，不是每个 micro batch 都 step。

5. 混合精度
   - 参数、激活、梯度可能使用 FP32、FP16、BF16 或 FP8。
   - 混合精度目标是减少显存和提升吞吐，但要控制数值稳定性。
   - TransformerEngine 的价值之一是在特定层和 kernel 中封装高性能低精度计算。

6. 分布式训练
   - Data Parallel：不同 rank 处理不同 batch shard，反向后同步梯度。
   - Tensor Parallel：把单层大矩阵切到多个设备上，层内需要通信。
   - Pipeline Parallel：把不同层放到不同设备上，micro batch 流水执行。
   - Day 01 只建立概念，不深入 Megatron 实现细节。

## 显存由什么组成

| 类型 | 说明 | 常见影响因素 |
| --- | --- | --- |
| Parameters | 模型权重 | 参数量、精度、是否切分 |
| Gradients | 反向传播得到的梯度 | 是否 zero/offload/shard |
| Optimizer states | Adam 的一阶/二阶矩等 | optimizer 类型、精度、ZeRO/shard |
| Activations | forward 中为 backward 保存的中间结果 | batch、seq、hidden、层数、checkpointing |
| Temporary buffers | kernel workspace、通信 buffer、框架缓存 | backend、算子实现、通信后端 |

## 面试表达版本

如果被问“Transformer 训练一步发生了什么”，可以这样回答：

> 一个 batch 先经过 embedding 和多层 Transformer block 做 forward，得到 logits 并计算 loss。随后 autograd 根据 loss 反向传播，得到每个参数的梯度。optimizer 用这些梯度更新参数。大模型场景下，真正复杂的是显存和通信：参数、梯度、optimizer state、activation 都要占显存；多卡训练还要在 backward 或层内做 collective 通信。TE 这类库主要出现在高性能线性层、attention、norm、低精度计算和 backend kernel 调用链里。

