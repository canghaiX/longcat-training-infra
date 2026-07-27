#!/usr/bin/env python3
"""Estimate rough Transformer training memory.

This is a learning tool, not a replacement for profiler output. It estimates
parameter, gradient, optimizer-state, and activation memory for a decoder-only
Transformer-like model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


DTYPE_BYTES = {
    "fp32": 4,
    "tf32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
}


@dataclass(frozen=True)
class ModelConfig:
    layers: int
    hidden: int
    ffn_hidden: int
    vocab_size: int
    seq_len: int
    batch_size: int
    tied_embeddings: bool
    gated_mlp: bool


@dataclass(frozen=True)
class PrecisionConfig:
    param_dtype: str
    grad_dtype: str
    optimizer_state_dtype: str
    activation_dtype: str
    optimizer_states_per_param: int


def bytes_for(num_elements: int, dtype: str) -> int:
    return num_elements * DTYPE_BYTES[dtype]


def estimate_params(config: ModelConfig) -> int:
    """Estimate decoder-only Transformer parameter count."""

    embedding = config.vocab_size * config.hidden
    lm_head = 0 if config.tied_embeddings else config.vocab_size * config.hidden

    # Attention: q, k, v projections plus output projection.
    attention = 4 * config.hidden * config.hidden

    # MLP: common SwiGLU/geglu-style MLP has gate, up, and down projections.
    mlp_matrices = 3 if config.gated_mlp else 2
    mlp = mlp_matrices * config.hidden * config.ffn_hidden

    # Two norms per block plus a final norm. Scale-only norm is the common case.
    norms = (2 * config.layers + 1) * config.hidden

    return embedding + lm_head + config.layers * (attention + mlp) + norms


def estimate_activation_elements(config: ModelConfig, checkpoint_ratio: float) -> int:
    """Approximate saved activation elements.

    A plain Transformer block saves multiple hidden-sized tensors for backward.
    The multiplier is intentionally simple so that the estimate is explainable.
    checkpoint_ratio=1.0 means save the full rough activation set; 0.5 means
    activation checkpointing cuts the saved activations roughly in half.
    """

    hidden_tokens = config.batch_size * config.seq_len * config.hidden
    per_layer_multiplier = 8
    return int(config.layers * hidden_tokens * per_layer_multiplier * checkpoint_ratio)


def format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024**3)
    mib = num_bytes / (1024**2)
    return f"{gib:.2f} GiB ({mib:.0f} MiB)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate rough decoder-only Transformer training memory."
    )
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--ffn-hidden", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--param-dtype", choices=DTYPE_BYTES, default="bf16")
    parser.add_argument("--grad-dtype", choices=DTYPE_BYTES, default="bf16")
    parser.add_argument("--optimizer-state-dtype", choices=DTYPE_BYTES, default="fp32")
    parser.add_argument("--activation-dtype", choices=DTYPE_BYTES, default="bf16")
    parser.add_argument("--optimizer-states-per-param", type=int, default=2)
    parser.add_argument("--checkpoint-ratio", type=float, default=1.0)
    parser.add_argument("--untie-embeddings", action="store_true")
    parser.add_argument("--plain-mlp", action="store_true")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--zero-stage",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Very rough ZeRO sharding model: stage 1 shards optimizer state, "
        "stage 2 also shards gradients, stage 3 also shards parameters.",
    )
    return parser.parse_args()


def shard_if_needed(num_bytes: int, devices: int, enabled: bool) -> int:
    if enabled and devices > 1:
        return num_bytes // devices
    return num_bytes


def main() -> None:
    args = parse_args()
    if not 0 < args.checkpoint_ratio <= 1:
        raise SystemExit("--checkpoint-ratio must be in (0, 1].")
    if args.devices < 1:
        raise SystemExit("--devices must be >= 1.")

    model = ModelConfig(
        layers=args.layers,
        hidden=args.hidden,
        ffn_hidden=args.ffn_hidden,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        tied_embeddings=not args.untie_embeddings,
        gated_mlp=not args.plain_mlp,
    )
    precision = PrecisionConfig(
        param_dtype=args.param_dtype,
        grad_dtype=args.grad_dtype,
        optimizer_state_dtype=args.optimizer_state_dtype,
        activation_dtype=args.activation_dtype,
        optimizer_states_per_param=args.optimizer_states_per_param,
    )

    params = estimate_params(model)
    activation_elements = estimate_activation_elements(model, args.checkpoint_ratio)

    param_bytes = bytes_for(params, precision.param_dtype)
    grad_bytes = bytes_for(params, precision.grad_dtype)
    optimizer_bytes = bytes_for(
        params * precision.optimizer_states_per_param,
        precision.optimizer_state_dtype,
    )
    activation_bytes = bytes_for(activation_elements, precision.activation_dtype)

    per_device_param = shard_if_needed(param_bytes, args.devices, args.zero_stage >= 3)
    per_device_grad = shard_if_needed(grad_bytes, args.devices, args.zero_stage >= 2)
    per_device_optimizer = shard_if_needed(
        optimizer_bytes, args.devices, args.zero_stage >= 1
    )
    per_device_total = (
        per_device_param + per_device_grad + per_device_optimizer + activation_bytes
    )

    print("Transformer memory estimate")
    print("===========================")
    print(f"Parameters: {params:,}")
    print(f"Devices: {args.devices}")
    print(f"ZeRO stage: {args.zero_stage}")
    print()
    print(f"Parameter memory: {format_bytes(per_device_param)}")
    print(f"Gradient memory: {format_bytes(per_device_grad)}")
    print(f"Optimizer memory: {format_bytes(per_device_optimizer)}")
    print(f"Activation memory: {format_bytes(activation_bytes)}")
    print("---------------------------")
    print(f"Estimated per-device total: {format_bytes(per_device_total)}")
    print()
    print("Notes:")
    print("- Temporary buffers, framework allocator cache, and communication buffers are not included.")
    print("- Activation memory is intentionally approximate; validate with a real profiler.")


if __name__ == "__main__":
    main()

