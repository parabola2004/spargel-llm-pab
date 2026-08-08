import math
import os
import socket
from collections.abc import Callable, Iterator

import torch

from spargel_llm.model import Config
from spargel_llm.train.basic import Item, compile_batch_data

# Learning rate scheduler

type LRFunc = Callable[[int], float]


def linear(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
    k = (y2 - y1) / (x2 - x1)
    return k * (x - x1) + y1


def cosine(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
    t = (x - x1) / (x2 - x1)
    A = (y1 - y2) / 2
    return (y1 + y2) / 2 + A * math.cos(t * math.pi)


# State classes


def setup(
    num_threads: int | None = None,
    float32_precision: str = "high",
) -> str:
    print(f"Host: {socket.gethostname()}")

    # Configure PyTorch
    if num_threads is None:
        num_threads = os.cpu_count() or 8
    torch.set_num_threads(num_threads)
    print(f"PyTorch will use {num_threads} CPU thread(s).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device_name = torch.cuda.get_device_name()
        print(f"Using device: {device} ({device_name})")
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        print(
            f"GPU memory: {free_mem / (1024**3):.2f} GiB free / "
            f"{total_mem / (1024**3):.2f} GiB total"
        )

        torch.set_float32_matmul_precision(float32_precision)
        print(f"Float32 matmul precision: {float32_precision}")
    else:
        print(f"Using device: {device}")

    return device


def warn_bf16(device: str, use_bf16: bool):
    if use_bf16 and device != "cuda":
        print("WARNING: device is not cuda, but BF16 autocast is enabled.")


def next_batches(
    iterator: Iterator[Item],
    batches: int,
    batch_size: int,
    seq_len: int,
    *,
    pad_id: int,
):
    items = []
    for _ in range(batch_size * batches):
        try:
            items.append(next(iterator))
        except StopIteration:
            break
    if len(items) == 0:
        return None

    return compile_batch_data(items, batch_size, seq_len, pad_id=pad_id)


def compute_param_counts(config: Config) -> tuple[int, int]:
    """Compute the number of trainable parameters implied by *config*.

    Returns ``(embedding_params, body_params)`` where *embedding_params*
    counts the input embedding and output projection (lm head), and
    *body_params* counts every other parameter (transformer blocks,
    final norm, positional encoding, etc.).
    """
    # ---- input / output embeddings ----
    embedding_params = config.vocab_size * config.dim  # nn.Embedding
    embedding_params += (
        config.dim * config.vocab_size + config.vocab_size
    )  # nn.Linear (weight + bias)

    # ---- per transformer block ----
    # attention: W_q + W_k + W_v + W_o
    attn_params = (
        config.num_head * config.dim * config.dim_key
        + config.num_head * config.dim * config.dim_key
        + config.num_head * config.dim * config.dim_value
        + config.num_head * config.dim_value * config.dim
    )
    # feed-forward: up (weight + bias) + down (weight + bias)
    ff_params = (
        config.dim * config.dim_ff_hidden
        + config.dim_ff_hidden
        + config.dim_ff_hidden * config.dim
        + config.dim
    )

    body_params = config.num_layer * (attn_params + ff_params)

    # RMSNorm currently uses a plain scalar 1, not an nn.Parameter.
    # PositionalEncoding uses non-persistent nn.Buffers.
    # Therefore neither contributes to the parameter count.

    return embedding_params, body_params


def format_flops(v: float) -> str:
    """Format a FLOP count into a human-readable string with SI prefix.

    >>> format_flops(1500)
    '1,500 FLOPs'
    >>> format_flops(2.5e9)
    '2.50 GFLOPs'
    """
    if v >= 1e18:
        return f"{v / 1e18:,.2f} EFLOPs"
    if v >= 1e15:
        return f"{v / 1e15:,.2f} PFLOPs"
    if v >= 1e12:
        return f"{v / 1e12:,.2f} TFLOPs"
    if v >= 1e9:
        return f"{v / 1e9:,.2f} GFLOPs"
    if v >= 1e6:
        return f"{v / 1e6:,.2f} MFLOPs"
    return f"{v:,} FLOPs"


def format_bytes(v: int) -> str:
    """Format a byte count with an appropriate binary unit.

    >>> format_bytes(42_016_768)
    '40.07 MiB'
    >>> format_bytes(26_662_092_800)
    '24.83 GiB'
    """
    if v >= 1024**4:
        return f"{v / (1024**4):,.2f} TiB"
    if v >= 1024**3:
        return f"{v / (1024**3):,.2f} GiB"
    if v >= 1024**2:
        return f"{v / (1024**2):,.2f} MiB"
    if v >= 1024:
        return f"{v / 1024:,.2f} KiB"
    return f"{v:,} bytes"
