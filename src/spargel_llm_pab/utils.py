import math
import os
import socket
from collections.abc import Callable, Iterator

import torch
from pyarrow.parquet import ParquetFile
from pydantic import BaseModel
from tokenizers import Tokenizer

from spargel_llm.model import Config
from spargel_llm.train.basic import Item, compile_batch_data
from spargel_llm.train.sft import Message, build_sft_item_chatml

# Learning rate scheduler

type LRFunc = Callable[[int], float]


def linear(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
    k = (y2 - y1) / (x2 - x1)
    return k * (x - x1) + y1


def cosine(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
    t = (x - x1) / (x2 - x1)
    A = (y1 - y2) / 2
    return (y1 + y2) / 2 + A * math.cos(t * math.pi)


def linear_warmup_cosine_decay(
    max_lr: float, max_steps: int, warmup_steps: int = 100
) -> LRFunc:
    assert max_steps >= warmup_steps

    def lr_func(step: int) -> float:
        if step < warmup_steps:
            return linear(0, warmup_steps, 0, max_lr, step)
        elif step < max_steps:
            return cosine(warmup_steps, max_steps, max_lr, 0, step)
        else:
            return 0

    return lr_func


# State classes


class PretrainDatasetState(BaseModel):
    group_index: int = 0
    offset: int = 0


class PretrainState(BaseModel):
    step: int = 0
    dataset: dict[str, PretrainDatasetState] = {}


class SFTDatasetState(BaseModel):
    index: int = 0


class SFTState(BaseModel):
    step: int = 0
    dataset: dict[str, SFTDatasetState] = {}


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


class SFTDataIterator(Iterator[Item]):
    def __init__(
        self,
        pf: ParquetFile,
        tokenizer: Tokenizer,
        seq_len: int,
        index: int = 0,
        *,
        pad_id: int,
        im_start_id: int,
        im_end_id: int,
        column_name: str = "messages",
        role_key: str = "role",
        content_key: str = "content",
    ):
        self.pf = pf
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.index = index
        self.pad_id = pad_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.column_name = column_name
        self.role_key = role_key
        self.content_key = content_key

        self._column = None
        self._start = 0
        self._end = 0

    def __iter__(self) -> SFTDataIterator:
        return self

    def __next__(self) -> Item:
        if self._column is None or self.index >= self._end:
            offset = 0
            for rg_idx in range(self.pf.metadata.num_row_groups):
                rg_rows = self.pf.metadata.row_group(rg_idx).num_rows
                if offset + rg_rows > self.index:
                    table = self.pf.read_row_group(rg_idx, columns=[self.column_name])
                    self._column = table.column(self.column_name).combine_chunks()
                    self._start = offset
                    self._end = offset + rg_rows
                    break
                offset += rg_rows
            else:
                raise StopIteration

        local_idx = self.index - self._start
        raw_messages = self._column[local_idx].as_py()
        messages = [
            Message(role=m[self.role_key], content=m[self.content_key])
            for m in raw_messages
        ]

        item = build_sft_item_chatml(
            messages,
            self.tokenizer,
            self.seq_len,
            pad_id=self.pad_id,
            im_start_id=self.im_start_id,
            im_end_id=self.im_end_id,
        )

        self.index += 1
        return item
