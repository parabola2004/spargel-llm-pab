import json
import math
import os
import shutil
import socket
import time
import uuid
from argparse import ArgumentParser
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, override

import torch
from pyarrow.parquet import ParquetFile
from pydantic import BaseModel
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from torch import Tensor
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from spargel_llm.model import Config, Model
from spargel_llm.train.basic import (
    BatchData,
    Item,
    compute_validation_metrics,
    train_step,
)
from spargel_llm.train.pretrain import ParquetConcatIterator

from .utils import (
    LRFunc,
    compute_param_counts,
    format_bytes,
    format_flops,
    next_batches,
    setup,
    warn_bf16,
)


# also used in SFT
class PretrainStatistics(BaseModel):
    time: float = 0.0
    tokens: int = 0


class PretrainDatasetState(BaseModel):
    group_index: int = 0
    offset: int = 0


class PretrainState(BaseModel):
    step: int = 0
    dataset: dict[str, PretrainDatasetState] = {}
    statistics: PretrainStatistics = PretrainStatistics()


class PretrainerBase:
    def __init__(
        self,
        model_config: Config,
        seq_len: int,
        batch_size: int,
        data_iterator: Iterator[Item],
        lr_func: LRFunc,
        model_state_path: str,
        optimizer_state_path: str,
        *,
        device: str,
        pad_id: int,
        use_bf16: bool = True,
        tensorboard_dir: str | None = None,
        log_period: int = 100,
        save_period: int = 1000,
        checkpoint_period: int = 4000,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data_iterator = data_iterator
        self.lr_func = lr_func
        self.model_state_path = model_state_path
        self.optimizer_state_path = optimizer_state_path
        self.device = device
        self.pad_id = pad_id
        self.use_bf16 = use_bf16
        self.tensorboard_dir = tensorboard_dir
        self.log_period = log_period
        self.save_period = save_period
        self.checkpoint_period = checkpoint_period

        print(f"Loading model state ({model_state_path})... ", end="")
        self.model = Model(model_config).to(device)
        self.model.load_state_dict(
            torch.load(model_state_path, weights_only=True, map_location=device)
        )
        print("ok.")

        print(f"Loading optimizer state ({optimizer_state_path})... ", end="")
        self.optimizer = AdamW(self.model.parameters())
        if os.path.isfile(optimizer_state_path):
            self.optimizer.load_state_dict(torch.load(optimizer_state_path))
            print("ok.")
        else:
            print("state file not found, use fresh optimizer.")

        self.val_batch_data: BatchData | None = None

        self.step = 0

        self.statistics = PretrainStatistics()

        self.writer: SummaryWriter | None = None

    def run(self, target_step: int):
        t_start = time.perf_counter()

        old_time = self.statistics.time

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        if target_step <= self.step:
            print("Nothing to do.")
            return

        # Open Tensorboard writer
        if self.tensorboard_dir is not None:
            self.writer = SummaryWriter(self.tensorboard_dir)
        writer = self.writer

        # Write run info
        run_id = uuid.uuid4()
        if writer is not None:
            start_info = {
                "event": "train_start",
                "id": str(run_id),
                "time": datetime.now().astimezone().isoformat(),
                "host": socket.gethostname(),
                "target_step": target_step,
                "seq_len": self.seq_len,
                "batch_size": self.batch_size,
                "use_bf16": self.use_bf16,
            }
            if self.device == "cuda":
                mem_free, mem_total = torch.cuda.mem_get_info(self.device)
                start_info["cuda"] = {
                    "device_name": torch.cuda.get_device_name(),
                    "mem_total": mem_total,
                    "mem_free": mem_free,
                }
            self.modify_start_info(start_info)
            writer.add_text("train/log", json.dumps(start_info), self.step)

        # Compute start validation loss
        if self.val_batch_data is not None:
            t_val_start = time.perf_counter()

            val_loss, val_perplexity = compute_validation_metrics(
                self.model,
                self.val_batch_data,
                pad_id=self.pad_id,
                use_bf16=self.use_bf16,
            )
            val_loss, val_perplexity = val_loss.item(), val_perplexity.item()

            t_val_end = time.perf_counter()
            val_time = t_val_end - t_val_start

            print(
                f"{self.step}: val_loss={val_loss:.6f}"
                f", val_ppl={val_perplexity:.4f}"
                f", val_time={val_time:.4f}"
            )
            if writer:
                writer.add_scalar("loss/validation", val_loss, self.step)
                writer.add_scalar("perplexity/validation", val_perplexity, self.step)
                writer.add_scalar("metric/time/validation", val_time, self.step)

        if writer is not None:
            writer.add_scalar("learning_rate", self.lr_func(self.step), self.step)

        # Train loop

        count = 0
        sum_loss = 0.0
        sum_perplexity = 0.0
        sum_time = 0.0
        num_tokens_valid = 0
        num_tokens_all = 0

        start_step = self.step
        for step in range(start_step, target_step):
            t_step_start = time.perf_counter()

            self.step = step

            self.on_step_start()

            # Update learning rate
            lr = self.lr_func(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            # Fetch batch data
            batch_data = next_batches(
                self.data_iterator, 1, self.batch_size, self.seq_len, pad_id=self.pad_id
            )
            if batch_data is None:
                print(f"Data exhausted, stop early at step {step}")
                self.on_data_exhausted()
                break
            batch_data = batch_data.to(self.device)

            loss = train_step(
                self.model,
                self.optimizer,
                batch_data,
                pad_id=self.pad_id,
                use_bf16=self.use_bf16,
            )

            loss = loss.item()
            perplexity = math.exp(loss)

            t_step_end = time.perf_counter()
            step_time = t_step_end - t_step_start

            count += 1
            sum_loss += loss
            sum_perplexity += perplexity
            sum_time += step_time

            step_tokens_valid = sum(batch_data.num_valid)
            num_tokens_valid += step_tokens_valid
            num_tokens_all += batch_data.batches * self.batch_size * self.seq_len

            if writer:
                writer.add_scalar("loss/train", loss, step)
                writer.add_scalar("perplexity/train", perplexity, step)
                writer.add_scalar("learning_rate", lr, step)
                writer.add_scalar("metric/time/train", step_time, step)

            self.on_step_end()

            self.step = step + 1

            if self.step % self.log_period == 0:
                avg_loss = sum_loss / count
                avg_perplexity = sum_perplexity / count
                avg_time = sum_time / count
                sum_loss = 0.0
                sum_perplexity = 0.0
                sum_time = 0.0
                count = 0

                if self.val_batch_data is not None:
                    t_val_start = time.perf_counter()

                    val_loss, val_perplexity = compute_validation_metrics(
                        self.model,
                        self.val_batch_data,
                        pad_id=self.pad_id,
                        use_bf16=self.use_bf16,
                    )
                    val_loss, val_perplexity = val_loss.item(), val_perplexity.item()

                    t_val_end = time.perf_counter()
                    val_time = t_val_end - t_val_start

                    print(
                        f"{self.step}: avg_loss={avg_loss:.6f}"
                        f", val_loss={val_loss:.6f}"
                        f", avg_ppl={avg_perplexity:.4f}"
                        f", val_ppl={val_perplexity:.4f}"
                        f", lr={lr:.2e}"
                        f", avg_time={avg_time:.4f}"
                        f", val_time={val_time:.4f}"
                    )
                    if writer:
                        writer.add_scalar("loss/validation", val_loss, self.step)
                        writer.add_scalar(
                            "perplexity/validation", val_perplexity, self.step
                        )
                        writer.add_scalar("metric/time/validation", val_time, self.step)
                else:
                    print(
                        f"{self.step}: avg_loss={avg_loss:.6f}"
                        f", avg_ppl={avg_perplexity:.4f}"
                        f", lr={lr:.2e}"
                        f", avg_time={avg_time:.4f}"
                    )

            t = time.perf_counter()
            self.statistics.time = old_time + (t - t_start)
            self.statistics.tokens += step_tokens_valid

            if writer and self.step != target_step:
                writer.add_scalar("statistics/tokens", self.statistics.tokens, step)
                writer.add_scalar("statistics/time", self.statistics.time, step)

            if self.step % self.save_period == 0 and self.step != target_step:
                self.save()
                valid_ratio = num_tokens_valid / num_tokens_all
                print(f"valid_ratio={valid_ratio:.4f}")
                num_tokens_valid = 0
                num_tokens_all = 0

            if self.step % self.checkpoint_period == 0 and self.step != target_step:
                self.make_checkpoint()
                if writer:
                    writer.close()
                    self.writer = writer = SummaryWriter(self.tensorboard_dir)

        t_end = time.perf_counter()
        elapsed = t_end - t_start

        self.statistics.time = old_time + elapsed

        self.save()
        self.make_checkpoint()

        if writer:
            writer.add_scalar("statistics/tokens", self.statistics.tokens, self.step)
            writer.add_scalar("statistics/time", self.statistics.time, self.step)

        if self.device == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
        else:
            peak_allocated = 0
            peak_reserved = 0

        print(f"Training completed. (time elapsed: {elapsed:.6f})")

        if self.device == "cuda":
            print(
                f"Peak GPU memory: "
                f"{peak_allocated / (1024**3):.2f} GiB allocated, "
                f"{peak_reserved / (1024**3):.2f} GiB reserved"
            )

        if writer:
            end_info: dict = {
                "event": "train_end",
                "id": str(run_id),
                "time": datetime.now().astimezone().isoformat(),
                "elapsed": round(elapsed, 6),
                "statistics": self.statistics.model_dump(),
            }
            if self.device == "cuda":
                end_info["cuda"] = {
                    "peak_allocated": peak_allocated,
                    "peak_reserved": peak_reserved,
                }
            self.modify_end_info(end_info)
            writer.add_text("train/log", json.dumps(end_info), self.step)
            writer.close()

    def save(self):
        print(f"Saving (step={self.step}).")
        torch.save(self.model.state_dict(), self.model_state_path)
        torch.save(self.optimizer.state_dict(), self.optimizer_state_path)

    def get_checkpoint_file_paths(self):
        return [self.model_state_path, self.optimizer_state_path]

    def make_checkpoint(self):
        print(f"Making checkpoint (step={self.step}).")
        for path in self.get_checkpoint_file_paths():
            path = Path(path).resolve()
            checkpoint_path = path.parent / f".{path.resolve().name}.{self.step}"
            shutil.copyfile(path, checkpoint_path)

    def modify_start_info(self, info: dict[str, Any]):
        pass

    def modify_end_info(self, info: dict[str, Any]):
        pass

    def on_step_start(self):
        pass

    def on_step_end(self):
        pass

    def on_data_exhausted(self):
        pass


class Pretrainer(PretrainerBase):
    def __init__(
        self,
        model_config: Config,
        seq_len: int,
        batch_size: int,
        dataset_name: str,
        lr_func: Callable[[int], float],
        pretrain_state_path: str,
        model_state_path: str,
        optimizer_state_path: str,
        *,
        device: str,
        pad_id: int,
        sep_id: int,
        loop_dataset: bool = True,
        use_bf16: bool = True,
        tensorboard_dir: str | None = None,
        validation_batches: int = 10,
        log_period: int = 100,
        save_period: int = 1000,
        checkpoint_period: int = 4000,
    ):
        self.dataset_name = dataset_name
        self.pretrain_state_path = pretrain_state_path
        self.loop_dataset = loop_dataset

        print(f"Loading pretrain state ({pretrain_state_path})... ", end="")
        with open(pretrain_state_path, "r") as f:
            self.state = PretrainState.model_validate_json(f.read())
        self.dataset_state = self.state.dataset.get(
            dataset_name, PretrainDatasetState()
        )
        print("ok.")

        # Prepare training data
        print(f"Loading training dataset ({dataset_name}.parquet)... ", end="")
        pf_train = ParquetFile(f"{dataset_name}.parquet")
        data_iterator_train = ParquetConcatIterator(
            pf_train,
            seq_len,
            self.dataset_state.group_index,
            self.dataset_state.offset,
            pad_id=pad_id,
            sep_id=sep_id,
            loop=loop_dataset,
        )
        print("ok.")

        super().__init__(
            model_config,
            seq_len,
            batch_size,
            data_iterator_train,
            lr_func,
            model_state_path,
            optimizer_state_path,
            device=device,
            pad_id=pad_id,
            use_bf16=use_bf16,
            tensorboard_dir=tensorboard_dir,
            log_period=log_period,
            save_period=save_period,
            checkpoint_period=checkpoint_period,
        )

        # override type
        self.data_iterator = data_iterator_train

        # Prepare validation data
        print(f"Loading validation data ({dataset_name}_val.parquet)... ", end="")
        pf_val = ParquetFile(f"{dataset_name}_val.parquet")
        data_iterator_val = ParquetConcatIterator(
            pf_val, seq_len, pad_id=pad_id, sep_id=sep_id
        )
        self.val_batch_data = next_batches(
            data_iterator_val, validation_batches, batch_size, seq_len, pad_id=pad_id
        )
        del data_iterator_val
        if self.val_batch_data is not None:
            print(
                f"number of batches: {self.val_batch_data.batches} (expected: {validation_batches})."
            )
            self.val_batch_data = self.val_batch_data.to(device)
        else:
            print("no data, not loaded.")

        self.step = self.state.step

        self.statistics = self.state.statistics

        self.last_group_index = -1

    @override
    def run(self, target_step: int):
        self.last_group_index = -1
        super().run(target_step)

    @override
    def save(self):
        super().save()
        with open(self.pretrain_state_path, "w") as f:
            f.write(self.state.model_dump_json())

    @override
    def get_checkpoint_file_paths(self):
        return super().get_checkpoint_file_paths() + [self.pretrain_state_path]

    @override
    def modify_start_info(self, info):
        info["dataset"] = {
            "name": self.dataset_name,
            "group_index": self.dataset_state.group_index,
            "offset": self.dataset_state.offset,
        }
        info["loop_dataset"] = self.loop_dataset

    @override
    def modify_end_info(self, info: dict[str, Any]):
        info["dataset"] = {
            "name": self.dataset_name,
            "group_index": self.dataset_state.group_index,
            "offset": self.dataset_state.offset,
        }

    @override
    def on_step_start(self):
        group_index = self.data_iterator.group_index
        if group_index != self.last_group_index:
            self.last_group_index = group_index
            if self.writer:
                self.writer.add_scalar(
                    f"dataset/{self.dataset_name}/group_index", group_index, self.step
                )

        if self.writer:
            self.writer.add_scalar(
                f"dataset/{self.dataset_name}/offset",
                self.data_iterator.offset,
                self.step,
            )

    @override
    def on_step_end(self):
        self.state.step = self.step + 1
        self.dataset_state = PretrainDatasetState(
            group_index=self.data_iterator.group_index, offset=self.data_iterator.offset
        )
        self.state.dataset[self.dataset_name] = self.dataset_state

    @override
    def on_data_exhausted(self):
        self.dataset_state = PretrainDatasetState()
        self.state.dataset[self.dataset_name] = self.dataset_state


def estimate_memory(
    config: Config,
    *,
    seq_len: int,
    batch_size: int,
    use_bf16: bool = True,
) -> dict:
    """Estimate peak GPU memory and training FLOPs.

    Covers model weights (fp32), gradients (fp32), AdamW optimizer state
    (fp32 momentum + variance), per-batch activations, and
    forward+backward FLOPs per training step.

    Returns a breakdown in bytes and FLOP counts.
    """
    embedding_params, body_params = compute_param_counts(config)
    total_params = embedding_params + body_params

    # fp32 master weights
    model_mem = total_params * 4
    # fp32 gradients
    grad_mem = total_params * 4
    # AdamW: fp32 momentum + fp32 variance
    optim_mem = total_params * 8

    # ---- shorthand aliases ----
    B = batch_size
    S = seq_len
    H = config.num_head
    D = config.dim
    d_k = config.dim_key
    d_v = config.dim_value
    d_ff = config.dim_ff_hidden
    n_layer = config.num_layer
    V = config.vocab_size

    # ---- precision constants ----
    # Autocast: matmuls / most pointwise ops → bf16 (2 B).
    # softmax & cross-entropy logits → fp32 (4 B) for numerical stability
    # (Inductor upcasts bf16→fp32 internally even under autocast).
    act_bytes = 2 if use_bf16 else 4
    fp32 = 4

    residual = B * S * D

    # ---- memory ----

    embed_act = residual * act_bytes

    # RMSNorm (use_fp32=True) saves its input for backward.  Inductor's
    # min-cut partitioner bans recomputation when dist_from_bw > 4, which
    # covers nearly all layers, so the saved tensor precision is determined
    # by a cost-model heuristic: store the saved input in bf16 when the
    # memory savings outweigh the backward recomputation cost of the fp32
    # cast.  Both savings and recomputation cost scale linearly with D, so
    # the decision boundary depends only on the per-sample token count B×S.
    if B * S >= 200_000:
        norm_bytes = act_bytes
    else:
        norm_bytes = fp32

    # Per-block activations saved for backward.
    per_block = (
        residual * norm_bytes  # norm1 input
        + residual * act_bytes  # norm1 output (for W_q/k/v grads)
        + B * H * S * (2 * d_k + d_v) * act_bytes  # Q, K, V
        + B * H * S * S * fp32  # attention softmax
        + B * H * S * d_v * act_bytes  # pre-W_o
        + residual * norm_bytes  # norm2 input
        + residual * act_bytes  # norm2 output (for FF up grad)
        + B * S * d_ff * act_bytes  # FF hidden (ReLU saves input)
    )

    all_blocks = n_layer * per_block

    # final_norm input + output (bf16, saved by lm_head)
    final_norm_act = residual * norm_bytes + residual * act_bytes

    # Cross-entropy stores logits in fp32 for numerical stability.
    logit_act = B * S * V * fp32

    peak_act = embed_act + all_blocks + final_norm_act + logit_act

    total = int(model_mem + grad_mem + optim_mem + peak_act)

    # ---- FLOPs (forward pass, per batch) ----

    # Q, K, V projections
    flops_qkv = 2 * B * H * S * D * (2 * d_k + d_v)
    # attention scores: Q @ K^T  → (B, H, S, S)
    flops_attn_scores = 2 * B * H * S * S * d_k
    # attention output: scores @ V  → (B, H, S, d_v)
    flops_attn_values = 2 * B * H * S * S * d_v
    # output projection
    flops_out_proj = 2 * B * H * S * d_v * D
    # feed-forward: up (D→d_ff) + down (d_ff→D)
    flops_ff = 4 * B * S * D * d_ff

    flops_attn_block = (
        flops_qkv + flops_attn_scores + flops_attn_values + flops_out_proj
    )
    flops_per_block = flops_attn_block + flops_ff
    flops_fwd = int(n_layer * flops_per_block + 2 * B * S * D * V)

    # Forward + backward ≈ 3× forward.
    flops_fwd_bwd = 3 * flops_fwd

    return {
        # params
        "total_params": total_params,
        "embedding_params": embedding_params,
        "body_params": body_params,
        # memory
        "model_mem": model_mem,
        "grad_mem": grad_mem,
        "optim_mem": optim_mem,
        "activation_mem": int(peak_act),
        "total": total,
        # FLOPs (all values are per batch)
        "flops_fwd_per_batch": flops_fwd,
        "flops_fwd_bwd_per_batch": flops_fwd_bwd,
        "flops_fwd_attn_per_block": int(flops_attn_block),
        "flops_fwd_ff_per_block": int(flops_ff),
        "flops_fwd_attn_scores_per_block": int(flops_attn_scores),
        "flops_fwd_attn_values_per_block": int(flops_attn_values),
    }


def report_memory_estimate(
    result: dict,
    seq_len: int,
    batch_size: int,
    use_bf16: bool,
) -> None:
    """Print a human-readable GPU memory and FLOPs estimate."""
    dtype_label = "BF16" if use_bf16 else "FP32"
    print("==== Memory Estimate ====")
    print(
        f"Parameters:  {result['total_params']:,} "
        f"(embedding: {result['embedding_params']:,}, "
        f"body: {result['body_params']:,})"
    )
    print(f"Precision:   {dtype_label} activations, FP32 master weights")
    print(f"Batch size:  {batch_size}")
    print(f"Seq length:  {seq_len}")
    print()
    print(f"  Model weights:     {format_bytes(result['model_mem'])}")
    print(f"  Gradients:         {format_bytes(result['grad_mem'])}")
    print(f"  Optimizer (AdamW): {format_bytes(result['optim_mem'])}")
    print(f"  Activations:       {format_bytes(result['activation_mem'])}")
    print("  ─────────────────────────────")
    print(f"  Estimated total:   {format_bytes(result['total'])}")

    # ---- FLOPs ----

    tokens_per_step = seq_len * batch_size
    flops_fwd_step = result["flops_fwd_per_batch"]
    flops_fwd_bwd_step = result["flops_fwd_bwd_per_batch"]

    print()
    print("==== FLOPs Estimate ====")
    print(f"  Tokens per step:           {tokens_per_step:,}")
    print(f"  Forward (per step):        {format_flops(flops_fwd_step)}")
    print(f"  Forward + backward (step): {format_flops(flops_fwd_bwd_step)}")
    print(f"  FLOPs per token:           {flops_fwd_bwd_step / tokens_per_step:,.1f}")


def validate(
    model_config: Config,
    seq_len: int,
    batch_size: int,
    dataset_name: str,
    batches: int,
    model_state_path: str,
    *,
    device: str,
    pad_id: int,
    sep_id: int,
    use_bf16: bool = True,
):
    print(f"Loading model state ({model_state_path})... ", end="")
    model = Model(model_config).to(device)
    model.load_state_dict(
        torch.load(model_state_path, weights_only=True, map_location=device)
    )
    print("ok.")

    print(f"Loading validation data ({dataset_name}_val.parquet)... ", end="")
    pf_val = ParquetFile(f"{dataset_name}_val.parquet")
    iterator_val = ParquetConcatIterator(pf_val, seq_len, pad_id=pad_id, sep_id=sep_id)
    val_batch_data = next_batches(
        iterator_val, batches, batch_size, seq_len, pad_id=pad_id
    )
    del iterator_val
    if val_batch_data is None:
        print("no data, cannot do validation.")
        return
    else:
        print(f"number of batches: {val_batch_data.batches} (expected: {batches}).")
        val_batch_data = val_batch_data.to(device)

    t_val_start = time.perf_counter()

    val_loss, val_perplexity = compute_validation_metrics(
        model, val_batch_data, pad_id=pad_id, use_bf16=use_bf16
    )
    val_loss, val_perplexity = val_loss.item(), val_perplexity.item()

    t_val_end = time.perf_counter()
    val_time = t_val_end - t_val_start

    print(f"val_loss = {val_loss:.6f}")
    print(f"val_ppl  = {val_perplexity:.4f}")
    print(f"val_time = {val_time:.4f}")

    num_tokens_valid = val_batch_data.mask.sum().item()
    num_tokens_all = val_batch_data.mask.numel()
    print(f"valid_ratio = {num_tokens_valid / num_tokens_all:.4f}")


@torch.compile
def generate_step(model: Model, input: Tensor) -> Tensor:
    return model(input)


def generate(
    model_state_path: str,
    tokenizer_path: str,
    seq_len: int,
    count: int,
    prompt: str,
    *,
    device: str,
    model_config: Config,
    pad_id: int,
    temperature: float = 0.5,
    use_bf16: bool = True,
):
    model = Model(model_config).to(device)
    model.load_state_dict(
        torch.load(model_state_path, weights_only=True, map_location=device)
    )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    prompt_tokens = tokenizer.encode(prompt).ids
    tokens = list(prompt_tokens)

    print("Sequence length:", seq_len)
    print("Max generation count:", count)
    print("Temperature:", temperature)
    print("Prompt:", repr(tokenizer.decode(tokens, skip_special_tokens=False)))
    print("Prompt token count:", len(tokens))
    print("********")

    start_pos = len(tokens)
    decode_stream = DecodeStream()

    print(tokenizer.decode(tokens, skip_special_tokens=False), end="")

    model.eval()

    for _ in range(count):
        input_tokens = tokens[-seq_len:]
        L = len(input_tokens)
        if L < seq_len:
            input_tokens = input_tokens + [pad_id] * (seq_len - L)
        assert len(input_tokens) == seq_len

        with (
            torch.no_grad(),
            torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16),
        ):
            logits = generate_step(model, torch.tensor(input_tokens, device=device))

        logits = logits[L - 1, :]  # get the new token
        probs = torch.softmax(logits / temperature, dim=-1)
        next = int(torch.multinomial(probs, num_samples=1).item())

        tokens.append(next)

        chunk = decode_stream.step(tokenizer, next)
        if chunk is not None:
            print(chunk, end="", flush=True)

    print()
    print("********")
    print("Generated token count:", len(tokens) - start_pos)


# CLI


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Pretrain CLI Tool", fromfile_prefix_chars="@")

    parser.add_argument(
        "--threads", "-t", type=int, help="number of CPU threads that PyTorch will use"
    )
    parser.add_argument(
        "--no-bf16", "-n16", action="store_true", help="disable bfloat16 autocast"
    )

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # init
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("pretrain_state")
    init_parser.add_argument("model_state")
    init_parser.add_argument("optimizer_state")

    # generate
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("model_state")
    gen_parser.add_argument("tokenizer")
    gen_parser.add_argument("seq_len", type=int, help="sequence length")
    gen_parser.add_argument("count", type=int, help="max generation count")
    gen_parser.add_argument("prompt")
    gen_parser.add_argument(
        "-temp",
        "--temperature",
        type=float,
        default=0.5,
        help="temperature for sampling",
    )

    # train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("pretrain_state")
    train_parser.add_argument("model_state")
    train_parser.add_argument("optimizer_state")
    train_parser.add_argument("dataset_name")
    train_parser.add_argument("target_step", type=int)
    train_parser.add_argument(
        "--tensorboard-dir", "-tb", required=True, help="TensorBoard write directory"
    )
    train_parser.add_argument("--loop-dataset", "-ld", default=True)
    train_parser.add_argument(
        "--estimate",
        "-es",
        action="store_true",
        help="only estimate and report GPU memory required for training,",
    )

    # validate
    val_parser = subparsers.add_parser("validate")
    val_parser.add_argument("model_state")
    val_parser.add_argument("seq_len", type=int, help="sequence length")
    val_parser.add_argument("batch_size", type=int, help="batch size")
    val_parser.add_argument("dataset_name")
    val_parser.add_argument("batches", type=int, help="max number of batches")

    return parser


def cli(
    model_config: Config,
    seq_len: int,
    batch_size: int,
    lr_func: LRFunc,
    *,
    pad_id: int,
    sep_id: int,
):
    parser = create_parser()
    args = parser.parse_args()
    device = setup(num_threads=args.threads)

    use_bf16 = not args.no_bf16
    print(f"Use BF16: {use_bf16}")
    warn_bf16(device, use_bf16)

    match args.action:
        case "init":
            state = PretrainState()
            with open(args.pretrain_state, "w") as f:
                f.write(state.model_dump_json())

            model = Model(model_config)
            torch.save(model.state_dict(), args.model_state)

            optimizer = AdamW(model.parameters())
            torch.save(optimizer.state_dict(), args.optimizer_state)

            print("Initialization done.")

        case "generate":
            generate(
                args.model_state,
                args.tokenizer,
                args.seq_len,
                args.count,
                args.prompt,
                device=device,
                model_config=model_config,
                pad_id=pad_id,
                temperature=args.temperature,
                use_bf16=use_bf16,
            )

        case "train":
            print(f"Sequence length: {seq_len}")
            print(f"Batch size: {batch_size}")

            if args.estimate:
                result = estimate_memory(
                    model_config,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    use_bf16=use_bf16,
                )
                report_memory_estimate(
                    result, seq_len=seq_len, batch_size=batch_size, use_bf16=use_bf16
                )
            else:
                trainer = Pretrainer(
                    model_config,
                    seq_len,
                    batch_size,
                    args.dataset_name,
                    lr_func,
                    args.pretrain_state,
                    args.model_state,
                    args.optimizer_state,
                    device=device,
                    pad_id=pad_id,
                    sep_id=sep_id,
                    loop_dataset=args.loop_dataset,
                    use_bf16=use_bf16,
                    tensorboard_dir=args.tensorboard_dir,
                )
                trainer.run(args.target_step)

        case "validate":
            print(f"Sequence length: {args.seq_len}")
            print(f"Batch size: {args.batch_size}")

            validate(
                model_config,
                args.seq_len,
                args.batch_size,
                args.dataset_name,
                args.batches,
                args.model_state,
                device=device,
                pad_id=pad_id,
                sep_id=sep_id,
                use_bf16=use_bf16,
            )
