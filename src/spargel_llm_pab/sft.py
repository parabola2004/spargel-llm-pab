import time
from argparse import ArgumentParser
from collections.abc import Callable
from typing import Any, override

import torch
from torch.optim import AdamW
from pyarrow.parquet import ParquetFile
from tokenizers import Tokenizer

from spargel_llm.model import Config, Model
from spargel_llm.train.basic import compute_validation_metrics

from .pretrain import PretrainerBase, estimate_memory, report_memory_estimate
from .utils import (
    LRFunc,
    SFTState,
    SFTDatasetState,
    SFTDataIterator,
    next_batches,
    setup,
)


class SFTTrainer(PretrainerBase):
    def __init__(
        self,
        model_config: Config,
        seq_len: int,
        batch_size: int,
        dataset_name: str,
        lr_func: Callable[[int], float],
        sft_state_path: str,
        model_state_path: str,
        optimizer_state_path: str,
        tokenizer_path: str,
        *,
        device: str,
        pad_id: int,
        im_start_id: int,
        im_end_id: int,
        use_bf16: bool = True,
        tensorboard_dir: str | None = None,
        log_period: int = 100,
        save_period: int = 1000,
        checkpoint_period: int = 4000,
    ):
        self.dataset_name = dataset_name
        self.sft_state_path = sft_state_path

        print(f"Loading SFT state ({sft_state_path})... ", end="")
        with open(sft_state_path, "r") as f:
            self.state = SFTState.model_validate_json(f.read())
        self.dataset_state = self.state.dataset.get(dataset_name, SFTDatasetState())
        print("loaded.")

        print(f"Loading tokenizer ({tokenizer_path})... ", end="")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        print("loaded.")

        # Prepare training data
        pf_train = ParquetFile(f"{dataset_name}.parquet")
        data_iterator_train = SFTDataIterator(
            pf_train,
            self.tokenizer,
            seq_len,
            self.dataset_state.index,
            pad_id=pad_id,
            im_start_id=im_start_id,
            im_end_id=im_end_id,
        )

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
        pf_val = ParquetFile(f"{dataset_name}_val.parquet")
        data_iterator_val = SFTDataIterator(
            pf_val,
            self.tokenizer,
            seq_len,
            self.dataset_state.index,
            pad_id=pad_id,
            im_start_id=im_start_id,
            im_end_id=im_end_id,
        )
        self.val_batch_data = next_batches(
            data_iterator_val, 10, batch_size, seq_len, pad_id=pad_id
        )
        del data_iterator_val
        if self.val_batch_data is not None:
            self.val_batch_data = self.val_batch_data.to(device)

        self.step = self.state.step

    @override
    def run(self, target_step: int):
        super().run(target_step)

    @override
    def save(self):
        super().save()
        with open(self.sft_state_path, "w") as f:
            f.write(self.state.model_dump_json())

    @override
    def get_checkpoint_file_paths(self):
        return super().get_checkpoint_file_paths() + [self.sft_state_path]

    @override
    def modify_start_info(self, info):
        info["dataset"] = {"name": self.dataset_name, "index": self.dataset_state.index}

    @override
    def modify_end_info(self, info: dict[str, Any]):
        info["dataset"] = {"name": self.dataset_name, "index": self.dataset_state.index}

    @override
    def on_step_start(self):
        if self.writer:
            self.writer.add_scalar(
                f"dataset/{self.dataset_name}/index",
                self.data_iterator.index,
                self.step,
            )

    @override
    def on_step_end(self):
        self.state.step = self.step + 1
        self.dataset_state = SFTDatasetState(index=self.data_iterator.index)
        self.state.dataset[self.dataset_name] = self.dataset_state

    @override
    def on_data_exhausted(self):
        self.dataset_state = SFTDatasetState()
        self.state.dataset[self.dataset_name] = self.dataset_state


def validate(
    model_config: Config,
    seq_len: int,
    batch_size: int,
    dataset_name: str,
    batches: int,
    model_state_path: str,
    tokenizer_path: str,
    *,
    device: str,
    pad_id: int,
    im_start_id: int,
    im_end_id: int,
    use_bf16: bool = True,
):
    print(f"Loading model state ({model_state_path})... ", end="")
    model = Model(model_config).to(device)
    model.load_state_dict(
        torch.load(model_state_path, weights_only=True, map_location=device)
    )
    print("loaded.")

    print(f"Loading tokenizer ({tokenizer_path})... ", end="")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("loaded.")

    pf_val = ParquetFile(f"{dataset_name}_val.parquet")
    iterator_val = SFTDataIterator(
        pf_val,
        tokenizer,
        seq_len,
        pad_id=pad_id,
        im_start_id=im_start_id,
        im_end_id=im_end_id,
    )
    val_batch_data = next_batches(
        iterator_val, batches, batch_size, seq_len, pad_id=pad_id
    )
    del iterator_val
    if val_batch_data is None:
        print("No data.")
        return
    else:
        print(f"Number of batches: {val_batch_data.batches} (expected: {batches}).")
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

    mask_true = val_batch_data.mask.sum().item()
    mask_total = val_batch_data.mask.numel()
    print(f"mask_ratio = {mask_true / mask_total:.4f}")


# CLI


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="SFT CLI Tool", fromfile_prefix_chars="@")

    parser.add_argument(
        "--threads", "-t", type=int, help="number of CPU threads that PyTorch will use"
    )
    parser.add_argument(
        "--no-bf16", "-n16", action="store_true", help="disable bfloat16 autocast"
    )

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # init
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("sft_state_path")
    init_parser.add_argument("optimizer_state_path")

    # train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("sft_state_path")
    train_parser.add_argument("model_state_path")
    train_parser.add_argument("optimizer_state_path")
    train_parser.add_argument("tokenizer_path")
    train_parser.add_argument("dataset_name")
    train_parser.add_argument("target_step", type=int)
    train_parser.add_argument(
        "--tensorboard-dir", "-tb", help="TensorBoard write directory"
    )
    train_parser.add_argument(
        "--estimate",
        "-es",
        action="store_true",
        help="only estimate and report GPU memory required for training,",
    )

    # validate
    val_parser = subparsers.add_parser("validate")
    val_parser.add_argument("model_state_path")
    val_parser.add_argument("tokenizer_path")
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
    im_start_id: int,
    im_end_id: int,
):
    parser = create_parser()
    args = parser.parse_args()
    device = setup(num_threads=args.threads)

    use_bf16 = not args.no_bf16
    print(f"Use BF16: {use_bf16}")
    if use_bf16 and device != "cuda":
        print("WARNING: device is not cuda, but BF16 autocast is enabled.")

    match args.action:
        case "init":
            state = SFTState()
            with open(args.sft_state_path, "w") as f:
                f.write(state.model_dump_json())

            model = Model(model_config)
            optimizer = AdamW(model.parameters())
            torch.save(optimizer.state_dict(), args.optimizer_state_path)

            print("Initialization done.")

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
                trainer = SFTTrainer(
                    model_config,
                    seq_len,
                    batch_size,
                    args.dataset_name,
                    lr_func,
                    args.sft_state_path,
                    args.model_state_path,
                    args.optimizer_state_path,
                    args.tokenizer_path,
                    device=device,
                    pad_id=pad_id,
                    im_start_id=im_start_id,
                    im_end_id=im_end_id,
                    use_bf16=use_bf16,
                    tensorboard_dir=args.tensorboard_dir,
                )
                trainer.run(args.target_step)

        case "validate":
            print(f"Sequence length: {seq_len}")
            print(f"Batch size: {batch_size}")

            validate(
                model_config,
                args.seq_len,
                args.batch_size,
                args.dataset_name,
                args.batches,
                args.model_state_path,
                args.tokenizer_path,
                device=device,
                pad_id=pad_id,
                im_start_id=im_start_id,
                im_end_id=im_end_id,
                use_bf16=use_bf16,
            )
