import json
import random
from argparse import ArgumentParser
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from spargel_llm.train.sft import Message
from spargel_llm_pab.sft import ShareGPTConversation, from_sharegpt

type Type = Literal["openhermes", "ultrachat"]


def transform_parquet(
    input_paths: list[Path], *, column_name="messages"
) -> Iterator[list[Message]]:
    for input_path in input_paths:
        print(f"Reading (Parquet): {input_path}")
        table = pq.read_table(str(input_path))
        column = table.column(column_name)
        for i in tqdm(range(len(column))):
            raw_messages = column[i].as_py()
            messages = [
                Message(role=m["role"], content=m["content"]) for m in raw_messages
            ]
            yield messages


def transform_openhermes(input_paths: list[Path]) -> Iterator[list[Message]]:
    """For OpenHermes-2.5"""
    for input_path in input_paths:
        print(f"Reading (OpenHermes): {input_path}")
        with open(input_path, "r") as f:
            raw_data = json.load(f)

        skipped = 0
        for item in tqdm(raw_data):
            try:
                conversation = ShareGPTConversation.model_validate(item)
                messages = from_sharegpt(conversation)
                yield messages
            except Exception as e:
                skipped += 1
                print(f"  Skipping item: {e}")

        if skipped:
            print(f"  Skipped {skipped} item(s) from {input_path}.")


def transform_ultrachat(input_paths: list[Path]) -> Iterator[list[Message]]:
    """For UltraChat"""
    for input_path in input_paths:
        print(f"Reading (UltraChat): {input_path}")
        with open(input_path, "r") as f:

            def yield_lines():
                while True:
                    line = f.readline()
                    if not line:
                        break
                    yield line

            for line in tqdm(yield_lines()):
                raw_data = json.loads(line)
                messages: list[Message] = []
                for i, text in enumerate(raw_data["data"]):
                    role = "user" if i % 2 == 0 else "assistant"
                    messages.append(Message(role=role, content=text))
                yield messages


def action_show(path: str, row: int | None = None):
    """Read and print a single row from a SFT Parquet file."""
    pf = pq.ParquetFile(path)

    if row is None:
        row = random.randrange(pf.metadata.num_rows)
    elif row < 0:
        row += int(pf.metadata.num_rows)

    if row < 0 or row >= pf.metadata.num_rows:
        raise IndexError(
            f"row index out of range (file has {pf.metadata.num_rows} rows)"
        )

    row_index = row
    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(rg_idx)
        if row < rg.num_rows:
            table = pf.read_row_group(rg_idx, columns=["messages"])
            raw_messages = table.column("messages")[row].as_py()
            break
        row -= rg.num_rows
    else:
        raise IndexError(f"row {row_index} not found")

    print(f"[{row_index}/{pf.metadata.num_rows}] messages={len(raw_messages)}")
    for i, m in enumerate(raw_messages):
        print(f"[{i}: {m['role']}]")
        print(m["content"])
        print()


def action_transform(
    input_paths: list[Path],
    output_path: Path,
    type: Type | None = None,
    *,
    batch_size: int | None = None,
):
    """Collect each file's conversations and write them out once per file.

    A single ParquetWriter is reused across files so that only one file's
    conversations are held in memory at a time; ``batch_size`` bounds the
    number of rows written per batch (ParquetWriter ``write_batch_size``).
    """
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    total_rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        for input_path in input_paths:
            match type:
                case "openhermes":
                    transform = transform_openhermes([input_path])
                case "ultrachat":
                    transform = transform_ultrachat([input_path])
                case None:
                    transform = transform_parquet([input_path])
                case _:
                    raise ValueError(f"unknown input type: {type}")

            rows = [
                [{"role": m.role, "content": m.content} for m in messages]
                for messages in transform
            ]
            if not rows:
                continue

            table = pa.table({"messages": rows})
            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_path),
                    table.schema,
                    compression="zstd",
                    write_batch_size=batch_size,
                )
            writer.write_table(table)
            total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # no valid conversations across all inputs: still create the output file
        pq.write_table(pa.table({"messages": []}), str(output_path), compression="zstd")
    print(f"Wrote {total_rows} rows -> {output_path}")


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="SFT Data CLI Tool", fromfile_prefix_chars="@")

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # show
    show_parser = subparsers.add_parser(
        "show", help="read and print a row from a SFT Parquet file"
    )
    show_parser.add_argument("path", help="Parquet file")
    show_parser.add_argument(
        "row",
        nargs="?",
        type=int,
        default=None,
        help="zero-based row index (negative counts from end; random if omitted)",
    )

    # transform
    transform_parser = subparsers.add_parser(
        "transform", help="transform input files into a Parquet dataset"
    )
    transform_parser.add_argument("output", help="output Parquet file")
    transform_parser.add_argument("inputs", nargs="+", help="input files")
    transform_parser.add_argument(
        "--type", "-t", choices=["openhermes", "ultrachat"], help="input file type"
    )
    transform_parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=None,
        help="maximum number of rows to write at a time "
        "(ParquetWriter write_batch_size)",
    )

    return parser


def cli():
    parser = create_parser()
    args = parser.parse_args()

    match args.action:
        case "show":
            action_show(args.path, args.row)
        case "transform":
            action_transform(
                [Path(p) for p in args.inputs],
                Path(args.output),
                args.type,
                batch_size=args.batch_size,
            )


if __name__ == "__main__":
    cli()
