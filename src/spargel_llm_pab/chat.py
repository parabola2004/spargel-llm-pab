import readline  # noqa: F401
from argparse import ArgumentParser

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from spargel_llm.model import Config, Model

from .pretrain import generate_step
from .utils import setup, warn_bf16


def chat(
    model_config: Config,
    context_length: int,
    model_state_path: str,
    tokenizer_path: str,
    *,
    device: str,
    pad_id: int,
    im_start_id: int,
    im_end_id: int,
    temperature: float = 0.5,
    use_bf16: bool = True,
    system_prompt: str | None = None,
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

    print("********")
    print("Context length:", context_length)
    print("Temperature:", temperature)
    print("********")

    newline_ids = tokenizer.encode("\n").ids
    role_ids_cache: dict[str, list[int]] = {
        "assistant": tokenizer.encode("assistant").ids
    }

    def get_role_ids(role: str):
        if role in role_ids_cache:
            return role_ids_cache[role]
        else:
            ids = tokenizer.encode(role).ids
            role_ids_cache[role] = ids
            return ids

    tokens: list[int] = []

    def add_message(tokens: list[int], role: str, content: str):
        tokens.append(im_start_id)
        tokens.extend(get_role_ids(role))
        tokens.extend(newline_ids)
        tokens.extend(tokenizer.encode(content).ids)
        tokens.append(im_end_id)
        tokens.extend(newline_ids)

    if system_prompt is not None:
        add_message(tokens, "system", system_prompt)
        print("[system]")
        print(system_prompt)
        print()

    seq_len = context_length

    while True:
        try:
            user_message = input("user > ")
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print()
            break

        print()

        if user_message in ["/q", "/quit", "/exit"]:
            break
        elif user_message == "/load":
            print(f"Loading model state ({model_state_path})... ", end="")
            model.load_state_dict(
                torch.load(model_state_path, weights_only=True, map_location=device)
            )
            print("loaded.")
            print()
            continue
        elif user_message == "/clear":
            tokens = []
            print("Context cleared.")
            print()
            if system_prompt is not None:
                add_message(tokens, "system", system_prompt)
                print("[system]")
                print(system_prompt)
                print()
            continue
        elif user_message == "/dump":
            print("**** DUMP ****")
            print(tokenizer.decode(tokens, skip_special_tokens=False))
            print("********")
            print()
            continue

        add_message(tokens, "user", user_message)

        # Generate
        decode_stream = DecodeStream()

        tokens.append(im_start_id)
        tokens.extend(get_role_ids("assistant"))
        tokens.extend(newline_ids)

        print("[assistant]")

        try:
            while True:
                if len(tokens) > context_length:
                    print("[CONTEXT FULL]")
                    break

                L = len(tokens)
                input_tokens = tokens + [pad_id] * (seq_len - L)
                assert len(input_tokens) == seq_len

                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=device, dtype=torch.bfloat16, enabled=use_bf16
                    ),
                ):
                    logits = generate_step(
                        model, torch.tensor(input_tokens, device=device)
                    )

                logits = logits[L - 1, :]  # get the new token
                probs = torch.softmax(logits / temperature, dim=-1)
                next = int(torch.multinomial(probs, num_samples=1).item())

                tokens.append(next)

                if next == im_end_id:
                    print("\n")
                    tokens.extend(newline_ids)
                    break
                else:
                    chunk = decode_stream.step(tokenizer, next)
                    if chunk is not None:
                        print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            print("\n[interrupted]\n")
            tokens.append(im_end_id)
            tokens.extend(newline_ids)


# CLI


def cli(
    model_config: Config,
    *,
    pad_id: int,
    im_start_id: int,
    im_end_id: int,
    default_context_length: int,
):
    parser = ArgumentParser(description="Chat CLI Tool", fromfile_prefix_chars="@")
    parser.add_argument("model_state_path")
    parser.add_argument("tokenizer_path")
    parser.add_argument(
        "context_length", type=int, nargs="?", default=default_context_length
    )
    parser.add_argument("--system-prompt", "-sys")
    parser.add_argument(
        "--temperature",
        "-temp",
        type=float,
        default=0.5,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--threads", "-t", type=int, help="number of CPU threads that PyTorch will use"
    )
    parser.add_argument(
        "--no-bf16", "-n16", action="store_true", help="disable bfloat16 autocast"
    )

    args = parser.parse_args()
    use_bf16 = not args.no_bf16

    device = setup(num_threads=args.threads)
    print(f"Use BF16: {use_bf16}")
    warn_bf16(device, use_bf16)

    chat(
        model_config,
        args.context_length,
        args.model_state_path,
        args.tokenizer_path,
        device=device,
        pad_id=pad_id,
        im_start_id=im_start_id,
        im_end_id=im_end_id,
        temperature=args.temperature,
        use_bf16=use_bf16,
        system_prompt=args.system_prompt,
    )
