from spargel_llm.model import Config

# Special tokens

UNK, PAD, SOT, EOT, SEP = range(5)
IM_START, IM_END = range(8, 10)


def default_model_config():
    return Config(
        vocab_size=8192,
        max_seq_len=4096,
        num_layer=8,
        num_head=8,
        dim=512,
        dim_key=64,
        dim_value=64,
        dim_ff_hidden=2048,
    )
