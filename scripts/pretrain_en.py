from spargel_llm_pab.config import PAD, SEP, default_model_config
from spargel_llm_pab.pretrain import cli
from spargel_llm_pab.utils import cosine, linear


def lr_func(step: int) -> float:
    max_lr = 1e-3
    warmup_steps = 100
    decay_start_step = 16000
    decay_steps = 4000
    total_steps = decay_start_step + decay_steps

    if step < 0:
        return 0
    elif step < warmup_steps:
        return linear(0, warmup_steps, 0, max_lr, step)
    elif step < decay_start_step:
        return max_lr
    elif step < total_steps:
        return cosine(decay_start_step, total_steps, max_lr, 0, step)
    else:
        return 0


if __name__ == "__main__":
    cli(
        default_model_config(),
        seq_len=512,
        batch_size=128,
        lr_func=lr_func,
        pad_id=PAD,
        sep_id=SEP,
    )
