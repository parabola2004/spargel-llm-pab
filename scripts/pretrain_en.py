from spargel_llm_pab.config import PAD, SEP, default_model_config
from spargel_llm_pab.pretrain import cli
from spargel_llm_pab.utils import linear_warmup_stable_cosine_decay

if __name__ == "__main__":
    cli(
        default_model_config(),
        seq_len=512,
        batch_size=128,
        lr_func=linear_warmup_stable_cosine_decay(
            max_lr=1e-3, max_steps=8000, warmup_steps=100, decay_steps=2000
        ),
        pad_id=PAD,
        sep_id=SEP,
    )
