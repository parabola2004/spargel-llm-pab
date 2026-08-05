from spargel_llm_pab.config import PAD, IM_START, IM_END, default_model_config
from spargel_llm_pab.sft import cli

if __name__ == "__main__":
    cli(
        default_model_config(),
        seq_len=1024,
        batch_size=16,
        lr_func=lambda _step: 1e-4,
        pad_id=PAD,
        im_start_id=IM_START,
        im_end_id=IM_END,
    )
