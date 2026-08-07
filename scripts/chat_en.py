from spargel_llm_pab.chat import cli
from spargel_llm_pab.config import IM_END, IM_START, PAD, default_model_config

if __name__ == "__main__":
    cli(
        default_model_config(),
        pad_id=PAD,
        im_start_id=IM_START,
        im_end_id=IM_END,
        default_context_length=1024,
    )
