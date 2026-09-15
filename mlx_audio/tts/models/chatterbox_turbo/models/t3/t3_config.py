# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

from dataclasses import dataclass
from typing import Optional

# GPT2 Small configuration for Nano model
GPT2_SMALL_CONFIG = {
    "activation_function": "gelu_new",
    "n_ctx": 8196,
    "n_embd": 768,
    "hidden_size": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 8196,
    "vocab_size": 50276,
    "layer_norm_epsilon": 1e-05,
    "attn_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "resid_pdrop": 0.1,
}

# GPT2 Medium configuration for Turbo model
GPT2_MEDIUM_CONFIG = {
    "activation_function": "gelu_new",
    "n_ctx": 8196,
    "n_embd": 1024,
    "hidden_size": 1024,
    "n_head": 16,
    "n_layer": 24,
    "n_positions": 8196,
    "vocab_size": 50276,
    "layer_norm_epsilon": 1e-05,
    "attn_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "resid_pdrop": 0.1,
}

GPT2_CONFIGS = {
    "GPT2_small": GPT2_SMALL_CONFIG,
    "GPT2_medium": GPT2_MEDIUM_CONFIG,
}


@dataclass
class T3Config:
    """Configuration for T3 TTS model."""

    # Text tokens
    start_text_token: int = 255
    stop_text_token: int = 0
    text_tokens_dict_size: int = 50276
    max_text_tokens: int = 2048

    # Speech tokens
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    speech_tokens_dict_size: int = 6563
    max_speech_tokens: int = 4096

    # Model architecture
    llama_config_name: str = "GPT2_medium"
    input_pos_emb: Optional[str] = None
    speech_cond_prompt_len: int = 375

    # Conditioning
    encoder_type: str = "voice_encoder"
    speaker_embed_size: int = 256
    use_perceiver_resampler: bool = False
    emotion_adv: bool = False

    @property
    def n_channels(self) -> int:
        """Get hidden size from the backbone this config names."""
        return self.gpt2_config["hidden_size"]

    @property
    def gpt2_config(self) -> dict:
        """The GPT2 hyperparameters for this variant."""
        return GPT2_CONFIGS[self.llama_config_name]

    @property
    def is_multilingual(self) -> bool:
        """Check if this is a multilingual model."""
        return self.text_tokens_dict_size == 2454

    @classmethod
    def nano(cls) -> "T3Config":
        """Create configuration for Nano TTS model.

        Nano is Turbo's architecture on a GPT2-small backbone: same tokenizers,
        same speech vocabulary, 768 channels over 12 blocks instead of 1024
        over 24.
        """
        return cls(
            text_tokens_dict_size=50276,
            speech_tokens_dict_size=6563,
            llama_config_name="GPT2_small",
            input_pos_emb=None,
            speech_cond_prompt_len=375,
            use_perceiver_resampler=False,
            emotion_adv=False,
        )

    @classmethod
    def turbo(cls) -> "T3Config":
        """Create configuration for Turbo TTS model."""
        return cls(
            text_tokens_dict_size=50276,
            speech_tokens_dict_size=6563,
            llama_config_name="GPT2_medium",
            input_pos_emb=None,
            speech_cond_prompt_len=375,
            use_perceiver_resampler=False,
            emotion_adv=False,
        )
