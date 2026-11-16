"""
Sestavení modelu z YAML konfigurace a tokenizéru.

    from dnalm.models.factory import build_model
    model = build_model("configs/model/S.yaml", tokenizer, window_bp=1024)
"""

from __future__ import annotations

from dnalm.tokenizers.base import PAD_ID, DNATokenizer

from .encoder import DNAForMaskedLM, EncoderConfig

_MODEL_KEYS = {
    "hidden_size",
    "num_layers",
    "num_heads",
    "intermediate_size",
    "rope_theta",
    "dropout",
    "attention_dropout",
    "norm_eps",
    "global_every",
    "local_window",
    "init_std",
}


def load_model_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def max_tokens(tokenizer: DNATokenizer, window_bp: int) -> int:
    """Horní mez počtu tokenů pro okno daného tokenizéru (+2 za [CLS] a [SEP])."""
    # nejhorší případ: každá báze samostatný token (zbytky u k-merů, N, krátké BPE tokeny)
    return window_bp + 2


def build_model(cfg: dict | str, tokenizer: DNATokenizer, window_bp: int = 1024) -> DNAForMaskedLM:
    if isinstance(cfg, str):
        cfg = load_model_config(cfg)
    params = {k: v for k, v in cfg.items() if k in _MODEL_KEYS}
    unknown = set(cfg) - _MODEL_KEYS - {"name", "description"}
    if unknown:
        raise ValueError(f"Neznámé klíče v konfiguraci modelu: {sorted(unknown)}")
    enc_cfg = EncoderConfig(
        vocab_size=tokenizer.vocab_size,
        max_position=max_tokens(tokenizer, window_bp),
        pad_token_id=PAD_ID,
        extra={"model": cfg.get("name"), "tokenizer": tokenizer.name, "window_bp": window_bp},
        **params,
    )
    return DNAForMaskedLM(enc_cfg)
