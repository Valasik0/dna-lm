"""
Výběr tokenizéru podle konfigurace.

    from dnalm.tokenizers import build_tokenizer, load_tokenizer, load_config

    cfg = load_config("configs/tokenizer/bpe_4096.yaml")
    tok = build_tokenizer(cfg)          # nový (BPE/Unigram je potřeba ještě natrénovat)
    tok = load_tokenizer("cesta/k/ulozenemu/tokenizeru")   # uložený
"""

from __future__ import annotations

import json
import os

from .base import CONFIG_FILE, DNATokenizer
from .kmer import KmerTokenizer
from .nucleotide import NucleotideTokenizer
from .subword import BPETokenizer, SubwordTokenizer, UnigramTokenizer

TOKENIZER_TYPES: dict[str, type[DNATokenizer]] = {
    "nucleotide": NucleotideTokenizer,
    "kmer": KmerTokenizer,
    "bpe": BPETokenizer,
    "unigram": UnigramTokenizer,
}

# klíče konfigurace, které nejsou parametry konstruktoru (týkají se tréninku)
_NON_INIT_KEYS = {"type", "train", "description"}


def load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "type" not in cfg:
        raise ValueError(f"{path}: chybí klíč 'type'")
    return cfg


def build_tokenizer(cfg: dict | str) -> DNATokenizer:
    """Vytvoří tokenizér z konfigurace (slovník nebo cesta k YAML)."""
    if isinstance(cfg, str):
        cfg = load_config(cfg)
    try:
        cls = TOKENIZER_TYPES[cfg["type"]]
    except KeyError:
        known = list(TOKENIZER_TYPES)
        raise ValueError(f"Neznámý typ tokenizéru {cfg['type']!r}, známé: {known}") from None
    params = {k: v for k, v in cfg.items() if k not in _NON_INIT_KEYS}
    return cls(**params)


def needs_training(tok: DNATokenizer) -> bool:
    return isinstance(tok, SubwordTokenizer) and not tok.is_trained


def load_tokenizer(directory: str) -> DNATokenizer:
    """Načte tokenizér uložený přes tok.save(directory)."""
    with open(os.path.join(directory, CONFIG_FILE)) as f:
        cfg = json.load(f)
    cls = TOKENIZER_TYPES[cfg.pop("type")]
    if issubclass(cls, SubwordTokenizer):
        return cls.from_directory(directory, **cfg)
    return cls(**cfg)
