"""
Příprava tokenizérů z YAML konfigurací. BPE a Unigram se natrénují na náhodném
vzorku trénovacích oken (nikdy na validaci a testu), ostatní se jen uloží.

    python -m dnalm.tokenizers.train \\
        --config configs/tokenizer/bpe_4096.yaml configs/tokenizer/bpe_600.yaml \\
        --fasta /content/genome/hg38.main.fa \\
        --windows /content/drive/MyDrive/dna-lm/data/processed/windows_1024/train.tsv.gz \\
        --out /content/drive/MyDrive/dna-lm/data/processed/tokenizers
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .registry import build_tokenizer, load_config, needs_training

DEFAULT_TRAIN = {"n_windows": 100_000, "seed": 42}  # ~7 GB RAM při tréninku BPE


def sample_sequences(fasta: str, windows: str, n: int, seed: int):
    """Iterátor sekvencí z n náhodně vybraných oken (pevný seed = reprodukovatelné)."""
    from dnalm.data.genome import GenomeWindows

    ds = GenomeWindows(fasta, windows)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    idx.sort()  # čtení po pořadí v souboru je rychlejší
    return (ds[int(i)] for i in idx), len(idx)


def prepare_tokenizer(cfg_path: str, fasta: str | None, windows: str | None, out_dir: str):
    cfg = load_config(cfg_path)
    tok = build_tokenizer(cfg)
    info = {"config": cfg_path, "name": tok.name, "type": cfg["type"]}
    if needs_training(tok):
        if not (fasta and windows):
            raise ValueError(f"{tok.name} je potřeba natrénovat: zadej --fasta a --windows")
        train_cfg = {**DEFAULT_TRAIN, **cfg.get("train", {})}
        seqs, n = sample_sequences(fasta, windows, train_cfg["n_windows"], train_cfg["seed"])
        print(f"[{tok.name}] trénink na {n:,} oknech...")
        t0 = time.time()
        tok.train(seqs, length=n)
        info.update(train_windows=n, seed=train_cfg["seed"], train_seconds=round(time.time() - t0))
        info["source_windows"] = os.path.abspath(windows)
    path = os.path.join(out_dir, tok.name)
    tok.save(path)
    info["vocab_size"] = tok.vocab_size
    with open(os.path.join(path, "train_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[{tok.name}] uloženo do {path} (slovník {tok.vocab_size} tokenů)")
    return tok


def main(argv=None):
    p = argparse.ArgumentParser(description="Příprava a trénink tokenizérů")
    p.add_argument("--config", nargs="+", required=True, help="jeden nebo více YAML souborů")
    p.add_argument("--fasta", help="hg38.main.fa (potřeba pro BPE/Unigram)")
    p.add_argument("--windows", help="train.tsv.gz (potřeba pro BPE/Unigram)")
    p.add_argument("--out", required=True, help="složka, kam se tokenizéry uloží")
    a = p.parse_args(argv)
    for c in a.config:
        prepare_tokenizer(c, a.fasta, a.windows, a.out)


if __name__ == "__main__":
    main()
