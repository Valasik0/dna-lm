"""End-to-end: genom -> okna -> trénink tokenizérů z YAML -> načtení a použití."""

import json
import os
import random

import yaml

from dnalm.data.genome import GenomeWindows, WindowConfig, make_windows
from dnalm.tokenizers import load_tokenizer
from dnalm.tokenizers.train import main as train_main


def test_train_from_configs(tmp_path):
    from pyfaidx import Fasta

    rng = random.Random(0)
    motif = "GGGCGGGGCC"
    fasta = tmp_path / "toy.fa"
    with open(fasta, "w") as f:
        for name in ["chr1", "chr2", "chr20", "chr21"]:
            seq = motif.join("".join(rng.choice("ACGT") for _ in range(30)) for _ in range(120))
            f.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i : i + 60] + "\n")
    Fasta(str(fasta))
    splits = {"train": ["chr1", "chr2"], "val": ["chr20"], "test": ["chr21"]}
    make_windows(str(fasta), str(tmp_path / "win"), WindowConfig(window=512, splits=splits), False)

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "bpe.yaml").write_text(
        yaml.safe_dump(
            {"name": "bpe_toy", "type": "bpe", "vocab_size": 80, "train": {"n_windows": 10}}
        )
    )
    (cfg_dir / "kmer.yaml").write_text(yaml.safe_dump({"name": "k6", "type": "kmer", "k": 6}))

    out = tmp_path / "tok"
    train_main(
        [
            "--config",
            str(cfg_dir / "bpe.yaml"),
            str(cfg_dir / "kmer.yaml"),
            "--fasta",
            str(fasta),
            "--windows",
            str(tmp_path / "win" / "train.tsv.gz"),
            "--out",
            str(out),
        ]
    )
    info = json.loads((out / "bpe_toy" / "train_info.json").read_text())
    assert info["train_windows"] == 10 and info["vocab_size"] <= 80
    assert os.path.exists(out / "k6" / "tokenizer_config.json")

    bpe = load_tokenizer(str(out / "bpe_toy"))
    val = GenomeWindows(str(fasta), str(tmp_path / "win" / "val.tsv.gz"))
    seq = val[0]
    assert bpe.decode(bpe.encode(seq)) == seq
    assert len(bpe.tokenize(seq)) < len(seq)  # BPE sekvenci zkrátí
