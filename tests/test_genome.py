"""Testy generování oken z genomu na malém syntetickém FASTA souboru."""

import gzip
import json
import random

import numpy as np
import pytest

from dnalm.data.genome import (
    GenomeWindows,
    WindowConfig,
    load_windows,
    make_windows,
    reverse_complement,
    window_stats,
)


def write_fasta(path, records, line_width=60):
    with open(path, "w") as f:
        for name, seq in records.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), line_width):
                f.write(seq[i : i + line_width] + "\n")
    from pyfaidx import Fasta

    Fasta(str(path))  # vytvoří .fai


@pytest.fixture
def genome(tmp_path):
    rng = random.Random(0)
    seqs = {}
    for name, length in [("chr1", 5000), ("chr2", 3000), ("chr20", 2000), ("chr21", 1500)]:
        seqs[name] = "".join(rng.choice("ACGTacgt") for _ in range(length))
    # blok N na chr1 (pozice 1000-1600) a jedno N na chr2
    s = seqs["chr1"]
    seqs["chr1"] = s[:1000] + "N" * 600 + s[1600:]
    s = seqs["chr2"]
    seqs["chr2"] = s[:10] + "N" + s[11:]
    path = tmp_path / "toy.fa"
    write_fasta(path, seqs)
    return str(path), seqs


SPLITS = {"train": ["chr1", "chr2"], "val": ["chr20"], "test": ["chr21"]}


def test_reverse_complement():
    assert reverse_complement("ACGTN") == "NACGT"
    assert reverse_complement("aacG") == "Cgtt"
    s = "ACGTTGCAAGGN"
    assert reverse_complement(reverse_complement(s)) == s


def test_window_stats_matches_naive():
    rng = random.Random(1)
    seq = "".join(rng.choice("ACGTNacgtn") for _ in range(777))
    w, stride = 100, 37
    st = window_stats(seq, w, stride)
    for k, s in enumerate(st["start"]):
        win = seq[s : s + w]
        assert st["n"][k] == sum(c in "Nn" for c in win)
        assert st["gc"][k] == sum(c in "GCgc" for c in win)
        assert st["lower"][k] == sum(c.islower() for c in win)
    assert st["start"][-1] + w <= len(seq)


def test_make_windows(genome, tmp_path):
    fasta, seqs = genome
    out = tmp_path / "win"
    cfg = WindowConfig(window=256, max_n_frac=0.01, splits=SPLITS)
    summary = make_windows(fasta, str(out), cfg, verbose=False)

    train = load_windows(out / "train.tsv.gz")
    val = load_windows(out / "val.tsv.gz")
    test = load_windows(out / "test.tsv.gz")

    # splity jsou na různých chromozomech
    assert set(train.chrom) == {"chr1", "chr2"}
    assert set(val.chrom) == {"chr20"} and set(test.chrom) == {"chr21"}

    # všechna okna mají správnou délku a leží uvnitř chromozomu
    for df in (train, val, test):
        assert ((df.end - df.start) == 256).all()
        for row in df.itertuples():
            assert 0 <= row.start and row.end <= len(seqs[row.chrom])

    # žádné okno nepřekročí limit N; okna přes blok N na chr1 jsou vyřazena
    assert (train.n_frac <= 0.01).all()
    chr1 = train[train.chrom == "chr1"]
    assert not ((chr1.start < 1600) & (chr1.end > 1000)).any()
    # okno s jediným N (1/256 < 1 %) zůstává
    assert ((train.chrom == "chr2") & (train.start == 0)).any()

    # posuny: train napůl překryvem, val/test bez překryvu
    assert np.diff(val.start).min() == 256
    assert summary["splits"]["train"]["stride"] == 128

    # GC podíl odpovídá sekvenci
    row = val.iloc[0]
    win = seqs["chr20"][row.start : row.end].upper()
    assert abs(row.gc_frac - (win.count("G") + win.count("C")) / 256) < 1e-3

    meta = json.loads((out / "summary.json").read_text())
    assert meta["splits"]["val"]["windows_kept"] == len(val)
    with gzip.open(out / "train.tsv.gz", "rt") as f:
        assert f.readline().strip().split("\t")[:3] == ["chrom", "start", "end"]


def test_genome_windows_access(genome, tmp_path):
    fasta, seqs = genome
    out = tmp_path / "win"
    make_windows(fasta, str(out), WindowConfig(window=256, splits=SPLITS), verbose=False)
    ds = GenomeWindows(fasta, str(out / "val.tsv.gz"))
    assert len(ds) > 0
    s0 = ds[0]
    assert s0 == seqs["chr20"][ds.start[0] : ds.end[0]].upper()
    assert ds.get(0, reverse=True) == reverse_complement(s0)
    assert set(s0) <= set("ACGTN")


def test_overlapping_splits_rejected(genome, tmp_path):
    fasta, _ = genome
    bad = {"train": ["chr1", "chr20"], "val": ["chr20"]}
    with pytest.raises(ValueError):
        cfg = WindowConfig(window=256, splits=bad)
        make_windows(fasta, str(tmp_path / "x"), cfg, verbose=False)
