"""
Práce s referenčním genomem: čtení sekvencí a generování oken pro předtrénink.

Okna se ukládají jen jako souřadnice (BED-like TSV, 0-based, polouzavřený interval
[start, end)), ne jako sekvence. Sekvence se z FASTA vyřezávají až při tokenizaci,
takže stejná okna slouží všem tokenizérům.

Příklad použití z příkazové řádky:

    python -m dnalm.data.genome \\
        --fasta /content/drive/MyDrive/dna-lm/data/raw/genome/hg38.main.fa \\
        --out /content/drive/MyDrive/dna-lm/data/processed/windows_1024 \\
        --window 1024
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

# Rozdělení podle chromozomů: validace a test jsou celé chromozomy,
# takže mezi splity nemůže dojít k překryvu sekvencí.
DEFAULT_SPLITS = {
    "train": [f"chr{i}" for i in range(1, 23) if i not in (20, 21)] + ["chrX"],
    "val": ["chr20"],
    "test": ["chr21"],
}

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

WINDOW_COLUMNS = ["chrom", "start", "end", "n_frac", "gc_frac", "repeat_frac"]


def reverse_complement(seq: str) -> str:
    """
    Reverzní komplement DNA sekvence (zachovává velikost písmen).
    Odpovídá stejnému úseku čtenému z druhého vlákna DNA: A<->T, C<->G, pozpátku.
    """
    return seq.translate(_COMPLEMENT)[::-1]


def read_fai(fasta_path: str) -> dict[str, int]:
    """Délky chromozomů z .fai indexu (bez načítání FASTA)."""
    lengths = {}
    with open(fasta_path + ".fai") as f:
        for line in f:
            name, length = line.split("\t")[:2]
            lengths[name] = int(length)
    return lengths


@dataclass
class WindowConfig:
    window: int = 1024
    train_stride: int | None = None  # None = window // 2 (okna se v tréninku překrývají napůl)
    eval_stride: int | None = None  # None = window (val/test bez překryvu)
    max_n_frac: float = 0.01  # okna s více N se zahodí
    splits: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_SPLITS))

    def stride_for(self, split: str) -> int:
        if split == "train":
            return self.train_stride or self.window // 2
        return self.eval_stride or self.window


def window_stats(seq: str, window: int, stride: int) -> dict[str, np.ndarray]:
    """
    Pro všechna okna dané délky a posunu spočítá podíl N, GC a repetic (malá písmena).
    Používá kumulativní součty, takže je to O(délka chromozomu) i pro miliony oken.
    GC se počítá jen z bází, které nejsou N.
    """
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    upper = arr & 0xDF  # převod na velká písmena (ASCII)
    is_n = upper == ord("N")
    is_gc = (upper == ord("G")) | (upper == ord("C"))
    is_lower = arr >= ord("a")

    def window_sums(mask: np.ndarray, starts: np.ndarray) -> np.ndarray:
        # int32 stačí: nejdelší chromozom (chr1) má ~249 milionů bp
        csum = np.concatenate(([0], np.cumsum(mask, dtype=np.int32)))
        return csum[starts + window] - csum[starts]

    if len(arr) < window:
        empty = np.array([], dtype=np.int64)
        return {"start": empty, "n": empty, "gc": empty, "lower": empty}
    starts = np.arange(0, len(arr) - window + 1, stride, dtype=np.int64)
    return {
        "start": starts,
        "n": window_sums(is_n, starts),
        "gc": window_sums(is_gc, starts),
        "lower": window_sums(is_lower, starts),
    }


def make_windows(fasta_path: str, out_dir: str, cfg: WindowConfig, verbose: bool = True) -> dict:
    """Vygeneruje okna pro všechny splity a uloží je jako <split>.tsv.gz + summary.json."""
    from pyfaidx import Fasta

    os.makedirs(out_dir, exist_ok=True)
    lengths = read_fai(fasta_path)
    genome = Fasta(fasta_path, as_raw=True, sequence_always_upper=False)
    w = cfg.window

    all_chroms = [c for chroms in cfg.splits.values() for c in chroms]
    if len(all_chroms) != len(set(all_chroms)):
        raise ValueError("Chromozom je ve více splitech")
    missing = set(all_chroms) - set(lengths)
    if missing:
        raise ValueError(f"Chromozomy chybí ve FASTA: {sorted(missing)}")

    summary = {"config": asdict(cfg), "fasta": os.path.abspath(fasta_path), "splits": {}}
    for split, chroms in cfg.splits.items():
        stride = cfg.stride_for(split)
        path = os.path.join(out_dir, f"{split}.tsv.gz")
        total = kept = 0
        per_chrom = {}
        with gzip.open(path, "wt") as f:
            f.write("\t".join(WINDOW_COLUMNS) + "\n")
            for chrom in chroms:
                seq = genome[chrom][:]
                st = window_stats(seq, w, stride)
                n_frac = st["n"] / w
                keep = n_frac <= cfg.max_n_frac
                acgt = w - st["n"]
                gc_frac = np.divide(st["gc"], acgt, out=np.zeros(len(acgt)), where=acgt > 0)
                rep_frac = st["lower"] / w
                for s, nf, gc, rp in zip(
                    st["start"][keep], n_frac[keep], gc_frac[keep], rep_frac[keep], strict=True
                ):
                    f.write(f"{chrom}\t{s}\t{s + w}\t{nf:.4f}\t{gc:.4f}\t{rp:.4f}\n")
                total += len(keep)
                kept += int(keep.sum())
                per_chrom[chrom] = int(keep.sum())
                if verbose:
                    print(f"  {split:5s} {chrom:6s} {int(keep.sum()):>9,} / {len(keep):>9,} oken")
                del seq
        summary["splits"][split] = {
            "chromosomes": chroms,
            "stride": stride,
            "windows_total": total,
            "windows_kept": kept,
            "bp_covered_unique": None if split == "train" else kept * w,
            "per_chrom": per_chrom,
            "file": os.path.basename(path),
        }
        if verbose:
            print(f"{split}: {kept:,} oken ponecháno z {total:,}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_windows(path: str):
    """Načte soubor s okny jako pandas DataFrame."""
    import pandas as pd

    return pd.read_csv(path, sep="\t")


class GenomeWindows:
    """
    Přístup k sekvencím oken: windows[i] vrátí sekvenci i-tého okna (velkými písmeny),
    get(i, reverse=True) její reverzní komplement.
    Základ pro tokenizaci a Dataset v PyTorch.
    """

    def __init__(self, fasta_path: str, windows_path: str, uppercase: bool = True):
        from pyfaidx import Fasta

        df = load_windows(windows_path)
        self.chrom = df["chrom"].to_numpy()
        self.start = df["start"].to_numpy()
        self.end = df["end"].to_numpy()
        self.uppercase = uppercase
        self.fasta_path = fasta_path
        self._genome = None
        self._Fasta = Fasta

    @property
    def genome(self):
        # otevírá se líně, aby šel objekt bezpečně předat do DataLoader workerů
        if self._genome is None:
            self._genome = self._Fasta(self.fasta_path, as_raw=True, sequence_always_upper=False)
        return self._genome

    def __len__(self) -> int:
        return len(self.start)

    def get(self, i: int, reverse: bool = False) -> str:
        seq = self.genome[self.chrom[i]][int(self.start[i]) : int(self.end[i])]
        if self.uppercase:
            seq = seq.upper()
        return reverse_complement(seq) if reverse else seq

    def __getitem__(self, i: int) -> str:
        return self.get(i)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generování oken z referenčního genomu")
    p.add_argument("--fasta", required=True, help="cesta k hg38.main.fa (s .fai indexem)")
    p.add_argument("--out", required=True, help="výstupní složka")
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--train-stride", type=int, default=None)
    p.add_argument("--eval-stride", type=int, default=None)
    p.add_argument("--max-n-frac", type=float, default=0.01)
    a = p.parse_args(argv)
    cfg = WindowConfig(
        window=a.window,
        train_stride=a.train_stride,
        eval_stride=a.eval_stride,
        max_n_frac=a.max_n_frac,
    )
    make_windows(a.fasta, a.out, cfg)


if __name__ == "__main__":
    main()
