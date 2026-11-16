"""
PyTorch datasety pro předtrénink.

WindowTokenDataset: okna z genomu, tokenizace za běhu (vhodné pro ladění a menší běhy).
Pro velké běhy se data předtokenizují (úkol 1.10) a čtou se rovnou ID tokenů.

RandomDNADataset: rovnoměrně náhodná DNA. Nejlepší možný výsledek je přesně 2 bity
na nukleotid, takže model, který je na ní lepší, dostává nápovědu (únik informace).
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from dnalm.tokenizers.base import DNATokenizer

from .genome import GenomeWindows


class WindowTokenDataset(Dataset):
    def __init__(
        self,
        fasta: str,
        windows: str,
        tokenizer: DNATokenizer,
        rc_prob: float = 0.0,
        max_windows: int | None = None,
        seed: int = 0,
    ):
        """
        rc_prob: pravděpodobnost, že se místo okna použije jeho reverzní komplement
                 (augmentace, 0 = vypnuto). Pro validaci vždy 0.
        max_windows: náhodná podmnožina oken (pevný seed), např. pro rychlou validaci.
        """
        self.windows = GenomeWindows(fasta, windows)
        self.tokenizer = tokenizer
        self.rc_prob = rc_prob
        n = len(self.windows)
        if max_windows and max_windows < n:
            rng = np.random.default_rng(seed)
            self.index = np.sort(rng.choice(n, size=max_windows, replace=False))
        else:
            self.index = np.arange(n)
        self.rng = np.random.default_rng(seed + 1)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> list[int]:
        reverse = self.rc_prob > 0 and self.rng.random() < self.rc_prob
        seq = self.windows.get(int(self.index[i]), reverse=reverse)
        return self.tokenizer.encode(seq)


class RandomDNADataset(Dataset):
    """Náhodné sekvence ACGT pevné délky (deterministické podle indexu)."""

    def __init__(self, tokenizer: DNATokenizer, n: int, length: int = 1024, seed: int = 0):
        self.tokenizer, self.n, self.length, self.seed = tokenizer, n, length, seed

    def __len__(self) -> int:
        return self.n

    def sequence(self, i: int) -> str:
        rng = np.random.default_rng((self.seed, i))
        return "".join(np.array(list("ACGT"))[rng.integers(0, 4, self.length)])

    def __getitem__(self, i: int) -> list[int]:
        return self.tokenizer.encode(self.sequence(i))
