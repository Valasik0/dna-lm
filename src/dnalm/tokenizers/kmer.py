"""
k-merová tokenizace s parametry k a stride.

- stride == k: nepřekrývající se k-mery (T3, jako Nucleotide Transformer). Hlavní varianta.
  Úsek kratší než k na konci sekvence a k-mery obsahující N se rozloží na jednotlivé
  nukleotidy, takže tokenizace je bezeztrátová.
- stride < k: překrývající se k-mery (T2, jako DNABERT). POZOR: při MLM sousední tokeny
  prozrazují maskovaný token (únik informace). Používá se jen jako kontrolní experiment.
"""

from __future__ import annotations

from itertools import product

from .base import UNK, DNATokenizer, specials_vocab
from .nucleotide import NUCLEOTIDES

# Nad k = 7 je slovník (4^k) nepoužitelný: k = 10 má ~1 milion tokenů, většinu z nich
# by model v tréninku viděl jen párkrát a embedding matice by byla větší než celý model.
MAX_K = 7


class KmerTokenizer(DNATokenizer):
    type_name = "kmer"

    def __init__(self, k: int = 6, stride: int | None = None, name: str | None = None):
        if not 1 <= k <= MAX_K:
            raise ValueError(f"k musí být 1–{MAX_K} (slovník 4^k: k={k} dá {4**k:,} tokenů)")
        stride = k if stride is None else stride
        if not 1 <= stride <= k:
            raise ValueError("stride musí být mezi 1 a k")
        super().__init__(name or (f"kmer{k}" if stride == k else f"kmer{k}_stride{stride}"))
        self.k, self.stride = k, stride
        v = specials_vocab()
        for kmer in ("".join(p) for p in product("ACGT", repeat=k)):
            v[kmer] = len(v)
        for n in NUCLEOTIDES:  # pro zbytky na konci a úseky s N
            if n not in v:
                v[n] = len(v)
        self._vocab = v

    @property
    def overlapping(self) -> bool:
        return self.stride < self.k

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab

    def _tokenize_with_spans(self, seq):
        k, s = self.k, self.stride
        tokens, spans = [], []
        if not self.overlapping:
            for start in range(0, len(seq), k):
                chunk = seq[start : start + k]
                if len(chunk) == k and "N" not in chunk:
                    tokens.append(chunk)
                    spans.append((start, start + k))
                else:  # zbytek nebo N: po jednotlivých bázích
                    for j, c in enumerate(chunk):
                        tokens.append(c)
                        spans.append((start + j, start + j + 1))
            return tokens, spans
        # překrývající se k-mery
        for start in range(0, len(seq) - k + 1, s):
            chunk = seq[start : start + k]
            tokens.append(UNK if "N" in chunk else chunk)
            spans.append((start, start + k))
        return tokens, spans

    def decode(self, ids) -> str:
        if not self.overlapping:
            return super().decode(ids)
        toks = [self.id_to_token[int(i)] for i in ids if int(i) not in self.special_ids]
        if not toks:
            return ""
        # první k-mer celý, z každého dalšího jen posledních `stride` bází ([UNK] -> N)
        toks = ["N" * self.k if t == UNK else t for t in toks]
        return toks[0] + "".join(t[-self.stride :] for t in toks[1:])

    def _params(self) -> dict:
        return {"k": self.k, "stride": self.stride}
