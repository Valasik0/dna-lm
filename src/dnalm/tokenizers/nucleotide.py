"""T1: nukleotidová tokenizace, každá báze je jeden token (jako HyenaDNA, Caduceus)."""

from __future__ import annotations

from .base import DNATokenizer, specials_vocab

NUCLEOTIDES = ["A", "C", "G", "T", "N"]


class NucleotideTokenizer(DNATokenizer):
    type_name = "nucleotide"

    def __init__(self, name: str | None = None):
        super().__init__(name)
        v = specials_vocab()
        for n in NUCLEOTIDES:
            v[n] = len(v)
        self._vocab = v

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab

    def _tokenize_with_spans(self, seq):
        return list(seq), [(i, i + 1) for i in range(len(seq))]

    def _params(self) -> dict:
        return {}
