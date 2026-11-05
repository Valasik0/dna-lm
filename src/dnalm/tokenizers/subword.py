"""
Podslovní tokenizace naučená z dat: BPE (T4, T5) a Unigram (T6).

Používá knihovnu HuggingFace `tokenizers` (rychlá implementace v Rustu).
Sekvence se nedělí na „slova“, celé okno je jeden řetězec, takže tokeny vznikají
přímo ze statistik DNA (stejně jako u DNABERT-2 a GROVER).
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from .base import CLS_ID, SEP_ID, SPECIAL_TOKENS, UNK, DNATokenizer

HF_FILE = "tokenizer.json"
ALPHABET = list("ACGTN")


class SubwordTokenizer(DNATokenizer):
    """Společný základ pro BPE a Unigram. Před použitím je potřeba train() nebo load."""

    type_name = "subword"

    def __init__(
        self,
        vocab_size: int = 4096,
        max_token_length: int | None = None,
        name: str | None = None,
        hf_tokenizer=None,
    ):
        super().__init__(name or f"{self.type_name}_{vocab_size}")
        self.target_vocab_size = vocab_size
        self.max_token_length = max_token_length
        self._hf = hf_tokenizer
        self._vocab_cache = None

    # --- trénink ---------------------------------------------------------------

    def _make_model_and_trainer(self, show_progress: bool):
        raise NotImplementedError

    def train(self, sequences: Iterable[str], length: int | None = None, show_progress=True):
        """Natrénuje slovník na sekvencích (iterátor řetězců, např. trénovací okna)."""
        from tokenizers import Tokenizer

        model, trainer = self._make_model_and_trainer(show_progress)
        tok = Tokenizer(model)
        tok.train_from_iterator((self.normalize(s) for s in sequences), trainer, length=length)
        for i, t in enumerate(SPECIAL_TOKENS):  # speciální tokeny musí mít ID 0..4
            if tok.token_to_id(t) != i:
                raise RuntimeError(f"{t} má ID {tok.token_to_id(t)}, očekáváno {i}")
        self._hf = tok
        self._vocab_cache = None
        if hasattr(self, "_id_to_token"):
            del self._id_to_token
        return self

    @property
    def is_trained(self) -> bool:
        return self._hf is not None

    def _require(self):
        if self._hf is None:
            raise RuntimeError(f"Tokenizér {self.name} není natrénovaný (train() nebo load).")

    # --- rozhraní DNATokenizer -------------------------------------------------

    @property
    def vocab(self) -> dict[str, int]:
        self._require()
        if self._vocab_cache is None:
            self._vocab_cache = self._hf.get_vocab()
        return self._vocab_cache

    def _tokenize_with_spans(self, seq):
        self._require()
        enc = self._hf.encode(seq, add_special_tokens=False)
        return list(enc.tokens), [tuple(o) for o in enc.offsets]

    def encode_many(self, seqs, add_special_tokens=True):
        """Rychlé dávkové kódování (paralelně v Rustu)."""
        self._require()
        encs = self._hf.encode_batch([self.normalize(s) for s in seqs], add_special_tokens=False)
        if add_special_tokens:
            return [[CLS_ID] + e.ids + [SEP_ID] for e in encs]
        return [e.ids for e in encs]

    def _params(self) -> dict:
        return {"vocab_size": self.target_vocab_size, "max_token_length": self.max_token_length}

    def _save_extra(self, directory: str) -> None:
        self._require()
        self._hf.save(os.path.join(directory, HF_FILE))

    @classmethod
    def from_directory(cls, directory: str, **params):
        from tokenizers import Tokenizer

        hf = Tokenizer.from_file(os.path.join(directory, HF_FILE))
        return cls(hf_tokenizer=hf, **params)


class BPETokenizer(SubwordTokenizer):
    """Byte Pair Encoding: opakovaně slučuje nejčastější dvojice tokenů."""

    type_name = "bpe"

    def _make_model_and_trainer(self, show_progress):
        from tokenizers import models, trainers

        kwargs = {}
        if self.max_token_length:
            kwargs["max_token_length"] = self.max_token_length
        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=ALPHABET,
            show_progress=show_progress,
            **kwargs,
        )
        return models.BPE(unk_token=UNK), trainer


class UnigramTokenizer(SubwordTokenizer):
    """Unigram: začne s velkým slovníkem a odebírá nejméně užitečné tokeny."""

    type_name = "unigram"

    def _make_model_and_trainer(self, show_progress):
        from tokenizers import models, trainers

        trainer = trainers.UnigramTrainer(
            vocab_size=self.target_vocab_size,
            special_tokens=SPECIAL_TOKENS,
            unk_token=UNK,
            initial_alphabet=ALPHABET,
            max_piece_length=self.max_token_length or 16,
            show_progress=show_progress,
        )
        return models.Unigram(), trainer
