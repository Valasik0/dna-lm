"""
Společné rozhraní všech DNA tokenizérů.

Každý tokenizér umí:
  - tokenize(seq)      -> seznam tokenů (řetězců)
  - spans(seq)         -> pro každý token interval [start, end) v nukleotidech
  - encode(seq)        -> seznam ID (volitelně s [CLS] a [SEP])
  - decode(ids)        -> zpět sekvence DNA
  - encode_batch(...)  -> numpy pole s paddingem a attention maskou
  - save(dir) / load   -> uložení a načtení (viz registry.load_tokenizer)

Speciální tokeny mají ve všech tokenizérech stejná ID (0–4), takže collator,
model i evaluace s nimi pracují stejně bez ohledu na tokenizaci.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import numpy as np

PAD, UNK, CLS, SEP, MASK = "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
SPECIAL_TOKENS = [PAD, UNK, CLS, SEP, MASK]  # pořadí = ID 0..4
PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID = range(len(SPECIAL_TOKENS))

CONFIG_FILE = "tokenizer_config.json"


class DNATokenizer(ABC):
    """Abstraktní základ. Podtřídy implementují _tokenize_with_spans a vocab."""

    type_name: str = "base"

    def __init__(self, name: str | None = None):
        self.name = name or self.type_name

    # --- co musí implementovat podtřída ----------------------------------------

    @abstractmethod
    def _tokenize_with_spans(self, seq: str) -> tuple[list[str], list[tuple[int, int]]]:
        """Rozdělí (už normalizovanou) sekvenci na tokeny a jejich intervaly."""

    @property
    @abstractmethod
    def vocab(self) -> dict[str, int]:
        """Slovník token -> ID, včetně speciálních tokenů na ID 0..4."""

    @abstractmethod
    def _params(self) -> dict:
        """Parametry potřebné k obnovení tokenizéru (ukládají se do configu)."""

    # --- společná logika -------------------------------------------------------

    @staticmethod
    def normalize(seq: str) -> str:
        """Velká písmena; cokoli mimo ACGT se převede na N."""
        seq = seq.upper()
        if set(seq) <= set("ACGTN"):
            return seq
        return "".join(c if c in "ACGT" else "N" for c in seq)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def id_to_token(self) -> dict[int, str]:
        if not hasattr(self, "_id_to_token"):
            self._id_to_token = {i: t for t, i in self.vocab.items()}
        return self._id_to_token

    @property
    def special_ids(self) -> list[int]:
        return list(range(len(SPECIAL_TOKENS)))

    @property
    def dna_token_ids(self) -> list[int]:
        """ID všech běžných (ne-speciálních) tokenů, např. pro náhodné náhrady v MLM."""
        return sorted(i for t, i in self.vocab.items() if t not in SPECIAL_TOKENS)

    def tokenize(self, seq: str) -> list[str]:
        return self._tokenize_with_spans(self.normalize(seq))[0]

    def spans(self, seq: str) -> list[tuple[int, int]]:
        """Interval [start, end) v nukleotidech pro každý token z tokenize(seq)."""
        return self._tokenize_with_spans(self.normalize(seq))[1]

    def encode(
        self, seq: str, add_special_tokens: bool = True, return_spans: bool = False
    ) -> list[int] | tuple[list[int], list[tuple[int, int]]]:
        tokens, spans = self._tokenize_with_spans(self.normalize(seq))
        v = self.vocab
        ids = [v.get(t, UNK_ID) for t in tokens]
        if add_special_tokens:
            ids = [CLS_ID] + ids + [SEP_ID]
            spans = [(0, 0)] + spans + [(0, 0)]  # speciální tokeny nepokrývají žádnou bázi
        return (ids, spans) if return_spans else ids

    def encode_many(self, seqs, add_special_tokens: bool = True) -> list[list[int]]:
        """Zakóduje seznam sekvencí (podtřídy můžou zrychlit, např. BPE v Rustu)."""
        return [self.encode(s, add_special_tokens=add_special_tokens) for s in seqs]

    def decode(self, ids) -> str:
        """Sloučí tokeny zpět do sekvence (speciální tokeny se přeskočí)."""
        toks = [self.id_to_token[int(i)] for i in ids if int(i) not in self.special_ids]
        return "".join(toks)

    def encode_batch(
        self,
        seqs: list[str],
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Zakóduje více sekvencí do pole stejné délky.
        Vrací input_ids a attention_mask (1 = token, 0 = [PAD]).
        Delší sekvence se oříznou na max_length (zachová se [SEP] na konci).
        """
        encoded = self.encode_many(seqs, add_special_tokens=add_special_tokens)
        length = max(len(e) for e in encoded) if max_length is None else max_length
        ids = np.full((len(seqs), length), PAD_ID, dtype=np.int32)
        mask = np.zeros((len(seqs), length), dtype=np.int8)
        for row, e in enumerate(encoded):
            if len(e) > length:
                e = e[: length - 1] + [SEP_ID] if add_special_tokens else e[:length]
            ids[row, : len(e)] = e
            mask[row, : len(e)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    # --- ukládání --------------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        cfg = {"type": self.type_name, "name": self.name, **self._params()}
        with open(os.path.join(directory, CONFIG_FILE), "w") as f:
            json.dump(cfg, f, indent=2)
        self._save_extra(directory)

    def _save_extra(self, directory: str) -> None:  # noqa: B027 (záměrně prázdné)
        """Podtřídy s natrénovaným slovníkem ho uloží sem."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, vocab_size={self.vocab_size})"


def specials_vocab() -> dict[str, int]:
    return {t: i for i, t in enumerate(SPECIAL_TOKENS)}
