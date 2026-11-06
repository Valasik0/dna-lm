"""
Statistiky tokenizérů.

Všechny funkce berou tokenizér a seznam sekvencí (např. vzorek validačních oken).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .base import SPECIAL_TOKENS, UNK, DNATokenizer


def tokenize_all(tok: DNATokenizer, seqs: list[str]) -> list[list[str]]:
    return [tok.tokenize(s) for s in seqs]


def summary(tok: DNATokenizer, seqs: list[str]) -> dict:
    """Souhrnné metriky jednoho tokenizéru na vzorku sekvencí."""
    toks = tokenize_all(tok, seqs)
    n_tokens = np.array([len(t) for t in toks])
    n_nt = np.array([len(s) for s in seqs])
    flat = [t for ts in toks for t in ts]
    counts = Counter(flat)
    lengths = np.array([len(t) for t in flat if t not in SPECIAL_TOKENS])
    dna_vocab = [t for t in tok.vocab if t not in SPECIAL_TOKENS]
    return {
        "tokenizer": tok.name,
        "vocab_size": tok.vocab_size,
        "tokens_per_kb": float((n_tokens / n_nt).mean() * 1000),
        "nt_per_token": float(n_nt.sum() / n_tokens.sum()),
        "token_len_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "token_len_max": int(lengths.max()) if len(lengths) else 0,
        "single_nt_share": float((lengths == 1).mean()) if len(lengths) else 0.0,
        "vocab_used_share": len([t for t in dna_vocab if counts[t] > 0]) / max(len(dna_vocab), 1),
        "unk_share": counts[UNK] / max(len(flat), 1),
    }


def token_length_distribution(tok: DNATokenizer, seqs: list[str]) -> Counter:
    """Kolikrát se vyskytl token dané délky (v nukleotidech)."""
    return Counter(len(t) for ts in tokenize_all(tok, seqs) for t in ts if t not in SPECIAL_TOKENS)


def top_tokens(tok: DNATokenizer, seqs: list[str], n: int = 20) -> list[tuple[str, int]]:
    """Nejčastější tokeny ve vzorku."""
    return Counter(t for ts in tokenize_all(tok, seqs) for t in ts).most_common(n)


def mutation_stability(
    tok: DNATokenizer, seqs: list[str], n_mutations: int = 5, seed: int = 0
) -> dict:
    """
    Jak moc jedna bodová mutace změní tokenizaci. Pro každou sekvenci se n-krát
    změní náhodná báze a spočítá se, kolik tokenů (token + jeho pozice) se změnilo.
    U k-merů je to vždy 1 token, u BPE se může přesegmentovat celé okolí.
    """
    rng = np.random.default_rng(seed)
    changed, resegmented_nt = [], []
    for s in seqs:
        s = tok.normalize(s)
        orig = set(zip(tok.tokenize(s), map(tuple, tok.spans(s)), strict=True))
        for _ in range(n_mutations):
            pos = int(rng.integers(len(s)))
            alt = rng.choice([b for b in "ACGT" if b != s[pos]])
            m = s[:pos] + alt + s[pos + 1 :]
            new = set(zip(tok.tokenize(m), map(tuple, tok.spans(m)), strict=True))
            diff = orig ^ new
            changed.append(len(orig - new))
            starts = [sp[0] for _, sp in diff]
            ends = [sp[1] for _, sp in diff]
            resegmented_nt.append(max(ends) - min(starts) if diff else 0)
    return {
        "tokenizer": tok.name,
        "tokens_changed_mean": float(np.mean(changed)),
        "tokens_changed_max": int(np.max(changed)),
        "resegmented_nt_mean": float(np.mean(resegmented_nt)),
    }


def repeat_token_lengths(tok: DNATokenizer, raw_seqs: list[str]) -> dict:
    """
    Průměrná délka tokenu (v bp) uvnitř repetic a mimo ně. Vstupem jsou sekvence
    se zachovanými malými písmeny (soft-masking). Token se počítá jako „v repetici“,
    když je většina jeho bází malými písmeny. Ukazuje, zda BPE repetice „zkomprimuje“.
    """
    rep, non = [], []
    for raw in raw_seqs:
        lower = np.frombuffer(raw.encode("ascii"), dtype=np.uint8) >= ord("a")
        for s, e in tok.spans(raw):
            if e <= s:
                continue
            (rep if lower[s:e].mean() > 0.5 else non).append(e - s)
    return {
        "tokenizer": tok.name,
        "len_in_repeats": float(np.mean(rep)) if rep else float("nan"),
        "len_outside_repeats": float(np.mean(non)) if non else float("nan"),
        "tokens_in_repeats_share": len(rep) / max(len(rep) + len(non), 1),
    }
