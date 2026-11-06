"""Testy tokenizérů: bezeztrátovost, speciální tokeny, intervaly, ukládání, konfigurace."""

import glob
import os
import random

import numpy as np
import pytest

from dnalm.tokenizers import (
    CLS_ID,
    MASK_ID,
    PAD_ID,
    SEP_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    BPETokenizer,
    KmerTokenizer,
    NucleotideTokenizer,
    UnigramTokenizer,
    build_tokenizer,
    load_config,
    load_tokenizer,
    needs_training,
)
from dnalm.tokenizers.kmer import MAX_K
from dnalm.tokenizers.stats import mutation_stability, repeat_token_lengths, summary

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "tokenizer")
MOTIF = "TTGACGTCAA"


def random_dna(n, rng, alphabet="ACGT"):
    return "".join(rng.choice(alphabet) for _ in range(n))


@pytest.fixture(scope="module")
def corpus():
    """Syntetická „DNA“ s opakovaným motivem, ať se BPE má co naučit."""
    rng = random.Random(0)
    seqs = []
    for _ in range(300):
        parts = [random_dna(rng.randint(5, 40), rng) for _ in range(6)]
        seqs.append(MOTIF.join(parts))
    return seqs


@pytest.fixture(scope="module")
def trained(corpus):
    bpe = BPETokenizer(vocab_size=120).train(corpus, show_progress=False)
    uni = UnigramTokenizer(vocab_size=100, max_token_length=12).train(corpus, show_progress=False)
    return {"bpe": bpe, "unigram": uni}


def all_tokenizers(trained):
    return [
        NucleotideTokenizer(),
        KmerTokenizer(k=6),
        KmerTokenizer(k=3),
        trained["bpe"],
        trained["unigram"],
    ]


TEST_SEQS = ["ACGTACGTAC", "A", "ACGTN" * 7, "NNNNNN", "acgtTTGACGTCAAggca", "ACGTAC" * 5 + "GG"]


def test_special_ids_same_everywhere(trained):
    assert (PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID) == (0, 1, 2, 3, 4)
    for tok in all_tokenizers(trained) + [KmerTokenizer(k=6, stride=1)]:
        for i, t in enumerate(SPECIAL_TOKENS):
            assert tok.vocab[t] == i, tok.name
        assert set(tok.dna_token_ids).isdisjoint(tok.special_ids)


@pytest.mark.parametrize("seq", TEST_SEQS)
def test_roundtrip_and_spans(trained, seq):
    norm = seq.upper()
    for tok in all_tokenizers(trained):
        ids = tok.encode(seq)
        assert ids[0] == CLS_ID and ids[-1] == SEP_ID
        assert tok.decode(ids) == norm, tok.name
        assert UNK_ID not in ids, tok.name
        tokens, spans = tok.tokenize(seq), tok.spans(seq)
        assert len(tokens) == len(spans) == len(ids) - 2
        # nepřekrývající se tokenizace: intervaly navazují a pokrývají celou sekvenci
        assert spans[0][0] == 0 and spans[-1][1] == len(seq)
        for (_s1, e1), (s2, _) in zip(spans, spans[1:], strict=False):
            assert e1 == s2
        for t, (s, e) in zip(tokens, spans, strict=True):
            assert norm[s:e] == t


def test_kmer_counts():
    tok = KmerTokenizer(k=6)
    seq = "ACGTAC" * 170 + "GTAC"  # 1 024 bp
    assert len(tok.tokenize(seq)) == 170 + 4  # 170 šestic + 4 jednotlivé báze na konci
    assert tok.vocab_size == 5 + 4096 + 5
    # k-mer s N se rozloží na jednotlivé báze
    assert tok.tokenize("ACGNTA") == list("ACGNTA")


def test_overlapping_kmer_is_leaky():
    """Překrývající se k-mery: maskovaný token jde dopočítat ze sousedů (připomínka vedoucího)."""
    tok = KmerTokenizer(k=3, stride=1)
    assert tok.tokenize("ACCGG") == ["ACC", "CCG", "CGG"]
    assert tok.spans("ACCGG") == [(0, 3), (1, 4), (2, 5)]
    assert tok.decode(tok.encode("ACCGGTTA")) == "ACCGGTTA"
    left, right = "ACC", "CGG"
    assert left[1:] + right[-1] == "CCG"  # prostřední token je plně určen sousedy


def test_bpe_learns_motif(trained):
    bpe = trained["bpe"]
    assert bpe.vocab_size <= 120
    longest = max((t for t in bpe.vocab if t not in SPECIAL_TOKENS), key=len)
    assert len(longest) >= 6
    # opakovaný motiv se zakóduje malým počtem tokenů
    assert len(bpe.tokenize(MOTIF)) <= 3


def test_save_load(tmp_path, trained, corpus):
    for tok in all_tokenizers(trained) + [KmerTokenizer(k=6, stride=1)]:
        d = tmp_path / tok.name
        tok.save(str(d))
        loaded = load_tokenizer(str(d))
        assert type(loaded) is type(tok) and loaded.name == tok.name
        for s in corpus[:20] + TEST_SEQS:
            assert loaded.encode(s) == tok.encode(s)


def test_encode_batch(trained):
    tok = trained["bpe"]
    seqs = ["ACGT" * 10, "ACG", MOTIF * 4]
    out = tok.encode_batch(seqs)
    ids, mask = out["input_ids"], out["attention_mask"]
    assert ids.shape == mask.shape and ids.shape[0] == 3
    for row, s in enumerate(seqs):
        n = int(mask[row].sum())
        assert list(ids[row, :n]) == tok.encode(s)
        assert (ids[row, n:] == PAD_ID).all()
    # oříznutí zachová [SEP] na konci
    cut = NucleotideTokenizer().encode_batch(["ACGTACGT"], max_length=5)["input_ids"][0]
    assert list(cut) == [CLS_ID, 5, 6, 7, SEP_ID]
    # dávkové kódování BPE = kódování po jednom
    assert tok.encode_many(seqs) == [tok.encode(s) for s in seqs]


def test_configs_build():
    paths = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    assert len(paths) >= 5
    for p in paths:
        cfg = load_config(p)
        tok = build_tokenizer(cfg)
        assert tok.name == cfg["name"]
        assert needs_training(tok) == (cfg["type"] in ("bpe", "unigram"))


def test_untrained_bpe_raises():
    with pytest.raises(RuntimeError):
        BPETokenizer(vocab_size=100).tokenize("ACGT")


def test_stats(trained, corpus):
    s = summary(KmerTokenizer(k=6), ["ACGTAC" * 10])
    assert s["nt_per_token"] == 6 and s["tokens_per_kb"] == pytest.approx(1000 / 6)
    s = summary(trained["bpe"], corpus[:50])
    assert 1 < s["nt_per_token"] < 20 and s["unk_share"] == 0
    st = mutation_stability(KmerTokenizer(k=6), ["ACGTAC" * 20], n_mutations=10)
    assert st["tokens_changed_mean"] == 1  # mutace u k-merů změní právě jeden token
    st = mutation_stability(NucleotideTokenizer(), ["ACGT" * 20], n_mutations=10)
    assert st["tokens_changed_mean"] == 1 and st["resegmented_nt_mean"] == 1


def test_normalize():
    assert NucleotideTokenizer.normalize("acgtRY") == "ACGTNN"
    assert np.array_equal(
        NucleotideTokenizer().encode_batch(["acgt"])["input_ids"],
        NucleotideTokenizer().encode_batch(["ACGT"])["input_ids"],
    )


def test_repeat_token_lengths():
    r = repeat_token_lengths(KmerTokenizer(k=3), ["acgacgACGACG"])
    assert r["len_in_repeats"] == 3 and r["len_outside_repeats"] == 3
    assert r["tokens_in_repeats_share"] == 0.5


@pytest.mark.parametrize("k", range(2, MAX_K + 1))
def test_kmer_series(k):
    tok = KmerTokenizer(k=k)
    assert tok.vocab_size == 5 + 4**k + 5
    seq = "ACGTTGCA" * 128  # 1 024 bp
    assert tok.decode(tok.encode(seq)) == seq
    assert len(tok.tokenize(seq)) == 1024 // k + 1024 % k


def test_kmer_too_large():
    with pytest.raises(ValueError):
        KmerTokenizer(k=MAX_K + 1)
