"""Testy modelu, maskování, metriky BPN a tréninkové smyčky (vše na CPU, malé)."""

import math
import random

import pytest
import torch

from dnalm.data.dataset import RandomDNADataset
from dnalm.models.encoder import DNAForMaskedLM, EncoderConfig
from dnalm.models.factory import build_model
from dnalm.models.mlm import BPNMeter, MLMCollator, token_lengths
from dnalm.tokenizers import MASK_ID, PAD_ID, KmerTokenizer, NucleotideTokenizer

DEBUG = {"hidden_size": 32, "num_layers": 2, "num_heads": 2, "intermediate_size": 64}


def tiny(vocab=10, **kw):
    return DNAForMaskedLM(EncoderConfig(vocab_size=vocab, max_position=128, **{**DEBUG, **kw}))


def test_shapes_attention_and_padding():
    m = tiny().eval()
    x = torch.randint(5, 10, (3, 40))
    x[2, 30:] = PAD_ID
    out = m(x, output_attentions=True, output_hidden_states=True)
    enc = out["encoder"]
    assert out["logits"].shape == (3, 40, 10)
    assert len(enc.attentions) == 2 and enc.attentions[0].shape == (3, 2, 40, 40)
    assert len(enc.hidden_states) == 3
    a = enc.attentions[1]
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
    assert a[2, :, :, 30:].abs().max() == 0  # na padding se nikdo nedívá
    with torch.no_grad():  # rychlá (SDPA) a explicitní cesta dávají stejný výsledek
        assert torch.allclose(m(x)["logits"], out["logits"], atol=1e-5)


def test_padding_does_not_change_output():
    m = tiny().eval()
    x = torch.randint(5, 10, (1, 20))
    padded = torch.cat([x, torch.zeros(1, 7, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a = m(x)["logits"]
        b = m(padded)["logits"][:, :20]
    assert torch.allclose(a, b, atol=1e-5)


def test_local_attention_window():
    m = tiny(global_every=2, local_window=8).eval()
    att = m(torch.randint(5, 10, (1, 30)), output_attentions=True)["encoder"].attentions
    assert att[1][0, :, 0, 10:].abs().max() == 0  # vrstva 1 lokální: dál než 4 tokeny nevidí
    assert att[0][0, :, 0, 10:].abs().max() > 0  # vrstva 0 globální


def test_save_load(tmp_path):
    m = tiny().eval()
    m.save(str(tmp_path / "m"))
    m2 = DNAForMaskedLM.load(str(tmp_path / "m")).eval()
    x = torch.randint(5, 10, (2, 15))
    assert torch.equal(m(x)["logits"], m2(x)["logits"])


def test_factory_parameter_counts():
    m = build_model(
        {
            "name": "S",
            "hidden_size": 384,
            "num_layers": 6,
            "num_heads": 6,
            "intermediate_size": 1536,
        },
        KmerTokenizer(k=6),
    )
    n = m.num_parameters()
    assert 13e6 < n["non_embedding"] < 15e6
    assert n["embedding"] == 4106 * 384
    with pytest.raises(ValueError):
        build_model({"hidden_sise": 1}, NucleotideTokenizer())


def test_collator_masking():
    tok = KmerTokenizer(k=3)
    col = MLMCollator(tok, mask_prob=0.15, seed=0)
    rng = random.Random(0)
    batch = [
        tok.encode("".join(rng.choice("ACGT") for _ in range(rng.randint(30, 300))))
        for _ in range(64)
    ]
    out = col(batch)
    ids, labels, att = out["input_ids"], out["labels"], out["attention_mask"]
    sel = labels != -100
    # masky jen na DNA tokenech, nikdy na speciálních/paddingu
    assert torch.isin(labels[sel], torch.tensor(tok.dna_token_ids)).all()
    assert (att[sel] == 1).all()
    frac = sel.sum() / torch.isin(labels.where(sel, ids), torch.tensor(tok.dna_token_ids)).sum()
    assert 0.10 < frac < 0.20
    # 80/10/10
    masked = (ids[sel] == MASK_ID).float().mean()
    assert 0.7 < masked < 0.9
    replaced = ids[sel][ids[sel] != MASK_ID]
    assert torch.isin(replaced, torch.tensor(tok.dna_token_ids)).all()
    # attention_mask podle původní sekvence, i když náhodná náhrada nikdy nedá [PAD]
    assert (att.sum(1) == torch.tensor([len(b) for b in batch])).all()
    # determinismus validace
    col.reseed()
    again = col(batch)
    assert torch.equal(again["input_ids"], ids) and torch.equal(again["labels"], labels)


def test_bpn_meter():
    tok = KmerTokenizer(k=2)
    lengths = token_lengths(tok)
    assert lengths[tok.vocab["AC"]] == 2 and lengths[MASK_ID] == 0
    # rovnoměrná předpověď přes 16 dinukleotidů = 4 bity na token = 2 bity na nukleotid
    dna = torch.tensor([i for t, i in tok.vocab.items() if len(t) == 2])
    logits = torch.full((100, tok.vocab_size), -1e9)
    logits[:, dna] = 0.0
    labels = dna[torch.randint(16, (100,))]
    m = BPNMeter(tok)
    m.update(logits, labels)
    r = m.compute()
    assert r["bpn"] == pytest.approx(2.0, abs=1e-4)
    assert r["loss"] == pytest.approx(math.log(16), abs=1e-4)


def test_random_dataset_deterministic():
    ds = RandomDNADataset(NucleotideTokenizer(), n=5, length=50, seed=1)
    assert ds.sequence(3) == ds.sequence(3) and ds.sequence(3) != ds.sequence(4)
    assert len(ds[0]) == 52


def test_eval_collator_masks_everything():
    tok = NucleotideTokenizer()
    col = MLMCollator(tok, mask_prob=0.5, p_mask=1.0, p_random=0.0, seed=0)
    out = col([tok.encode("ACGT" * 50)])
    sel = out["labels"] != -100
    assert (out["input_ids"][sel] == MASK_ID).all()


def test_no_mask_without_padding():
    """Bez paddingu se maska nepředává (rychlé jádro attention), s paddingem je bool."""
    m = tiny()
    enc = m.encoder
    full = torch.ones(2, 10, dtype=torch.long)
    assert enc._attn_bias(full, 10, "cpu", True) is None
    padded = full.clone()
    padded[1, 7:] = 0
    mask = enc._attn_bias(padded, 10, "cpu", True)
    assert mask.dtype == torch.bool and not mask[1, 0, 0, 8]
    assert enc._attn_bias(full, 10, "cpu", False).dtype == torch.bool  # lokální okno


def test_autocast_bf16_cpu():
    """Pod autocastem (bf16) projde forward i backward a výsledek je konečný."""
    m = tiny()
    x = torch.randint(5, 10, (2, 30))
    labels = torch.full_like(x, -100)
    labels[:, ::3] = x[:, ::3]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(x, labels=labels)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
