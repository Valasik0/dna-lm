"""
Maskované jazykové modelování: maskování dávek a metrika bits per nucleotide.

Maskování (opravená verze kolatoru z původního notebooku):
  - vybere se `mask_prob` (15 %) DNA tokenů, nikdy speciální tokeny ani padding
  - z vybraných: 80 % -> [MASK], 10 % -> náhodný DNA token, 10 % beze změny
  - náhodné náhrady jen z běžných DNA tokenů (ne [PAD], [CLS], ...)
  - attention_mask se počítá z původní sekvence
  - seed: deterministické masky (validace: v každé epoše stejné)

Pro vyhodnocení se používá p_mask=1.0, p_random=0.0 (všechny vybrané pozice -> [MASK]).
S pravidlem 80/10/10 by model u 20 % pozic viděl skutečný nebo náhodný token a mohl by ho
částečně opsat: i na čistě náhodné DNA by vyšlo ~1,91 bitu na nukleotid místo 2.
"""

from __future__ import annotations

import math

import torch

from dnalm.tokenizers.base import MASK_ID, PAD_ID, SPECIAL_TOKENS, DNATokenizer


class MLMCollator:
    def __init__(
        self,
        tokenizer: DNATokenizer,
        mask_prob: float = 0.15,
        p_mask: float = 0.8,
        p_random: float = 0.1,
        seed: int | None = None,
    ):
        self.mask_prob, self.p_mask, self.p_random = mask_prob, p_mask, p_random
        self.special = torch.tensor(tokenizer.special_ids)
        self.dna_ids = torch.tensor(tokenizer.dna_token_ids)
        self.seed = seed
        self.generator = None
        self.reseed()

    def reseed(self) -> None:
        """Znovu nastaví generátor (pro validaci volat před každým vyhodnocením)."""
        if self.seed is not None:
            self.generator = torch.Generator().manual_seed(self.seed)

    def _rand(self, shape):
        return torch.rand(shape, generator=self.generator)

    def pad(self, batch: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        n = max(len(x) for x in batch)
        ids = torch.full((len(batch), n), PAD_ID, dtype=torch.long)
        for i, x in enumerate(batch):
            ids[i, : len(x)] = torch.as_tensor(x, dtype=torch.long)
        return ids, (ids != PAD_ID).long()

    def __call__(self, batch: list[list[int]]) -> dict[str, torch.Tensor]:
        ids, attention_mask = self.pad(batch)
        labels = ids.clone()
        maskable = ~torch.isin(ids, self.special)
        selected = (self._rand(ids.shape) < self.mask_prob) & maskable
        # každá sekvence má aspoň jeden maskovaný token (jinak by nepřispěla ke ztrátě)
        for row in torch.nonzero(selected.sum(1) == 0).flatten().tolist():
            cand = torch.nonzero(maskable[row]).flatten()
            if len(cand):
                j = cand[torch.randint(len(cand), (1,), generator=self.generator)]
                selected[row, j] = True
        labels[~selected] = -100

        r = self._rand(ids.shape)
        to_mask = selected & (r < self.p_mask)
        to_random = selected & (r >= self.p_mask) & (r < self.p_mask + self.p_random)
        ids[to_mask] = MASK_ID
        n_rand = int(to_random.sum())
        if n_rand:
            pick = torch.randint(len(self.dna_ids), (n_rand,), generator=self.generator)
            ids[to_random] = self.dna_ids[pick]
        return {"input_ids": ids, "attention_mask": attention_mask, "labels": labels}


def token_lengths(tokenizer: DNATokenizer) -> torch.Tensor:
    """ID tokenu -> počet nukleotidů, které token reprezentuje (speciální = 0)."""
    lengths = torch.zeros(tokenizer.vocab_size, dtype=torch.long)
    for tok, i in tokenizer.vocab.items():
        lengths[i] = 0 if tok in SPECIAL_TOKENS else len(tok)
    return lengths


class BPNMeter:
    """
    Bits per nucleotide na maskovaných pozicích: součet -log2 p(správný token) / počet bází
    v maskovaných tokenech. Srovnatelné napříč tokenizacemi (ztráta na token není).
    Náhodná DNA = 2 bity. U překrývajících se k-merů je hodnota zkreslená (únik informace),
    právě to ukazuje test na náhodné DNA.
    """

    def __init__(self, tokenizer: DNATokenizer):
        self.lengths = token_lengths(tokenizer)
        self.reset()

    def reset(self):
        self.bits = 0.0
        self.nucleotides = 0
        self.tokens = 0
        self.correct = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """logits: (N, V) pro maskované pozice, labels: (N,)"""
        logp = torch.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, labels[:, None]).squeeze(1)
        self.bits += float(nll.sum()) / math.log(2)
        self.nucleotides += int(self.lengths.to(labels.device)[labels].sum())
        self.tokens += int(labels.numel())
        self.correct += int((logits.argmax(-1) == labels).sum())

    def compute(self) -> dict[str, float]:
        return {
            "bpn": self.bits / max(self.nucleotides, 1),
            "loss": self.bits * math.log(2) / max(self.tokens, 1),
            "accuracy": self.correct / max(self.tokens, 1),
            "masked_tokens": self.tokens,
        }
