"""
DNA encoder ve stylu ModernBERT, implementovaný přímo v PyTorch.

Architektura (každá volba odpovídá současnému stavu BERT-like modelů):
  - token embedding -> LayerNorm (bez biasu) -> dropout
  - N bloků s pre-norm uspořádáním:
        x = x + Attention(LN(x))       # self-attention s RoPE
        x = x + GeGLU-MLP(LN(x))
  - závěrečná LayerNorm
  - MLM hlava: Linear -> GELU -> LayerNorm -> dekodér sdílející váhy s embeddingem

Attention:
  - RoPE (rotary position embedding): pozice se kódují rotací dotazů a klíčů, attention
    tak přirozeně pracuje s relativní vzdáleností tokenů. Žádná tabulka pozic se neučí.
  - Výpočet přes torch.nn.functional.scaled_dot_product_attention (na GPU FlashAttention).
  - Pro analýzu vysvětlitelnosti (output_attentions=True) se attention počítá explicitně
    a vrací matice tvaru (batch, hlavy, délka, délka) pro každou vrstvu.
  - Volitelně lokální attention (posuvné okno) ve vybraných vrstvách, jako ModernBERT.
    Výchozí nastavení je globální attention ve všech vrstvách, aby srovnání tokenizací
    nezáviselo na tom, kolik bp odpovídá oknu v tokenech.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG_FILE = "model_config.json"
WEIGHTS_FILE = "model.pt"


@dataclass
class EncoderConfig:
    vocab_size: int
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    intermediate_size: int = 1536
    max_position: int = 1056  # max. tokenů včetně [CLS] a [SEP]
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    norm_eps: float = 1e-5
    pad_token_id: int = 0
    # lokální attention: každá `global_every`-tá vrstva globální, ostatní s oknem `local_window`
    global_every: int = 1  # 1 = všechny vrstvy globální
    local_window: int = 128  # šířka okna lokální attention (v tokenech)
    init_std: float = 0.02
    extra: dict = field(default_factory=dict)  # volné pole pro poznámky (tokenizér apod.)

    def __post_init__(self):
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size musí být dělitelné num_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("dimenze hlavy musí být sudá (RoPE)")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def is_global(self, layer: int) -> bool:
        return layer % self.global_every == 0


# --------------------------------------------------------------------------- RoPE


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_position).float()
        freqs = torch.outer(t, inv_freq)  # (L, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (L, D)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, length: int, dtype: torch.dtype):
        return self.cos[:length].to(dtype), self.sin[:length].to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, L, D), cos/sin: (L, D)"""
    return x * cos + _rotate_half(x) * sin


# --------------------------------------------------------------------------- bloky


class Attention(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, attn_bias, output_attentions: bool):
        b, n, _ = x.shape
        h, d = self.cfg.num_heads, self.cfg.head_dim
        q, k, v = self.qkv(x).view(b, n, 3, h, d).permute(2, 0, 3, 1, 4)  # 3 × (B, H, L, D)
        # RoPE v přesnosti q (pod autocastem fp16/bf16), jinak by se q převedlo na fp32
        # a PyTorch by nemohl použít úsporné jádro attention
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        p = self.cfg.attention_dropout if self.training else 0.0
        if output_attentions:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
            if attn_bias is not None:  # bool maska: True = smí se dívat
                scores = scores.masked_fill(~attn_bias, torch.finfo(scores.dtype).min)
            probs = scores.softmax(dim=-1)
            ctx = F.dropout(probs, p=p, training=self.training) @ v
        else:
            # bez masky (žádný padding) může PyTorch použít FlashAttention nebo
            # memory-efficient attention, které nedrží celou matici L × L v paměti
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=p)
            probs = None
        ctx = ctx.transpose(1, 2).reshape(b, n, h * d)
        return self.out(ctx), probs


class GeGLU(nn.Module):
    """Feed-forward se „bránou“: GELU(x W1) ⊙ (x W2), pak projekce zpět."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        a, gate = self.wi(x).chunk(2, dim=-1)
        return self.wo(self.drop(F.gelu(a) * gate))


class Block(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.mlp = GeGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, attn_bias, output_attentions):
        a, probs = self.attn(self.attn_norm(x), cos, sin, attn_bias, output_attentions)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.mlp_norm(x)))
        return x, probs


# --------------------------------------------------------------------------- modely


@dataclass
class EncoderOutput:
    last_hidden_state: torch.Tensor  # (B, L, H)
    hidden_states: list[torch.Tensor] | None = None  # embedding + výstup každé vrstvy
    attentions: list[torch.Tensor] | None = None  # (B, heads, L, L) pro každou vrstvu


class DNAEncoder(nn.Module):
    """Samotný encoder: tokeny -> kontextové vektory."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.embed_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.embed_drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_position, cfg.rope_theta)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.num_layers))
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)

    def _attn_bias(self, attention_mask, length, device, global_layer):
        """
        Bool maska (B, 1, 1|L, L), True = smí se dívat (ne padding, v okně lokální attention).
        Vrací None, když není co maskovat (globální vrstva, žádný padding): pak běží
        nejrychlejší a paměťově nejúspornější jádro attention.
        """
        has_padding = not bool(attention_mask.all())
        if global_layer and not has_padding:
            return None
        keep = attention_mask[:, None, None, :].bool()  # (B,1,1,L)
        if not global_layer:
            i = torch.arange(length, device=device)
            window = (i[:, None] - i[None, :]).abs() <= self.cfg.local_window // 2
            keep = keep & window[None, None]
        return keep

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> EncoderOutput:
        b, n = input_ids.shape
        if n > self.cfg.max_position:
            raise ValueError(f"Sekvence má {n} tokenů, model zvládne {self.cfg.max_position}")
        if attention_mask is None:
            attention_mask = (input_ids != self.cfg.pad_token_id).long()
        x = self.embed_drop(self.embed_norm(self.embed(input_ids)))
        cos, sin = self.rope(n, x.dtype)
        biases = {}
        hidden, attns = ([x] if output_hidden_states else None), ([] if output_attentions else None)
        for i, layer in enumerate(self.layers):
            g = self.cfg.is_global(i)
            if g not in biases:
                biases[g] = self._attn_bias(attention_mask, n, x.device, g)
            x, probs = layer(x, cos, sin, biases[g], output_attentions)
            if output_hidden_states:
                hidden.append(x)
            if output_attentions:
                attns.append(probs)
        x = self.final_norm(x)
        if output_hidden_states:
            hidden[-1] = x
        return EncoderOutput(last_hidden_state=x, hidden_states=hidden, attentions=attns)


class MLMHead(nn.Module):
    def __init__(self, cfg: EncoderConfig, embedding: nn.Embedding):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.embedding = embedding  # sdílené váhy (weight tying)

    def forward(self, h):
        return F.linear(self.norm(F.gelu(self.dense(h))), self.embedding.weight, self.bias)


class DNAForMaskedLM(nn.Module):
    """Encoder + MLM hlava. Vrací logity pro každou pozici (nebo jen pro maskované)."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = DNAEncoder(cfg)
        self.head = MLMHead(cfg, self.encoder.embed)
        self.apply(self._init_weights)
        # výstupní projekce bloků menší, ať se reziduální proud na začátku nerozkmitá
        for layer in self.encoder.layers:
            nn.init.normal_(layer.attn.out.weight, std=cfg.init_std / math.sqrt(2 * cfg.num_layers))
            nn.init.normal_(layer.mlp.wo.weight, std=cfg.init_std / math.sqrt(2 * cfg.num_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    def forward(self, input_ids, attention_mask=None, labels=None, **kw):
        out = self.encoder(input_ids, attention_mask, **kw)
        h = out.last_hidden_state
        if labels is not None:  # logity jen pro maskované pozice (úspora paměti)
            sel = labels != -100
            logits = self.head(h[sel])
            loss = F.cross_entropy(logits.float(), labels[sel])
            return {"loss": loss, "logits": logits, "labels": labels[sel], "encoder": out}
        return {"logits": self.head(h), "encoder": out}

    # --- parametry, ukládání ----------------------------------------------------

    def num_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        emb = self.encoder.embed.weight.numel()
        return {"total": total, "embedding": emb, "non_embedding": total - emb}

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, CONFIG_FILE), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        torch.save(self.state_dict(), os.path.join(directory, WEIGHTS_FILE))

    @classmethod
    def load(cls, directory: str, map_location="cpu") -> DNAForMaskedLM:
        with open(os.path.join(directory, CONFIG_FILE)) as f:
            cfg = EncoderConfig(**json.load(f))
        model = cls(cfg)
        state = torch.load(os.path.join(directory, WEIGHTS_FILE), map_location=map_location)
        model.load_state_dict(state)
        return model
