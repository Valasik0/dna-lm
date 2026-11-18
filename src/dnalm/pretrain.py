"""
MLM předtrénink jednoho modelu podle YAML konfigurace.

    python -m dnalm.pretrain --config configs/pretrain/S_bpe_4096.yaml
    python -m dnalm.pretrain --config ... --set train.budget_nt=5e6   # rychlý test

Délka tréninku se zadává rozpočtem viděných nukleotidů (train.budget_nt), ne epochami,
aby byla srovnatelná napříč tokenizacemi. Počet kroků = budget_nt / (batch × accum × okno).

Výstupy v P.checkpoints/<name>/:
  last.pt         stav tréninku pro pokračování (model, optimizér, scheduler, krok, RNG)
  best/, final/   uložený model (model_config.json + model.pt) a tokenizér
  metrics.csv     průběh tréninku a validace
  config.yaml     použitá konfigurace
Spuštění se stejnou konfigurací automaticky pokračuje od last.pt (např. po pádu Colabu).
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os

# méně fragmentace paměti GPU (musí být nastaveno před prvním použitím CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from dnalm.data.dataset import RandomDNADataset, WindowTokenDataset
from dnalm.models.factory import build_model, load_model_config
from dnalm.models.mlm import BPNMeter, MLMCollator
from dnalm.tokenizers import build_tokenizer, load_config, load_tokenizer, needs_training
from dnalm.utils.paths import Paths, get_paths

DEFAULTS = {
    "data": {"window": 1024, "source": "genome", "rc_prob": 0.0, "val_windows": 2000},
    "train": {
        "budget_nt": 2.0e9,
        "batch_size": 128,
        "grad_accum": 1,
        "lr": 6.0e-4,
        "weight_decay": 0.01,
        "betas": [0.9, 0.98],
        "warmup_frac": 0.05,
        "min_lr_frac": 0.1,
        "grad_clip": 1.0,
        "mask_prob": 0.15,
        "precision": "auto",
        "eval_every": 1000,
        "save_every": 1000,
        "log_every": 50,
        "num_workers": 2,
        "seed": 42,
    },
}


# --------------------------------------------------------------------------- konfigurace


def _merge(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _coerce_numbers(x):
    """PyYAML (YAML 1.1) čte např. 5.0e7 nebo 2e9 jako text; převedeme to na číslo."""
    if isinstance(x, dict):
        return {k: _coerce_numbers(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_coerce_numbers(v) for v in x]
    if isinstance(x, str):
        try:
            return float(x) if any(c in x for c in ".eE") else int(x)
        except ValueError:
            return x
    return x


def _parse_value(v: str):
    import yaml

    return _coerce_numbers(yaml.safe_load(v))


def load_run_config(path: str | None = None, overrides: list[str] | None = None, **cfg) -> dict:
    import yaml

    if path:
        with open(path) as f:
            cfg = _merge(yaml.safe_load(f), cfg)
    cfg = _merge(DEFAULTS, cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = _parse_value(value)
    cfg = _coerce_numbers(cfg)
    for k in ("name", "tokenizer", "model"):
        if k not in cfg:
            raise ValueError(f"V konfiguraci chybí '{k}'")
    return cfg


def resolve_tokenizer(name: str | dict, P: Paths):
    """Uložený tokenizér z P.tokenizer(name), jinak z configs/tokenizer/<name>.yaml.
    Lze zadat i přímo slovník s konfigurací (např. {"type": "kmer", "k": 3, "stride": 1})."""
    if isinstance(name, dict):
        tok = build_tokenizer(name)
        if needs_training(tok):
            raise ValueError("Tokenizér zadaný slovníkem nesmí vyžadovat trénink")
        return tok
    saved = P.tokenizer(name)
    if (saved / "tokenizer_config.json").exists():
        return load_tokenizer(str(saved))
    tok = build_tokenizer(load_config(str(P.configs / "tokenizer" / f"{name}.yaml")))
    if needs_training(tok):
        raise FileNotFoundError(f"Tokenizér {name} není natrénovaný ({saved}), spusť notebook 03")
    return tok


def resolve_model_config(model: str | dict, P: Paths) -> dict:
    if isinstance(model, dict):
        return model
    return load_model_config(str(P.configs / "model" / f"{model}.yaml"))


# --------------------------------------------------------------------------- pomocné


class ResumableSampler(Sampler):
    """Náhodné pořadí pro každou epochu (seed + epocha); umí začít uprostřed (resume)."""

    def __init__(self, n: int, seed: int, start: int = 0):
        self.n, self.seed, self.start = n, seed, start

    def __iter__(self):
        i = self.start
        while True:
            epoch, offset = divmod(i, self.n)
            perm = np.random.default_rng((self.seed, epoch)).permutation(self.n)
            for j in perm[offset:]:
                yield int(j)
            i = (epoch + 1) * self.n

    def __len__(self):
        return 2**62


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(precision: str, device: torch.device):
    """
    auto: bf16 na GPU, které ho umí (A100, L4, H100), fp16 na starších (T4, V100),
    fp32 na Macu (MPS) a CPU. fp16 potřebuje GradScaler (viz train), bf16 ne.
    """
    if precision == "fp32":
        return None
    if device.type == "cuda":
        # bf16 jen tam, kde ho GPU umí hardwarově (Ampere a novější: A100, L4, H100);
        # torch.cuda.is_bf16_supported() vrací True i pro emulaci na T4, která je pomalá
        native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
        if precision == "bf16" or (precision == "auto" and native_bf16):
            return torch.bfloat16
        if precision in ("auto", "fp16"):
            return torch.float16
    if precision == "bf16" and device.type == "cpu":
        return torch.bfloat16
    return None


def lr_at(step: int, total: int, cfg: dict) -> float:
    warm = max(1, int(cfg["warmup_frac"] * total))
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    floor = cfg["min_lr_frac"]
    return cfg["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def make_datasets(cfg: dict, tok, P: Paths):
    d, seed = cfg["data"], cfg["train"]["seed"]
    if d["source"] == "random":
        n = int(d.get("random_windows", 100_000))
        train = RandomDNADataset(tok, n=n, length=d["window"], seed=seed)
        val = RandomDNADataset(tok, n=d["val_windows"], length=d["window"], seed=seed + 10_000)
        return train, val
    fasta = str(d.get("fasta") or P.fasta)
    train = WindowTokenDataset(
        fasta,
        str(d.get("train_windows") or P.windows("train")),
        tok,
        rc_prob=d["rc_prob"],
        max_windows=d.get("max_train_windows"),
        seed=seed,
    )
    val = WindowTokenDataset(
        fasta,
        str(d.get("val_windows_file") or P.windows("val")),
        tok,
        max_windows=d["val_windows"],
        seed=seed,
    )
    return train, val


@torch.no_grad()
def evaluate(model, loader, collator, tok, device, dtype) -> dict:
    model.eval()
    collator.reseed()
    meter = BPNMeter(tok)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            out = model(**batch)
        meter.update(out["logits"], out["labels"])
    model.train()
    return meter.compute()


METRICS_HEADER = ["step", "split", "loss", "bpn", "accuracy", "lr", "nt_per_s", "seen_nt"]


def ensure_metrics_header(path: Path) -> None:
    """
    Zajistí hlavičku v metrics.csv. Když běh spadne hned po vytvoření souboru (např. OOM),
    hlavička se nemusí zapsat na disk a další běh by připisoval data bez ní.
    """
    header = ",".join(METRICS_HEADER)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(header + "\n")
        return
    content = path.read_text()
    if not content.startswith("step,"):
        path.write_text(header + "\n" + content)


def _save_atomic(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def _set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])


# --------------------------------------------------------------------------- trénink


def train(cfg: dict, P: Paths | None = None, out_dir: str | None = None, verbose: bool = True):
    import yaml

    P = P or get_paths(window=cfg["data"]["window"])
    tc = cfg["train"]
    out = Path(out_dir) if out_dir else P.checkpoints / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    seed = tc["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tok = resolve_tokenizer(cfg["tokenizer"], P)
    model = build_model(resolve_model_config(cfg["model"], P), tok, window_bp=cfg["data"]["window"])
    device = pick_device()
    dtype = pick_dtype(tc["precision"], device)
    model.to(device)

    tokens_per_step = tc["batch_size"] * tc["grad_accum"]
    total_steps = max(1, int(tc["budget_nt"] // (tokens_per_step * cfg["data"]["window"])))

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": tc["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=tc["lr"],
        betas=tuple(tc["betas"]),
        fused=device.type == "cuda",
    )

    # fp16 má malý rozsah: ztráta se před backward násobí, aby gradienty nepodtekly k nule
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)

    step, best_bpn = 0, float("inf")
    ckpt = out / "last.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        step, best_bpn = state["step"], state.get("best_bpn", best_bpn)
        _set_rng_state(state["rng"])
        if verbose:
            print(f"Pokračuji od kroku {step} ({ckpt})")

    train_ds, val_ds = make_datasets(cfg, tok, P)
    collate = MLMCollator(tok, mask_prob=tc["mask_prob"])
    # validace: všechny vybrané pozice -> [MASK] (bez 10 % ponechaných a 10 % náhodných),
    # jinak by metrika obsahovala „nápovědu“ z ponechaných tokenů (na náhodné DNA 1,91 bitu místo 2)
    val_collate = MLMCollator(tok, mask_prob=tc["mask_prob"], p_mask=1.0, p_random=0.0, seed=seed)
    nw = tc["num_workers"]
    sampler = ResumableSampler(len(train_ds), seed, start=step * tokens_per_step)
    train_loader = DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        sampler=sampler,
        collate_fn=collate,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc["batch_size"], collate_fn=val_collate, num_workers=0
    )

    if verbose:
        n = model.num_parameters()
        print(
            f"{cfg['name']}: tokenizér {tok.name} ({tok.vocab_size} tokenů), "
            f"parametry {n['total'] / 1e6:.2f} M "
            f"(bez embeddingů {n['non_embedding'] / 1e6:.2f} M)\n"
            f"zařízení {device}, přesnost {dtype or 'fp32'}, kroků {total_steps:,}, "
            f"dávka {tokens_per_step} oken = {tokens_per_step * cfg['data']['window']:,} bp"
        )

    log_path = out / "metrics.csv"
    ensure_metrics_header(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)

    model.train()
    it = iter(train_loader)
    meter = BPNMeter(tok)
    t0, nt_since = time.time(), 0
    while step < total_steps:
        lr = lr_at(step, total_steps, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(tc["grad_accum"]):
            batch = {k: v.to(device, non_blocking=True) for k, v in next(it).items()}
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                res = model(**batch)
            scaler.scale(res["loss"] / tc["grad_accum"]).backward()
            meter.update(res["logits"].detach(), res["labels"])
            nt_since += batch["input_ids"].shape[0] * cfg["data"]["window"]
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        seen = step * tokens_per_step * cfg["data"]["window"]

        if step % tc["log_every"] == 0 or step == total_steps:
            m = meter.compute()
            speed = nt_since / max(time.time() - t0, 1e-9)
            log.writerow([step, "train", m["loss"], m["bpn"], m["accuracy"], lr, speed, seen])
            log_f.flush()
            if verbose:
                print(
                    f"krok {step:>7,}/{total_steps:,}  loss {m['loss']:.4f}  bpn {m['bpn']:.4f}  "
                    f"acc {m['accuracy']:.3f}  lr {lr:.2e}  {speed / 1e6:.2f} M bp/s"
                )
            meter.reset()
            t0, nt_since = time.time(), 0

        last = step == total_steps
        if step % tc["eval_every"] == 0 or last:
            v = evaluate(model, val_loader, val_collate, tok, device, dtype)
            log.writerow([step, "val", v["loss"], v["bpn"], v["accuracy"], lr, "", seen])
            log_f.flush()
            if verbose:
                print(
                    f"  validace: loss {v['loss']:.4f}  bpn {v['bpn']:.4f}  acc {v['accuracy']:.3f}"
                )
            if v["bpn"] < best_bpn:
                best_bpn = v["bpn"]
                model.save(str(out / "best"))
                tok.save(str(out / "best" / "tokenizer"))
        if step % tc["save_every"] == 0 or last:
            _save_atomic(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "best_bpn": best_bpn,
                    "rng": _rng_state(),
                },
                ckpt,
            )

    model.save(str(out / "final"))
    tok.save(str(out / "final" / "tokenizer"))
    log_f.close()
    return {"steps": step, "best_val_bpn": best_bpn, "out_dir": str(out)}


def main(argv=None):
    p = argparse.ArgumentParser(description="MLM předtrénink DNA modelu")
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="přepsání hodnot, např. train.lr=3e-4")
    a = p.parse_args(argv)
    cfg = load_run_config(a.config, a.set)
    print(train(cfg))


if __name__ == "__main__":
    main()
