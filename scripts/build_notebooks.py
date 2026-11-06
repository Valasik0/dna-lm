# ruff: noqa: E501  (dlouhé řádky jsou obsah buněk notebooků)
"""
Generuje notebooky v notebooks/ ze zdrojů v tomto souboru.

Všechny notebooky začínají stejnou úvodní buňkou (BOOTSTRAP), která se nemění:
v Colabu stáhne kód z GitHubu, lokálně použije repo, ve kterém notebook leží,
a pak zavolá dnalm.utils.colab.setup(). Nastavení prostředí se upravuje v kódu
(src/dnalm/utils/), ne v notebookách.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def bootstrap(genome: bool = True) -> str:
    return f"""
# Úvodní buňka (stejná ve všech notebookách, neměnit).
# Colab: stáhne aktuální kód z GitHubu. Lokálně (Mac, školní GPU): použije nainstalované repo.
import importlib.util, sys
if importlib.util.find_spec("google.colab"):
    !git clone -q https://github.com/Valasik0/dna-lm.git /content/dna-lm 2>/dev/null || git -C /content/dna-lm pull -q
    !git -C /content/dna-lm log --oneline -1
    if "/content/dna-lm/src" not in sys.path:
        sys.path.insert(0, "/content/dna-lm/src")

from dnalm.utils.colab import setup
P = setup(window=WINDOW, genome={genome})
"""


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": s.strip("\n").splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_MD = md("""
## Prostředí

Funguje v Colabu (prohlížeč i VS Code) i lokálně. V Colabu připojí Google Drive,
stáhne kód z GitHubu a zkopíruje genom na rychlý lokální disk. Jinde se cesty
řídí proměnnou `DNALM_ROOT`. Všechny cesty jsou v objektu `P` (`P.fasta`,
`P.windows("train")`, `P.tokenizer("bpe_600")`, `P.figures`, ...).

Ve VS Code s Colab rozšířením: pokud se Drive nepřipojí, použij příkaz
`Colab: Mount Google Drive to Server...` a buňku spusť znovu.
""")


# =============================================================================
# 01 Stažení genomu
# =============================================================================

NB01_HELPERS = r'''
import gzip
import hashlib
import json
import os
import shutil
import socket
import time
import urllib.request
from datetime import datetime, timezone

from tqdm.auto import tqdm


def check_dns(urls):
    """Vypíše, které servery jde přeložit na IP (diagnostika síťových chyb)."""
    for u in urls:
        host = u.split("/")[2]
        try:
            print(f"OK     {host} -> {socket.gethostbyname(host)}")
        except OSError as e:
            print(f"CHYBA  {host}: {e}")


def with_mirrors(func, mirrors, retries=3, wait=10):
    """Zavolá func(base_url) postupně pro každé zrcadlo, s několika pokusy. Vrací base_url."""
    last = None
    for base in mirrors:
        for attempt in range(1, retries + 1):
            try:
                print(f"[{base.split('/')[2]}] pokus {attempt}/{retries}")
                func(base)
                return base
            except OSError as e:  # URLError, DNS, timeout, přerušené spojení
                last = e
                print(f"   selhalo: {e}")
                time.sleep(wait)
    raise RuntimeError(f"Žádné zrcadlo nefunguje, poslední chyba: {last}")


def download(url, dest):
    """Stáhne soubor s progress barem; pokud už existuje celý, přeskočí."""
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        if os.path.exists(dest) and os.path.getsize(dest) == total:
            print(f"Už staženo: {dest}")
            return
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)
        ) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def expected_md5(md5_url, filename):
    text = urllib.request.urlopen(md5_url).read().decode()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == filename:
            return parts[0]
    raise ValueError(f"{filename} není v {md5_url}")


def filter_fasta(src_gz, dst_fa, keep):
    """
    Projde gzip FASTA jedním průchodem, ponechá jen chromozomy z `keep`,
    zapíše nekomprimovanou FASTA, vytvoří k ní .fai index a vrátí statistiky.
    """
    keep = set(keep)
    stats, fai = [], []
    cur = None
    offset = 0

    def close(c):
        if c is not None:
            stats.append({k: c[k] for k in ("chrom", "length", "n_count", "softmasked")})
            fai.append(f"{c['chrom']}\t{c['length']}\t{c['offset']}\t{c['lb']}\t{c['lb'] + 1}")

    with gzip.open(src_gz, "rt") as fin, open(dst_fa, "w") as fout:
        writing = False
        for line in tqdm(fin, desc="Filtrování", unit=" řádků", mininterval=5):
            if line.startswith(">"):
                close(cur)
                cur = None
                name = line[1:].split()[0]
                writing = name in keep
                if writing:
                    header = f">{name}\n"
                    fout.write(header)
                    offset += len(header)
                    cur = {"chrom": name, "length": 0, "n_count": 0, "softmasked": 0,
                           "offset": offset, "lb": None, "short_seen": False}
                continue
            if not writing:
                continue
            seq = line.rstrip("\n")
            n = len(seq)
            if cur["lb"] is None:
                cur["lb"] = n
            elif cur["short_seen"] or n > cur["lb"]:
                raise ValueError(f"Nestejná délka řádků v {cur['chrom']}, .fai by byl špatně")
            if n < cur["lb"]:
                cur["short_seen"] = True
            cur["length"] += n
            cur["n_count"] += seq.count("N") + seq.count("n")
            cur["softmasked"] += sum(seq.count(b) for b in "acgtn")
            fout.write(seq + "\n")
            offset += n + 1
        close(cur)

    with open(dst_fa + ".fai", "w") as f:
        f.write("\n".join(fai) + "\n")
    missing = keep - {s["chrom"] for s in stats}
    if missing:
        raise ValueError(f"Chybí chromozomy: {sorted(missing)}")
    return stats
'''

NB01 = notebook(
    [
        md("""
# 01 Stažení referenčního genomu (GRCh38 / hg38), úkol 1.1

Jednorázový notebook. Stáhne **UCSC hg38 analysis set** (bez alternativních kontigů,
repetice soft-maskované malými písmeny), ověří MD5, ponechá jen hlavní chromozomy,
vytvoří `.fai` index a uloží vše do `P.genome_dir` (v Colabu na Google Drive).

Stačí CPU runtime. Trvá zhruba 10–20 minut, na Drive zabere asi 4 GB.
"""),
        md("## 1. Nastavení"),
        code("""
WINDOW = 1024   # jen kvůli jednotné úvodní buňce, tady se nepoužívá

# Hlavní server UCSC a evropské zrcadlo (stejné soubory); zkouší se v tomto pořadí
MIRRORS = [
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/analysisSet",
    "https://hgdownload-euro.soe.ucsc.edu/goldenPath/hg38/bigZips/analysisSet",
]
FA_GZ = "hg38.analysisSet.fa.gz"
OUT_FA = "hg38.main.fa"

# chr1–22, X, Y a mitochondrie. Výběr pro trénink (1–22 + X) se dělá až při generování oken.
KEEP = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
"""),
        SETUP_MD,
        code(bootstrap(genome=False)),
        md("## 2. Pomocné funkce"),
        code(NB01_HELPERS),
        md("## 3. Stažení a kontrola MD5"),
        code("""
WORK_DIR = str(P.scratch / "genome")     # v Colabu rychlý lokální disk, jinde = cílová složka
OUT_DIR = str(P.genome_dir)
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

check_dns(MIRRORS)
gz_path = os.path.join(WORK_DIR, FA_GZ)
BASE_URL = with_mirrors(lambda base: download(f"{base}/{FA_GZ}", gz_path), MIRRORS)

md5_expected = expected_md5(f"{BASE_URL}/md5sum.txt", FA_GZ)
md5_actual = md5sum(gz_path)
print("očekávané MD5:", md5_expected)
print("spočítané MD5:", md5_actual)
assert md5_actual == md5_expected, "MD5 nesedí, soubor smaž a stáhni znovu"
"""),
        md("## 4. Filtrování chromozomů, index a statistiky"),
        code("""
import pandas as pd

fa_path = os.path.join(WORK_DIR, OUT_FA)
t0 = time.time()
stats = filter_fasta(gz_path, fa_path, KEEP)
print(f"Hotovo za {time.time() - t0:.0f} s")

df = pd.DataFrame(stats)
df["n_frac"] = df["n_count"] / df["length"]
df["softmasked_frac"] = df["softmasked"] / df["length"]
df = df.set_index("chrom").loc[KEEP]
print(f"Celkem: {df['length'].sum():,} bp, N: {df['n_count'].sum():,}, "
      f"soft-masked: {df['softmasked'].sum() / df['length'].sum():.1%}")
df
"""),
        md("## 5. Uložení"),
        code("""
meta = {
    "source": f"{BASE_URL}/{FA_GZ}",
    "assembly": "GRCh38 (UCSC hg38 analysis set, no alt contigs)",
    "md5_fa_gz": md5_actual,
    "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "chromosomes": KEEP,
    "total_bp": int(df["length"].sum()),
    "softmasking": "lowercase = RepeatMasker + Tandem Repeats Finder",
}
with open(os.path.join(WORK_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
df.to_csv(os.path.join(WORK_DIR, "chrom_stats.csv"))

if os.path.abspath(WORK_DIR) != os.path.abspath(OUT_DIR):
    for name in [FA_GZ, OUT_FA, OUT_FA + ".fai", "metadata.json", "chrom_stats.csv"]:
        src = os.path.join(WORK_DIR, name)
        print(f"Kopíruji {name} ({os.path.getsize(src) / 1e9:.2f} GB)...")
        shutil.copy(src, os.path.join(OUT_DIR, name))
print(sorted(os.listdir(OUT_DIR)))
"""),
        md("## 6. Kontrola: čtení přes pyfaidx"),
        code("""
from pyfaidx import Fasta

genome = Fasta(str(P.fasta_stored))   # použije uložený .fai
print(list(genome.keys()))
print(len(genome["chr1"]), "bp na chr1")
print(genome["chr1"][1_000_000:1_000_120])
"""),
    ]
)


# =============================================================================
# 02 Okna
# =============================================================================

NB02 = notebook(
    [
        md("""
# 02 Generování oken z genomu (úkol 1.3)

Rozřeže genom na okna o délce `WINDOW`, vyřadí okna s více než 1 % N a rozdělí je
podle chromozomů: **trénink** chr1–22 a X kromě chr20 a chr21, **validace** chr20,
**test** chr21. Ukládají se jen souřadnice oken a jejich GC obsah, podíl N a repetic,
sekvence se vyřezávají až při tokenizaci. Stačí CPU runtime, trvá pár minut.

Předpoklad: hotový notebook 01.
"""),
        md("## 1. Nastavení"),
        code("""
WINDOW = 1024          # délka okna v bp
MAX_N_FRAC = 0.01      # max. podíl N v okně
"""),
        SETUP_MD,
        code(bootstrap()),
        md("## 2. Generování oken"),
        code("""
import time
from dnalm.data.genome import WindowConfig, make_windows

cfg = WindowConfig(window=WINDOW, max_n_frac=MAX_N_FRAC)
t0 = time.time()
summary = make_windows(str(P.fasta), str(P.windows_dir), cfg)
print(f"Hotovo za {time.time() - t0:.0f} s")
"""),
        md("## 3. Přehled"),
        code("""
import pandas as pd
from dnalm.data.genome import load_windows

rows = []
for split, info in summary["splits"].items():
    df = load_windows(P.windows(split))
    chroms = info["chromosomes"]
    rows.append({
        "split": split,
        "chromozomy": ", ".join(chroms) if len(chroms) < 3 else f"{len(chroms)} chromozomů",
        "posun_bp": info["stride"],
        "oken": len(df),
        "vyřazeno_N": info["windows_total"] - info["windows_kept"],
        "GC_průměr": round(df.gc_frac.mean(), 3),
        "repetice_průměr": round(df.repeat_frac.mean(), 3),
    })
overview = pd.DataFrame(rows).set_index("split")
overview.to_csv(P.windows_dir / "overview.csv")
overview
"""),
        md("## 4. Kontrola: sekvence několika oken"),
        code("""
from dnalm.data.genome import GenomeWindows

val = GenomeWindows(str(P.fasta), str(P.windows("val")))
print(len(val), "validačních oken")
for i in [0, len(val) // 2, len(val) - 1]:
    s = val[i]
    print(val.chrom[i], val.start[i], val.end[i], len(s), s[:60] + "...")
print("reverzní komplement prvního okna:", val.get(0, reverse=True)[:60] + "...")
"""),
    ]
)


# =============================================================================
# 03 Tokenizéry
# =============================================================================

NB03 = notebook(
    [
        md("""
# 03 Tokenizéry: trénink a analýza (úkoly 1.5–1.9)

1. Připraví tokenizéry z `configs/tokenizer/`. BPE a Unigram se natrénují na náhodném
   vzorku trénovacích oken (100 tisíc oken, ~2 min a ~7 GB RAM na tokenizér),
   nukleotidy a k-mery se jen uloží.
2. Spočítá jejich statistiky na vzorku validačních oken (chr20).
3. Vykreslí grafy do práce (PDF + PNG).

Předpoklad: hotový notebook 02. Stačí CPU runtime.
"""),
        md("## 1. Nastavení"),
        code("""
WINDOW = 1024

# Hlavní tokenizace + kontrolní překrývající se 6-mery. Unigram (T6) je volitelný: přidej "unigram_4096".
TOKENIZERS = ["nucleotide", "kmer6_overlap", "kmer6", "bpe_600", "bpe_4096"]
RETRAIN = False      # True = natrénovat znovu, i když už tokenizér uložený je
N_SAMPLE = 2000      # počet validačních oken pro statistiky
"""),
        SETUP_MD,
        code(bootstrap()),
        md("## 2. Příprava a trénink tokenizérů"),
        code("""
import time
from dnalm.tokenizers import load_tokenizer
from dnalm.tokenizers.train import prepare_tokenizer

P.tokenizers_dir.mkdir(parents=True, exist_ok=True)
tokenizers = {}
for name in TOKENIZERS:
    path = P.tokenizer(name)
    if (path / "tokenizer_config.json").exists() and not RETRAIN:
        tokenizers[name] = load_tokenizer(str(path))
        print(f"[{name}] načten z {path}")
    else:
        t0 = time.time()
        tokenizers[name] = prepare_tokenizer(
            str(P.configs / "tokenizer" / f"{name}.yaml"),
            str(P.fasta), str(P.windows("train")), str(P.tokenizers_dir),
        )
        print(f"[{name}] hotovo za {time.time() - t0:.0f} s")
for name, tok in tokenizers.items():
    print(f"{name:15s} slovník {tok.vocab_size:>6,} tokenů")
"""),
        md("## 3. Vzorek validačních oken"),
        code("""
import numpy as np
from dnalm.data.genome import GenomeWindows

# vzorek validačních oken (chr20); raw = se zachovanými malými písmeny (repetice)
val_raw = GenomeWindows(str(P.fasta), str(P.windows("val")), uppercase=False)
rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(val_raw), size=min(N_SAMPLE, len(val_raw)), replace=False))
raw_seqs = [val_raw[int(i)] for i in idx]
seqs = [s.upper() for s in raw_seqs]
print(f"{len(seqs)} validačních oken po {len(seqs[0])} bp")
"""),
        md("""
## 4. Souhrnné statistiky

- `tokens_per_kb`: kolik tokenů dá 1 000 bp (délka vstupu modelu, a tedy cena výpočtu)
- `nt_per_token`: průměrný počet bází na token
- `single_nt_share`: podíl tokenů o délce 1 bp
- `vocab_used_share`: jaká část slovníku se ve vzorku skutečně použije
- `tokens_changed_mean`: kolik tokenů změní jedna bodová mutace (stabilita tokenizace)
- `resegmented_nt_mean`: jak dlouhý úsek se kolem mutace přesegmentuje (bp)
- `len_in_repeats`, `len_outside_repeats`: průměrná délka tokenu v repeticích a mimo ně
"""),
        code("""
import pandas as pd
from dnalm.tokenizers.stats import mutation_stability, repeat_token_lengths, summary
from dnalm.utils.plotting import label_for

rows = []
for name, tok in tokenizers.items():
    r = summary(tok, seqs)
    r.update({k: v for k, v in mutation_stability(tok, seqs[:200]).items() if k != "tokenizer"})
    r.update({k: v for k, v in repeat_token_lengths(tok, raw_seqs).items() if k != "tokenizer"})
    rows.append(r)
stats = pd.DataFrame(rows).set_index("tokenizer")
P.results("tokenizers").mkdir(parents=True, exist_ok=True)
stats.to_csv(P.results("tokenizers") / "tokenizer_stats.csv")

view = stats[["vocab_size", "tokens_per_kb", "nt_per_token", "token_len_max", "single_nt_share",
              "vocab_used_share", "tokens_changed_mean", "resegmented_nt_mean",
              "len_in_repeats", "len_outside_repeats"]].rename(index=label_for)
view.round(3)
"""),
        md("## 5. Grafy"),
        code("""
import matplotlib.pyplot as plt
from dnalm.utils.plotting import apply_style, color_for, label_for, save_figure
apply_style()

# Tokeny na 1 kb: jedna míra napříč tokenizacemi -> sloupcový graf, hodnota přímo u sloupce
order = stats.sort_values("tokens_per_kb", ascending=False).index
fig, ax = plt.subplots(figsize=(6, 3.2))
bars = ax.barh([label_for(n) for n in order], stats.loc[order, "tokens_per_kb"],
               color=[color_for(n) for n in order], height=0.6)
ax.bar_label(bars, labels=[f"{v:,.0f}".replace(",", " ") for v in stats.loc[order, "tokens_per_kb"]],
             padding=4, fontsize=9, color="#0b0b0b")
ax.invert_yaxis()
ax.set_xlabel("tokenů na 1 000 bp")
ax.set_title("Délka vstupu modelu podle tokenizace", loc="left")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, stats["tokens_per_kb"].max() * 1.15)
save_figure(fig, "tokens_per_kb", str(P.figures))
plt.show()
"""),
        code("""
from dnalm.tokenizers.stats import token_length_distribution

# Rozdělení délek tokenů jen pro naučené tokenizace (u nukleotidů a k-merů je délka pevná).
# Osa jen do MAX_LEN bp, podíl delších tokenů je v legendě (dlouhé tokeny = repetice).
learned = [n for n in tokenizers if n.startswith(("bpe", "unigram"))]
MAX_LEN = 20
fig, ax = plt.subplots(figsize=(6, 3.4))
for name in learned:
    dist = token_length_distribution(tokenizers[name], seqs)
    total = sum(dist.values())
    xs = list(range(1, MAX_LEN + 1))
    tail = sum(c for length, c in dist.items() if length > MAX_LEN) / total
    ax.plot(xs, [dist.get(x, 0) / total for x in xs], marker="o", markersize=4,
            color=color_for(name), label=f"{label_for(name)}, delší než {MAX_LEN} bp: {tail:.1%}")
ax.set_xlabel("délka tokenu (bp)")
ax.set_ylabel("podíl výskytů tokenů")
ax.set_xticks(range(1, MAX_LEN + 1, 2))
ax.set_title("Rozdělení délek tokenů na validačních oknech", loc="left")
ax.legend(fontsize=8.5)
save_figure(fig, "token_length_distribution", str(P.figures))
plt.show()
"""),
        md("## 6. Co se naučené tokenizace naučily"),
        code("""
from dnalm.tokenizers.stats import top_tokens

for name in learned:
    tok = tokenizers[name]
    top = top_tokens(tok, seqs, n=15)
    longest = sorted((t for t in tok.vocab if not t.startswith("[")), key=len, reverse=True)[:10]
    print(f"=== {label_for(name)}")
    print("nejčastější:", ", ".join(f"{t} ({c})" for t, c in top))
    print("nejdelší:   ", ", ".join(t if len(t) <= 40 else t[:37] + "..." for t in longest))
"""),
        md("## 7. Ukázka tokenizace jednoho úseku"),
        code("""
# Stejný úsek rozdělený všemi tokenizacemi (podklad pro obrázek v kapitole Tokenizace)
example = seqs[0][:60]
print(f"{'sekvence':26s} {example}")
for name, tok in tokenizers.items():
    print(f"{label_for(name):26s} {' | '.join(tok.tokenize(example))}")
"""),
    ]
)


NOTEBOOKS = {
    "01_download_genome.ipynb": NB01,
    "02_make_windows.ipynb": NB02,
    "03_tokenizers.ipynb": NB03,
}


def main(out_dir: str | None = None) -> None:
    out = Path(out_dir) if out_dir else NB_DIR
    out.mkdir(parents=True, exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        with open(out / name, "w") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("zapsáno:", out / name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
