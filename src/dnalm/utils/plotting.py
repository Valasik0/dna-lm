"""
Jednotný styl grafů pro diplomovou práci.

Každá tokenizace a každý model má pevnou barvu ve všech grafech (barva sleduje
entitu, ne pořadí). Grafy se ukládají jako PDF pro LaTeX.

    from dnalm.utils.plotting import apply_style, color_for, save_figure
    apply_style()
    ax.plot(x, y, color=color_for("bpe_4096"), label=label_for("bpe_4096"))
    save_figure(fig, "tokenizer_lengths")   # -> <FIG_DIR>/tokenizer_lengths.pdf
"""

from __future__ import annotations

import os

# Validovaná kategorická paleta (pořadí je pevné, nikdy se necykluje)
PALETTE = [
    "#2a78d6",  # 1 modrá
    "#eb6834",  # 2 oranžová
    "#1baf7a",  # 3 tyrkysová
    "#eda100",  # 4 žlutá
    "#e87ba4",  # 5 purpurová
    "#008300",  # 6 zelená
    "#6250d6",  # 7 fialová
    "#e34948",  # 8 červená
]

# Pevné přiřazení barev entitám. Nové tokenizace/modely se přidávají sem, ne ad hoc.
ENTITY_COLORS = {
    "nucleotide": PALETTE[0],
    "kmer6": PALETTE[1],
    "bpe_600": PALETTE[2],
    "bpe_4096": PALETTE[3],
    "unigram_4096": PALETTE[4],
    "kmer6_overlap": PALETTE[6],
}

ENTITY_LABELS = {
    **{f"kmer{k}": f"{k}-mery" for k in range(2, 8)},
    "nucleotide": "Nukleotidy (T1)",
    "kmer6_overlap": "6-mery s překryvem (T2)",
    "kmer6": "6-mery (T3)",
    "bpe_600": "BPE 600 (T4)",
    "bpe_4096": "BPE 4 096 (T5)",
    "unigram_4096": "Unigram 4 096 (T6)",
}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

FIG_DIR = os.environ.get("DNALM_FIG_DIR", "results/figures")


# Řada k-merů (k = 2..7) je veličina s pořadím -> sekvenční škála jednoho odstínu
# (oranžová jako kmer6), od světlé pro malé k po tmavou pro velké k.
KMER_SERIES = {
    2: "#f9e0d0",
    3: "#f6c4a8",
    4: "#f2a079",
    5: "#ee8452",
    6: "#eb6834",  # stejná jako kmer6 v ostatních grafech
    7: "#c9521f",
}


def color_for(name: str, series: bool = False) -> str:
    """Barva entity. series=True: pro k-mery barva podle k (graf řady k = 2..7)."""
    if series and name.startswith("kmer") and name[4:].isdigit():
        return KMER_SERIES.get(int(name[4:]), TEXT_SECONDARY)
    if name in ENTITY_COLORS:
        return ENTITY_COLORS[name]
    return TEXT_SECONDARY


def label_for(name: str) -> str:
    return ENTITY_LABELS.get(name, name)


def apply_style() -> None:
    """Nastaví matplotlib: tenké čáry, nevýrazná mřížka, bez horního a pravého rámu."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (6.0, 3.6),
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "pdf.fonttype": 42,  # písma vložená jako TrueType (čitelné a upravitelné v PDF)
        }
    )


def save_figure(fig, name: str, directory: str | None = None) -> str:
    """Uloží graf jako PDF (pro LaTeX) a PNG (pro rychlý náhled)."""
    directory = directory or FIG_DIR
    os.makedirs(directory, exist_ok=True)
    base = os.path.join(directory, name)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=200)
    return base + ".pdf"
