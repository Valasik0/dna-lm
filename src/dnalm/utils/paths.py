"""
Cesty k datům, modelům a výsledkům na jednom místě, pro všechna prostředí.

Kořen dat (ROOT) se určí takto (první, co platí):
  1. proměnná prostředí DNALM_ROOT (školní GPU, lokální Mac, testy)
  2. Google Colab: /content/drive/MyDrive/dna-lm (Google Drive)
  3. jinak složka data_root/ v repu (je v .gitignore)

Struktura pod kořenem odpovídá složkám na Google Drive:

    ROOT/
      data/raw/genome/hg38.main.fa(.fai)
      data/processed/windows_1024/{train,val,test}.tsv.gz
      data/processed/tokenizers/<název>/
      checkpoints/
      results/<oblast>/
      results/figures/
      logs/

Velké soubory, které se často čtou (genom), se v Colabu kopírují na rychlý lokální
disk (SCRATCH = /content). Jinde se čtou přímo z ROOT.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

COLAB_DRIVE_ROOT = "/content/drive/MyDrive/dna-lm"
COLAB_SCRATCH = "/content"
REPO_ROOT = Path(__file__).resolve().parents[3]


def in_colab() -> bool:
    import importlib.util

    return importlib.util.find_spec("google.colab") is not None


def detect_environment() -> str:
    if os.environ.get("DNALM_ROOT"):
        return "custom"
    if in_colab():
        return "colab"
    return "local"


@dataclass(frozen=True)
class Paths:
    root: Path
    scratch: Path
    environment: str
    window: int = 1024

    # --- genom -----------------------------------------------------------------
    @property
    def genome_dir(self) -> Path:
        return self.root / "data" / "raw" / "genome"

    @property
    def fasta_stored(self) -> Path:
        """Genom tam, kde je trvale uložený (Drive nebo DNALM_ROOT)."""
        return self.genome_dir / "hg38.main.fa"

    @property
    def fasta(self) -> Path:
        """Genom ke čtení. V Colabu lokální kopie (viz colab.setup), jinde přímo uložený."""
        if self.environment == "colab":
            return self.scratch / "genome" / "hg38.main.fa"
        return self.fasta_stored

    # --- okna a tokenizéry -----------------------------------------------------
    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def windows_dir(self) -> Path:
        return self.processed / f"windows_{self.window}"

    def windows(self, split: str) -> Path:
        return self.windows_dir / f"{split}.tsv.gz"

    @property
    def tokenizers_dir(self) -> Path:
        return self.processed / "tokenizers"

    def tokenizer(self, name: str) -> Path:
        return self.tokenizers_dir / name

    # --- modely a výsledky -----------------------------------------------------
    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    def results(self, area: str = "") -> Path:
        return self.root / "results" / area if area else self.root / "results"

    @property
    def figures(self) -> Path:
        return self.root / "results" / "figures"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # --- repo ------------------------------------------------------------------
    @property
    def repo(self) -> Path:
        return REPO_ROOT

    @property
    def configs(self) -> Path:
        return REPO_ROOT / "configs"

    def describe(self) -> str:
        return (
            f"prostředí: {self.environment}\n"
            f"data:      {self.root}\n"
            f"genom:     {self.fasta}\n"
            f"okna:      {self.windows_dir}\n"
            f"kód:       {self.repo}"
        )


def get_paths(window: int = 1024) -> Paths:
    env = detect_environment()
    if env == "custom":
        root = Path(os.environ["DNALM_ROOT"]).expanduser()
        scratch = Path(os.environ.get("DNALM_SCRATCH", root))
    elif env == "colab":
        root, scratch = Path(COLAB_DRIVE_ROOT), Path(COLAB_SCRATCH)
    else:
        root = scratch = REPO_ROOT / "data_root"
    return Paths(root=root, scratch=scratch, environment=env, window=window)
