"""
Příprava prostředí pro notebooky. Volá se z první buňky každého notebooku:

    from dnalm.utils.colab import setup
    P = setup()

V Colabu připojí Google Drive, doinstaluje chybějící knihovny a zkopíruje genom
na rychlý lokální disk. Jinde (Mac, školní GPU) jen vrátí cesty. Úpravy nastavení
se dělají tady, ne v notebookách.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

from .paths import Paths, get_paths

# knihovny, které na Colabu nejsou předinstalované (import název -> pip balíček)
COLAB_EXTRA = {"pyfaidx": "pyfaidx"}


def _ensure_packages(packages: dict[str, str]) -> None:
    missing = [pip for mod, pip in packages.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("Instaluji:", ", ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def _mount_drive() -> None:
    if os.path.isdir("/content/drive/MyDrive"):
        return
    from google.colab import drive

    try:
        drive.mount("/content/drive")
    except Exception as e:  # ve VS Code rozšíření se Drive připojuje příkazem
        raise RuntimeError(
            "Google Drive se nepodařilo připojit. Ve VS Code použij příkaz "
            "'Colab: Mount Google Drive to Server...' a spusť buňku znovu."
        ) from e


def copy_genome(P: Paths) -> None:
    """Zkopíruje genom (+ .fai) na rychlý lokální disk, pokud tam ještě není."""
    if P.fasta == P.fasta_stored:
        return
    P.fasta.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["", ".fai"]:
        src, dst = str(P.fasta_stored) + ext, str(P.fasta) + ext
        if not os.path.exists(dst):
            print(f"Kopíruji {src} -> {dst}")
            shutil.copy(src, dst)


def setup(window: int = 1024, genome: bool = True, verbose: bool = True) -> Paths:
    """Připraví prostředí a vrátí objekt s cestami (P.fasta, P.windows('train'), ...)."""
    P = get_paths(window=window)
    if P.environment == "colab":
        _mount_drive()
        _ensure_packages(COLAB_EXTRA)
    P.root.mkdir(parents=True, exist_ok=True)
    if genome:
        if not P.fasta_stored.exists():
            raise FileNotFoundError(
                f"Genom nenalezen: {P.fasta_stored}. Nejdřív spusť notebook 01_download_genome."
            )
        copy_genome(P)
    if verbose:
        print(P.describe())
    return P
