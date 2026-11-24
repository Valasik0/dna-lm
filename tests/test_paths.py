"""Testy cest a přípravy prostředí mimo Colab."""

import pytest

from dnalm.utils.colab import setup
from dnalm.utils.paths import get_paths


def test_custom_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DNALM_ROOT", str(tmp_path))
    P = get_paths()
    assert P.environment == "custom"
    assert P.fasta == P.fasta_stored == tmp_path / "data/raw/genome/hg38.main.fa"
    assert P.windows("train") == tmp_path / "data/processed/windows_1024/train.tsv.gz"
    assert P.tokenizer("bpe_600") == tmp_path / "data/processed/tokenizers/bpe_600"
    assert P.figures == tmp_path / "results/figures"
    assert (P.configs / "tokenizer" / "bpe_600.yaml").exists()
    assert get_paths(window=512).windows_dir.name == "windows_512"


def test_setup_without_genome(tmp_path, monkeypatch):
    monkeypatch.setenv("DNALM_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        setup(verbose=False)
    P = setup(genome=False, verbose=False)
    assert P.root.exists()


def test_setup_colab_branch(tmp_path, monkeypatch):
    """Simulace Colabu: Drive připojený, genom se zkopíruje na rychlý lokální disk."""
    import importlib.machinery
    import sys
    import types

    from dnalm.utils import colab, paths

    drive_root, scratch = tmp_path / "drive", tmp_path / "content"
    genome = drive_root / "data/raw/genome"
    genome.mkdir(parents=True)
    (genome / "hg38.main.fa").write_text(">chr1\nACGT\n")
    (genome / "hg38.main.fa.fai").write_text("chr1\t4\t6\t4\t5\n")

    calls = []
    fake_colab = types.ModuleType("google.colab")
    fake_colab.__spec__ = importlib.machinery.ModuleSpec("google.colab", None)
    fake_colab.drive = types.SimpleNamespace(mount=lambda p: calls.append(p))
    fake_google = types.ModuleType("google")
    fake_google.__spec__ = importlib.machinery.ModuleSpec("google", None)
    fake_google.colab = fake_colab
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    monkeypatch.delenv("DNALM_ROOT", raising=False)
    monkeypatch.setattr(paths, "COLAB_DRIVE_ROOT", str(drive_root))
    monkeypatch.setattr(paths, "COLAB_SCRATCH", str(scratch))
    monkeypatch.setattr(colab, "_ensure_packages", lambda pkgs: None)

    P = colab.setup(verbose=False)
    assert P.environment == "colab"
    assert P.fasta == scratch / "genome/hg38.main.fa"
    assert P.fasta.read_text() == ">chr1\nACGT\n"
    assert (scratch / "genome/hg38.main.fa.fai").exists()
    assert P.windows("val") == drive_root / "data/processed/windows_1024/val.tsv.gz"


def _fake_drive(tmp_path, monkeypatch, folder="My Drive"):
    from dnalm.utils import paths

    root = tmp_path / "CloudStorage" / "GoogleDrive-student@example.com" / folder / "dna-lm"
    genome = root / "data/raw/genome"
    genome.mkdir(parents=True)
    (genome / "hg38.main.fa").write_text(">chr1\nACGT\n")
    (genome / "hg38.main.fa.fai").write_text("chr1\t4\t6\t4\t5\n")
    monkeypatch.delenv("DNALM_ROOT", raising=False)
    monkeypatch.delenv("DNALM_NO_DRIVE", raising=False)
    monkeypatch.setattr(paths, "DRIVE_SYNC_BASE", tmp_path / "CloudStorage")
    monkeypatch.setattr(paths, "LOCAL_SCRATCH", tmp_path / "cache")
    monkeypatch.setattr(paths, "in_colab", lambda: False)
    return root


@pytest.mark.parametrize("folder", ["My Drive", "Můj disk"])
def test_drive_for_desktop(tmp_path, monkeypatch, folder):
    """Mac s Google Drive pro počítač: stejná data jako v Colabu, genom zkopírovaný lokálně."""
    root = _fake_drive(tmp_path, monkeypatch, folder)
    P = setup(verbose=False)
    assert P.environment == "drive" and P.root == root
    assert P.checkpoints == root / "checkpoints"
    assert P.fasta == tmp_path / "cache/genome/hg38.main.fa"
    assert P.fasta.read_text() == ">chr1\nACGT\n"


def test_dnalm_root_beats_drive(tmp_path, monkeypatch):
    _fake_drive(tmp_path, monkeypatch)
    monkeypatch.setenv("DNALM_ROOT", str(tmp_path / "own"))
    assert get_paths().environment == "custom"
    monkeypatch.delenv("DNALM_ROOT")
    monkeypatch.setenv("DNALM_NO_DRIVE", "1")
    assert get_paths().environment == "local"
