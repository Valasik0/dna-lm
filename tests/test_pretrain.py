"""End-to-end test tréninku: loss klesá, checkpoint, pokračování, uložení modelu."""

import csv
import random

import pytest

from dnalm.data.genome import WindowConfig, make_windows
from dnalm.models.encoder import DNAForMaskedLM
from dnalm.pretrain import load_run_config, lr_at, train
from dnalm.utils.paths import get_paths

TINY_MODEL = {"hidden_size": 32, "num_layers": 2, "num_heads": 2, "intermediate_size": 64}


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Malý genom s opakovaným motivem (model se má co naučit) + okna."""
    from pyfaidx import Fasta

    monkeypatch.setenv("DNALM_ROOT", str(tmp_path))
    rng = random.Random(0)
    genome = tmp_path / "data/raw/genome"
    genome.mkdir(parents=True)
    with open(genome / "hg38.main.fa", "w") as f:
        for name in ["chr1", "chr2", "chr20", "chr21"]:
            seq = "".join(
                "GGGCGGGGCC" if rng.random() < 0.3 else rng.choice("ACGT") * 2 for _ in range(1500)
            )
            f.write(
                f">{name}\n" + "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60)) + "\n"
            )
    Fasta(str(genome / "hg38.main.fa"))
    P = get_paths(window=256)
    splits = {"train": ["chr1", "chr2"], "val": ["chr20"], "test": ["chr21"]}
    make_windows(str(P.fasta), str(P.windows_dir), WindowConfig(window=256, splits=splits), False)
    return P


def run_cfg(**train_over):
    return load_run_config(
        name="t",
        tokenizer="nucleotide",
        model={"name": "tiny", **TINY_MODEL},
        data={"window": 256, "val_windows": 16},
        train={
            "budget_nt": 256 * 8 * 60,
            "batch_size": 8,
            "lr": 3e-3,
            "eval_every": 30,
            "save_every": 30,
            "log_every": 10,
            "num_workers": 0,
            "warmup_frac": 0.1,
            **train_over,
        },
    )


def read_metrics(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_lr_schedule():
    cfg = {"lr": 1.0, "warmup_frac": 0.1, "min_lr_frac": 0.1}
    assert lr_at(0, 100, cfg) == pytest.approx(0.1)
    assert lr_at(9, 100, cfg) == pytest.approx(1.0)
    assert lr_at(99, 100, cfg) == pytest.approx(0.1, abs=0.01)


def test_train_learns_and_saves(data_root, tmp_path):
    res = train(run_cfg(), P=data_root, out_dir=str(tmp_path / "run"), verbose=False)
    assert res["steps"] == 60
    rows = read_metrics(tmp_path / "run/metrics.csv")
    tr = [float(r["bpn"]) for r in rows if r["split"] == "train"]
    assert tr[-1] < tr[0] - 0.05  # model se něco naučil
    assert len([r for r in rows if r["split"] == "val"]) == 2
    m = DNAForMaskedLM.load(str(tmp_path / "run/final"))
    assert m.cfg.extra["tokenizer"] == "nucleotide"
    assert (tmp_path / "run/best/tokenizer/tokenizer_config.json").exists()


def test_resume(data_root, tmp_path):
    out = str(tmp_path / "run")
    # 1. běh přerušíme po 30 krocích (rozpočet na 30 kroků, pak ho navýšíme na 60)
    train(run_cfg(budget_nt=256 * 8 * 30), P=data_root, out_dir=out, verbose=False)
    res = train(run_cfg(), P=data_root, out_dir=out, verbose=False)
    assert res["steps"] == 60
    steps = [int(r["step"]) for r in read_metrics(f"{out}/metrics.csv") if r["split"] == "train"]
    assert steps == sorted(steps) and steps[-1] == 60


def test_random_dna_source(data_root, tmp_path):
    cfg = run_cfg(budget_nt=256 * 8 * 5)
    cfg["data"].update(source="random", random_windows=100, val_windows=8)
    res = train(cfg, P=data_root, out_dir=str(tmp_path / "rnd"), verbose=False)
    assert res["steps"] == 5


def test_all_run_configs_valid():
    """Všechny konfigurace v configs/pretrain jdou načíst a mají existující model a tokenizér."""
    import glob
    import os

    from dnalm.models.factory import load_model_config
    from dnalm.tokenizers import build_tokenizer, load_config

    root = os.path.join(os.path.dirname(__file__), "..", "configs")
    paths = sorted(glob.glob(os.path.join(root, "pretrain", "*.yaml")))
    assert paths
    for p in paths:
        cfg = load_run_config(p)
        load_model_config(os.path.join(root, "model", f"{cfg['model']}.yaml"))
        t = cfg["tokenizer"]
        build_tokenizer(t if isinstance(t, dict) else load_config(f"{root}/tokenizer/{t}.yaml"))


def test_yaml_scientific_numbers(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\ntokenizer: kmer6\nmodel: S\ntrain:\n  budget_nt: 5.0e7\n  lr: 6.0e-4\n")
    cfg = load_run_config(str(p), ["train.batch_size=8", "train.warmup_frac=1e-2"])
    assert cfg["train"]["budget_nt"] == 5e7 and isinstance(cfg["train"]["budget_nt"], float)
    assert cfg["train"]["lr"] == 6e-4 and cfg["train"]["batch_size"] == 8
    assert cfg["train"]["warmup_frac"] == 0.01
    assert cfg["name"] == "x" and cfg["tokenizer"] == "kmer6"


def test_metrics_header_repair(tmp_path):
    from dnalm.pretrain import ensure_metrics_header

    p = tmp_path / "metrics.csv"
    p.write_text("")
    ensure_metrics_header(p)
    assert p.read_text().startswith("step,split")
    p.write_text("50,train,1.0,0.5,0.9,0.001,1000,512\n")
    ensure_metrics_header(p)
    rows = read_metrics(p)
    assert rows[0]["split"] == "train" and len(rows) == 1
    ensure_metrics_header(p)  # opakované volání nic nezdvojí
    assert len(read_metrics(p)) == 1


def test_trim_metrics_after_resume(tmp_path):
    from dnalm.pretrain import METRICS_HEADER, trim_metrics

    path = tmp_path / "metrics.csv"
    rows = [",".join(METRICS_HEADER)]
    rows += [f"{s},train,1,1,0,0,0,0" for s in (450, 500, 550, 600)]
    rows += ["500,val,1,1,0,0,0,0"]
    path.write_text("\n".join(rows) + "\n")
    trim_metrics(path, 500)
    steps = [ln.split(",")[0] for ln in path.read_text().splitlines()[1:]]
    assert steps == ["450", "500", "500"]
