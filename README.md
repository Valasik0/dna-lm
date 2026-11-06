# dna-lm

Kód k diplomové práci **Modelování DNA sekvencí pomocí jazykových modelů**.

Práce porovnává strategie tokenizace DNA (nukleotidy, k-mery, BPE), trénuje vlastní MLM modely
(moderní BERT encoder), srovnává je s předtrénovanými modely (DNABERT, DNABERT-2, GROVER,
Nucleotide Transformer, Caduceus), testuje embeddingy pro vyhledávání podobných sekvencí
a analyzuje vysvětlitelnost přes attention a atribuční metody.

## Struktura

```
configs/tokenizer/   YAML konfigurace tokenizérů (nucleotide, kmer6, bpe_600, bpe_4096, ...)
src/dnalm/
  data/genome.py     genom, okna, splity (train chr1–22+X kromě 20/21, val chr20, test chr21)
  tokenizers/        tokenizéry se společným rozhraním, trénink BPE, statistiky
  utils/paths.py     všechny cesty k datům na jednom místě (Colab / lokál / školní GPU)
  utils/colab.py     setup() pro notebooky: Drive, knihovny, kopie genomu
  utils/plotting.py  jednotný styl grafů, pevné barvy tokenizací
notebooks/           01 stažení genomu, 02 okna, 03 tokenizéry
scripts/             build_notebooks.py (notebooky se generují odtud)
tests/               pytest, běží na CPU
```

## Spuštění

**Google Colab** (prohlížeč i VS Code rozšíření): otevři notebook, vyber Colab runtime a spusť
buňky. Úvodní buňka sama stáhne kód z GitHubu a připojí Google Drive. Data jsou na Drive
v `MyDrive/dna-lm/`.

**Lokálně / školní GPU:**

```bash
pip install -e ".[dev]"
export DNALM_ROOT=/cesta/k/datum     # složka se stejnou strukturou jako MyDrive/dna-lm
pytest
```

Bez `DNALM_ROOT` se data hledají v `data_root/` v repu (je v `.gitignore`).

## Notebooky

Notebooky se generují ze skriptu, aby měly všechny stejnou úvodní buňku:

```bash
python scripts/build_notebooks.py
```

Změny nastavení prostředí patří do `src/dnalm/utils/`, ne do notebooků. Úvodní buňka notebooků
se nemění.

## Data

Data, checkpointy a logy nejsou v repozitáři (viz `.gitignore`):

```
MyDrive/dna-lm/
  data/raw/genome/hg38.main.fa(.fai)         UCSC hg38 analysis set, chr1–22, X, Y, M
  data/processed/windows_1024/                okna 1 024 bp: train/val/test.tsv.gz
  data/processed/tokenizers/<název>/          uložené tokenizéry
  checkpoints/  results/  results/figures/  logs/
```
