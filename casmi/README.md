# Enveda CASMI 2026: Molecule ID from Mass Spectra (`casmi/`)

Kaggle code competition [`enveda-CASMI26-molecule-id-mass-spectra`](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).

| | |
|---|---|
| Task | Predict 2D structures of small molecules from MS/MS spectra |
| Metric | MRR@25 on the InChIKey first block, after RemoveStereochemistry and tautomer canonicalisation (RDKit 2026.03.3) |
| Format | Notebook rerun on the hidden test: ~400 molecules, ~1,500 timsTOF spectra; ≤ 9 h; no internet |
| Deadlines | Merge Dec 7, final Dec 14 2026 |

Research and plan: [`../docs/research_2026-10-09/CASMI_NOTEBOOKS.md`](../docs/research_2026-10-09/CASMI_NOTEBOOKS.md),
[`../docs/research_2026-10-09/CASMI_DISCUSSIONS.md`](../docs/research_2026-10-09/CASMI_DISCUSSIONS.md),
[`../docs/research_2026-10-09/MASTER_PLAN.md`](../docs/research_2026-10-09/MASTER_PLAN.md) §2.

## Layout

```
kaggle/c1..c5/   submitted notebooks + kernel-metadata.json (private kernels on kragglenote2forwork)
tools/prep_fork.py   turns a pulled public notebook into a private fork with optional exact string edits
```

| Folder | Kaggle kernel | Source | Change |
|---|---|---|---|
| `c1` | `casmi-c1-v32-anchor` | `haideptry/0-350-casmi-2026-v32-ensemble` (V1.1 lineage) | none |
| `c2` | `casmi-c2-v32-pcjoin` | c1 + `huseyinemreaksoy` PubChem JOIN | PubChem 50 deep, join when library max < 0.7, 30 ppm filter, per-molecule fallback to c1 |
| `c3` | `casmi-c3-huseyin-glmh` | `huseyinemreaksoy/casmi26-v4n-fusion-pubchem-on-public-0-421` | `GL_MH_ONLY=True, POST_ICE_LAM=0.0, FILL_25=True` |
| `c4` | `casmi-c4-v18-cpu` | `amanatar/casmi26-v18` (CPU) | none |
| `c5` | `casmi-c5-flexon` | `flexonafft/casmi26-public-baseline-adaptation-experiments` | none |

Scores: [`../SUBMISSION_LOG.md`](../SUBMISSION_LOG.md).

## How to rebuild and push a fork

```bash
export KAGGLE_API_TOKEN=...          # never commit it
kaggle kernels pull <owner>/<slug> -p src_dir -m
python3 -I casmi/tools/prep_fork.py src_dir out_dir <new-slug> "<title>" '[["old text","new text"]]'
kaggle kernels push -p out_dir       # max 2 concurrent GPU sessions per account
kaggle competitions submit -c enveda-CASMI26-molecule-id-mass-spectra -k kragglenote2forwork/<new-slug> -v <N> -f submission.csv -m "..."
```

In this competition `-f` must be `submission.csv`. Commit runs score only 12 smoke molecules; the full prediction
happens on Kaggle's hidden rerun (about 3.5–4.5 h on T4).

**Licences:** every frontier notebook uses `ahmedberatozer/*` datasets ("Other" licence, ruling pending, topic 745841).
Before choosing finals, re-check prize compliance.
