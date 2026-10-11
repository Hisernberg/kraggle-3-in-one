# DataVerse: Detecting Emotions from Hindi Speech (`bhav`)

Competition: [`dataverse-detecting-emotions-from-hindi-speech`](https://www.kaggle.com/competitions/dataverse-detecting-emotions-from-hindi-speech)
(SRCASW-BhavVaani). 4 classes (angry, happy, neutral, sad), macro-F1, final score = 0.4 public + 0.6 private,
5 submissions/day, deadline 2026-10-12 18:30 UTC, CSV upload in `sample_submission.csv` format (`Id,emotion`, lowercase).

Plan and submission ladder: [`PLAN.md`](PLAN.md). Submissions: [`../SUBMISSION_LOG.md`](../SUBMISSION_LOG.md).

## Data facts (from `scripts/eda.py`, `samelen.py`, `dupgraph.py`, `cropsearch.py`, `ext_overlap.py`)

* 826 train / 209 test clips, all 16 kHz mono, 1.3-11.2 s (median 2.9 s). Classes balanced (192-217 each).
  10 % of files are 32-bit PCM; storage format and Id order carry no label signal (adjacent-Id label agreement = chance).
* **Duplicates are the dominant structure.** Many clips are copies of another clip with the *exact same sample
  count* (re-saved, gain-changed or noise-added: waveform correlation > 0.9). 467 of 1,035 files sit in 227
  duplicate clusters (mostly pairs). Train-train duplicate pairs share the label 98.8 % of the time (one
  conflicting cluster: Ids 75/636/878).
* **78 of 209 test clips (37 %) have a duplicate in train**, none with conflicting labels; 16 more test clips are
  duplicated only inside test. Crops/sub-segments do not occur (exhaustive FFT search), and no competition clip is
  taken from the four public Hindi/Indian emotion datasets on Kaggle (best match correlation 0.877).
* Consequence for validation: a random stratified K-fold reproduces the test situation (duplicates split across
  folds). We report two numbers per model: **novel** = macro-F1 of the model alone on OOF rows without a
  duplicate in the training folds (566 of 826 rows, the part that decides the ranking), and **all** = macro-F1
  of the full pipeline (duplicate label copied when one exists, model otherwise). Mean over 3 fold seeds.

## Public notebooks (4) and discussions (3, administrative only)

| Notebook | Approach | Reported |
|---|---|---|
| `kunalinfinite/01-dataverse-training` + `02-...-inference` | fine-tuned WavLM Base+/Large, 5 folds, 8 s crops; duplicates forced into the same fold | OOF 0.789, public 0.833 |
| `avikdas567/acoustic-source-filter-...` | 184 hand-crafted features + frozen `superb/wav2vec2-base-superb-er` mean/std, MLP+LGBM+XGB, class-threshold tuning | - |
| `shivamgravity/dataverse-dehs-baseline` | librosa MFCC/chroma/mel/contrast means, XGBoost | - |

None of them uses the duplicate structure explicitly, picks a layer of the SSL encoder, or uses an emotion-pretrained
encoder (emotion2vec). Leaderboard top is 0.96544 (three teams tied), then 0.95387 (two teams).

## Pipeline

1. `bhav/kaggle/extract/bhav_extract.py` (Kaggle T4) and `scripts/extract_local.py` / `extract_e2v.py` (CPU):
   per-layer mean and std pooled hidden states of speech foundation models -> `work/emb/emb_<model>.npz`.
2. `scripts/probe.py`: per-layer logistic-regression probes.
3. `scripts/blend.py` + `make_sub.py`: OOF/test probabilities per feature set, log-probability blend, duplicate
   override, test-test duplicate clusters averaged, submission CSV.
