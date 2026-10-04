# External expert-labelled validation data (declared external data)

Public sets from the UMUD data repository (osf.io/xbawc, CC-BY-4.0, Ritsche et al.):
- `osf`: "Expert Analysed Benchmark Image Datasets" / benchmark_dataset_architecture_v0.1.0 (35 images, up to 7 raters).
- `neuage`: "Example Annotated Images" (Young/Old_ANONYMIZED_annotated_v0.1.0, Esaote VL, one expert). Expert PA/FL
  parsed from the red annotation lines by the public CC0 Kaggle dataset `ngtmduc/umud-code` (`artifacts/neuage_expert.csv`).
- `gm`: benchmark_dataset_architecture_GM_dynamic_v0.1.0 (calf-raise video, 3 raters, even frames; FL in px / 7.75 px/mm).

`labels.csv`: expert reference values. `features.csv`: geometry features of our segmentation models on these images.
Regenerate with `scripts/external_bench.py infer|labels` and score with `scripts/eval_external.py`.
