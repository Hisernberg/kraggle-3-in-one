# RSNA Knee: hardened Apex submission notebook

Private kernel `kragglenote2forwork/rsna-knee-apex-hardened`, forked from `ranjeet258/rsna-knee-apex-b-under-150`.
Plan: [`../../docs/research_2026-10-09/KNEE_NOTEBOOKS.md`](../../docs/research_2026-10-09/KNEE_NOTEBOOKS.md) §5–6.

**Model legs** (each computed once per run):

| Leg | Model |
|---|---|
| R384 | nartaa CoAtNet SWA with anatomical mirror TTA. The only mandatory leg |
| C224 | nartaa 224-crop CoAtNet |
| CNXT | goodpjw2008 2.5D ConvNeXt reader, 3 folds |
| MaVIT | hengck23 MaVIT-288 |

**Output files** written to `/kaggle/working`:

| File | Content |
|---|---|
| `sub_k1.csv` | Apex exact: per-target `rank((1-w)·rank(R384) + w·rank(CNXT))` with Apex V3 weights |
| `sub_k2.csv` | CoAtNet side = `rank(0.7·R384 + 0.3·C224)`, then the k1 routing |
| `sub_k3.csv` | k2 + MaVIT at `HK_W_MAVIT` rank weight. Failed MaVIT rows keep their k2 value |
| `submission.csv` | Copy of the file chosen by `VARIANT` |

**Hardening:**
- Each optional leg runs in its own process group with a timeout. A failure sets its weight to 0.
- R384-only files are written as soon as R384 finishes.
- Output follows `sample_submission` order; NaN is filled with 0.5.
- Nothing raises after R384.
- Decoder installs only log.
- fp16 only.
- The pinned image is py3.12 (`docker_image` in the metadata).

## Versions submitted on 2026-10-09

Kaggle rejected `-f sub_k2.csv` (HTTP 400), so each variant is its own notebook version, with only the constants
below changed:

| Version | `VARIANT` | `HK_W_MAVIT` | Submission |
|---|---|---|---|
| v1 | k1 | 0.12 | K1 |
| v2 | k2 | 0.12 | K2 |
| v3 | k3 | 0.12 | K3 |
| v4 | k3 | 0.20 | K4 |

The commit run on the 3 public studies showed all four legs valid. Mean Spearman correlation between legs:
R384–CNXT 0.75, R384–MaVIT 0.75, CNXT–MaVIT 0.46.
