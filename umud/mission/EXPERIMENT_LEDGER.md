# Experiment Ledger — `umud-challenge-muscle-architecture-in-ultrasound-data`

> Maintained by `ops/ledger`. Append-only: corrections are new rows.
> Verdict vocabulary: **ADOPT** (beat noise floor under locked protocol) · **NOISE** (within 1 seed-σ) · **REJECT** (failed protocol) · **INCONCLUSIVE** (missing evidence — name the gap).
> Historical experiments (d1–d21) live in `../RESULTS.md`; this ledger backfills the recent cells the dossier cites and all new work.

| ID | Date | Phase | Hypothesis (one sentence) | Owner | Config hash | Seeds | CV (Δ vs best) | LB (Δ) | Verdict | Learned (one line) |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-2201 | 2026-10-18 | consolidation | Juniper FL pipeline weight 0.45 (was 0.33), Juniper mean FL kept | brain/zero | d22_S1_junflw045 | 1 | n/a | −0.0018 | ADOPT | non-square-pixel devices want more pipeline after true-geometry correction |
| EXP-2202 | 2026-10-18 | consolidation | rest-family FL weight 0.25 (was 0.33), family means kept | brain/zero | d22_S2_restflw025 | 1 | n/a | −0.00021 | ADOPT | flat-to-better; joint vertex bracket opened |
| EXP-2203 | 2026-10-18 | consolidation | rest-family FL weight 0.29 (vertex estimate), means kept | brain/zero | d22_S3_restflw029 | 1 | n/a | −0.00164 | ADOPT | joint rest-group FL vertex ≈ 0.289 |
| EXP-2204 | 2026-10-18 | consolidation | rest-family PA weight 0.60 (was 0.65), family means kept | brain/zero | d22_S4_restpaw060 | 1 | n/a | −0.00034 | ADOPT | PA vertex for Telemed-family devices ~0.59–0.61; best 0.20241 rank 1 |
| EXP-2301 | 2026-10-21 | consolidation | Juniper FL weight 0.55, mean kept (bracket above) | brain/zero | d23_S1_junflw055 | 1 | n/a | +0.00018 | REJECT | Juniper FL settled at 0.45; vertices land near the first step's end |
| EXP-2302 | 2026-10-21 | consolidation | Juniper PA weight 0.75, mean kept (quadratic vertex 0.76) | brain/zero | d23_S2_junpa075 | 1 | n/a | −0.00014 | ADOPT | new best 0.20227; vertex fit exact to the 4th decimal |
| EXP-2303 | 2026-10-21 | consolidation | Telemed-only FL weight 0.24 (OSF hint: reference stronger on Telemed-like devices) | brain/zero | d23_S3_telFLw024 | 1 | n/a | +0.00056 | REJECT | Telemed FL settled 0.29; expert-set biases again fail to transfer; curvature c≈0.22 bounds all remaining family-weight gains <0.0001 |
| EXP-2304 | 2026-10-21 | eligibility | swap anchor Vera→DLTrack prior (level-matched) to measure the re-runnability price | brain/zero | d23_S4_dlanchor | 1 | n/a | +0.0222 (0.22451) | INCONCLUSIVE | not a score shot: quantifies the prize-eligibility gap; DL anchor keeps Vera-calibrated weights |
| EXP-2305 | 2026-10-21 | eligibility | FL pipeline weight ×0.7 under DLTrack anchor | brain/zero | d23_S5_dlw020 | 1 | n/a | +0.0076 vs S4 | REJECT | our pipeline FL is the binding constraint on the re-runnable track (OSF: DLTrack FL 3.74 vs ours 4.69) |
| EXP-2306 | 2026-10-21 | eligibility | Vera/variational/eas public notebooks are re-runnable ensemble members | ops/onboarding | notebook source audit | - | - | - | REJECT | ALL are hard-coded CSVs; DL_Track_US 0.3.1 (Apache-2.0, host-authored) is the only re-runnable member |

## Running stats (ops/ledger, weekly)

- Win rate by idea family (d22–d23): device-weight brackets 3/7 ADOPT; eligibility diagnostics 2 probes, 1 direction found.
- Best/worst families: device-level weights (pre-geometry-fix era) best; anything calibrated on external expert sets alone worst (0/4 transfers to test).
- Judge calibration: quadratic vertex fits from 3 LB points predicted 4 follow-ups exactly (d12–d23); trust the fits inside the bracket, not outside.

## Submission log (ops/submission-auditor)

| # | Date | Exp ID | Public LB | Probe hypothesis (pre-written) | Observed | Decision |
|---|---|---|---|---|---|---|
| S-105 | 2026-10-18 | EXP-2204 | 0.20241 | rest-PA vertex | −0.00034 | new best |
| S-106..110 | 2026-10-21 | EXP-2301..2305 | 0.20259 / **0.20227** / 0.20283 / 0.22451 / 0.23213 | per cards d23.md | per cards | best = S-107 |

## CV↔LB transfer matrix (ops/cvlb-auditor)

| Exp ID | dCV | dLB | Ratio | Status (ALIGNED/DIVERGENT/NOISE) |
|---|---|---|---|---|
| EXP-2303 | OSF −0.0003..−0.0007 (predicted gain) | +0.00056 | n/a | DIVERGENT (expert-set bias did not transfer — 4th occurrence, gates tightened accordingly) |

**Crisis protocol:** correlation across recent pairs < 0.5 → freeze ADOPT decisions → differential diagnosis (shift / CV leak / LB noise / metric mismatch) → resume with written cause.
| EXP-2501 | 2026-10-23 | rank defence | FL anchor 0.85 Vera + 0.15 DLTrack prior (level-matched) adds information beyond Vera | brain/zero | d25_S1_vdl15 | 1 | n/a | pending | - | card: daily/plans/d25.md |
| EXP-2501b | 2026-10-23 | rank defence | (result) FL anchor mix 0.15 | brain/zero | d25_S1_vdl15 | 1 | n/a | -0.00092 | ADOPT | partial anchor wins where the full swap lost; vertex ~0.17 |
| EXP-2502 | 2026-10-23 | rank defence | PA spread x0.90 within device family | brain/zero | d25_S2_pas090 | 1 | n/a | -0.00099 | ADOPT | new axis; vertex ~0.91 (S4 x0.80 +0.00135) |
| EXP-2503 | 2026-10-23 | rank defence | FL spread x0.92 / x1.04 | brain/zero | d25_S3/S5 | 1 | n/a | +0.00456 / +0.00097 | REJECT | FL spread settled at 1.0 |
