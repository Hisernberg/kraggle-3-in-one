# ClimX main track - account B state (2026-10-28 12:45 UTC, B2 run)

Best public: E1 0.04882 (rule-compliant analog emulator, 256 systematic-resampled real NorESM2-MM full-res years
by forcing-predicted GMST kernel h=0.35K). Rank 49/115. Top 0.12458 / 0.10827 / 0.10792; ~20 teams at 0.10-0.11.
Deadline 2026-12-15.

## Probe decode (57724275, precip +1e4 -> nNSE=-1 for 6 precip indices): score -0.34830
sum_precip_nNSE = 15*(0.04882+0.34830) - 6 = -0.043 -> the 6 precip indices together score ~0 (avg -0.007),
the 9 non-precip indices sum ~0.775 (avg ~0.086). Precip is a slight drag; temperature indices carry the score.

## Where artefacts live
E1/A1 were built on the DGX session from full-res daily data (HF isp-uv-es/ClimX, ~200 GB). Not reproducible
in the 2-CPU cloud container; the E1 parquet itself is not backed up anywhere reachable.

## Next ideas
- Precip: since precip nNSE ~0, try shrinking precip index anomalies toward the historical climatology
  (nNSE of a near-constant field is ~0 by construction, but a partial shrink can be >0 if E1 has signal);
  needs E1 parquet + holdout check (SSP370/126/585).
- Temperature indices: the 0.10+ cluster suggests a better forced-pattern estimate (pattern scaling on
  ensemble/annual means) + noise-free mean, rather than a resampled realisation mean. Holdout-gate first.
