# Squares in Squares (account B)

Competition: `squares-in-squares` — pack n unit squares (n = 1..1000) into the smallest enclosing square.
Deadline 2027-07-18. **Lower is better.**

## Exact scoring (from the verbatim metric `sis_metric.py`, Kaggle notebook metric/squares-in-squares)
- Submission CSV columns: `id, n, square_id, x, y, angle` — 500 500 rows (n rows for each n, square_id 0..n-1).
  x, y = square centre; **angle in radians**; values must be plain decimal literals (sci notation ok), |x|,|y| bounded, |angle| bounded.
- Per puzzle: L = max(max_x - min_x, max_y - min_y) over all corners (translation invariant, measured at 60 digits).
  score(n) = L / sqrt(n) - 1. Total = sum over n = 1..1000.
- Infeasible / malformed / missing puzzle -> charged the grid ceiling ceil(sqrt(n))/sqrt(n) - 1 (no crash).
- Feasibility: float SAT with 1e-8 "undecided" band, undecided pairs certified with mpmath interval arithmetic at
  50/100/200/500 digits. **Touching is allowed** (overlap <= 0); any certified positive overlap or an undecidable
  pair at 500 digits = infeasible. So contacts must be exact in the decimal strings (or have a small positive gap).

## Tools here
- `validate.py sub.csv [n,..]` — runs the official metric (all n or a subset), prints score and infeasible n. ~15-25 min for all n on 2 CPUs.
- `fastL.py a.csv ...` — float per-n L / score screen (seconds). Use for merging; certify with validate.py.
- `merge.py out.csv base.csv cand...` — per-n min-L merge of candidate files (float screen).
- `quench.py file n secs` — penalty (SAT penetration) L-BFGS compression prototype + centre-scaling repair.
- `best/best_L.csv` — per-n best table (L, score, source); `best/best_submission.csv.gz` — the packing file.
- `records_ellsworth.txt` — best-known records (Friedman/Ellsworth page, n <= 178).

## State / findings (2026-10-28)
- Our best public 22.2626642 = public merge (busyaprime 33 notebook). LB top 22.0787 / 22.1131 / 22.1499.
- Public packings already beat every Ellsworth record they differ from (n=51, 84, 103, 126, 176, ...). Records give nothing.
- 422 non-square n use the trivial integer side L = ceil(sqrt n) (score 7.29); 547 n are non-integer (score 14.98).
  Frontier: the largest deficiency (k+1)^2 - n still packed below k+1 is ~0.8k (k=30: 26).
- No n violates monotonicity (L(n) <= L(m), m>n) and no n is improved by adding a grid L-border to a smaller packing.
- Local penalty quench on n=51 / 108: packings are jammed (penalty ~ ds^2 for every shrink ds); local moves give nothing.
  The 0.18 gap to rank 1 needs new constructions (topology), most likely in the mid/large n tilted-band region.
