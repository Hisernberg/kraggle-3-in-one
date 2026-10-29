# Account B (kragglenote2forwork) - daily competition loop

Branch `acctb-daily` of Hisernberg/kraggle-3-in-one. Every scheduled run commits and pushes here.
Goal: rank 1-3 in each competition. Up to 5 submissions per competition per UTC day, one at a time,
each verified (format check + local/OOF evidence) before it is spent.

## Division of work (scheduled tasks, all UTC)

| task | competitions | cadence |
|---|---|---|
| "Account B hourly" (since 2026-10-29) | squares-in-squares, 2026-ieee-big-data-traffic-flow-bench, umud-challenge-muscle-architecture-in-ultrasound-data, memecoins-pump-or-dump, misconception-detection-in-k-12-student-math-workings, climx-main-track | hourly at :12 |

At most one submission per competition per run; fill every daily slot by 23:59 UTC (5/day, Squares 10/day - check
`kaggle competitions submissions`). The old B1-B4 tasks no longer exist.

**Account B no longer enters name-that-llm, google-verifiable-quantum-circuit-compilation or neurips-2026-smart-buildings**
(decided 2026-10-29): account A works those, and the same operator must not run both accounts in one competition.

## Starting point (2026-10-28 11:30 UTC)

| competition | deadline | our best public | rank | LB top 3 |
|---|---|---:|---:|---|
| UMUD (lower better) | 11-14 | 0.19888 | 3 | 0.18297 / 0.18704 / 0.19888 |
| Traffic Flow Bench | 11-07 | 0.88384 (L1) | 58 | 0.92817 / 0.92651 / 0.92406 |
| Misconception K-12 | 11-17 | 0.64900 (V7) | 88 | 0.83300 / 0.83183 / 0.82900 |
| ClimX main | 12-15 | 0.04882 (E1) | 49 | 0.12458 / 0.10827 / 0.10792 |
| Squares in Squares (lower better) | 2027-07-18 | none yet | - | 22.080 / 22.113 / 22.150 |
| Name That LLM / Quantum / Smart Buildings | Jan 2027 | see Kaggle | - | - |

Earlier work: `umud/`, `trafficflow/` (branch `claude/dgx-umud-traffic-solar` history). Misconception/ClimX
history lives only in the Kaggle submission descriptions - read them before building the next one.

## Rules every run follows

1. Read this file + `ACCOUNT_B_RUNLOG.md`, then `kaggle competitions submissions <comp>` - never spend a slot
   another run already used today, never submit the same file twice.
2. One change per submission, written down first (hypothesis, expected score, kill rule). Learn from every result:
   append the lesson to the competition's section in `ACCOUNT_B_LESSONS.md`.
3. Prefer CPU; GPU only when the model needs it, and as short as possible.
4. **Kaggle rules:** account B never uses code, outputs or notes from account A (gwey1025, repos acco-cyber/*) and never
   submits to NTL / Quantum / Smart Buildings. No private sharing between teams.
5. Near each deadline: make sure the two final selections are the best-validated, not just the best public.
6. Never print tokens in logs or commits.

## Improvement ladder (user instruction 2026-10-29: every submission should beat the last)

* Each slot goes to the candidate with the best evidence of beating the competition's current best (CV/offline gain,
  or a credited public result above ours). Diagnostics only when they unblock a bigger gain, and say so in the card.
* After each score: new best -> it becomes the base for the next card; worse -> revert and record why in LESSONS.
* **UMUD (rank 3 -> target rank 1, gap 0.016 to 0.18297):** per-device weight/spread tuning is at the +-0.0001 noise
  floor, so stop those probes. Gains must come from a better measurement signal: (1) FL - the weakest estimator
  (pipeline vs reference 0.89 on the metric scale): stronger fascicle-orientation model / longer-fascicle
  extrapolation, test on expert sets first; (2) PA from a new fascicle model that passes the held-out-mask gate;
  (3) the prize-eligible line (no hard-coded public CSV, currently ~0.30): re-implement the reference's MT/FL steps
  in a re-runnable member so the final selection is both eligible and close to the best. Final selections
  (11-14): A = best public, B = best eligible.
