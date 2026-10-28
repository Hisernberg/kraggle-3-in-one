# Account B (kragglenote2forwork) - daily competition loop

Branch `acctb-daily` of Hisernberg/kraggle-3-in-one. Every scheduled run commits and pushes here.
Goal: rank 1-3 in each competition. Up to 5 submissions per competition per UTC day, one at a time,
each verified (format check + local/OOF evidence) before it is spent.

## Division of work (scheduled tasks, all UTC)

| task | competitions | cadence |
|---|---|---|
| B1 UMUD + Traffic | umud-challenge-muscle-architecture-in-ultrasound-data, 2026-ieee-big-data-traffic-flow-bench | every 4 h at :07 |
| B2 Misconception + ClimX | misconception-detection-in-k-12-student-math-workings, climx-main-track | every 4 h at :17 |
| B3 Squares in Squares | squares-in-squares | every 4 h at :27 |
| B4 Independent | name-that-llm, google-verifiable-quantum-circuit-compilation, neurips-2026-smart-buildings | every 4 h at :37 |

6 runs per day per task -> at most one submission per competition per run, max 5 per day.

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
4. **Kaggle rules:** account B never uses code, outputs or notes from account A (gwey1025, repos
   acco-cyber/*). In NTL / Quantum / Smart Buildings it works only from public notebooks/datasets and its own ideas.
   No private sharing between teams.
5. Near each deadline: make sure the two final selections are the best-validated, not just the best public.
6. Never print tokens in logs or commits.
