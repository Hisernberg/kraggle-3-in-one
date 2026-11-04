# RESTORE PLAYBOOK — UMUD campaign (works from this folder alone after any sandbox recycle)

The campaign lives in two durable places: this `download/umud_d32/` folder and the public GitHub repo
`Hisernberg/kraggle-3-in-one` (everything through d25) — plus `download/umud_d32/` for the d26–d32 layer that
was never pushed. Credentials are the ONLY thing that cannot be restored from files: after a sandbox recycle,
`upload/secrets.env` must be re-provided by the user (KAGGLE_TOKEN, GITHUB_TOKEN; optional HF_TOKEN).

## One-command recovery (any fresh session)

    bash /home/z/my-project/scripts/bootstrap_umud.sh

Does: load credentials (upload/secrets.env or .secrets/env.sh) → clone/pull the repo → restore the d32 layer from
this folder → rebuild the best submission from its args → require a file-level byte-match against the scored CSV.
Verdicts: READY (submissions possible) / READY-OFFLINE (builds only) / FAILED (do not work; fix first).
Verified 2026-11-03: byte-match TRUE (10,807/10,807 bytes) from a cold clone.

## Daily rhythm — 3 batches (2+2+1 slots), standing orders in cards/SCHEDULE_3BATCH.md

- Batch 1 (08:10 Shanghai / 06:10 Dhaka, right after the UTC quota reset): the 2 top-conviction queued cards.
- Batch 2 (14:00 Shanghai): follow-ups per batch-1 decision rules.
- Batch 3 (21:00 Shanghai): best remaining card + bookkeeping (commit, refresh this folder, A/B records).
- All three can also run back-to-back in one session (~25 min total): scores arrive in ~75–90 s each and the
  decision rules apply sequentially. The time spacing only exists so a fresh session can react to the day's
  first results.
- Mechanical execution: `bash /home/z/my-project/scripts/run_batch.sh 1|2|3` (bootstrap + quota check +
  manifest-driven submit + score log). The manifest for batch 1 (d33_S1_gh60) is pre-written in
  `work/kraggle-3-in-one/umud/daily/batches/`; later batches are written by the preceding session.
- The thinking (card construction, decision rules, gates) follows cards/SCHEDULE_3BATCH.md — any fresh session
  can execute a batch from it in <30 min. Say "run today's batches" and it happens.

## What is where

- `submission/BEST_d33_S1_gh60.csv` (+ `.args`) — current best, public 0.19748 (rank 6, 2026-11-04). Verification anchor remains BEST_d32_S5_gh45 (byte-exact).
- `submission/d32_S3_gh30.csv`, `d32_S4_hedgeB2026.csv` (candidate B, 0.21150), `reconstructed_d31_combo.csv`.
- `submission/ref_custom_gh30|gh45.csv` — reference mixes; `cards/` — d32.md, d33.md, SCHEDULE_3BATCH.md.
- `scripts/` — recon_chain.py (verified build helpers), gate_members.py, projection_gate.py, verify_d25.py,
  build scripts; plus the campaign-level bootstrap_umud.sh / run_batch.sh in /home/z/my-project/scripts/.

## Known limitations (stated plainly)

- No scheduler facility is available in this agent environment (cron tool absent — tested twice), so the daily
  trigger is: one short user message per day (or per batch). Everything else is turnkey.
- Submissions need KAGGLE_TOKEN re-provided after every sandbox recycle (put it in upload/secrets.env — that
  path survives recycling; the file is user-provided input, never printed by the scripts).
- The d26–d31 daily artifacts exist only as described in cards/d32.md and the submission descriptions; the d32+
  layer is fully restored here.
