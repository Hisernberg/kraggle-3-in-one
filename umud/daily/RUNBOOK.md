# Daily UMUD loop (5 submissions per UTC day)

A routine fires into the Claude session at 00:05 UTC each day. Each run:

1. `python daily/run.py status`: quota, leaderboard, current best, experiment queue.
2. Submission protocol v3 (user rule 2026-10-07: research -> one shot -> analyse -> plan; never a burst of five).
   See "Submission protocol" below. Every shot needs a pre-registered card in `daily/plans/dNN.md` before it is
   submitted; after each score the card gets its outcome and the next card is chosen from its decision rules.
3. For each shot: `python daily/run.py build dN_Sk_tag [blend overrides]`, then
   `python daily/run.py submit dN_Sk_tag "dN Sk: what + why"`. `submit` waits for the score, appends to `history.csv`,
   copies the CSV to `submissions/`, and updates `state.json` when there is a new best (later builds start from it).
4. Update `state.json` (settled findings and queue), add a Day-N table to `RESULTS.md` and the README status block,
   then commit and push.

Inputs (`daily/inputs/`) are the frozen pipeline predictions, features, clip groups and the public reference CSV,
so a fresh container needs only the repo plus Kaggle credentials (`~/.kaggle/access_token` for KGAT_ tokens, `~/.kaggle/kaggle.json`, or the
`KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables). Never commit credentials.


## Submission protocol (v3, 2026-10-07)

**Objective.** The final rank is decided on the private split (~206 rows). The public LB (~103 rows) is a small,
noisy measurement: one row moved by 10 mm FL changes the score by ~0.0027, 1 deg PA by ~0.0005, 1 mm MT by ~0.0011.
So LB deltas below ~0.001 are noise. Changes that touch only a few rows are tested on whichever of them happen to be
public, so they tell us little about the private split.

**Shot card (written before submitting, in `daily/plans/dNN.md`).**
- Change: exact diff vs the current best (targets, rows, magnitude), single factor only.
- Hypothesis and mechanism: why the private split should improve, not just the public one.
- Offline evidence: expert sets (OSF / NeuAge / GM, paired bootstrap), geometry checks, blind visual checks,
  earlier LB results it builds on.
- Prediction: expected sign and range of the LB delta.
- Decision rules: what happens next if it is better (>= 0.001), flat, or worse.

**Cadence.**
- S1 at 00:05 UTC is the card prepared the day before.
- Afterwards: write the outcome into the card, update `state.json` (settled, queue), and do a research block of
  at least one hour before the next shot (resume with `send_later`). Two shots in the same hour only if the second
  is a pre-registered branch of the first card's decision rule and needs no new analysis.
- Unused slots are fine. A slot is spent only on a card with evidence or a pre-registered diagnostic.

**Gates (reject offline, no LB shot).**
- Estimator changes must beat the current one on at least two expert sets with a paired bootstrap CI excluding 0,
  and must not be a selection artefact (compare on identical rows).
- Row-level changes need either physics or blind visual evidence with a validated accuracy, and are reported as
  public-only evidence.
- No public-LB fitting of many free parameters (per-row or per-device sweeps without offline support).

**Portfolio.** Prefer global, private-transferable changes (estimators, calibration levels) over row fixes.
Each week: re-rank the workstreams by realised gains, check new discussions and notebooks, and plan GPU use.

**Final selection (2026-11-14).** A = best public among global-change submissions; B = robust hedge (less reliance
on the hard-coded reference). Write both choices down a week ahead.
