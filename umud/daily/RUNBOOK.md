# Daily UMUD loop (5 submissions per UTC day)

A routine fires into the Claude session at 00:05 UTC each day. Each run:

1. `python daily/run.py status`: quota, leaderboard, current best, experiment queue.
2. Gated, one shot at a time (user rule, 2026-10-05): submit only a shot with a written hypothesis and local
   evidence (expert sets, model agreement, geometry consistency, diff vs best). After each score, write down what it
   means and research before choosing the next shot; follow the decision tree in `state.json`. Unused slots are fine
   when no shot has evidence. No random sweeps, no single-row public probes.
3. For each shot: `python daily/run.py build dN_Sk_tag [blend overrides]`, then
   `python daily/run.py submit dN_Sk_tag "dN Sk: what + why"`. `submit` waits for the score, appends to `history.csv`,
   copies the CSV to `submissions/`, and updates `state.json` when there is a new best (later builds start from it).
4. Update `state.json` (settled findings and queue), add a Day-N table to `RESULTS.md` and the README status block,
   then commit and push.

Inputs (`daily/inputs/`) are the frozen pipeline predictions, features, clip groups and the public reference CSV,
so a fresh container needs only the repo plus Kaggle credentials (`~/.kaggle/access_token` for KGAT_ tokens, `~/.kaggle/kaggle.json`, or the
`KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables). Never commit credentials.
