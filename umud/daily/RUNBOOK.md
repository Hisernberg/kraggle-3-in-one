# Daily UMUD loop (5 submissions per UTC day)

A routine fires into the Claude session at 00:05 UTC each day. Each run:

1. `python daily/run.py status`: quota, leaderboard, current best, experiment queue.
2. Pick 5 shots with a reason. Go sequentially and let each result decide the next: queue items, followed up by
   what the previous score says. No random sweeps, and no single-row public probes unless nothing better is left.
3. For each shot: `python daily/run.py build dN_Sk_tag [blend overrides]`, then
   `python daily/run.py submit dN_Sk_tag "dN Sk: what + why"`. `submit` waits for the score, appends to `history.csv`,
   copies the CSV to `submissions/`, and updates `state.json` when there is a new best (later builds start from it).
4. Update `state.json` (settled findings and queue), add a Day-N table to `RESULTS.md` and the README status block,
   then commit and push.

Inputs (`daily/inputs/`) are the frozen pipeline predictions, features, clip groups and the public reference CSV,
so a fresh container needs only the repo plus Kaggle credentials (`~/.kaggle/kaggle.json`, or the
`KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables). Never commit credentials.
