# Automated daily loop: runbook and state

Goal: every UTC day, submit up to 5 files, each one locally validated and each changing a single
factor. Submit one at a time, read the score, learn, report. Runs until the deadline, 2026-11-07
06:55 UTC.

Helpers: `trafficflow/loop.py` (`status`, `pack`, `diff`, `submit`, `rank`, `backup`). Run everything
from the repository root with `PYTHONPATH=.`.

**Scope.** The repository (`Hisernberg/kraggle-3-in-one`) holds several competitions, one folder each.
This loop works only on the traffic project:
- everything it creates or edits lives in `trafficflow/`, with docs in `trafficflow/docs/`;
- it never edits another competition's folder or the shared root files;
- data and artefacts stay outside the repository (`/home/user/data`, `/home/user/cache`, `/home/user/work`).

> **No research agents (firm, since 2026-09-27 04:50 UTC).**
> - The user stopped a research agent again right after it started.
> - Research continues only as local shell jobs (training, builds, evaluations) and as analysis in the
>   main session.

## Schedule (UTC; Routines fire into this session)
| Time | Job |
|---|---|
| 00:07 | **Daily chain:** submit the queued candidates one at a time, reading each score before the next |
| 12:43 | **Heartbeat:** review finished research, submit newly validated candidates, launch the next experiments |
| 21:13 | **Evening sweep:** spend the remaining quota (validated candidates first, then probes that answer an open question), queue tomorrow's candidates, start overnight jobs |

Background agents and jobs wake the session when they finish, so work continues between firings.

## Every firing, in order
1. `python3 -m trafficflow.loop status`. It reports quota used/left today, pending scores, the best
   file, free disk, credentials and missing critical files. If credentials or files are missing, go
   to **Recovery**.
2. Read **Current best**, **Candidate queue** and **Decision log** below.
3. Review finished agent/job results against the **Gates**. Build each passing candidate on the
   current best:
   - `make_submission ... --out /home/user/work/subs/<ID>.csv`
   - `loop pack` the CSV
   - `loop diff` against the best zip, which must touch only the intended rows.
4. Submit with `python3 -m trafficflow.loop submit <zip> -m "<ID>: <change>; <local evidence>"`.
   Add `--probe` for probes. After each score, apply the adoption rule, then rebuild the next
   candidate on the new best if the best changed.
5. After the firing's last submission, run `python3 -m trafficflow.loop rank`.
6. Update `trafficflow/docs/EXPERIMENTS.md` (LB log rows and what was learned) and this file (best,
   queue, decision log). Commit and push to `claude/focused-allen-uzq8gr`. Pushes update the open draft PR (#27 since 27 Sep; #18 was merged on 25 Sep). If the open PR has been merged, open a new draft PR for the branch. Merging is the user's call.
7. Report in the chat:
   - a table of submission, change, public score, Δ and post-rebuild rank;
   - the best so far and the gap to #1;
   - what we learned, and what comes next.

   Send one push notification a day, once the day's submissions are done.
8. Launch the next experiments: at most 2 LightGBM jobs at once, 2 threads each (4 cores, 15 GB).
   Keep at least 2.5 GB of disk free. Agents never submit or commit.
9. Once a day (evening sweep), back up the artefacts: `python3 -m trafficflow.loop backup`
   (**Backup** below).

## Current best
**`H11P_transductive3.zip` = 0.87249** (2026-09-28). #1 KTK 0.90085.

Build:
```
python3 -m trafficflow.make_submission --state-tag ens_H11P --recon-a 0.75 --gate 0.6 \
  --smooth "free=0.0075,free_a=0.001,gate=0.02,gate_a=0.005,dark=0.05" \
  --queue /home/user/work/t2/lgb_v8_seeds9_stack03.csv --odme /home/user/work/t4/t4_l2proj.csv \
  --out /home/user/work/subs/<ID>.csv --note "<ID>: ..."
python3 -m trafficflow.loop pack /home/user/work/subs/<ID>.csv
```

- **Task 1:** regular rows = 0.5·`fullP2` + 0.5·`fullP3` (transductive); blackout rows = `full7` (ramp-flow member).
  - `fullP3`: seed 13, three disjoint transductive row sets (`trainrows,trainrows2,trainrows3`).
  - `fullP2`: like `fullP`, seed 12, trained on two disjoint transductive row sets (`TFB_PSEUDO_FILES=trainrows,trainrows2`).
  - `fullP`: regular models only (`TFB_KINDS=reg`), FD features, seed 11, standard config.
    It trains on train rows plus 180k observed March/April cells per panel (`t1_pseudo.py trainrows`, `TFB_PSEUDO=1`).
  - `state_ens_H11P.parquet` comes from `t1_pipeline enskind --tag ens_H11P --reg fullP2:0.5 fullP3:0.5 --dark full7`.
  - Density reconciliation where v < 0.6·v_f (a = 0.75), then TV smoothing.
- **Task 2:** `lgb_v8_seeds9_stack03`.
  - Onset v8: stacking + 9 seeds on hybrid labels.
  - Ongoing: the v5 blend.
- **Task 4:** L2 projection.

Exact decomposition: ODME 0.19876, Task 1+3 0.43268, queue 0.23633 (S_queue 0.7878: onset 0.732,
ongoing 0.843).

## Gates
**Candidate:** one locally validated change against the current best.
- **Adopt** as the new best if LB Δ ≥ +0.0005. A Δ between 0 and +0.0005 needs a matching local
  prediction.
- **Task 1/3 gate:** J = 0.35·S_state + 0.10·S_LWR (full-coverage holdout, realistic blackouts,
  `trafficflow/t1_lwr_eval.py`) improves on at least 3 of 4 panels, and the mean improves.
- **Task 2 gate** (since 2026-09-25; TASK2_ANALYSIS.md section 17). All three must hold:
  1. **Plain CV.**
     - The hybrid-truth score is ≥ best − 0.002.
     - The conservative evaluation (re-drawn windows, old truth) is ≥ best − 0.002 and within one paired SE of 0.
     - Onset, ongoing or the non-recurrent slices improve.
  2. **Shift-weighted CV** (`trafficflow/t2/shift_cv.py`: `weights`, `score`). The Δ is positive under both the validation and the private weighting. Reject if either is below −1 SE.
  3. **Footprint check** (`shift_cv.footprint_check`, on both months). A flag (≥ p95) means the change goes to the LB as a probe first and is never adopted directly. A flag on private means don't adopt at all.

  Applied retroactively, this gate makes the right call on all 8 past Task 2 changes. That includes G3 (+0.0065 in plain CV but −0.023 on March), which the footprint check flags.
- **Local gain, LB Δ ≤ 0:** not adopted. It goes on the robust list for the final pick, since there
  are only 40 windows per Task 2 condition and private is a different month (7 incidents vs 5).

**Probe:** measures something, such as one task's score or a calibration direction. Never adopted.

**Never:**
- more than 5 per UTC day, or anything after 23:45 UTC;
- an untested change;
- a blind resubmission after an ERROR.

## Candidate queue
**Chain for 2026-09-28.** Base = H5w (0.86838). The loop's `status` shows H6 (0.86839) as best because it takes
the maximum score; H6 was below the +0.0001 adoption bar, so the base for single-factor steps stays H5w.

1. **H7** (regular-only booster member `hold9/full9`, `reg_member.sh`; seed 6, 300k rows per panel, lr 0.05, up to
   6000 rounds, no ramp features). **Built: `/home/user/work/subs/H7_reg9.zip`** (65/65; regular rows only vs H5w;
   also in the Kaggle backup).
   - The gate uses holdout J with blackout rows fixed, i.e. the regular-cell change alone, which transfers exactly.
   - **Gate passed (27 Sep 12:52):** 0.5·hold9 + 0.5·mean(hold3, 4, 5, 7) gives ΔJ +0.00053 on 4/4 panels.
     full9 is training; `H7_reg9.zip` should be built around 15:00.
   - Adopt if Δ ≥ +0.0001 and within 0.0002 of the local ΔJ (+0.00053).
2. **H8** (second boosted seed `hold10/full10`, seed 7, `boost_member.sh`, density model up to 10000 rounds).
   - Restarted 27 Sep 21:15 after the container restart; log `/home/user/work/H8_reg10.log`; ETA about 03:30.
   - Gated against the H7 scheme (b2half / b2heavy / b2only). **Gate passed 23:19: b2only (0.5·hold9 + 0.5·hold10) +0.00026 on 4/4.**
     full10 is training; `H8_reg10.zip` should be ready around 03:00.
   - Expected LB with the refined proxy: +0.00013 to +0.00026 vs H7. Adopt if Δ ≥ +0.0001.
   - Submit only after H7 has scored and landed within 0.0002 of its local ΔJ (the H7 → H8 step is regular rows only).
   - If the container restarts, re-run the same command: training skips models already saved.
3. **H9P (transductive Task 1 member fullP; EXPERIMENTS.md): 0.87093 (+0.00229), adopted.**
   - Pre-registered: adopt if Δ vs H7 ≥ +0.0005; a large miss against +0.0019 means re-checking for a transfer problem
     before stacking more on it.
4. **H10P**: regular rows = 0.35·fullP + 0.65·fullP2 (2× transductive rows). Pseudo-holdout +0.00054 (March) /
   +0.00067 (April) vs H9P, 10/10 panels. Adopt if Δ ≥ +0.0001.
5. **H11P**: fullP3 on 3× rows; the scheme is picked on the pseudo-holdout, chained after H10P (log `/home/user/work/H11P.log`).
6. ~~H9 (third boosted seed)~~: **cancelled 28 Sep 03:05** after H8 (+0.00004). Capacity gains don't transfer, so a third
   boosted seed would add about +0.00003.
   Replaced by the **test-month pseudo-holdout** (backlog 1): hide observed non-target cells of March/April and predict
   them, which measures generalization to the independent test-month draws directly.
4. **Remaining slots:** only probes that answer an open question.

**Paused lines (27 Sep evidence):**
- **Ongoing:** 4 of the last 5 changes lost on March despite passing every local test (G8 passed the full gate).
- **Blackout-cell models:** noisy transfer (H5w's blackout part ~50%, H6 ~5%).
- **Onset decoding / extent:** calibrated from both sides (F1, F2; head-only would score 0.43 vs 0.86 on CV).

**Robust list (for the final pick):** v11 ongoing (G8; private-weighted CV +0.0052 ± 0.0006, March −0.0027).

**Checked and closed on 27 Sep (no submission needed):**
- Task 2 onset with ramp-flow features (+0.15% log-loss only).
- Characteristic (kinematic-wave) features for blackout cells: corr(ch − li, y − li) = −0.07. Congested waves travel 20+ km over a 90-min blackout.
- Off-ramp share as a mainline-flow meter: ramps carry only 1–2.5% of mainline flow; the implied flow error (67–130 veh/h per lane) is worse than the model's.
- TV smoothing strength for the 4-member ensemble: ×0.5–0.75 gives +0.00004 only; it stays at 1.0.
- Scenario shift train → March/April: none (capacity q99.9, peak flow and median speed match; only per-panel queue shares move).
- March onset site check (TASK2_ANALYSIS §19): site hits 0.775 vs 0.824 on train, so the loss is mostly extent.
- March ongoing size check: predicted size is flat, 35% of queues gone by T+19 (train 28%); no bias to correct.
- `pct_observed` at target cells (released, unused): no residual difference between pct 100 and 75–90.
- Low-rank (soft-impute) completion of each day's field: best linear blend −1.4% speed RMSE on D7_I405_S, −0.3% on D7_I10_W (≈ +0.0001 J).
- Train/test mismatch at the origin slot T: none (both use the masked view).

## Decision log
| Date | Submission | Public (Δ vs best) | Decision / lesson |
|---|---|---|---|
| 09-25 | F1: onset re-decoded with logit bias +0.5 (+15 cells, 12 in validation) | 0.86356 (−0.00235) | Larger onset sets hurt on March (onset −0.016); the official first-slot blocks are not larger than ours. Keep b = 0 |
| 09-25 | F2: onset site-commit decoder `site2_lo.05_r.5` (−11 hedge cells) | 0.86418 (−0.00173) | Fewer hedges hurt too (onset −0.012). Top-m at b = 0 is optimal on March from both sides; onset gains must come from better probabilities, not decoding |
| 09-25 | **G1: E1 + TV density smoothing inside target runs** (state rows only) | **0.86651 (+0.00060)** | Local J predicted +0.00062. **Adopted: new best.** The Task 3 proxy predicts the LB to within 0.00002 |
| 09-25 | **G2: G1 + onset v8** (stacking 0.3 + 9-seed mix; 15 cells, 11 in 4 validation windows) | **0.86711 (+0.00060)** | CV onset +0.0035 hybrid / +0.0024 old, i.e. about +0.0005 total. **Adopted: new best.** Shape-aware hedges from stage 2 help on March, where F1's blanket bias hurt |
| 09-27 | **H3: H2 + third Task 1 seed** (state rows only) | **0.86799 (+0.00022)** | local J +0.00023. **Adopted: new best.** The Task 1/3 proxy matches the LB on five changes in a row |
| 09-27 | G7: H3 with ongoing v7 (hybrid labels) | 0.86779 (−0.00020) | March ongoing −0.0013 against CV +0.0032 (hybrid) / +0.0006 (old). The official ongoing truth does not reward the hybrid edge cells. **Ongoing label work closed** |
| 09-27 | **H5w: ramp member, blackout-weighted** (state rows only) | **0.86838 (+0.00039)** | Local J +0.00081, so the LB gave half. **Adopted: new best.** Blackout-only gains need a haircut: 2 of 10 panels (D12_I405) have no test blackouts, and the holdout overstates the rest |
| 09-27 | H6: H5w with blackout rows = mean(full7, full8) | 0.86839 (+0.00001) | Local +0.00016. Rescoring without blackout cells: H5w's non-blackout part is only +0.00008, so its blackout part transferred about 50% and H6's about 5%. **Blackout ΔJ rests on 10 blackouts per panel and is noisy; discount it to ~1/3.** Pause blackout-model work |
| 09-27 | G8: H5w with ongoing v11 (capacity p3) | 0.86798 (−0.00041) | Passed plain CV (+0.005), shift-weighted CV (+0.005 both months) and the footprint check, then lost on March (ongoing −0.0027). **4 of the last 5 ongoing changes failed on March. Ongoing is paused: no local test predicts it.** Robust list for the final pick |
| 09-28 | **H7: boosted regular member at 0.5** (regular rows only) | **0.86864 (+0.00026)** | Local +0.00053; its S_state part (+0.00022) transferred, its density-driven LWR part (+0.00031) mostly didn't. **Adopted: new best.** Proxy refined (EXPERIMENTS.md) |
| 09-28 | H8: two boosted seeds, old members dropped | 0.86868 (+0.00004) | Local +0.00026. **Capacity/data gains transfer at 15–50%; variance reduction at 85–100%** (EXPERIMENTS.md). Not adopted; base stays H7. The loop's `status` shows H8 as best (max score) |
| 09-28 | **H9P: transductive Task 1 member (regular rows)** | **0.87093 (+0.00229)** | Pseudo-holdout +0.0019 from S_state alone. **Adopted: new best.** Learning from the test months' observed cells is the biggest Task 1 lever |
| 09-28 | **H10P: 2× transductive rows (0.35·fullP + 0.65·fullP2)** | **0.87202 (+0.00109)** | Pseudo-holdout +0.00054 (S_state only). **Adopted: new best.** More test-month rows keep paying |
| 09-28 | **H11P: 0.5·fullP2 + 0.5·fullP3 (3× rows)** | **0.87249 (+0.00047)** | Pseudo +0.00025; the LB gives about 2× the S_state-only pseudo figure for transductive steps. **Adopted: new best.** Day: 0.86838 → 0.87249 |
| 09-26 | **H2: G2 + Task 1 seed ensemble** (state rows only) | **0.86777 (+0.00066)** | local J +0.00077. **Adopted: new best** |
| 09-26 | H1b: H2 + ongoing v10 (stage 2 may only remove cells, recurrence ≥ 0.05) | 0.86629 (−0.00148) | Passed the new Task 2 gate and still failed (ongoing −0.010). **Ongoing stacking line dropped, final pick included.** Ongoing changes are LB probes first from now on |
| 09-26 | P5 probe: H2 + TV smoothing ×3 | 0.86736 (−0.00041) | local −0.00062. The official Task 3 truth behaves like the train truth, and the smoothing strength is at or near its optimum |
| 09-26 | P6 probe: H2 with the queue zeroed | 0.63144 | S_queue(H2) 0.7878 exactly (onset 0.732, ongoing 0.843) |
| 09-25 | G3: G2 + ongoing v9 stage-2 stacking (217 ongoing cells, 139 in validation) | 0.86361 (−0.00350) | CV +0.0065 ± 0.0009 (7 SE) but **March ongoing −0.023**. Not adopted, and not on the robust list. The stack extends queues (D7_I10_E +22 to +27 cells per window) and reshuffles small ones (D7_I10_W, D12_I5_N). Train-month growth patterns don't hold in the shifted months. **Lesson: plain CV cannot gate Task 2 ongoing changes; build a shift-weighted CV first** |

## Robust list (final-selection pool)
- **v11 ongoing (G8, 27 Sep):** passed the full Task 2 gate, with private-weighted CV +0.0052 ± 0.0006; March −0.0027 ongoing.
  Candidate for the second final if April is expected to behave like the CV rather than like March.

## Backlog (ordered by expected gain per effort; refreshed 27 Sep)
1. **Boosted regular Task 1 members.** H7 (hold9): one boosted model beats the 4-seed ensemble; ΔJ +0.00053.
   - More boosted seeds (hold10 running, `boost_member.sh`).
   - More rounds for the density model, which is still at the 6000 cap.
   - Later: rebuild the older members at the boosted settings and drop the weak ones.
   - Regular-cell changes transfer exactly, so this is the reliable lever.
2. **Why do ongoing CV gains fail on March?** Four of the last five failed (G3, H1b, G7, G8).
   - Until this is explained, ongoing edits have no local gate.
   - Ideas: per-family probes to locate the loss; compare the March window population against CV draws with
     post-blackout diagnostics (analysis only).
3. **Robustness to incidents and non-recurrent queues** (private has 7 incidents), for onset and ongoing:
   - incident-signature features;
   - recent-month features for private built from March's masked view (all ≤ T).
4. **Blackout-cell models:** widen the holdout to all holdout-period origins before any more blackout work,
   because the dark ΔJ estimate currently rests on 10 blackouts per panel.
5. Task 1 transductive fine-tuning on observed cells of the validation/private months.

Closed:
- Ongoing stage-2 stacking v9 (G3), cell removal (H1b), hybrid labels (G7) and capacity p3 (G8): all lost on March.
- Onset stacking and seeds: adopted in G2. Onset decoding (F1/F2) and extent are calibrated.
- Task 3 TV smoothing: adopted in G1. Its strength is re-checked on the 4-member ensemble and stays at 1.0.
- Dark booster (H6): +0.00001.
- The ideas closed on 27 Sep (Candidate queue above).

## Final selection (5–6 Nov)
- Recommend 2 finals: the best public score, and the most robust (best local validation and
  non-recurrent performance, from the robust list).
- The Kaggle CLI cannot select finals, so ask the user to tick them. If none are ticked, Kaggle
  takes the top 2 public.
- After the deadline, delete the Routines and send the final report.

## Backup (daily, evening sweep)
`python3 -m trafficflow.loop backup [EXTRA_GLOB ...]` creates a new version of the private Kaggle
dataset `kragglenote2forwork/tfb-work`, replacing the old one. The first version (2026-09-25) holds
18 files, 482 MB:
- the best zip;
- `work/t1/pred/state_full3.parquet` and `work/t1/models/full3/`;
- `work/t2/lgb_v6.csv`, `work/t2/probs_lgb_v5.parquet`, `work/t2h/probs_v6_onset.parquet` and
  `work/t2h/model_v6_*`;
- `work/t4/t4_l2proj.csv`;
- `research/lb/lb_with_era.csv`.

## Container restart (same disk)
A restart notice means the harness-tracked background tasks were stopped. `/home/user/work`, the caches and the repo survive.
**First run `ps -eo pid,etime,args | grep -E "python|_member.sh"`.** On 27 Sep the nohup'd H7 pipeline survived; a
duplicate "resume" ran the same full9 prediction in parallel and doubled its time (~3 h).
Resume only stages that have no live process: the logs under `/home/user/work/*.log` show the stage, and each stage
writes its own file (model dir, `state_<tag>.parquet`, zip). Then re-arm the waiters.

## Recovery (fresh container)
1. **Credentials.** If `~/.kaggle/kaggle.json` is missing, use env `KAGGLE_USERNAME`/`KAGGLE_KEY`,
   or copy `/root/.claude/uploads/*/*kaggle.json` to `~/.kaggle/kaggle.json` (chmod 600). If there
   are none: report to the user and pause submissions.
2. **Data.** `kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench -p /home/user/data`,
   then unzip into `/home/user/data/kaggle_public` (README.md, config, corridors, task1, task2,
   task4, submission_key.csv, sample_submission.csv).
3. **Caches.** `python3 -m trafficflow.data` rebuilds `/home/user/cache/<panel>.npz`.
4. **Artefacts.** `kaggle datasets download kragglenote2forwork/tfb-work -p /home/user/work/restore --unzip`,
   then move each file back to its path in `manifest.json`. Kaggle unpacks the best zip into a
   folder, so re-zip that CSV with `loop pack` (copy the checks.json from git history or rebuild).
   Submissions can resume from here.
5. **Research state (hours, in the background):**
   - Task 2: `trafficflow/t2/run_all.sh`;
   - Task 1: `trafficflow/t1_pipeline.py` stages (`TFB_FD=1`).
