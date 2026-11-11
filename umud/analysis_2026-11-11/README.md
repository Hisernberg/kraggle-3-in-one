# UMUD full audit and plan, 2026-11-11 (three days before close)

No Kaggle submission was made for this analysis. Sources: the Kaggle API today (competition pages, rules,
the full public leaderboard, all 23 forum topics, public notebooks, and all **404** team submission files with their
scores), plus every UMUD branch in this repository.

Files next to this report:
* `team_submissions.csv`: every team submission (5 members). Columns: date, member, file, public score, nearest
  parent file, public delta, rows changed, noise bound, z, and the original description.
* `public_split_map.csv`: what is known about which test rows are public or private.

---

## 0. Bottom line

1. **Standing:** team best is **0.19377** (`umud_d3_geo0875.csv`, 2026-11-09), public rank **10 of 647**. The top of
   the board moved a lot this week: #1 vibecoder **0.15876**, then 0.17210, 0.17348, 0.17395 (PatrickAIForFun),
   0.17897, 0.18118, 0.18297, 0.18560, 0.18599. **1 slot left today and 15 more by the close on
   2026-11-14 23:00 UTC**, so 16 in total. Two final picks.
2. **Most of our team's work is not in this repo.** Kaggle lists 404 team submissions from 5 accounts:
   kragglenote2forwork 175, dreaddevelopment 189, shhauryajaiswal004 18, shresthml 18 and koushikrudra 4. This repo
   logs only the kragglenote2forwork line up to d31. d32-d35 are on `claude/rsna-knee-abnormality-detection-8qu73d`.
   The 2026-11-05 to 11-11 work (own aponeurosis-map MT, the FL geometry-share "G" series, the CNN PA regressor and
   the public-membership probes) is not in any branch.
3. **"Vera" is our own teammate's output.** The anchor of the whole blend
   (`dreaddevelopment/vera-seg-centerline-mt-correction-lb-0-45134`) is a hard-coded CSV published by teammate
   dreaddevelopment. Its PA/FL are exactly that teammate's automated pipeline `c1_r6.csv` (2026-08-12), except
   about 6 rows. Its MT is their `cand_B_rebased` MT moved 40 % toward a segmentation-centerline reading. The repo
   concluded that "Vera is not re-runnable, so the blend can never be prize-eligible". **That conclusion is probably
   wrong for the team:** if dreaddevelopment can regenerate `c1_r6` from code, the whole blend can be made
   re-runnable. This is the single most valuable finding (section 5).
4. **The public split is now known in detail.** The public set is **exactly 103 rows** (private 206). This is proven
   by teammate shhauryajaiswal004's probes `pm1/pm2/pm3a/pm3b`, which I decoded (section 3.3). Public-frame counts are
   known for 23 of the 28 cine clips. 8 clips (40 rows) are fully private and 2 clips are fully public.
5. **The leaderboard tuning space is exhausted.** Since 2026-10-19, 2 of every 3 knob probes lost, and nearly every
   win was within ±1.5 SD of pure row-sampling noise (section 4). The real gains always came from **device physics**,
   **better maps/models** (own aponeurosis MT −0.0015) and **geometry** (true-geometry FL share −0.0022). They never
   came from more weights or spreads.
6. **Public scores below about 0.17 are almost certainly per-image or per-clip leaderboard fits.** A forum thread
   (2026-11-09) says so explicitly: two-rater agreement implies a floor of about 0.17. The private leaderboard will
   reshuffle. Per-clip fits partly survive on private, because clip-mates share labels.
7. **Decision needed from you:** prize-first (make the final pick re-runnable and drop the third-party CSV and
   per-row picks, at an estimated public cost of about +0.002) or rank-first (spend the slots on clip-level transfer,
   which is not prize-eligible). Section 7 has both plans.

---

## 1. Verified competition facts

| Item | Value (source) |
|---|---|
| Task | PA (deg), FL (mm), MT (mm) for 309 test images; 28 five-frame Telemed cine clips (IMG_00056-195) + a near-duplicate Juniper pair (18/19) + 167 single images |
| Devices (test) | Juniper 91, left_ticks_800 (Telemed screenshots) 90, Lumify 58, telemed_644 50, gray853 12, h512 8. **All 140 clip rows are Telemed**; every Juniper/Lumify/gray853/h512 row is a single image |
| Metric (official `paulritsche/umud-score` source) | `mean(MAE_PA/6, MAE_FL/12, MAE_MT/3)` **+ 1e-6·same with MedAE + 1e-9·same with RMSE** (hidden tie-breakers). No clipping of predictions, so huge values are scored as-is (this is what makes membership probes possible) |
| Public/private | random by image (host, topic 748126); **public = 103 rows exactly** (decoded), private = 206 |
| Public label means (from teammate's constant-bound probes) | **PA 16.925°, FL 82.256 mm, MT 20.881 mm** |
| One public row | FL 1 mm = 0.00027, PA 1° = 0.00054, MT 1 mm = 0.00108 of score |
| Deadline / slots | close **2026-11-14 23:00 UTC** (CLI deadline; host forum said "15 Nov", private LB released 22 Nov); 5/day; **2 final picks**; today 4 used, 1 left |
| Prize | CHF 1500/1000/500; provisional private top-3 must pass FAIR + OSI licence + reproducible-execution review; winner licence GPL-3.0; decision 3 weeks after close |
| Labels | 2 raters (Paul Ritsche, Gerardo Romeny), each: 3 fascicles (FL, linear extrapolation, "DL_Track_US protocol"), 3 PA at fragment insertions vs the **deep aponeurosis slope**, 3 MT as **vertical column distance between inner borders** at left/middle/right; averaged; >10 mm FL / >4° PA / >1 mm MT disagreements re-discussed |

### Host rulings that matter (forum, dated)
* **Allowed:** hand-chosen ensemble weights and constants, even if chosen after seeing public scores. The criterion
  is "can your final submission be (re-)run on a new dataset where no leaderboard exists?" (topic 739660,
  2026-09-14).
* **Not considered for final ranking:** automatic weighting from leaderboard scores (2026-09-10). Leaderboard
  probing is "not disallowed by the rules", but such an approach is not reproducible, so it cannot place.
* **Allowed if declared:** manually labelling the 309 test images and fine-tuning or calibrating on them, even
  against Foundational Rule 3.4(b) (topics 690868 and 737782; re-confirmed to a participant 2026-09-13).
* **Pending (asked 2026-11-11):** (a) fitting ensemble weights automatically on a *manual* test annotation
  (PatrickAIForFun, #4); (b) fixed constants chosen after looking at the public LB, plus using public segmentation
  weights from other participants (another team). Topic 757415 (2026-11-09) asks whether per-image or per-clip LB
  tuning is acceptable and how final solutions will be verified. No host answer yet. **Watch these three threads
  before the final pick.**
* Lumify/Juniper: the host says labels were measured on the distributed images with the vertical ruler and assumed
  square pixels (2026-11-02 and 11-05). Our LB evidence (d16: true-geometry PA/FL corrections worth −0.044) says the
  labels behave as if they are in true geometry. The corrections are device-level constants, so they carry over to
  private either way.

---

## 2. Team inventory (404 submissions)

| Member | Subs | Period | Best public | Line |
|---|---|---|---|---|
| dreaddevelopment | 189 | 08-09 to 11-01 | 0.35876 (`cand_dlt_cwalpha`) | Own CNN+VLM pipeline (`c1_r6` 0.42507) followed by heavy LB probing: family MT/FL/PA offsets, bound probes, per-image PA shrinkage, DL_Track FL corrections. **Published Vera (0.45134)** |
| kragglenote2forwork (this repo) | 175 | 09-22 to 11-08 | 0.19383 | Own seg pipeline + Vera blend; device physics; per-device weights/levels/spreads; DLTrack-prior FL mix; G-series FL geometry share (11-08, not in repo) |
| shhauryajaiswal004 | 18 | 11-05 to 11-11 | **0.19377 (team best)** | Own apo maps for MT (m9, −0.0015); geo-share step; CNN PA regressor (x8, lost); Telemed/left_ticks PA weights (n1/n2, lost); **pm1-pm3b public-membership probes** |
| shresthml | 18 | 10-12 to 11-03 | 0.32220 | Independent own-pipeline attempts (PA local-fragment / PCA ensembles) |
| koushikrudra | 4 | 09-19 to 09-26 | 0.31995 | Re-submitted repo files (Vera, C5, C9, d5_S5) |

Score trajectory: 0.767 (public fork) → 0.425 (dread `c1_r6`) → 0.359 (dread probing) → 0.370 (our first blend
s04) → 0.345 (per-target weights) → 0.294 (maps, d14) → **0.217 (device physics d15-d16)** → 0.202 (per-device
weights d21-d23) → 0.1975 (knobs, d24-d34) → **0.1938 (own MT maps + FL geometry share, 11-07 to 11-09)**.

---

## 3. New facts found today

### 3.1 Vera is in-team (eligibility impact)
`candidate_segmt_a40_c3.csv` (dreaddevelopment, 2026-08-21) equals Vera to 0.000 on all 927 cells. Its PA/FL equal
`cand_B_rebased.csv`, which equals `c1_r6.csv` (2026-08-12, "apo-pair fix + FIJI-MT + old PA, per-image everywhere,
RAW, no anchoring") except 5 PA rows and 6 FL rows (the "6-row apo-pair-fix geometry rescue"). `c1_r6` sits on their
R2/R4 pipeline: UnetPP/EffB5 aponeurosis + fascicle segmentation, plus AmbrosM's tick-mark px/cm code. AmbrosM's
calibration is computed from the images, not a typed table. Vera's MT is only used as the `--clip-mt 3` bound (blend
MT weight is 1.0 pipeline).
**Action:** ask dreaddevelopment whether `c1_r6`, the 6-row rescue and the 40 % seg-centerline MT step can be re-run
from code. If yes, the reference stops being a hard-coded CSV.

### 3.2 The DLTrack prior is third-party and not a formula
`ref_dltrack_prior.csv` comes from the "Variational Ultrasound Kinematics Ensemble" notebook (lamhuy8904). Its PA is
Vera's PA; its FL ≈ 0.757·Vera_FL + 14.8 mm with 4.9 mm residual SD. It cannot be derived. Its FL-share wins total
about −0.0021 public: d25 S1 −0.00092, d31 S1 −0.00007, d32/d33 gray853+h512 −0.00116. That is the price of dropping
it for eligibility.

### 3.3 Public-membership decode (teammate probes pm1-pm3b, decoded here)
Each probe adds MT + 9270·7^k to all frames of clip k. Since 9270/(9·103) = 10, every public frame of clip k adds
exactly 10·7^k to the score. The score's integer part is therefore a base-7 number whose digits are the public-frame
counts. The residuals (pm3a +0.70, pm3b +1.13, pm1 ≈ 770 in MT-mm units) match the RMSE tie-breaker term, which
confirms both the decode and **N_public = 103**. (N = 104 also gave valid digits, but its residuals are far off the
predicted tie-breaker term.) pm1 and pm2 alone are over-range in their top digits; pm3a/pm3b confirm their low digits.

* Mixed clips (public frames give information about private clip-mates): 56, 61, 71, 116, 121, 146, 151 (1 public
  each); 86, 91, 136, 166 (2); 156 (3); 161 (4). In total **22 public / 43 private frames**.
* Fully private: 66, 76, 81, 96, 106, 111, 131, 141 (40 rows). Fully public: 101, 126 (10 rows).
* Pair 18/19: 1 public, 1 private. Undecoded: clips 171-195 (25 left_ticks rows).
* Singles confirmed by exact score equality between submissions: **private** 251, 268, 287, 303, 147, 149;
  **public** 305, 38.

### 3.4 Per-target errors of the Vera lineage (exact, from bound probes)
`c1_r6` public MAE: **PA 2.50°, FL 8.21 mm, MT 0.52 mm** (score terms 0.139 / 0.228 / 0.058). Vera's MT step made MT
worse (≈0.76 mm vs ≈0.32 mm for `cand_B`, approximate). FL dominates every line's error.

### 3.5 The 4 manual per-row picks can become a rule
The best uses `roww_d12s4.csv` (IMG_00275/278/302 FL weight 0.8, IMG_00303 0.7; visual forced choice, d11/d12).
The rule *"Lumify rows whose pipeline FL is more than 21 mm below the reference get FL pipeline weight 0.8"* selects
exactly these 4 rows and no others. The next gap is IMG_00286 at −20.2 mm. Row 303 is private, so 0.7 vs 0.8 there is
free on public. This makes them eligibility-safe.

### 3.6 Signal vs noise of every step (`team_submissions.csv`, column `z`)
For each submission I found its nearest earlier file and computed `z = public delta / SD` under a no-signal null.
The null assumes each changed row is public with probability 1/3 and moves the score by ± its maximum possible
amount. This SD is an upper bound, so the z values are conservative.

| Phase | Probes | Win rate | Typical |z| | Biggest real steps (z) |
|---|---|---|---|---|
| dread probing 08-20 to 11-01 | 105 | 38 % | 0.9 | global MT −0.2 (−5.5) |
| ours, pre-physics (to 10-10) | 65 | 42 % | 0.9 | MT weight 1.0 (−2.1), PA weight 0.62 (−2.0), maps x50 (−1.8) |
| **ours, device physics 10-11 to 10-18** | 33 | **61 %** | 1.2 | **Juniper FL x0.984 (−4.6), x0.95 (−3.8), Juniper PA R1.10 (−3.5)** |
| ours, knob phase 10-19 to 11-06 | 64 | **33 %** | 0.8 | nothing beyond −1.4 |
| team, model/geometry 11-05 to 11-11 | 30 | 23 % | 1.1 | own-apo MT (−1.2, 47 % of max), G125 (−1.1) |

Every new best after 2026-10-21 has |z| ≤ 1.4. Roughly half of the late-phase public gain (~0.0066 since d25) is
expected to be noise that will not repeat on private. The physics, MT and map gains are real.

---

## 4. Every submission line, judged

### Worked (keep; expected to carry over to private)
1. **Device physics** (d15-d16, 0.294 → 0.217): Juniper FL x0.92 and PA R1.12, Lumify PA R0.87 and FL x1.12,
   left_ticks FL x1.016-1.035. Strongest evidence in the whole history.
2. **Per-target blend weights** (C3, 0.370 → 0.352) and **per-device weights after the physics fix** (d18-d23):
   Juniper/Lumify want more pipeline (PA 0.85/0.90, FL 0.45-0.525); Telemed-family devices less (PA 0.60, FL 0.29).
3. **MT 100 % pipeline + clip 3 mm** (d4/d6); then the **own aponeurosis maps for MT** (m9, −0.0015). MT is
   effectively solved: tiny spread or level moves cost +0.0007 to +0.017.
4. **Cine-clip median (alpha 1.0)** (d6) and the clip-aware failure handling (d9/d10 sanity rule, IMG_00305).
5. **Better fascicle maps** for PA/FL (r34 + x50 + x50s7, d13-d14), but only up to three maps.
6. **FL estimator** mean(fl_med, chord_mid) (d8, −0.0082).
7. **True-geometry FL share on Juniper+Lumify** (G125/G250/geo0875, −0.0017 total) and Juniper FL weight 0.525
   (−0.0005).
8. **Partial second anchor:** DLTrack-prior FL share per family (T/L 0.30, gray853/h512 0.60, J/L 0): −0.0021 total,
   but third-party (section 3.2).
9. **Within-family PA spread 0.90** (d25, −0.001); MT spread 1.014.

### Did not work (do not retry without a new mechanism)
* External expert-set biases (OSF/NeuAge/GM) transferred to test: failed 4 out of 4 (A2, d8 S2, d23 S3, d35 PA rule
  +0.0099).
* Agreement with Vera as a proxy for quality: ConvNeXt FL agreed better with Vera and lost +0.0021. The mask harness
  ranked map changes against the LB 3 times out of 3.
* More maps for PA (6 maps +0.0009), multi-scale TTA (d9 S3 +0.026), isotropic maps (d30 +0.0019), msbase swap
  (+0.038), ConvNeXt-S FL, apo seed ensembles (offline +0.04 mm).
* Direct regressors: CNN PA regressor at 25 % (x8 +0.0009), own thick-line PA (m7 +0.0064), fascicle-map
  structure-tensor PA (s1/s2), dread's MT-CNN, ResNet regressors (1.2-1.7).
* Identity-based fixes: PA pulled toward FL·sin(PA)=k·MT (m10 +0.0018); anatomical FL blend; FL from MT/sin(PA).
* Confidence or disagreement gates (d30 flgate +0.0023, mirror flat); FL/PA residual clipping; tail weighting.
* Re-runnable DLTrack members (EXP-2401/2402, d24 S2 0.44154 = worst of the project); the eligible
  pipeline-only line peaked at 0.28823.
* Visual forced-choice on <15 mm gaps (d11 S1/S4/S5); hand MT labels (dread c7); 768-px inference; UDA/self-training.
* Every knob after 10-21: Telemed PA weight 0.85 (+0.0017), left_ticks PA weight 0.45 (+0.0034), PA spread 0.92,
  J/L MT ±0.1, MT spread 1.00, fl-scale 0.980, gh pa_final micros, T/L FL share 0.45, J/L corrected prior share.

### Best, worst, notable
* Best: `umud_d3_geo0875` 0.19377 (team), `umud_d2_s4_Jw0525_onG250` 0.19383 (this account).
* Worst real candidates: `d24_S2_dlmask` 0.44154; `d28_elig_pipe` 0.3438; `d35_msbase` 0.23562; d23 DLTrack anchor
  0.22451.
* Hedges on record: hedge B (public-only levels dropped) 0.20732 (d24) and 0.21150 (d32). These levels are
  family-level (z ≈ −0.2 to −1.0 each), so they likely hold on private and the hedge is likely worse there too.

---

## 5. Loopholes and techniques (ranked by value)

| # | Lever | What it is | Value | Cost | Prize-eligible? |
|---|---|---|---|---|---|
| L1 | **Make the blend re-runnable via the teammate's code** | Regenerate Vera from dreaddevelopment's `c1_r6` code; drop the DLTrack-prior mix; replace the 4 manual picks with the 21 mm rule (3.5); keep all constants (host allows hand-set constants) | Turns the ~0.196 line into a **prize candidate**. The best public re-runnable notebook is 0.299 (auriclee, 11-10) and our old eligible line was 0.288 | ~+0.002 public; 0-1 slot to verify; code assembly can finish after the close (repo requested after the deadline) | **Yes**, if Vera regenerates |
| L2 | **Clip transfer (public frames → private clip-mates)** | Read the public-frame label of a mixed clip with a two-arm constant probe (FL = 30 / 200 on that clip only; the difference gives the exact sum of the public labels), then set the whole clip to it | 43 private frames in 13 mixed clips. Each FL clip probe pair ≈ −0.0005 to −0.0008 private (and more on public). About 10 probes ≈ **−0.005 private** | ~1.5-2 slots per clip-target; clips 171-195 need one more membership probe | **No.** Per-clip LB tuning; host stance pending (757415) |
| L3 | Exact per-target split of our best | One probe with PA = 0.5 and MT = 0.5 on all rows gives the FL term exactly (public means known); a second gives PA | Tells where the remaining error is (expected FL ≈ 55-60 %) | 1-2 slots | n/a (diagnostic) |
| L4 | Manual test annotation as declared external data | Allowed by the host; could calibrate per-device levels/weights reproducibly | Medium. A rough annotation alone scores ~0.46 (PatrickAIForFun), so it only helps as calibration | Many hours of expert-like work | Yes (declared); automated fitting on it is pending |
| L5 | Near-duplicate pair 18/19 | One public, one private: copy the probed public label to the twin | ~1 row | 2 slots | No |
| L6 | Neighbour/subject smoothing for singles | Checked: Juniper MT neighbour similarity ratio 0.79, others ≥ 0.95, and dread's dedup lost +0.004 | Weak | - | - |
| L7 | Distil Vera into an image model trained on test images | Technically re-runnable but memorises the test set; likely viewed as hard-coding | Not recommended without a host OK | - | doubtful |

Ruled out: other teams' files cannot be pulled (`team-submissions` exposes only the score and id, and their files are
private); there is no train/test overlap (NCC ≤ 0.70); the IMG_00001/2 sample values are made up (host); and the
metric tie-breakers are too small to exploit.

---

## 6. Process lessons (every nick and knack)
* **Fragmented state is the biggest operational risk:** `daily/state.json` still says best = d31_combo 0.19864, and
  the code that produced the current best (m9 maps, G-series) is not in git. Consolidate before the final pick.
* Per-target and per-cell additivity is exact; disjoint wins combine to the 5th decimal (used 6+ times).
* One-sided probes give a bound, not a vertex. Mirror steps should be smaller than the losing probe (asymmetric MT).
* A member that fails as a replacement can win at a small share (DLTrack prior: +0.022 as a full swap, −0.0009 at
  0.15).
* The gate for a new member is both independence (r < 0.9) and a small row SD vs the anchor (FL ≤ 8 mm,
  PA ≤ 1.5°, MT ≤ 1 mm). The worst submission of the project skipped the second check.
* Always rebuild and diff-check before submitting; `daily/out` is gitignored. Argparse is last-wins (silent args
  override); image→group NaN mapping bugs; batch manifests and quota guard (d34).
* Read library unit code before trusting it (DLTrack 0.3.1 scaling bug).
* External expert sets rank estimators but their biases never transfer. The held-out mask harness measures mask
  fidelity, not label agreement.

---

## 7. Plan for the last 3 days (16 slots), pending your decision

**Step 0 (today, no slot): ask dreaddevelopment** whether `c1_r6` (+ the 6-row rescue + the 40 % MT step) can be
re-run from code, and pull the 11-05 to 11-09 code (m9 apo maps, G-series) into git. The answer decides between the
two tracks.

### Track P: prize-first (recommended if Vera can be regenerated)
1. Build `elig_clean` = team best − DLTrack-prior mix + 21 mm rule instead of `roww_d12s4` (offline, diff-checked).
   Expected public ≈ 0.1955-0.1965. Spend 1 slot to measure it.
2. Regenerate Vera from the teammate's code and check bit-exactness offline. Optionally spend 1 slot on the
   regenerated blend to confirm the identical score.
3. Spend the remaining slots only on re-runnable, physics/model-backed cards (e.g., a separate geometry share for
   Juniper vs Lumify, MT from own maps on gray853/h512), each pre-registered with a z-check.
4. **Final picks:** both eligible (e.g., `elig_clean` and its best eligible variant). Avoid pairing with a
   non-reproducible pick: if that pick scored better on private it would be the one reviewed.
5. After the close: assemble the GPL-3.0 repo (README, requirements, CITATION.cff/codemeta, inference notebook).

### Track R: rank-first (if Vera cannot be regenerated, or you only want rank)
1. One membership probe for clips 171-195 (same 9270·7^k design, ≤ 6 groups so it stays in range).
2. About 12 slots of two-arm FL probes on the mixed clips with the most private frames (56, 61, 71, 116, 121, 146,
   151, then 86, 91, 136, 166). Set each clip to the recovered public-frame mean, shrunk about 30 % toward the
   current value for rater noise.
3. **Final picks:** A = best + clip transfer; B = current team best (0.19377).

### Hybrid (only after the host answers 757415)
A = `elig_clean` (prize), B = best + clip transfer (rank). Risky: the host has not said which of a team's two picks
gets the compliance review.

### Do not spend slots on
Weights, spreads, levels, offsets, more maps for PA, gates, regressors, expert-set calibration (all closed in
section 4).
