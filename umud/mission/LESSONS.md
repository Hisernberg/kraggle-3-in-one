# Lessons — mistakes, learnings, improvements (append-only)

> One line per lesson, newest section per day. Sources: daily cards, RESULTS.md, kernel logs, LB probes.
> The leviathan ratchet rule applies to process too: never silently repeat a mistake.

## 2026-10-21 (day 23 + evening session)

**Mistakes made and fixed:**
1. `daily/run.py submit` on a fresh container hung 10 min polling a missing CSV — `daily/out/` is gitignored, so a
   "built" submission listed in the queue does NOT exist after a container recycle. **Fix: always `run.py build` before
   `submit`; verify the CSV exists and diff-check it against the best before every submission.**
2. Family-flag name mismatch (`left_ticks` vs the actual flag `left_ticks_800`) produced a wrong level-match scale in a
   first build. **Fix: always resolve family flags from the actual args (`--fam-fl-scale <flag>`), not from family stems.**
3. Two-step level-matching must rebuild from the SAME base args as the reference (S5 step 1 was built from state args
   instead of S4 args → wrong scales; caught by the final diff check). **Fix: the diff check after every build is mandatory.**
4. Kernel whack-a-mole (4 failed pushes): nested repo path (`sys.path` must point at the repo ROOT that CONTAINS the
   package), GUI deps imported by `__init__.py` (bypass with stub packages + importlib), Keras 3 rejecting TF2.x H5
   (`TF_USE_LEGACY_KERAS=1` + `tf-keras`), and test images being 251 tif + 58 png (glob both). **Fix: reproduce the
   import chain locally before pushing a kernel; assert expected file counts in the kernel.**

**Learnings (verified):**
1. Every public reference (Vera 0.45134, variational 0.49332, eas "LB 0.374") is a hard-coded CSV disguised as a
   notebook; the variational one embeds a DLTrack prior whose PA is EXACTLY Vera's PA. There is no re-runnable public
   member — DL_Track_US 0.3.1 (host-authored, Apache-2.0, weights on OSF 7mjsc) is the only viable one.
2. The host's prize criterion is "re-runnable on a new dataset where no leaderboard exists" (topic 744191 + Overview);
   manual blend weights are explicitly allowed. LB-probed automatic weighting is not.
3. Per-target additivity held to the 5th decimal again (Juniper PA vertex predicted −0.00015, measured −0.00014).
4. Expert-set biases failed to transfer to test for the 4th time (d23 S3) — the offline gates exist for a reason.
5. Curvature bounds from a single level-matched probe (S3) bound ALL remaining family-weight gains <0.0001 —
   the blend-weight space is exhausted; further Track-A gains need a better estimator, not different weights.

**Improvements shipped:**
- `umud/mission/` war room (dossier + ledger backfilled with the d22/d23 experiments).
- `umud/kaggle/build_dltrack_kernel.py` — self-contained DLTrack re-run kernel (OSF weights, DLTrack v0.3.1 code,
  umud scale fallback, mask dumping). EXP-2401.
- `umud/scripts/build_dl_blend.py` — automated two-pass level-matched anchor-swap builder (ends the manual 2-step dance).
- Pre-registered d24 cards (hedge B S1 + gated DL-rerun shots S2–S5).

## 2026-10-22 (day 24)

**Mistakes made and fixed:**
1. Adopted a blend member on independence alone (r 0.33) without checking row-level deviation magnitude -> S2 scored
   0.44154 (worst of the project). **New gate: a member must have BOTH r < 0.9 vs the incumbent AND row-level sd vs the
   anchor <= ~8 mm (FL) / ~1.5 deg (PA) / ~1 mm (MT). Compute the deviation stats BEFORE spending a slot.**
2. Inferred "pipeline FL beats the prior" from a one-sided probe (d23 S5) and spent S3 on the untested direction -
   both directions lose; the vertex was already settled. **One-sided probes give a bound, not a vertex; never infer a
   vertex direction from a single side.**
3. Kaggle competition data downloads die mid-stream at ~0.5-1.3 GB in this environment - worked around by using the
   kernel-side data + committed frozen inputs instead of local copies.

**Learnings (verified):**
1. Blend weights transfer across anchors of similar quality (both FL directions lose from 0.29/0.45 on the DL anchor
   too). The member's row-level quality is the only Track-B lever left.
2. DLTrack 0.3.1 re-runs cleanly headless (customtkinter stub, TF_USE_LEGACY_KERAS=1, tif+png glob); its PA correlates
   0.912 with the prior/Vera PA; its own scale/geometry layer is unreliable on these devices (use umud scale + masks).
3. hedge B = 0.20732 (+0.00505): the three LB-only levels are worth exactly that on public - the private-split risk of
   keeping them is now quantified for the final decision.

## 2026-10-22 (night session, EXP-2402)

**Mistakes made and fixed:**
1. Trusted doCalculations' internal unit scaling in EXP-2401 without reading its code - the v0.3.1 scaling line
   (`fasc_l / (calib_dist/spacing)` on a Python list) crashes whenever calib is passed, and with calib absent it
   returns pixel-space values. EXP-2401's "bar"-scaled FL/MT were silent garbage (only PA was valid). **Rule: read the
   library's unit-conversion code before trusting its outputs; validate one image end-to-end against physics.**
2. Six kernel pushes for one experiment (input-shape, GUI deps, json syntax, % escaping, trailing paren, calib). The
   fixes were cheap but the loop was slow (~4 h). **Rule: smoke-test the full import + one-image path locally (or in a
   tiny CPU kernel with a 10-image slice) before the full push.**

**Learnings (verified):**
1. DLTrack 0.3.1's models segment the UMUD test devices imperfectly: letterboxed crops break its aponeurosis
   detection on 47% of images; full-frame squashed input works but its geometry is unit-broken. Its PA still lands
   within 2.75 deg of the prior.
2. Member bar for the blend (empirically): FL row-sd vs the anchor <= 8 mm AND coverage >= ~95%. Nothing re-runnable
   reached it tonight; the prior's FL (sd ~7) embeds author magic we have not reproduced.
3. The blend weights are anchor-invariant (S3 + d23 S5 bracket the same vertex), so Track-B improvement is purely a
   member-quality problem now - or a rules clarification from the host.

## 2026-10-22 (late session, d25 prep)

**Learnings:**
1. The leaderboard is now a knife-edge: #2 Kainwang Roger 0.20231 vs our 0.20227 (20:44 UTC). Rank defence needs a
   new information source, not more Vera-anchor weights (exhausted).
2. The DLTrack prior passes the d24 member gate as a PARTIAL anchor (FL sd 6.7 mm vs Vera, r 0.40 with our deviations);
   it was only tested as a full swap. A 0.15 share costs at most ~+0.0005 if it carries nothing (quadratic from the full
   swap's +0.0222) -> d25 S1 card (`daily/plans/d25.md`).

**Improvements:**
- `daily/inputs/ref_mix_vdl15.csv` / `ref_mix_vdl30.csv` (Vera with FL mixed toward the DLTrack prior) + level-matched
  builds via `scripts/build_dl_blend.py --member ...` (FL-only diff verified: 309 rows, PA/MT identical, family means 0).

## 2026-10-23 (day 25)
**Learnings (verified):**
1. Partial second anchor: DLTrack-prior FL at 0.15 share = -0.00092, although the full swap cost +0.0222. A member
   that fails as a replacement can still win at a small share; always probe the share near 0 before rejecting a member.
2. Within-family dispersion is a separate axis from weights and levels: PA spread x0.90 = -0.00099 (vertex ~0.91);
   FL spread is at its vertex (x0.92 +0.00456, x1.04 +0.00097).
3. The two-rater variance estimate pointed the right way for PA but was biased low by ~0.2 (it assumes independent
   errors, which the LB-tuned weights already contradict). Use it for direction only.
**Process:** 5/5 slots in 5 minutes, all pre-registered branches; one branch swapped for a better-EV card after a
quadratic vertex estimate (recorded in the card before submitting).
