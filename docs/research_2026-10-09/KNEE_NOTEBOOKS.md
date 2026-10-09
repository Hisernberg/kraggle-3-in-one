# RSNA Knee: forensic audit of public notebooks and today's submission plan

*Snapshot: 2026-10-09, around 14:55–15:20 UTC. Account `kragglenote2forwork`. Research only: nothing was pushed, submitted or committed.*

**Evidence labels used throughout:**
- **[M]** measured by me through the Kaggle API, or read directly from code or logs.
- **[A]** measured by a notebook author on the public LB, per their own write-up (an author claim, but a specific one: they cite submission rows or "our run").
- **[C]** a claim in a title or description with no supporting evidence.
- **[E]** my estimate.

---

## 0. The three things that matter

1. **Your pending submission will fail. [M]**
   - Submission `57012226` is a fork of `heliosli/rsna-knee-abnormality-detection`, version 1.
   - The fork's notebook version is in state `KernelWorkerStatus.ERROR`. Its log ends with `RuntimeError: Cannot resolve rsna-knee-compact-v1: []`, raised in cell 3 (`find_mount(C_SLUG, 'manifest.json')`).
   - The cause: heliosli **retired the dataset `heliosli/rsna-knee-compact-v1` today at 13:50 UTC**, 52 minutes before your submission. It is now 296 bytes, and its README reads: *"This dataset no longer ships model files. The earlier versions have been retired and are not accessible. Notebooks that relied on them will not run."*
   - Your fork's metadata does not even list that dataset. Its code is byte-identical to heliosli's (same sha1 `8bc08822`).
   - Expect this submission to come back as an error with no score. Treat it as lost.

2. **The whole "Helios" branch (claimed 0.951 to 0.954) is now dead for everyone. [M]**
   - Affected notebooks: `heliosli/rsna-knee-abnormality-detection`, `heliosli/rsna-knee-blend-gold`, `evgendvorkin/rsna-versia-7`, `matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed` (current version), `rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-95`, `sameerk2004/rsna-knee-sil-d5d-compact0951`, `spark328/rsna-dinosaur-v5-top`, `hitarthjain0/rsna-knee-apex-grandmaster-blend-gold`, and the `ayodejiibrahimlateef/*helios*` notebooks.
   - All of them hash-assert the retired bundle.
   - `kiyoshiohno/rsna-knee-template` (#2 in score order) runs code from two private datasets that I cannot see. It also attaches the heliosli code dataset, so it is probably dead as well (unverified).
   - `romantamrazov/rsna-knee-dinosaur-v5` was switched to a "PUBLIC ONLY apex950" path in its 14:45 run, which suggests its author reacted the same way.
   - No re-upload of the compact bundle exists. I searched for `knee compact` and `rsna knee`, sorted by date. A re-upload would be a licence and ethics problem in any case.

3. **The reproducible public frontier is 0.950, and top-10 (0.962) is not reachable today.**
   - The best you can reasonably expect from public assets is **0.950 (likely) to 0.951 (possible)**.
   - That moves you from rank 1954 to roughly 270–720. 455 teams sit at exactly 0.950.
   - Top-50 needs 0.957, top-100 needs 0.954 and top-10 needs 0.962 [M]. These require a privately trained, decorrelated second family. No such family exists in public form any more.

---

## 1. Leaderboard [M]

Full public LB downloaded at 2026-10-09T14:57:53: 5,586 teams. You are at rank 1954 with 0.942 (5 submissions, the last on 2026-09-21).

| Public score | Best rank | Teams at score | Worst rank |
|---|---|---|---|
| 0.964 | 1 | 3 | 3 |
| 0.963 | 4 | 5 | 8 |
| 0.962 | 9 | 2 | 10 |
| 0.961 | 11 | 6 | 16 |
| 0.960 | 17 | 10 | 26 |
| 0.959 | 27 | 10 | 36 |
| 0.958 | 37 | 9 | 45 |
| 0.957 | 46 | 8 | 53 |
| 0.956 | 54 | 18 | 71 |
| 0.955 | 72 | 14 | 85 |
| 0.954 | 86 | 15 | 100 |
| 0.953 | 101 | 29 | 129 |
| 0.952 | 130 | 42 | 171 |
| 0.951 | 172 | 95 | 266 |
| **0.950** | **267** | **455** | **721** |
| 0.949 | 722 | 128 | 849 |
| 0.948 | 850 | 34 | 883 |
| 0.945 | 956 | 88 | 1043 |
| 0.943 | 1201 | 699 | 1899 |
| 0.942 | 1900 | 168 | 2067 |

**Score needed for each target:**

| Target | Score needed |
|---|---|
| Top-10 | 0.962 |
| Top-25 | 0.960 |
| Top-50 | 0.957 |
| Top-100 | 0.954 |
| Top-150 | 0.952 |
| Top-200 / top-250 | 0.951 |

**What the board tells us:**
- Ties are broken by submission time, so a new 0.950 today lands near **rank ~721**.
- The 455-team spike at 0.950 is the fork pile of the Apex / Raptor+ConvNeXt recipe.
- The 0.951 bucket of 95 teams is mostly Helios forks scored before the retirement. That suggests Helios really did add about +0.001.

**Top of the board:**
- CentralyDS, Sergey Bryansky and Ilya Kolb are at 0.964. Pa3฿aJluHа, joinT fusion, Zhengru Li, Wassim Dobbi and rhoskeri are at 0.963. YOKKOISHO and dalab are at 0.962.
- The median submission count in the top-50 is 131.
- On the Efficiency LB (`ryanholbrook/...efficiency-lb` output [M]), Pa3฿aJluHа scores 0.962 with an efficiency-rank-2 model and Scott Willis 0.959 at efficiency rank 1. The leaders' models are fast and strong, and none of them is public.

Comparison with 2026-09-22: the top went from 0.958 to 0.964, and the public plateau went from 0.943 to 0.950. The public ecosystem gained +0.007, while the top gained +0.006.

---

## 2. Lineage tree (verified by code diff / line-set Jaccard / sha1 of code cells [M])

```
pilkwang baseline (DINOv2) ─┬─ prvsiyan / mattiaangeli "dinosaurs" ── Speedy Raptors 0.943 stack
                            │   (20 DINOv2-S + A5 DINOv3 + RadImageNet heads + dreaddevelopment CoAtNet views)
                            │   ├─ jiweiliu fast-2xT4, maverickss26, skarin repro (0.943 [A])
                            │   ├─ pjmathematician d4-blend / d4-lite, yamadan96 d4-public0946 ...
                            │   ├─ xianhan "0.957 candidate", qinghaowang1 "fuad0957"  (titles = local diag numbers; code = 0.943 stack) [M]
                            │   └─ goodpjw2008 stack + own 2.5D ConvNeXt reader 30% → 0.944 [A]  (reader alone 0.929 [A])
                            │        (medvax, sujanmajhisuzan tri-specialist = copies)
                            │
dreaddevelopment Raptor (CoAtNet-RMLP-2 MIL) ── nartaa retrain on all data + OAI external MRIs, SWA
   ├─ nartaa/rsna-knee-0949-anatomical-mirror : R384 SWA, 320 crop, K94, plain+anatomical-mirror TTA → 0.949 [A]
   ├─ nartaa/rsna-knee-0945-efficient-224crop : C224 SWA, 0.8 crop→224, single view → 0.945 [A]
   │
   ├─ APEX family (R384 + goodpjw CNXT 3 folds, per-target rank weights 0.03–0.35, try/except fallback to R384)
   │    sujanmajhisuzan/apex-grandmaster-stack == matterhorn3838/rsna-knee-d4 == rian55/floor-velci
   │       == newwang12/versia-5 == shuheitak/versia5-fork   (identical code) ............ 0.950 [A, several]
   │    ├─ haideptry V3/V4/V5 (V4 heavier weights = 0.950 [A]; V5 = gold-58 grid search 0–0.55)
   │    ├─ sujanmajhisuzan apex-v5 / hitarthjain0 v5 (V5 weights .04–.48, CNXT 256+320 multires + offset + flip TTA) [C 0.952+]
   │    ├─ romantamrazov dinosaur-v5 V15 (RESEARCH_W + quality gate; RAISES if reader fails)
   │    ├─ kozykappa gold-gated triple reader (+C224, weights chosen on gold-58 with bootstrap gate) == sameerk2004 triple224
   │    ├─ ranjeet258 apex-b (+C224 at 0.3 inside CoAtNet side, NO fallback, pip check=True)
   │    │    └─ rabari9999 lb-0-95 (ranjeet + heavier weights + 50/50 linear/probit rank fusion + jitter) [C 0.95+]
   │    ├─ karttikjangid05 "does blend improve": Apex 0.7 + 0945 notebook 0.3 → 0.950; Apex alone 0.950;
   │    │    Apex 0.6 + tonylica C 0.4 → 0.948  [A]
   │    └─ waterjoe / rabari dual-reader-0950 / sameerk dual0950 (Apex copies with cosmetic changes)
   │
   ├─ goodpjw2008 0-949-coatnet-stack-blend: R384 0.6 + 0.943 stack 0.4 → 0.949; R384 + own readers 30% → 0.948 [A]
   ├─ prvsiyan bee's knees: R384 mirror arm 0.949; R384 + CNXT uniform 0.30 → 0.948 (two runs) [A]
   │
   └─ HELIOS family (R384 0.7 + heliosli compact convnext_small 4 folds "E_lrprior" 0.3, rank) — DEAD since 13:50 UTC
        heliosli/blend-gold (S5) 0.951 [A via versia-7 write-up] == matterhorn v2 (current) == versia-7
        heliosli/rsna-knee-abnormality-detection (S6, dual-arch bundle) ["0.954 research" per romantamrazov, C]
        rabari dual-engine, sameerk compact0951, spark328 v5-top, hitarthjain blend-gold, ayodeji h01–h05,
        kragglenote2forwork/rsna-knee-abnormality-detection (YOURS → ERROR) [M]

Independent, unmeasured: hengck23/mavit-288-baseline-01 (MaVIT encoder, 288 px, 6 views × 8 slices,
  own checkpoint 00000008.pth, MIT, ~1.5 s/study single GPU, no public score found) [M code / score unknown]
```

---

## 3. Per-notebook forensic table (main notebooks)

| Notebook | Public LB | Members / checkpoints | Blend | TTA / post-processing | Runtime (hidden) | Fragility |
|---|---|---|---|---|---|---|
| nartaa/rsna-knee-0949-anatomical-mirror | 0.949 [A] | `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (sha 7e5315da, CoAtNet-RMLP-2, 96-slice/5-slot volume, 140 mm, K94 windows, 320 centre crop) | single model | plain + anatomical mirror (sagittal: reverse slice channels; cor/ax: width flip) | 26 min submit-to-score [A]; 3 studies 21 s [M] | hash assert only; robust |
| nartaa/0945-efficient-224crop | 0.945 [A] | `..._c08_r224_swa.pt` | single | single view, 0.8 crop→224 | 7.2 min [A] | robust |
| **sujanmajhisuzan/apex-grandmaster-stack** (= matterhorn d4, rian55, versia-5) | **0.950 [A: karttik "Apex alone 0.950 in our run", haideptry V3, sameerk]** | R384 + goodpjw `cnxt_v0_fold{0,1,2}.pt` (ConvNeXt-T, 6 slots × 16 windows, 384 px / 153.6 mm, transformer + per-finding attention) | per-target rank: cnxt weight Baker's .35, Contusion .30, MedOA .25, ACL .25, MedMen .20, MCL .20, LatMen/Fx/LatOA/PFOA .08, Effusion .04, Synovitis .03; re-ranked | R384 mirror TTA; CNXT none | ~1 h [E]; 3 studies: R384 21 s + CNXT 27 s [M log] | **Safe**: fusion inside try/except, falls back to R384 0.949. pip decoder install is `check=False` and fails on the current py3.13 image (no cp313 wheels) [M log] (see §6) |
| haideptry v2-vs-apex V5 | V3 and V4 = 0.950 [A]; V5 unknown | Apex + mattiaangeli resgated dataset (only to read gold-58) | gold-58 grid search of per-target weights in [0, 0.55] | – | ~1 h | gold-58 weights: **overfit risk** (58 studies, and R384 was trained on "alldata", which probably includes gold-58) |
| hitarthjain0 / sujanmajhisuzan apex V5 | [C] "0.952+/0.954"; unknown | Apex | V5 weights .04–.48 + micro-jitter | CNXT at 256 **and 320** px (320 is off the training resolution), offset 0/+1, flip | ~1.5 h | tiered fallbacks; the AUC bar charts in the notebook are hard-coded, not measured |
| ranjeet258 apex-b-under-150 | unknown (≥0.950 by score order) | R384 + C224 (0.3 inside CoAtNet side) + CNXT | Apex V3 weights | as members | ~1.2 h | **No fallback by design** + `pip ... check=True` + decoder import `check=True`: ran on py3.12.13 [M log]; on a py3.13 image it would crash |
| rabari9999/rsna-knee-lb-0-95 | [C] "0.95+" | ranjeet + heavier weights (LatOA .28, Baker's .45) | 0.5·linear-rank + 0.5·probit-rank + 1e-4 jitter | – | ~1.2 h | same as ranjeet |
| kozykappa gold-gated triple | unknown | R384 + CNXT + C224 | chosen at run time on gold-58 (bootstrap gate) | – | ~1.5 h (runs R384 on gold-58 too) | complex; gold-58 leakage risk |
| romantamrazov dinosaur-v5 (V15, 14:45 today) | unknown; earlier versions 0.950–0.951 | Apex | RESEARCH_W + per-study quality gating (valid slots, fold std) | – | ~1 h | **Raises if CNXT fails** ("REFUSED 0.949 DOWNGRADE") → crash on any hidden-set reader failure |
| karttikjangid05 does-blend | **0.950 [A]** | Apex (stage 0) + nartaa 0945 notebook (stage 1) | outer rank 0.7/0.3 | – | ~1.3 h | good fallback, **but** `_validate` requires output row order == sample_submission order |
| goodpjw2008 0-949-coatnet-stack-blend | **0.949 [A]** | R384 + 0.943 stack | rank 0.6/0.4 | – | ~7 h [A] | heavy; stack needs exactly 2 T4s |
| prvsiyan bee's knees | 0.949 (mirror arm) [A]; +CNXT 0.30 → 0.948 [A] | R384 (+ CNXT inactive) | – | – | – | – |
| heliosli/* and all Helios forks | 0.951 (S5) [A]; S6 unknown | R384 + compact bundle | 0.7/0.3 rank | – | – | **DEAD**: dataset retired [M] |
| kiyoshiohno/rsna-knee-template | unknown (#2 in score order) | thin launcher `exec()` of `rsna_knee_pipeline.py` from a private dataset | unknown | unknown | unknown | not forkable; probably Helios-dependent |
| hengck23/mavit-288-baseline-01 | unknown | MaVIT (ICCV'25 MALA), 288 px, 6×8 slices, 130 mm crop, 1–99 % window | single | none | ~35 min single GPU [E from 1.5 s/study] | bare `except:` writes zeros for failed studies |

The full inventory of the 39 notebooks pulled is in **Appendix A**: score-order position, sources, assert/raise counts, accelerator, internet setting and code hash.

**Settings common to all of them:**
- Every competitive notebook runs on T4×2 with internet off. No P100 or CPU-only notebook is competitive.
- No notebook hard-codes test-set predictions.

**LB-probing exposure:**
- The Apex per-target weights are hand-set. They are justified by gold-58 AUCs and by a few LB ticks.
- The V4/V5 and rabari weights are explicitly LB-chased.
- haideptry V5 and kozykappa fit their weights on gold-58.

---

## 4. Measured blend arithmetic: what explains 0.943 → 0.949 → 0.950 → 0.951

Every row below is an author-reported public LB result **[A]**. The public-LB resolution is about ±0.001 to 0.002: roughly 400 public studies, with one tick being 0.001.

| On top of… | Change | LB | Delta |
|---|---|---|---|
| old 0.943 stack | +goodpjw CNXT reader 30% | 0.944 | +0.001 |
| — | **nartaa R384 single model** (new labels "blendjev" + **OAI external MRIs** + SWA + anatomical mirror TTA) | **0.949** | **+0.006**: the real jump |
| R384 | + 0.943 stack at 40% | 0.949 | 0 |
| R384 | + goodpjw own readers (0.933 + 0.929) at 30% | 0.948 | −0.001 |
| R384 | + CNXT, uniform 0.30 (prvsiyan, 2 runs) | 0.948 | −0.001 |
| R384 | + CNXT, **per-target** 0.03–0.35 (Apex V3) | **0.950** | +0.001 |
| Apex V3 | heavier routing V4 (.04–.45) | 0.950 | 0 |
| Apex | + 0945 C224 notebook at 0.3 (outer) | 0.950 | 0 |
| Apex | + tonylica DINO/RadImageNet "C" at 0.4 | 0.948 | −0.002 |
| R384 | + Helios compact (independently trained, report-derived labels, native DICOM geometry) at 0.3 | 0.951 | +0.002 |

**Interpretation:**
1. The 0.943 → 0.949 step is a **better core model**. It comes from external OAI data and better labels, not from blending.
2. Every public second model is about 0.016–0.020 behind. At a weight large enough to matter it hurts, and at a weight small enough not to hurt it adds about 0.001.
3. Per-target routing works because it limits the weak reader to the findings where it is reportedly competitive: Baker's, Contusion, Medial OA and ACL.
4. The only real public step beyond 0.950 came from a **decorrelated, independently trained** model (Helios), and that model is gone.
5. The correlation of R384 with C224 is high: same team, same labels, same training data, a different crop and resolution. This explains why C224 adds roughly 0.

**Independent families still available, with their decorrelation potential:**

| Family | Solo public LB | Decorrelation from R384 | Assessment |
|---|---|---|---|
| goodpjw CNXT | 0.929 [A] | different arch, preprocessing and labels | **already in Apex** |
| 0.943 DINO/RadImageNet stack | 0.943 [A] | different, but measured null at 40% | 7 h runtime |
| tonylica C | – | – | measured −0.002 at 40% |
| hengck23 MaVIT-288 | unknown | different backbone, crop, slice sampling and probably labels | **the only unmeasured candidate**; if roughly 0.93, a weight of 0.10–0.15 is about neutral to +0.001 [E] |
| nartaa C224 | 0.945 [A] | same training recipe | low diversity |

---

## 5. Four candidates for today (ranked)

**Overall expectation.** Expected public LB for all four is **0.949–0.951 [E]**. None reaches the top-10, which would require +0.012. The goal today is to bank 0.950, and with luck 0.951, moving you from rank 1954 to about 270–720.

**Building all four from one notebook.** Build all candidates as **one notebook that computes every leg once and writes several files**:
- `submission.csv` (the default) plus `sub_c1.csv` … `sub_c4.csv`.
- The CLI documents `kaggle competitions submit -c rsna-knee-abnormality-detection -k <user>/<slug> -v <N> -f sub_cX.csv -m ...`. Its help text says `-f` is "the name of the output file produced by a kernel (for code competitions)" [M].
- **Unverified for this competition:** some code competitions only accept `submission.csv`. Test it first with the C1 file.
- If it is refused, add a `VARIANT = "c1"` constant that selects which file is copied to `submission.csv`, and save four versions. Each commit run on the 3 public studies costs only about 5–10 min of GPU session [M: R384 21 s, CNXT 27 s, C224 34 s on 3 studies, plus image start and pip].
- Each submission triggers its own hidden rerun on Kaggle's side. I believe that does not draw on your weekly quota, but this is unverified.
- Your quota resets on Saturday 2026-10-10 at 00:00 UTC.

**The base to fork:** `ranjeet258/rsna-knee-apex-b-under-150`. It already has separate legs: C224 cell, R384 cell, CNXT, and fusion with `W_C224`.

**Required hardening (lessons from your four lost submissions):**
- Change `pip install ... check=True` and the `import libjpeg, openjpeg` probe to `check=False`, and only log the result.
- Wrap the C224 leg, the CNXT leg and any extra leg in try/except. A failed leg sets its weight to 0, and R384 stays mandatory.
- Write `submission.csv` in `sample_submission` order, plus a final schema check that only logs rather than raising when it can fall back.
- **Pin the environment**:
  - In the editor, choose "Pin to original environment" of ranjeet's version, which ran on py3.12.13 with decoders available [M log].
  - Otherwise add a commit-time decoder smoke test that decodes a JPEG-Lossless and a JPEG-2000 *train* DICOM and prints the outcome. On the py3.13 image the goodpjw cp311/cp312 decoder wheels do not install [M].
  - The CNXT leg would then silently skip JPEG-compressed series on the hidden set.
- Keep the accelerator at **GPU T4 ×2**, internet off, and the datasets `nartaa/rsna-knee-publication-swa-weights-20261007` and `goodpjw2008/rsna-knee-2-5d-convnext-reader` attached. For C4, also attach `hengck23/hengck23-rnsa-knee`.

| # | What | Exact recipe | Expected public LB | Hidden runtime | Crash risk |
|---|---|---|---|---|---|
| **C1 (bank it)** | Apex exact | R384 mirror + CNXT 3 folds; Apex V3 weights {Baker's .35, Contusion .30, MedOA .25, ACL .25, MedMen .20, MCL .20, LatMen .08, Fx .08, LatOA .08, PFOA .08, Eff .04, Syn .03}; per-target `rank((1-w)·rank(R)+w·rank(CNXT))`; fallback to R384 | **0.950** (several [A]; floor 0.949 on fallback) | ~1 h | low (<5% [E]) |
| **C2** | Apex + C224 inside the CoAtNet side | `coat = rank(0.7·rank(R384)+0.3·rank(C224))`, then the C1 routing against CNXT (= ranjeet `W_C224=0.3`, `W_OWN=0`) | 0.950 (0.949–0.951) [E]; karttik's outer 0.7/0.3 version measured 0.950 [A] | ~1.2 h | low after hardening |
| **C3** | C2 + better weak-reader TTA | the CNXT leg at native res only (no 320): average of plain + horizontal flip and window offsets 0/+1 (4 passes); C1 routing unchanged | 0.950 (0.949–0.951) [E]; TTA on a 0.929 reader might add +0.002–0.004 solo, which is about +0.0005 in the blend | ~1.6 h | low–medium (new code; keep a fallback to the plain CNXT preds) |
| **C4 (information shot)** | C2 + independent MaVIT leg | third leg hengck23 MaVIT-288: `final = rank((1−0.12)·C2_score + 0.12·rank(MaVIT))` for all findings (or 0.15 on Effusion/Synovitis/Baker's/Contusion, 0.08 elsewhere); if MaVIT fails, it degenerates to C2 | 0.948–0.951 [E]; unknown solo; first measurement of an independent family on top of 0.950 | ~1.8 h | medium (unknown checkpoint behaviour; its bare `except` writes zeros for failed studies, so treat all-zero rows as missing: give them rank 0.5 or C2's value) |

**Why not other variants:**
- Weight-only LB chasing (V4/V5, rabari, probit fusion) has measured or claimed results of 0.950 and adds overfit risk.
- The 0.943 stack costs 7 h and measured null.
- Uniform reader weights measured 0.948.
- Helios forks are impossible now.

**Order of submission:**
1. Submit C1 first (−f test included) to bank 0.950.
2. Then C2 and C4.
3. Keep C3 last, or skip it and save the slot: C3 vs C2 is below LB resolution.

---

## 6. Risks and failure modes found in the code [M]

**Crash and silent-degradation risks:**

| Risk | Where | Mitigation |
|---|---|---|
| Retired private dataset → `RuntimeError` | every Helios fork, **including yours** | do not fork anything with `C_SLUG='rsna-knee-compact-v1'`, or with an empty `""` entry in `dataset_sources` |
| `pip install --no-index` of cp311/cp312 wheels on the py3.13 image | all Apex-family notebooks; fatal in ranjeet/rabari (`check=True`) | `check=False`, pin the original environment, smoke-test decoders on train DICOMs |
| "Refuse downgrade" raises | romantamrazov V15, ranjeet/rabari "no fallback on purpose", helios `>3% Raptor failures` | replace with fallback-and-log |
| Output row-order assertion vs `sample_submission` | karttik `_validate` | write in `sample_submission` order |
| `nunique()>1` asserts per column | Apex fusion (inside try → falls back, OK) | fine on 1,300 studies |
| subprocess `timeout=2*3600` on the Raptor stage | helios | not an issue (R384 ≈ 26 min) |
| Gold-58 used for selection | haideptry V5, kozykappa; R384 trained on "alldata", so gold-58 is likely in training | never pick weights on gold-58 |
| Fabricated "AUC trajectory" arrays in plots | hitarthjain0 / sujanmajhisuzan V5 | ignore those numbers |

---

## 7. Final-selection strategy (private LB, 2 picks by 2026-10-22)

**The constraints:**
- The public LB is about 30% (roughly 400 studies). Differences of ≤0.002 between correlated blends are noise.
- The private LB is about 70% (roughly 900 studies).

**Two picks:**
1. **Pick 1:** the most-evidenced robust blend, which is C1 (Apex exact) unless C2 or C4 beats it by **≥0.002** on public. Per-target routing has many independent replications at 0.950. Uniform alternatives measured 0.948, so routing is not merely LB noise.
2. **Pick 2:** a structurally different scored submission. Choose C4 (MaVIT) if it scores ≥0.950, otherwise C2. This diversifies across the second-family risk.

**Do not pick:**
- Anything whose weights were tuned on gold-58 or by LB ticks (V4/V5/rabari) over C1 at equal public score.
- Any version without fallbacks: a scored version is safe, but avoid it in further iterations.

**A scored submission is already safe.** Hidden-set predictions are computed once at scoring time. A later dataset retirement cannot break an already-scored submission.

**Beyond today (honest):**
- Moving above 0.951 needs what Helios had: an independently trained second family on good report labels, ideally with external data. Examples are the repo's `kneemri` arms or a CoAtNet trained on the 96-slice corpus you already built (`rsna-knee-corpus96-part*`).
- That is the only lever with public evidence of being worth more than +0.001.
- It needs offline training and does not fit into today.

---

## 8. Uncertainties

- The public scores of the Apex variants other than C1 (ranjeet, rabari, hitarthjain, kozykappa) are unknown. The Kaggle API's score ordering does not return the numbers; their positions in it (#9–#16, interleaved with karttik's 0.950) suggest about 0.950.
- **Unverified:**
  - whether Kaggle accepts a non-`submission.csv` output file for this competition
  - whether hidden reruns count toward the weekly GPU quota
  - the MaVIT solo score
  - whether the py3.13 base image already contains `pylibjpeg-libjpeg` (if it does not, the CNXT legs of today's 0.950 forks may decode fewer series on the hidden set)
- The S6 "0.954" is hearsay from romantamrazov's notes.
- The daily limit is assumed to be 5, with the errored submission counting as one, which leaves 4.

---

## Appendix A — inventory of notebooks pulled (39 with code) [M]

The score-order position comes from the `--sort-by scoreDescending` listing; 999 means outside the top 100 or not listed. Families are inferred from data sources and code:
- **R384 / C224:** nartaa CoAtNet SWA checkpoints
- **CNXT:** goodpjw ConvNeXt reader
- **STACK943:** the old DINO/RadImageNet/dreaddevelopment stack
- **HELIOS-COMPACT:** retired bundle
- **PRIVATE-DS:** a private dataset that the API shows as `""`

| score-order # | kernel | last run (UTC) | votes | families | #ds | asserts | raises | internet | accel | code sha1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | heliosli/rsna-knee-abnormality-detection | 2026-10-08 18:51 | 12 | R384+HELIOS-COMPACT(dead) | 3 | 36 | 45 | False | T4 | 8bc08822 |
| 2 | kiyoshiohno/rsna-knee-template | 2026-10-09 11:46 | 11 | R384+CNXT+PRIVATE-DS | 5 | 0 | 1 | False | T4 | 9ce5b674 |
| 3 | heliosli/rsna-knee-blend-gold | 2026-10-08 10:47 | 126 | R384+HELIOS-COMPACT(dead) | 3 | 34 | 36 | False | T4 | 16816165 |
| 4 | evgendvorkin/rsna-versia-7 | 2026-10-09 13:30 | 343 | R384+HELIOS-COMPACT(dead) | 3 | 34 | 36 | False | T4 | a4a85236 |
| 5 | romantamrazov/rsna-knee-dinosaur-v5 | 2026-10-09 14:45 | 146 | R384+CNXT | 2 | 49 | 10 | False | T4 | f7f805f5 |
| 6 | matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed | 2026-10-08 08:32 | 97 | R384+HELIOS-COMPACT(dead) | 3 | 34 | 36 | False | T4 | 16816165 |
| 7 | matterhorn3838/rsna-knee-d4 | 2026-10-08 08:33 | 15 | R384+CNXT | 2 | 17 | 7 | False | T4 | cb4d75f5 |
| 8 | sujanmajhisuzan/rsna-knee-apex-grandmaster-stack | 2026-10-07 22:23 | 129 | R384+CNXT | 2 | 17 | 7 | False | T4 | cb4d75f5 |
| 9 | haideptry/rsna-knee-v2-vs-apex-comparative-study | 2026-10-08 06:50 | 60 | R384+CNXT | 3 | 24 | 8 | False | T4 | d503554c |
| 10 | kozykappa/rsna-knee-gold-gated-triple-reader | 2026-10-08 04:26 | 19 | R384+C224+CNXT | 2 | 30 | 9 | False | T4 | ebdea430 |
| 11 | hitarthjain0/rsna-knee-apex-grandmaster-stack-v5 | 2026-10-08 18:35 | 34 | R384+CNXT | 2 | 24 | 8 | False | T4 | f6016903 |
| 12 | rabari9999/rsna-knee-lb-0-95 | 2026-10-09 07:27 | 25 | R384+C224+CNXT | 2 | 24 | 9 | False | T4 | 773b923b |
| 13 | ranjeet258/rsna-knee-apex-b-under-150 | 2026-10-08 10:57 | 6 | R384+C224+CNXT | 2 | 24 | 9 | False | T4 | 2294d6aa |
| 14 | waterjoe/rsna-knee-v2-velciraptor-dinosaur-speed | 2026-10-08 12:42 | 2 | R384+CNXT | 2 | 18 | 7 | False | T4 | 51abf0f1 |
| 15 | rian55/rsna-floor-velci | 2026-10-08 14:48 | 3 | R384+CNXT | 2 | 17 | 7 | False | T4 | cb4d75f5 |
| 16 | karttikjangid05/does-blend-improves-score-lb-0-950 | 2026-10-09 14:53 | 2 | R384+C224+CNXT | 2 | 21 | 29 | False | T4 | a135e892 |
| 17 | newwang12/rsna-versia-5 | 2026-10-08 21:29 | 0 | R384+CNXT(+unused private ds) | 3 | 17 | 7 | False | T4 | 0ea10766 |
| 18 | sameerk2004/rsna-knee-sil-d5c-triple224 | 2026-10-08 21:39 | 3 | R384+C224+CNXT | 2 | 30 | 9 | False | T4 | ebdea430 |
| 19 | shuheitak/rsna-knee-versia5-fork | 2026-10-09 06:58 | 0 | R384+CNXT | 2 | 17 | 7 | False | T4 | 0ea10766 |
| 20 | rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-95 | 2026-10-09 11:46 | 3 | R384+HELIOS-COMPACT(dead) | 3 | 34 | 36 | False | T4 | b76310de |
| 21 | prvsiyan/the-bee-s-knees-final-rsna-push | 2026-10-08 15:40 | 85 | R384+CNXT+STACK943 | 14 | 0 | 32 | False | T4 | 79830eb0 |
| 22 | nartaa/rsna-knee-0949-anatomical-mirror | 2026-10-07 20:28 | 72 | R384 | 1 | 13 | 6 | False | T4 | a23c2319 |
| 24 | goodpjw2008/rsna-knee-0-949-coatnet-stack-blend-lb-0-949 | 2026-10-08 12:20 | 2 | R384+STACK943 | 15 | 48 | 177 | False | T4 | 44ee5dee |
| 25 | sujanmajhisuzan/rsna-knee-apex-grandmaster-v5 | 2026-10-08 06:49 | 2 | R384+CNXT | 2 | 24 | 8 | False | T4 | 9a35424c |
| 28 | spark328/rsna-dinosaur-v5-top | 2026-10-09 11:42 | 2 | R384+HELIOS-COMPACT(dead) | 3 | 83 | 44 | False | T4 | 6523708b |
| 32 | pjmathematician/rsna-knee-d4-blend | 2026-09-28 00:14 | 261 | PRIVATE-DS+STACK943 | 17 | 39 | 181 | False | T4 | 1da7e343 |
| 36 | sujanmajhisuzan/rsna-knee-tri-specialist-superstack | 2026-10-06 11:56 | 106 | CNXT+STACK943 | 16 | 37 | 188 | False | T4 | 8d25b6e0 |
| 37 | nartaa/rsna-knee-0945-efficient-224crop | 2026-10-07 20:57 | 16 | C224 | 1 | 4 | 5 | False | T4 | ce8b048f |
| 39 | goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944 | 2026-10-06 01:29 | 148 | CNXT+STACK943 | 15 | 35 | 177 | False | T4 | 22578132 |
| 40 | medvax/rsna-knee-independent-2-5d-reader-trial | 2026-10-05 05:40 | 62 | CNXT+STACK943 | 15 | 48 | 175 | False | T4 | 90bdf097 |
| 91 | hitarthjain0/rsna-knee-apex-grandmaster-blend-gold | 2026-10-09 08:33 | 0 | R384+HELIOS-COMPACT(dead) | 3 | 33 | 36 | False | T4 | d584ccc8 |
| – | ayodejiibrahimlateef/rsna-knee-h04-helios-targetwise-20261009 | 2026-10-09 09:38 | 2 | R384+HELIOS-COMPACT(dead) | 3 | 34 | 36 | False | T4 | 43adcc0f |
| – | hengck23/mavit-288-baseline-01-1-5sec-study-1gpu | 2026-10-08 06:14 | 26 | MAVIT | 1 | 0 | 2 | False | T4 | e0e4f37f |
| – | **kragglenote2forwork/rsna-knee-abnormality-detection (yours)** | 2026-10-09 14:42 | 0 | R384 (+compact required, not attached) → **ERROR** | 2 | 36 | 45 | False | T4 | 8bc08822 |
| – | qinghaowang1/knee-fuad0957 | 2026-10-09 01:20 | 0 | STACK943 | 13 | 26 | 75 | False | T4 | 6dcd9d6e |
| – | rabari9999/rsna-knee-sota-2-5d-convnext-dual-reader-0950 | 2026-10-08 14:15 | 1 | R384+CNXT | 3 | 24 | 8 | False | T4 | fc97bcd7 |
| – | ryanholbrook/rsna-knee-abnormalities-efficiency-lb | 2026-10-08 21:53 | 537 | (efficiency LB) | 0 | 0 | 0 | True | – | e77b6ff3 |
| – | sameerk2004/rsna-knee-sil-d5b-dual0950 | 2026-10-08 21:14 | 0 | R384+CNXT | 2 | 24 | 8 | False | T4 | f832ec2d |
| – | sameerk2004/rsna-knee-sil-d5d-compact0951 | 2026-10-09 11:50 | 2 | R384+HELIOS-COMPACT(dead) | 3 | 33 | 35 | False | – | 2b9374a5 |
| – | xianhan/rsna-knee-fast-parent-0-957-candidate | 2026-10-08 03:14 | 3 | STACK943 | 13 | 26 | 75 | False | T4 | 6dcd9d6e |

**Not pulled:** about 120 lower-priority notebooks. The Kaggle API rate-limited to about 1 call per minute (HTTP 429) after the first burst. All priority notebooks named in the brief were pulled.

**Raw materials** (under `scratchpad/research/knee/`):

| What | Path |
|---|---|
| Kernel lists | `lists/*.csv` |
| Public LB CSV | `lb/` |
| Efficiency LB | `out/eff/full_leaderboard.csv` |
| Your fork's error log | `out/user_fork/` |
| Apex / ranjeet logs | `out/` |
| Notebook sources | `nb/` |
| Flattened code | `txt/<kernel>/all.py` |
| Notes | `notes.txt` |
| Parser tools | `tools/` |
