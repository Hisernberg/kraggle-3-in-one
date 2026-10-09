# CASMI 2026 (enveda-CASMI26-molecule-id-mass-spectra): forensic review of the public notebooks

Prepared by "Prof. Fragment, Lineage Hunter" on 2026-10-09. This is research only: nothing was submitted, no kernel was pushed and the git repo was not touched.

**How to read the numbers.**
- **[M] (measured):** I got the value myself from the Kaggle API or from code and data I parsed.
- **[C] (claimed):** the value comes from notebook text or a forum post and I could not verify it.
- **The API does not return notebook scores.** The "score order" below is the rank of a notebook in `--sort-by scoreDescending` [M]. It shows which notebook scores higher, not the actual score.

Artifacts (all under `scratchpad/research/casmi/`):
- `nb/` holds 85 pulled notebooks; each `*.py.txt` is a cell dump made with `tools/nb2py.py`.
- `out/nb_table.csv` lists every notebook with its score-order position, votes, GPU flag, extra datasets and detected flags (`tools/table.py`).
- `tools/sim.py` compares notebooks (normalised-line Jaccard) and prints the diff of any pair (`python3 -I tools/sim.py diff A B`).
- `out/embedded/` holds the modules embedded in the notebooks (engine 2, PubChem channel, `pc_join.py`, `fusion_core.py`), extracted with AST literal-eval and never executed.
- `topics/` holds the forum threads; `topics/pages_clean.txt` holds the competition pages.

---

## 0. Executive summary

1. **This is a code competition.** The visible `test.parquet` (400 molecules, 1,213 spectra [M]) is a placeholder made from training spectra. Kaggle reruns the notebook on the hidden test (about 1,500 spectra of about 400 molecules). Limits: 9 h, no internet, T4 GPU allowed. All of this is from the competition pages [M].
   - **Public LB:** about one third of the test, roughly 133 molecules. One molecule moved to rank 1 is worth about +0.0075. The standard error of the difference between two submissions is about 0.016 [C, two independent forum measurements].
   - **Rerun noise:** byte-identical notebooks have scored 0.005 to 0.013 apart [C: prvsiyan; flexonafft got 0.420 versus 0.433 for the same V1.1 pipeline].
2. **Metric: MRR@25** [M, official metric notebook v14].
   - Each answer is up to 25 SMILES separated by `;`.
   - Matching uses an InChIKey-14 key, computed after `Chem.RemoveStereochemistry` and then RDKit 2026.03.3 `TautomerEnumerator.Canonicalize`.
   - Only the first match counts. Empty entries are dropped. The solution side may hold several accepted SMILES separated by commas.
   - **Metric update of 2026-10-07:** stereochemistry is now stripped *before* the tautomer step. No public notebook breaks because of it.
   - **Rerun-data fix of 2026-10-07:** an electron-mass offset (+0.55 mDa) on positive-mode peaks was removed. All submissions were rescored, so older LB numbers moved by about ±0.01.
3. **The "class-1 leak" is dead.** The host states [M, thread 745715] that IDs do not overlap between the dummy test and the rerun test.
   - **Will fail:** `imranarif536/casmi26-v44-pairtail-locked-top1` and `nursrijan/forensics-lab-dual-shield-two-ranker-sota` hard-code the 400 visible `molecule_id`s with unguarded `dict[...]` lookups. On remapped IDs that raises a KeyError, which means a submission error. Do not fork them.
   - **No-op:** `lszlst/casmi26-b425` hard-codes the IDs behind a guarded `.get`, so on the hidden test it does nothing.
   - **Not a leak:** the `lszlst probe-top1/top2` notebooks only truncate lists to measure acc@1 and acc@2.
4. **One family holds the whole public frontier.** It descends from ahmedberatozer v4n plus seyitkaangunes engine fusion; the current lead is obstacledeveloper V1.1. Every top-35 public notebook uses the same 14 to 16 datasets.
   - **Best displayed score:** about 0.433 [C], obstacledeveloper V1.1. haideptry v32, which copies V1.1, is #1 by score order [M]. Realistic expectation is 0.425 to 0.44.
   - **Most other "improvements" are noise:** the sovereign, apex, MassBank, FIORA, 4-seed, diversify and adduct-reconciliation variants are all inside the noise band, and several rank below the base.
5. **One proven orthogonal ingredient is missing from the frontier:** the **PubChem JOIN** from huseyinemreaksoy (+0.011 [C], 0.409 → 0.420 in its own branch). It hands up to 50 PubChem-only proposals to the main LightGBM ranker. The frontier instead places them only in fixed slots, plus a promotion rule.
6. **Leaderboard context.** Top teams scored 0.484, 0.481 and 0.459 [M, team-submissions API]. In an API snapshot taken earlier today (≈14:53 UTC), #10 was 0.448 and #20 was 0.440 [M]. The +0.04 to 0.05 gap above the public frontier cannot come from knob-tuning. It must be a structural ingredient: class-2 retrieval, class-3 de novo generation, or better rankers or forward models.

---

## 1. Data and submission format [M]

- **`test.parquet` columns:** molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities, base_peak_intensity, adduct, ionization_mode, instrument_type (always timsTOF), precursor_mz, collision_energy_orig / _ev / _units.
- **Visible test:** 79% [M+H]+, 16% [M-H]-, plus formate, Na, NH4, K and Cl adducts. There are 1 to 9 spectra per molecule (mean 3.0).
- **Hidden-test adducts:** the hidden set uses 10 adducts, including `[M-H2O+H]+` and `[M-2H2O+H]+`, which are absent from the visible file. Engine 2 (`pv.py`) handles both.
- **`sample_submission.csv`:** `molecule_id,smiles` with 25 `;`-separated guesses (24 separators).
- **`train.parquet`:** about 2.5 M spectra and about 275 k structures (3.0 GB). It is a compilation of MassBank, GNPS, MoNA, RIKEN, MS-DIAL, Enveda-180 and enveda-np-examples. Only 46% of it is timsTOF.
- **Novelty classes** (data page):
  - **Class 1:** public reference spectra exist (library search).
  - **Class 2:** no public spectra, but the structure is in PubChem or COCONUT.
  - **Class 3:** not in PubChem (de novo).
  - **Class mix:** hidden by the host ("hope there are enough novel structures"). A forum estimate puts it at 16 / 45 / 39%. That would cap a retrieval-only system at about 0.61.
- **Submission rules:** 5 submissions per day [M, `submission-limits`]. Today: 0 used, 5 remaining. The user's best is 0.328, from haideptry's fast cosine baseline v1 (2026-09-21) [M].

## 2. Rules and licences that change which notebooks are usable

**Approved by the host** [M, threads 745841, 744338, 743569, 745148]:
- ICEBERG/GLACIER (MassSpecGym checkpoints, MIT)
- DreaMS
- ChEBI and LIPID MAPS
- Enveda-180 Zenodo (CC BY 4.0, including the 816 structures that are absent from train)
- Models trained on `train.parquet` and redistributed artifacts ("can be treated effectively as open-source")

**Forbidden:** NIST and anything derived from it [M, 744470]. The host also said the end-of-competition compliance check "can remove teams from the leaderboard entirely, not just disqualify from prizes".

**Unresolved:**
- **ahmedberatozer datasets** (`casmi26-v4b-models`, `-v3-models`, `-v2-pool`, `-fpnet-full1`, `-pubchem-tier`, `-glacier`, `-iceberg`) are licensed "Other" with a non-commercial note [M, metadata]. They are train- and COCONUT-derived, which the host's general rule calls fine. Code licence is unclear. **The whole public frontier depends on them.**
- **FRIGID checkpoints** are CC BY-NC (thread 746430, no answer). `lszlst b431-denovo` uses them, and its two de novo datasets are **private** (empty `dataset_sources` entries [M]). Forks therefore silently run without de novo.
- **Other external libraries** (licences read from dataset metadata [M]):
  - `samartalwar/casmi-2026-spectral-library-massbankharmonized` (rajrajak v220–v240, vibhanshus, jyxxxx v240) is **CC-BY-NC-SA-4.0**. A notebook that uses it is non-commercial, so it is prize-ineligible and may be removed from the leaderboard. Avoid it.
  - `takumuhata/casmi26-extlib` (lehau v30/v32, nursrijan v12) is labelled CC0-1.0, but the upstream provenance of its 200 k spectra is undocumented.
  - FIORA (`ahmedberatozer/casmi26-fiora-os-v1`, lehau v32) is the MSnLib-v7-only checkpoint (CC BY 4.0 data, MIT code). It looks clean.
  - hengck23 DreaMS dataset: MIT.
  - prvsiyan COCONUT bank: CC BY 4.0.
- `eveliaveldrine/rdkit-wheels` and `metric/rdkit-2026-3-3-wheel` are needed only for environment pinning.

## 3. Lineage tree (from code diffs, `tools/sim.py`)

```
host: inversion/casmi-denovo-tutorial (spectrum->BPE-SMILES transformer; weak; 303 votes)
haideptry fast-spectral-cosine baseline (0.29-0.33, 240 votes)  <- the user's current 0.328
prvsiyan analog-propagation (0.34) -- megayak two-rankers-one-engine (0.337) -- llccqq624 next-direct w088 (0.341)
      (becomes "ENGINE 2": COCONUT+ChEBI/LIPID MAPS+train pool, analog propagation, MetFrag-lite, 2x HistGBR, BIO+AFIX)
ahmedberatozer v2 -> v3 -> v4b (FPNet A, 0.354) -> v4f (+ICEBERG 0.358) -> v4g (0.366) -> v4h (+GLACIER)
      -> v4i (PubChem tier N1=5000) -> v4l (ICE/GL lambda=1, 0.380) -> v4m (fpnet_full1) -> v4n (0.384)
   |-- wangpenghua v17..v29: popularity prior in PubChem list + pool popularity mu=0.15 + GLACIER [M+H]+ only (0.415 [C], private pop arrays)
   |      `-- dmitriigluzdov publishes the popularity arrays (dataset) + "From Spectra to Structures" (PC join + v18 settings; 0.411 [C])
   |-- seyitkaangunes e5c / v4n engine-fusion-union: RRF(alpha=0.6, K=3) with ENGINE 2 + forward models on the union (0.399 [C])
        |-- huseyinemreaksoy v4b-0.409 (pop 0.25, frag-gap, ICE/GL off for lib hits) -> v4n Fusion + PubChem JOIN 0.420/0.421 [C]
        |      |-- guhongbin v408 (verbatim copy), senanuretin base-* (4 variants), huaiansun opt-D (FP_BANK ens, FUSE_SKIP_LIB 0.97)
        |      `-- gengsr fusion-glacier-MH 0.413 -> lehau v26 (0.413) / v28 / v29 pairtail / v30 extlib (0.418 [C]) / v32 FIORA tri-forward; remagen v29 xgb
        |-- bobthebot v10 apex (pool pop 0.15, 0.401) -> matterhorn supernext (GL [M+H]+, 0.405) -> imranarif v42-v44 PairTail + HARD-CODED TOP-1 (LEAK, 0.417) -> nursrijan forensics v12 (LEAK)
        `-- lszlst "CLAW" b4xx: b415c ICE fix, b416 PubChem PROMOTION (0.417), b420/b421 GLACIER-only (+0.015), b425 leak probe, b428-b431, probe-top1/2
               `-- bobthebot v18 Sovereign Zenith (promotion + BASE_ICE off + GL-only on fused list + DreaMS mass tail) 0.425 [C, replayed by uninhibitedscholar]
                     |-- obstacledeveloper V1 (=v18, 0.425) -> V1.1 metric-audit & safe tail (RDKit pin + metric-identity dedup + overflow tail) 0.433 [C]
                     |      |-- haideptry v32 "0.350+" (COPY of V1.1; non-fatal stage checks, cp-tag wheel pick, py3.12 image)  score-order #1 [M]
                     |      |-- flexonafft adaptation (+cross-formula GLACIER overlay on weak-lib [M+H]+)  #2 [M]  (their own V1.1 replay: 0.420 [C])
                     |      `-- evgendvorkin "Enveda CASMI 2026" (V1.1 re-documented, 162 votes)  #35
                     |-- amanatar v18 (= v18 + CPU fallback, GPU DISABLED)  #3 [M];  spark328 top-public (copy of amanatar)
                     |-- arman1o1 v11 (= v18, IS_RERUN=True)  #4;  nursrijan w/-sovereign, goodjane chem-v3-1 (copies)
                     |-- uninhibitedscholar own153 (own 153-feature rankers) 0.417 [C] (regression)
                     |-- ghazaros v19 clean-fusion (+adduct reconciliation, clean-entropy, Enveda-180 DreaMS cosine interleave) -> synthreaper v22  #6/#11
                     `-- rajrajak v170..v240: v180 = 4-seed engine 2, N_ANALOG 250, adduct reconciliation, rank-1 shields, formula diversify, pool pop 0.25 -> 0.428 [C by ahmedberatozer]
                            |-- ahmedberatozer v6a (v180 ported to py3.12), lszlst b429 / probe-top1/2 / b431-denovo (+FRIGID rank-2 hedge), matterhorn nextgen,
                            |   lavinwins "LB top 1" (= v180 with comments rewritten, no new code), arman v13
                            `-- v200-v240: MassBank-harmonized promotion, MetFrag-direct... (all rank BELOW v180 in score order)
hengck23 DreaMS demo + dataset (dreams_np_* embeddings/SMILES used for tails and retrieval)
```

## 4. What each top notebook does

The frontier pipeline (V1.1 / v18 family) has these stages. The engine code is in `ahmedberatozer/casmi26-v4b-models` and `out/embedded/obstacle/`.

1. **Engine 1, v4n.** Library built from `train.parquet` (cached spectra). Candidate pool of 711 k structures: 274 k train structures plus 437 k COCONUT, within 10 ppm of the median neutral mass, plus "generate=True" derivative candidates made by editing library analogs (the class-3 attempt).
   - **Features:** FPNet (a transformer that predicts substructure-fingerprint bits from the spectrum; `fpnet_full1`), the fe_v4 families (derivation priors, fragmentation 2.0, analog-structure relations, DreaMS-FP views) and library/analog matches.
   - **Ranking:** a 4-booster LightGBM, top 60 kept.
2. **Pool popularity re-rank:** `z(ranker) + 0.15·(log1p #PubChem substances + log1p #PubMed)`. It applies to pool rows only.
3. **PubChem-only channel:** 105.9 M PubChem structures (150–1250 Da, stereo-stripped). FPNet A+B shortlist of 5,000, then a full score, then popularity 0.25, then the top 25 keys absent from the pool.
4. **ICEBERG + GLACIER forward simulators** (MassSpecGym, MIT). `BASE_ICE=False`, so base candidates are not sent; the ICE input holds engine-2 and PubChem candidates. GLACIER sees [M+H]+ spectra only.
5. **Gated PubChem merge.**
   - If `lib_max ≥ 0.9`, the list is left untouched.
   - Otherwise PubChem proposals go into slots 2/4/6/8/10 (when the fingerprint-score margin over the pool, `rel`, exceeds 600) or 4/8/12/16/20.
   - **CLAW promotion:** PubChem top-1 goes to rank 1 if (`S>6` and `pop≥5`) or `rel>250`.
6. **Engine 2:** prvsiyan analog propagation plus two HistGBR rankers over COCONUT, ChEBI/LIPID MAPS and train. BIO and AFIX features, top 40.
7. **Fusion:** RRF with `1/(3+r_main) + 0.6/(3+r_eng2)`, top 40. Then a GLACIER-only z-rerank inside same-formula groups (`CELL19_ICE=False`), then the top 25.
8. **V1.1 additions:**
   - RDKit pinned to 2026.03.3.
   - De-duplication with the exact metric identity (`safe_extend`).
   - Tail filled from ranked overflow, then engine 2, then PubChem, then DreaMS mass-matched SMILES.
   - Monotonic assertions.
   - Commit runs a 12-molecule smoke test (`IS_RERUN` is detected by test signature or env var); the rerun processes everything.

**Runtime** [M from commit logs; C for the rerun]:
- Commit: 1,082 s for obstacle and 1,166 s for haideptry on T4. Fixed engine-2 overhead is about 640 s, plus about 7 s per molecule.
- Rerun: about 3.5–4.5 h on T4 [C]. ICE budget 5,400 s, GLACIER budget 4,000 s.

**Table: key notebooks**

| Notebook (score order [M]) | Claimed LB | Lineage, what is new | Extra data | Validity notes |
|---|---|---|---|---|
| haideptry/0-350-casmi-2026-v32-ensemble (#1) | title "0.350+" is stale; code = V1.1 | V1.1 copy; ICE/GL stage failures only WARN (V1.1 asserts); cp-tag wheel selection; py3.12 image | std 16 | ok; best anchor |
| flexonafft/...adaptation-experiments (#2) | "confirmed 0.420", overlay unscored | V1.1 + cross-formula GLACIER over ≤125 candidates for weak-lib [M+H]+ molecules | std | ok; higher variance |
| amanatar/casmi26-v18 (#3) | — | bob v18 + CPU fallback, **GPU off** (ICE/GL on CPU within budgets, so fewer molecules rescored) | std | ok; no GPU quota; CPU runtime near limit |
| arman1o1/...-v11 (#4) | — | bob v18 with `IS_RERUN=True` (full run on commit) | std | ok; commit costs about 4 h GPU |
| obstacledeveloper/...-v1-1 (#5) | **0.433** (V1 0.425) | v18 + RDKit pin + metric dedup + safe tail | std | fatal assert if ICE/GL stage ≠ ok |
| synthreaper v22 (#6) / ghazaros v19 (#11) | "personal best 0.433" | v18 + adduct reconciliation, clean-entropy tail, Enveda-180 DreaMS cosine interleave (ranks 8–24), formula cap | std | ok |
| rajrajak99 v180 (#7) | 0.428 [C by ahmedberatozer v6a] | 4 seeds, N_ANALOG 250, shields, diversify, pop 0.25 | std | ok; no gain over v18 |
| bobthebot369 v18 (#8) | 0.425 | promotion + GL-only + DreaMS tail | std | ok |
| huseyinemreaksoy v4n fusion + PubChem (#19) | **0.420** (0.399→0.404→0.404→0.409→0.420 ablation ladder) | config-driven (`CFG`), `fusion_core.py`, **PC JOIN 50, lib<0.7**, frag-gap, ICE/GL lib gate, pop 0.25 | + `dmitriigluzdov/casmi26-fold-safe-fpnet` | ok; cleanest codebase for experiments |
| rajrajak v220–v240 (#17,18,33,34) | "target 0.47–0.49" | MassBank-harmonized cosine promotion tiers | samartalwar MassBank (**CC-BY-NC-SA**) | licence problem; all rank below v180 |
| lszlst b431-denovo (#29) | pre-registered +0.003…+0.066 | v180 + FRIGID DLM conditioned on MIST fingerprint, best at rank 2 (hit 43/149 pool-absent [C]) | 2 private datasets | **cannot be reproduced; CC BY-NC** |
| lehau007 v30 / v32 (#54/#80) | 0.418 / — | huseyin branch + takumuhata extlib (200 k spectra, bonus 0.05); v32 + FIORA (neg mode) | extlib, FIORA | licence/provenance check; no LB gain |
| wangpenghua v29 (#43) | 0.415 (private arrays) | GLACIER [M+H]+, pop priors | dmitrii arrays (0.393 with them [C]) | ok |
| dmitriigluzdov from-spectra-to-structures (#56) | v36 0.411 | huseyin JOIN + v18 settings; identity-disjoint folds and NP-panel benchmark datasets | own datasets | ok; best documentation |
| imranarif v44 (#36), nursrijan forensics v12 (#57) | 0.417 / — | PairTail LambdaRank + **hard-coded 400 visible IDs (unguarded)** | extlib, FIORA | **will error or is invalid** |
| lavinwins zero-drift "LB top 1" (#26) | — | v180 with all comments and docstrings rewritten | std | no new content |
| prvsiyan analog-propagation | 0.340 (twice) | ENGINE 2 origin; reports rerun noise 0.005–0.007 | — | ok |
| inversion host tutorial | — | spectrum→SMILES BPE transformer (de novo starter) | — | ok |

"std" means the standard set of 14 to 16 datasets: ahmedberatozer v2-pool / v3 / v4b / fpnet-full1 / iceberg / glacier / pubchem-tier, prvsiyan fp-models / ranker-features / coconut / chebi-lipidmaps, megayak simulated rows, dmitrii popularity prior, hengck23 DreaMS, the metric rdkit wheel and eveliaveldrine rdkit wheels.

**Forum ablations worth keeping** [C]:
- **GLACIER-only beats ICE+GL:** b421 0.417 versus b420 0.402.
- **Base-list ICE off:** +0.009.
- **Forward-model weight 2.0 everywhere:** 0.352 (it destroys library hits).
- **Blind PubChem expansion:** 0.335 → 0.205.
- **PubChem in the tail only:** +0.001.
- **Promotion rule disagrees between sources:** lszlst +0.024, dmitrii −0.012.
- **Pool popularity 0.15→0.25:** +0.005 for huseyin, −0.001 for dmitrii.
- **Fragment re-score on all molecules:** 0.390 (hurts); "gap" mode is neutral or positive.
- **Filler or invalid SMILES** once zeroed a submission (old metric). Keep every guess a valid SMILES.

## 5. Agreement and coverage of submission files: not meaningful, so not computed

Published outputs come from the commit run. That run is a 12-molecule smoke test on the train-derived placeholder, and the other 388 rows are 'CCO' placeholders. Only arman1o1 v11 runs all 400. Those visible molecules are by construction class-1 train spectra, so their agreement says nothing about the hidden test.

Output folders also hold thousands of files (`ice_site`), which makes them expensive to list under the 429 rate limits. I pulled haideptry's outputs: in the smoke audit 1 of 12 molecules changed and ICE/GL scored 11 of 12 [M].

**Real complementarity is architectural:**
- The huseyin branch (JOIN, ICE+GL on base, frag-gap) and the V1.1 branch (promotion, GL-only, safe tail) differ in how PubChem candidates reach rank 1 and in how hard forward models rerank. That is the main source of decorrelation for final selection.
- Engine 2 versus engine 1 is already fused in every frontier notebook.

## 6. Five submissions for today (ranked)

Each one is a Copy & Edit followed by "Submit". Commit runs take about 20 min on T4 (smoke). The scoring rerun takes about 3.5–4.5 h on T4. Start all five early; they can queue in parallel.

**S1. Anchor.** `haideptry/0-350-casmi-2026-v32-ensemble`, unchanged.
- Pin the original environment (py3.12 image `…bf3fc647d461`) and attach the same 16 datasets.
- Expected public score: 0.43–0.44. V1.1 displays 0.433 [C], and this copy is #1 by score order [M].
- Why this rather than obstacle V1.1: the code is identical, but here ICE/GL failures do not crash the run.
- Validity after the metric/rerun update: fine. It uses the exact metric identity and no leak. Licence risk is limited to the ahmedberatozer datasets, which affects everybody.

**S2. Best expected value.** S1 plus the huseyin PubChem JOIN. Engineering takes about 1–2 h; reuse `out/embedded/huseyin/EMBED__pc_join.py` and `EMBED__ranker_infer.py` verbatim. Changes:
- **(a) Cell 8:** write `CORE.replace('K = 25', 'K = 50')`, so the PubChem list is 50 deep.
- **(b) Cell 10/11:** write `pc_join.py` (and `ranker_infer.py`, which is only needed for non-public rankers) to `/kaggle/working`. Then:
  ```python
  from casmi import frag as fragmod
  pc_join.install(E, V, 'first')
  ```
  Call `install` right after `V = v1engine.V1FE(...)`. This is huseyin cell 14.
- **(c) Cell 13 loop:** port huseyin cell 16's join block verbatim (lines ~535–590 of its `.py.txt`). It does the following:
  - Gate: `pc_join.library_max(E, spectra, target) < 0.7` (only applied on the rerun).
  - Select: `select_injection(p, 50)`.
  - Build: `recs = build_records(..., chem, fragmod, P.bits, fz=...)`.
  - Mass filter: keep `|mass − target| ≤ 30 ppm`.
  - Inject: `set_injection(E, recs)`, then `V.run(...)`, falling back to no injection if the joined run fails.
  - Track: `inj = injected_mask(E, C)`, plus the `_out` map so the PubChem SMILES form is kept in the list.
  - Memory: clear the `fragnet`/`frag2` per-SMILES caches every 50 molecules (about 33 MB per molecule at N=50).
  - Ranker: keep `rank_score` (the public ranker, `RANKER='public'`). Injected rows sit last in candidate order, so stable ties already put them behind pool rows.
  - Pool popularity: injected rows have `pid < 0`, so the popularity re-rank leaves them in place with a `|gen` formula tag. huseyin's code behaves the same way.
- **(d)** After the base lists, truncate every `PC[mid]` to its first 25 entries (`pc`, `pc_fz`, `pc_keys`; huseyin keeps a separate `PC_DEEP`). The ICE input, slot merge and promotion then behave as before.
- Keep the promotion rule. Proposals that the ranker already placed are skipped by `merge()`.
- Expected: +0.005 to +0.012 over S1 [C: +0.011 in huseyin's branch], giving about 0.44–0.45. Runtime: +30–60 min. Validity: fine, with moderate engineering risk. Smoke-test the commit and check the log line "PubChem join".

**S3. Second family, for diversity.** `huseyinemreaksoy/casmi26-v4n-fusion-pubchem-on-public-0-421` with a single edit: extend its `CFG.update({...})` line.
- **Add:** `'GL_MH_ONLY': True, 'POST_ICE_LAM': 0.0, 'FILL_25': True`. These are the v29, b421 and V1.1 lessons: GLACIER on [M+H]+ only, GLACIER-only after fusion, and safe fill to 25.
- **Optional:** `'ICE_LAM': 0.0` mimics V18's base-ICE-off. Leave it out on the first try.
- Expected: 0.425–0.44. Engineering: trivial. Runtime 3–4 h on T4. Validity: fine. Different architecture from S1/S2, so it is the best hedge for the 67% private set.

**S4. Zero-GPU bank.** `amanatar/casmi26-v18`, unchanged (CPU notebook).
- #3 by score order [M], so it ties or beats V1.1 within noise. It is bob v18 on CPU, where ICE/GLACIER run on CPU within the same time budgets.
- Expected: 0.42–0.44. It costs no GPU quota.
- Risk: CPU runtime is close to the 9 h limit. It completed for its author, but if it times out the submission fails. It is also correlated with S1.

**S5. High variance.** `flexonafft/casmi26-public-baseline-adaptation-experiments`, unchanged.
- It is V1.1 plus a cross-formula GLACIER re-ordering of up to 125 candidates on weak-library [M+H]+ molecules. #2 by score order [M], but which version produced that score is unknown.
- Expected: 0.42–0.45. GPU T4, about 4.5 h.
- If S2 is ready in time, use this slot instead for "S2 without the promotion rule": the JOIN may make promotion redundant, and the two sources disagree on its sign.

Do not submit:
- the leak notebooks (v44, forensics v12);
- the FRIGID fork (b431: weights are private and CC BY-NC);
- the rajrajak MassBank and "target 0.47+" notebooks (no gain by order; the library is CC-BY-NC-SA);
- lavinwins (a renamed copy).

**What today can realistically reach:** public 0.44–0.45 at best. Top-10 needs about 0.448 (#10 in the earlier API snapshot) and the user cited ≈0.455 now. One noise-lucky run of S2 could get there, but a stable top-10 needs the structural work in §7.

## 7. Two-month path to top-10 (0.46 now, probably 0.48–0.50 by December)

1. **Week 1: build a measurement harness.** With a ±0.016 public LB, offline validation is mandatory.
   - Use huseyin's `VALIDATION` mode (enveda-np-examples held out, all spectra purged, class-2-like).
   - Use dmitriigluzdov's `casmi26-identity-disjoint-validation-folds` and `natural-product-panel-benchmark`.
   - Cache stage dumps (`DUMP_STAGES`) and re-fuse offline (`fusion_core.build_submission` is a pure function).
   - Report coverage@25, top-1 and MRR separately for class-1-like, class-2-like and pool-absent molecules.
2. **Weeks 1–2: class-2 retrieval (about 45% of molecules).**
   - Tune JOIN depth and gates jointly with promotion and slots offline.
   - Add an NP-likeness or COCONUT-proximity prior to the PubChem shortlist.
   - Learn μ (popularity) per lib_max bucket.
   - Rank PubChem candidates with the same LightGBM, adding a "source" feature (instead of the 'first' order trick).
3. **Weeks 2–4: rankers and fingerprints.**
   - Retrain FPNet on identity-disjoint folds with timsTOF weighting, plus Enveda-180 Zenodo (allowed; 816 extra structures).
   - Retrain the ranker as LambdaRank (`truncation_level=25`) on simulated class-2 rows built with *the same* candidate generators used at inference.
   - Average seeds; seed noise is ±0.006 [C].
   - Feed GLACIER and ICEBERG scores as ranker features instead of post-hoc z-sums. Fixed λ is fragile: λ=2 dropped the score by 0.05 [C].
4. **Weeks 3–6: class 3, the biggest untapped mass** (forum estimate about 39%; retrieval ceiling 0.61).
   - Train a de novo generator conditioned on the predicted fingerprint and the inferred formula. Use MIST-style encoder → DLM/DiffMS-style decoder, retrained from Apache/MIT code on `train.parquet`. That keeps the licence clean: no FRIGID CC BY-NC weights.
   - Add biotransformation edits of top library analogs (glycosides, methylation, hydroxylation, acylation shifts that match Δm).
   - Insert one generated candidate as a **rank-2 hedge** only when lib_max is low and the PubChem margin is low. b431's own math: hit at rank 1 for 33 of 149 pool-absent molecules [C]; a novelty simulation overstates reach about 3× [C].
   - Expected gain if real: +0.02 to +0.05. This is probably what separates 0.48 from 0.44.
5. **Weeks 4–7: forward models.**
   - Fine-tune GLACIER/ICEBERG (MIT MassSpecGym weights) on the timsTOF subset of train.
   - Add negative-mode coverage (16% of the test is [M-H]-, and there the forward models are currently blind).
6. **Week 8: final selection and compliance.**
   - Select one max-public run and one different-family robust run.
   - Make every stage non-fatal with fallbacks, keep runtime under 6 h, and pin the image and RDKit.
   - Audit licences: no FRIGID weights, NIST, MassBank NC records or unknown-provenance libraries. Retrain any ahmedberatozer artifact you keep if the host's licence answer turns negative.

## 8. Discussion facts used (thread IDs)

- 746954: rerun data fixed (electron mass).
- 746878: metric RemoveStereochemistry.
- 745715: leak thread; host says no ID overlap.
- 745841: approved inputs; compliance check can remove teams from the leaderboard.
- 744470: NIST banned.
- 742274: class 3 is absent from PubChem and COCONUT; `A;;B` counts as 2 guesses.
- 744556: solutions are stereo-stripped.
- 743569 and 745148: Enveda-180 Zenodo allowed.
- 743254, 742088, 741659: noise floor, isomer errors account for about 98% of in-pool misses, pool-dilution evidence.
- 746430: FRIGID licence open.
- 743395: positive-mode one-electron offset.
- 744011: train is 46% timsTOF versus test 100%.
