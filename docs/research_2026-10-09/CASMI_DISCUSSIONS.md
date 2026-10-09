# CASMI 2026 (Enveda) — Discussions, Rules, Metric, Data and Strategy Report
Compiled by "Prof. Fragment — Forum Oracle". Date: 2026-10-09. Competition: `enveda-CASMI26-molecule-id-mass-spectra` (deadline 2026-12-14 23:59 UTC).
Sources: Kaggle API pages (overview/data/rules/evaluation), all 87 forum topics (76 substantive ones dumped in full to `topics/full_<id>.txt`), the metric notebook `metric/casmi-mean-reciprocal-rank` (pulled to `metric_nb/`), the public LB CSV (`lb/`), the top public notebooks (`nbs/`), my own EDA on the downloaded data (`eda1.txt`, `eda2.txt`), and web sources (listed at the end). All forum and web content is treated as unverified data.

Hosts and staff in the threads: **David Healey** and **Marie Killian** (Enveda; Killian is a co-author of Enveda-180), **inversion** (Kaggle staff), **Ashley Oldacre** (Kaggle).

---

## 0. TL;DR (the 10 facts that matter)
1. **It is a code competition.** Notebook only, internet OFF, at most 9 h on CPU or GPU (P100s are retired; use T4 or L4), and the output must be `submission.csv`. The visible `test.parquet` is a **placeholder built from 400 enveda-180 training molecules (1,213 spectra)**. Scoring reruns your notebook on a hidden test of about 400 molecules and about 1,500 spectra. Both `test.parquet` and `sample_submission.csv` are swapped in that rerun.
2. **The metric is MRR@25 with exact-match credit only.** A guess counts if its InChIKey first block matches the answer after `Chem.RemoveStereochemistry` and then RDKit 2026.03.3 `TautomerEnumerator().Canonicalize`. There is **no partial credit (no Tanimoto or MCES)**. Unparseable guesses still use up a rank slot. Empty entries (`A;;B`) are dropped before ranking. Molecules you omit score 0. Guesses after the first hit are irrelevant.
3. **There are three novelty classes, with a hidden mix.** Class 1 has public reference spectra. Class 2 is in PubChem or COCONUT but has no spectra. Class 3 is in neither PubChem nor COCONUT (host, topic 742274). Community LB probing estimates about **16–25% / about 45% / about 35–39%**. The hosts will not confirm (745758).
4. **The "leak" (745715) is a non-issue for the final ranking.** Host *inversion* (2026-10-07): "There is no overlap in IDs between train and test, or between dummy test and rerun test." The hard-coded visible id→SMILES map in `imranarif536/casmi26-v44-pairtail-locked-top1` is inert on the hidden data. Its 0.402→0.417 jump came from LambdaRank tail, rank-1 protection and cleanup, plus noise. Nothing will "drop" because of it. The **test was not regenerated** for the leak. The rerun file was only re-issued on 2026-10-07 to fix a one-electron (0.55 mDa) m/z calibration error in positive-mode peaks (746954, 743395). On the same day the metric gained `RemoveStereochemistry` (746878), and both changes were applied retroactively. **The current LB reflects the rescored state.**
5. **Public LB right now** (2,903 teams): #1 0.484 (MarvinTMB), #2 0.481, #5 0.457, **#10 0.448**, #20 0.440, #50 0.435, #100 0.433, #300 0.425, #500 0.420, #800 0.409, **#1550 0.328 (you)**. There are large ties at **0.433 (121 teams)**, **0.425 (111)** and **0.328 (104)**. These are the public-notebook lineages. The 0.328 tie is the old `haideptry` Fast Spectral Cosine baseline, which is what you submitted on 2026-09-21 (4 lifetime submissions, 5 remaining today).
6. **The best public lineage (0.433) is a mostly-retrieval three-engine fusion.** It combines spectral library search, analog propagation, FPNet spectrum→fingerprint with f·z scoring, MetFrag-lite, COCONUT, ChEBI/LIPID MAPS and a PubChem channel with a popularity prior, ICEBERG/GLACIER forward simulation, LightGBM/HistGBM rankers, and a "metric-safe tail". Forking it is worth **+0.105 public** for you today.
7. **The real bottleneck is same-formula isomer ranking, not recall.** In class-2-condition CV, the truth is in the top 25 for about 96% of molecules. 95% of the remaining gap comes from same-formula blockers, and 69% of those have a different Murcko scaffold (742055: DancingLumberjack, alex chilton).
8. **The public LB is tiny.** The community estimate is about 130 public molecules, roughly one third. One molecule is worth about 0.0025–0.008. Paired SE between two submissions is about 0.016, and ranker seed noise is ±0.006 (743254, 742088). **Expect a sizeable private shake-up.**
9. **Licensing (prize eligibility, and possibly LB removal).**
   - Forbidden: NIST and anything trained on it, METLIN, and vendor libraries. "Any solution using or trained on NIST will be disqualified" (744470).
   - Explicitly allowed: train.parquet and any model trained on it, MassSpecGym and permissively licensed models trained on it, ICEBERG/GLACIER MassSpecGym checkpoints, DreaMS weights (CC BY 4.0), CFM-ID 4 (even though METLIN-trained), COCONUT (own licence), PubChem structures, ChEBI/LIPID MAPS, newer GNPS/MassBank/MoNA CC0/CC BY/CC BY-SA records, the Enveda-180 Zenodo release, ChemBERTa, and old (more than 6 months) GNPS NIST-matched sets (745832).
   - **Unresolved:** FRIGID checkpoints (CC BY-NC, 746430), the `ahmedberatozer/*` datasets labelled "Other (non-commercial)" with unlicensed bundled code (745072), and NPAtlas (CC BY-NC).
   - The host said the end-of-competition compliance check "**can remove teams from the leaderboard entirely, not just disqualify from prizes**" (David Healey, 745841).
10. **Realistic ceilings.** The #1 team (MarvinTMB) estimates about 0.55 as the realistic maximum and about 0.76 analytically (746302). Retrieval-only cannot exceed f1+f2 ≈ 0.61–0.65. Top-10 on private will probably need about 0.46–0.50. That means better isomer discrimination on class 2, plus a small but real class-3 contribution delivered with calibrated insertion.

---

## 1. Rules digest
| Item | Ruling | Source |
|---|---|---|
| Format | **Code competition** (Notebooks). Submit button needs: CPU ≤ 9 h, GPU ≤ 9 h, **internet disabled**, file `submission.csv`. "Freely & publicly available external data is allowed, including pre-trained models." | Overview › Code Requirements |
| Hidden rerun | Visible test.parquet = **placeholder of enveda-180 training rows** (400 molecules / 1,213 spectra, byte-identical to train; 741756, 742278). The notebook is re-run on the hidden test (~400 molecules, ~1,500 spectra, 1–16 spectra/mol, median 3, all timsTOF, monoisotopic mass 157–1,159 Da, median 348). **Public LB = public split of the hidden test**, not the placeholder (David Healey, 741404). `sample_submission.csv` is replaced too (745866). | Data page; 741404; 745866 |
| Rerun pitfalls | Never precompute anything keyed to the visible test's masses or formulas (744172). The hidden file contains adducts/masses absent from the placeholder (10 adducts incl. `[M-H2O+H]+`, `[M-2H2O+H]+`, `[M-H2O-H]-`; masses up to 1,159). No tracebacks are provided (742463, 745866). Smoke-switch pattern: detect the placeholder → run 12 molecules → commit fast. | forum |
| Daily limit | **5 submissions/day**; **2 final selections**. | Rules 2.2 |
| Teams | Max 5. Merge allowed if combined submissions ≤ 5 × days running. **Team merger + entry deadline: 2026-12-07.** Final: **2026-12-14** 23:59 UTC. | Rules 2.1, Timeline |
| Public/private split | Not disclosed (745103 unanswered). Community estimate ~130 public molecules (~1/3) (Udam Liyanage, 743254). | 743254 |
| External data | Allowed if publicly available and equally accessible at no or minimal cost ("Reasonableness Standard"). Commercial-use restriction for prizes: "Solutions built on data or tools that cannot be commercially licensed will not be eligible to win" (David Healey, Welcome 741359). | Rules 2.6; 741359 |
| Winner licence | **MIT** (OSI, no commercial limits). Must deliver training and inference code plus an environment description. Rule 2.5.a.3 exempts incompatible input data/models from the open-source grant, but the host's welcome post says such solutions are not prize-eligible. | Rules 2.5, 2.8 |
| Competition data licence | CC BY-NC 4.0 nominally. Host: train.parquet "can be treated effectively as open-source and any model trained on it is fine, redistributed artifacts are fine"; the NC clause protects the **test** data (745841, 743236, 742265). enveda-np-examples is open source as well (742193). | 745841 |
| Compliance enforcement | End-of-competition compliance check "can remove teams from the leaderboard entirely" (745841). Scope (only final selections, or all submissions?) was asked in 745072 and is unanswered. Enforcement for non-prize medal teams is unclear (744470: greySnow, M Sato). | 745841, 744470 |
| Private compute | Training on private cloud such as Hugging Face is OK (David Healey, 741876). | 741876 |
| Hand labelling | Prohibited (general rule 4.b). | Rules |

### Licence whitelist and blacklist (host statements)
- **OK:** train.parquet and derivatives (742265, 743236, 745841). MassSpecGym and permissively licensed models trained on it (742991). ICEBERG/GLACIER MassSpecGym checkpoints (745841; "approved by name in 744338"). **DreaMS weights** (745604, 745841). **CFM-ID 4 stock METLIN-trained models** ("published by a major lab… assume permission", 743774), CFM-ID spectra of COCONUT (741912). LGPL runtime deps (741912). **PubChem structures** (741857). **COCONUT** (743234). ChEBI/LIPID MAPS (745841). Newer GNPS/MassBank/MoNA releases incl. CC BY-SA (745228). **Enveda-180 Zenodo 21346580 (CC BY 4.0)** (743569, 745148). ChemBERTa (742011). GNPS libraries containing old, small NIST-matched data if public for more than 6 months (745832).
- **NOT OK:** NIST in any form (744470). METLIN and vendor (Thermo/Bruker/SCIEX) libraries (742193, 742991). Weights distributed in violation of their data licence (742193).
- **Open or ambiguous:**
  - FRIGID Zenodo 19685145 checkpoints (CC BY-NC; code Apache-2.0; retraining on allowed data is the safe path; 746430).
  - `ahmedberatozer/casmi26-{v4b-models,v3-models,v2-pool,fpnet-full1,pubchem-tier,iceberg,glacier}`, licence "Other (non-commercial)", bundled `casmi/fe_v4` code without an OSI licence (745072, 742265). Every top public notebook depends on these.
  - The ICEBERG PubChem atlas, predicted by a NIST'23-trained model (745134, unanswered; treat as NOT OK).
  - NPAtlas is CC BY-NC (web) and not asked. HMDB-derived data is NC.

---

## 2. The metric, exactly
From the metric notebook (`metric/casmi-mean-reciprocal-rank`, live grader = v13/14 per Marie Killian, 742265):

```
key(s) = InChIKey( TautomerEnumerator().Canonicalize( RemoveStereochemistry( MolFromSmiles(s) ) ) )[:14]   # RDKit 2026.03.3
score  = mean over solution molecules u of  1/rank_u   (first guess whose key ∈ accepted_keys(u); else 0)
```
- **Submission rows:** `molecule_id,smiles` with ≤25 guesses joined by `;`, best first. Each id at most once. No NaN. More than 25 guesses → error.
- **Missing ids** score 0. Extra or unknown ids are ignored.
- **Parsing:** guesses are `strip()`ped and empty strings dropped *before* ranks are assigned. Unparseable or invalid SMILES **consume their rank slot** but cannot score. (The "filler → whole submission 0.000" report in 742088 is not reproduced by the code. Fernando Faria checked it locally and the hosts are "investigating". Never pad with junk anyway.)
- A heavy-atom-composition gate is applied before canonicalisation. Protomers and charge variants of the right skeleton still match.
- The solution field may hold **several comma-separated accepted SMILES** ("indistinguishable isomers"). This is currently not known to be used.
- **2026-10-07 update (746878):** `RemoveStereochemistry` now runs before tautomer canonicalisation on both sides. Before this, a stereocentre could pin an enone and send the same skeleton to two different canonical tautomers (example in 744556: `KOJSSZPTECJRNW` vs `SRMRLHNLDDQJLJ`). Consequences:
  1. Stereo is now truly free.
  2. **Deduplicate your 25 slots with exactly this key.** ~4.4% of NP-library structures change key under tautomer canonicalisation, and ~8% of holdout molecules had a tautomer twin under a different PubChem key (742042, 743254).
  3. The shipped `inchikey14` column in train is *not* the metric key. Recompute it.
- **No partial credit.** "Class 1/2/3" does not change the scoring. Every molecule counts equally (macro-average over molecules, not spectra). The classes only describe how hard a molecule is to reach.

### Maximising expected MRR under uncertainty
- With one correct answer and calibrated probabilities p_i over candidates, E[RR] = Σ p_(r)/r. By the rearrangement inequality, **sorting by posterior probability is optimal**. There is no "safe structure" or abstain trick, and no penalty for wrong guesses. **Always fill all 25 slots** with distinct metric keys. The tail is free.
- **Insertion rule** for a new (for example de novo) candidate with probability q at position k, given the current list p_1..p_25:
  - ΔE = q/k − Σ_{j=k}^{24} p_j·(1/j − 1/(j+1)) − p_25/25.
  - Pushing a confident rank-1 hit down costs p_1/2. So insert generated structures at rank 1 **only when q > the database top-1's posterior** (that is, when retrieval confidence is low, e.g. weak library/analog evidence or a poor FPNet fit). Otherwise put them in ranks 10–25, where displacement costs are about 0.004 per unit p.
  - This is exactly why "generation hurts before it helps" (741659). It is a calibration problem, not a metric problem.
- **Per-molecule aggregation:** score candidates by fusing evidence from all spectra of the molecule (CE ladder, merged spectra, both polarities, multiple adducts). Multi-adduct neutral masses agree to a median of 1.3 ppm (741423), so consensus mass is a strong formula anchor.
- **Formula first:** the correct formula is recovered more than 95% of the time (745468). Errors are almost always same-formula isomers (743254: ~98%).
- **Priors matter.** A PubChem popularity prior (log #substances + log #PubMed) gave +0.004 to +0.010 LB in public notebooks. This is legitimate, because hidden-test class-2 molecules are biased toward documented natural products. Curated-DB membership flags helped CV but hurt LB (743254), so beware selection bias.

---

## 3. The "leak" threads
| Topic | Claim | Resolution |
|---|---|---|
| 741756 (Nipon Sriwasut, −5), 742278 (Odim David Emeka, −4) | All 1,213 visible test spectra are byte-identical to enveda-180 train rows. | By design: the data page says the visible file is a placeholder from train (c-number, PC Jimmmy). No impact. |
| **745715** (NicholasOOO, +22) | `imranarif536/casmi26-v44-pairtail-locked-top1` V3 (0.417) hard-codes the 400 visible id→SMILES pairs and forces them to rank 1, "unguarded". Its previous version scored 0.402, and about two dozen teams hit 0.416–0.418 with forks. The thread asks for an id remap. | **inversion (Kaggle), 2026-10-07: "There is no overlap in IDs between train and test, or between dummy test and rerun test. It's a relatively low-observation test set… careful that they don't interpret noise with causation."** So the mapping never fires on hidden data. The +0.015 came from the LambdaRank tail, rank-1 protection and cleanup (credited in the lineage table of the 0.433 notebook) plus noise. **No regeneration of the test; no expected score drop for those forks.** |
| 746954 (inversion) | "Rerun test dataset has been updated to fix a tiny (mass of an electron) error in some of peak masses"; all submissions were rescored and published 2026-10-08. | Root cause in 743395 (Bryce Hedelius): positive-mode calibrant used neutral Na (22.989769) instead of Na⁺, so positive-mode m/z read **+0.49–0.55 mDa high**. The **train enveda-180 positive spectra probably still carry the offset**. If you match fragment formulas at ≤5 mDa, subtract ~0.5 mDa from e180 positive-mode m/z. |
| 744446 | 22 exact-duplicate spectra with conflicting labels (msdial vs riken, e.g. bergapten/xanthotoxin). | Label noise. Drop or merge them in training. |
| 745148 | 816 Enveda-180 Zenodo structures are absent from train. | Marie Killian: "likely filtered out in the train.parquet preparation"; the Zenodo data is fine to use. Not test molecules. |

**Implications:** no public notebook will lose points because of the id map. The real fragilities are (a) rerun crashes on unseen adducts and masses, and (b) the licence status of the `ahmedberatozer` assets.

---

## 4. Data EDA and domain insights
### 4.1 Files (my download)
- `train.parquet`: 3.03 GB, **2,539,608 spectra, 275,810 InChIKey14**, 18 columns (**no molecule_id/spectrum_id**, 742042).
- `test.parquet`: 1,213 rows, 400 molecules, 12 columns.
- `sample_submission.csv`.

| ingest_lib | spectra | structures | notes |
|---|---|---|---|
| enveda-180 | 1,153,785 | 182,941 | timsTOF, drug-like synthetic. 180,653 structures appear **only** here. Masses 268–391 (5–95%). Precursor ppm abs median 0.98, p99 4.4. |
| pluskal_ms2 (MSnLib) | 527,581 | 46,821 | Orbitrap, NCE |
| riken | 347,171 | 15,892 | plant specialised metabolites |
| gnps | 220,849 | 45,750 | heterogeneous; formate-adduct mislabels |
| massbank / mona / spectraverse / msdial | 101,727 / 92,416 / 50,933 / 40,765 | 9,180 / 11,681 / 9,631 / 9,127 | |
| drug_plus / masaryk | 2,545 / 652 | 2,539 / 416 | |
| **enveda-np-examples** | **1,184** | **250** | **Same instrument and pipeline as the hidden test.** All 250 structures also appear in other libraries, so it is not a clean holdout unless you purge by key (741597, 742055). |

- **Non-e180 (natural-product-ish) structures:** 92,869. Masses 192–841 (5–95%), median 351. Only 2,288 structures overlap between e180 and the rest.
- **Adducts:**
  - Train contains dimers (`[2M+Na]+` 190k, `[2M+H]+` 92k, `[2M-H]-` 56k). The test has **no dimers**.
  - np-examples adducts: [M+H]+ 569, [M-H]- 276, [M+CH2O2-H]- 105, [M+NH4]+ 98, [M+Na]+ 80, [M-H2O+H]+ 24, [M+K]+ 16, [M-2H2O+H]+ 8, [M+Cl]- 7.
  - Expect the hidden test to look like np-examples (more NH4/Na/formate/water-loss), **not** like the placeholder (79% [M+H]+).
- **Collision energies:**
  - Placeholder: 20/40/60 eV plus merged [20,40,60].
  - np-examples is a different ladder: [40] 213, [20,50] 209, [60] 200, [35] 124, [20] 114, [80] 110, [20,40,60,80] 100, [20,40,60] 81, [0..80] 15.
  - **Do not hard-code a CE ladder.**
- **Spectra per molecule:** placeholder mean 3.03 (max 9); np-examples mean 4.74 (max 17); hidden 1–16 (median 3). Resample CV to about 3 spectra per molecule (742055: count-matched CV changes effect sizes ~3×).
- **Polarity:** 52% of np-examples molecules have both polarities (130/250), versus 24% in the placeholder (97/400).
- **Precursor accuracy:** np-examples ppm abs median 1.65, p95 3.7, p99 6.6. A **±5–10 ppm window** covers it. Width barely matters, because the window is full of exact-mass isomers anyway (743254).
- **Isomer density** (train structures only): for np-examples molecules, the number of train structures sharing the formula has quantiles 10/25/50/75/90% = 1/3/6/16/33, and within ±5 ppm it is 2/4/10/24/44. With COCONUT plus PubChem it becomes ~83 candidates per window, ~66 of them same-formula (742055). PubChem slices reach thousands (~6,000 per query, 742088).
- **Noise:** test spectra have a median of 230 raw peaks, but 93% are below 1% relative intensity. After a 1% floor the median is 18 peaks; at 0.1% it is 52 (741641, 741423). The precursor peak is present in only 56% of spectra. M+1 isotope ratio gives the carbon count to ±25% for ~41% of molecules, but only +0.011 MRR (743254 thread).
- **No CCS or ion mobility** in the test (743368, 743569), even though Enveda-180 Zenodo has it. CCS is therefore only usable as a training auxiliary.
- **Label noise:** 6.9% of NP-library rows have a labelled-adduct neutral mass more than 10 ppm off, mostly formate mislabels in GNPS/MassBank. Enveda's own labels are clean: 0/250 misses (742087). Do not hypothesise alternative adducts at test time.

### 4.2 Which subset is solvable how
| Class | Community share estimate | Best method | Achievable MRR (forum CV) | Contribution |
|---|---|---|---|---|
| 1: public spectra exist | **~16–25%**. Library-only LB 0.151–0.158 ⇒ f1≈0.17 (741597). A "library twin at rank 1" probe gave 0.275 public (742088), so possibly up to ~0.28. | Spectral library search: entropy/cosine, adduct-shifted, timsTOF-preferred, DreaMS kNN. Merge evidence over all spectra. | **0.90–0.94** (741404: 0.907; 745952: 0.933) | ≈0.15–0.24 |
| 2: in PubChem/COCONUT, no spectra | ~45% | Candidate retrieval: formula/mass window over COCONUT ∪ train ∪ PubChem (∪ ChEBI/LIPID MAPS). Rank with FPNet f·z, analog propagation, ICEBERG/GLACIER/CFM-ID forward simulation, MetFrag-lite parsimony, popularity prior, LambdaRank. | 0.52–0.63 now (741597, 745952: 0.630). Oracle same-formula separation → 0.99 (742055) | ≈0.24–0.28 now; +0.1 possible |
| 3: in neither PubChem nor COCONUT | ~35–39% (hosts: "enough novel structures to make it worth working on", 745758) | De novo or analog-edit generation, then calibrated insertion. FRIGID-type: MassSpecGym top-1 16–18%, NPLIB1 20–25% (known formula). MS2Mol: EnvedaDark close-match 21%. Exact match on truly novel NPs is far lower. | ~0.0 for public notebooks. 745952 reports 0.347 in a "class-3 mode", which is optimistic. Udam: simulated novelty overstates reach ~3×. | 0 → 0.02–0.05 realistic |

The arithmetic: 0.433 public ≈ 0.20 (class 1) + 0.23 (class 2) + ~0 (class 3). MarvinTMB's 0.484 implies a better class-2 ranking and/or some class-3 hits.

---

## 5. State-of-the-art method map (2024–2026) and offline feasibility on Kaggle
| Component | Method(s) | Expected accuracy | Offline in Kaggle? / licence status |
|---|---|---|---|
| Spectral similarity / library | matchms; flash **entropy similarity** (Li & Fiehn 2023); modified cosine; **MS2DeepScore 2** and Spec2Vec (Apache/MIT); **DreaMS** embeddings (Nat Biotech 2025; weights **CC BY 4.0**, Zenodo 10997887, 2.6 GB; host-approved) | Class-1 MRR ~0.93. DreaMS AUC 0.88 for same-molecule vs same-formula pairs, but **below chance on the hard blocked isomer pairs** (742055). | Yes. All fit on T4/L4 or CPU. |
| Formula annotation | SIRIUS (needs a login plus CSI:FingerID web service: **not usable offline, academic licence**); **MIST-CF** (MIT; public NPLIB1 weights; positive mode only); BUDDY; simple decomposition plus isotope or multi-adduct consensus | Formula correct >95% here (745468) | MIST-CF/BUDDY yes; SIRIUS no. |
| Spectrum → fingerprint | **FPNet** (public, 6,930 bits, f·z scoring; `fpnet_full1` +0.013); **MIST** (Goldman 2023, MIT; MassSpecGym checkpoints in FRIGID's Zenodo, CC BY-NC); CSI:FingerID (closed) | MassSpecGym retrieval hit@1: MIST 14.6%, JESTR 15.6% (mass candidates) | Retrain your own MIST/FPNet on train.parquet + MassSpecGym + Enveda-180 Zenodo (all allowed). |
| Joint embedding retrieval | **JESTR** (contrastive mol↔spectrum), MVP, CLERMS-type | ~+5–10% relative over MIST on MassSpecGym | Yes; train yourself. |
| Forward simulation (rerank) | **ICEBERG** (ms-pred, MIT; MassSpecGym ckpt OK), **GLACIER** (Coley 2026, arXiv 2606.29161; claims 70% top-1 simulated-retrieval on MassSpecGym), **CFM-ID 4** (LGPL; OK), FraGNNet, MassFormer, GrAFF-MS (Enveda; NIST-trained ⇒ not OK), 3DMolMS | +0.007 (ICE/GL λ) in public notebooks. GLACIER-only rerank beat ICE+GL. ICEBERG as a learned feature +0.0015 (742055). | Yes, but budget the GPU time: ICEBERG over ~60 candidates × 400 molecules fits in 9 h. |
| Combinatorial fragmentation | MetFrag-lite parsimony: penalise 2-bond cuts ×0.6, linear intensity, 0.005 Da tolerance, H shift −2..+3 | +0.023 held-out on panels, ~0 on LB (743254) | Yes |
| De novo / generative | **FRIGID** (ICML 2026; MassSpecGym top-1 16–18%, NPLIB1 19.8–25%; code Apache, **weights CC BY-NC** ⇒ retrain), **DiffMS** (MIT), **MADGEN** (scaffold-conditioned), MSNovelist (needs CSI fingerprints), Spec2Mol, **MS2Mol** (Enveda; not released), MBGen, FlowMS (2026) | Top-1 exact on truly novel NPs ≪10% | FRIGID inference 6.6 s/spectrum on an A6000 ⇒ ~1–2 h for 400 molecules on L4/T4 at reduced rounds. Feasible. |
| Analog edits (cheap class-3 generator) | Apply ±O, ±CH2, ±hexose, ±acetyl, ±2H, prenyl, sulfate, methylation to strong spectral/structural analogs; filter by exact formula; score with FPNet + simulator | Did not help on LB yet (742055 variant E: 0.337→0.335) without calibration | Yes |
| Rankers | HistGBM/LightGBM, LambdaRank (+0.016 with rank-1 protection), two-engine RRF fusion | | Yes |
| Priors | PubChem substance and PMID counts (popularity), COCONUT organism/occurrence, NPClassifier class priors, NP-likeness | +0.004 to +0.010 | Yes (PubChem is public domain) |

**Candidate databases**

| Database | Licence | Notes |
|---|---|---|
| PubChem (~120M) | public domain, host-approved | Formula-filter to a mass band and ship as a Kaggle Dataset. Raw blind expansion hurts: 0.335→0.205 (741597). |
| COCONUT 2.0 | **CC0** (data) | Approved |
| LOTUS | Wikidata CC0 | 97.5% already contained in COCONUT (742055) |
| NPAtlas | **CC BY-NC** | Adds ~1.1k structures. Avoid for prize. |
| ChEBI / LIPID MAPS | approved | |
| HMDB | NC terms | Avoid |
| PubChemLite CC0 | | +386k structures not in COCONUT (743254) |

**Spectral libraries**
- MassBank: per-record licences; keep CC0/BY/BY-SA.
- GNPS: mostly CC0. Drop NIST14-matches unless the old-dataset exception applies.
- MoNA: mostly CC BY; drop NC records.
- MSnLib: CC BY.
- **Enveda-180 Zenodo**: CC BY 4.0, 8 files up to 2.6 GB, includes CCS/RT. Use it for pretraining and calibration.
- MassSpecGym 1.5: MIT; ships a 4M-molecule candidate file and 2.5M/50M pretraining SMILES. Note the "MassSpecGym in the Wild" paper (2606.19624): 17 of 26 papers had evaluation leaks.

---

## 6. Strategy
### 6.1 Five submissions TODAY (2026-10-09)
Goal: get off 0.328 and set reproducible anchors. Do not tune on public-LB noise.
1. **Fork the 0.433 lineage unchanged.** That is `obstacledeveloper/casmi26-v1-1-metric-audit-and-safe-tail` or `evgendvorkin/enveda-casmi-2026` ("Forensics Lab", 162 votes, same 16 datasets). It confirms the rerun completes within 9 h with the current GPU type. Expected ~0.433 (rank ≈100).
2. **`haideptry/0-350-casmi-2026-v32-ensemble`** (top public by score, same datasets; check its claimed score in the notebook output). This is a second anchor and a candidate final selection.
3. **The same lineage with a probability-calibrated tail fill.** Fill every empty slot with distinct metric keys (overflow candidates from the PubChem and Engine-2 channels). Add analog-edit candidates **only at ranks ≥16**. This is expected to be neutral-to-positive and is low risk.
4. **Seed-averaged ranker** (4 seeds × 2 priors already exists; average the scores rather than vote). This reduces the ±0.006 noise and makes it a better final-selection candidate.
5. **Keep one in reserve** or, for the "clean" track, submit a version that **drops the `ahmedberatozer` NC-labelled datasets** (use COCONUT and train-only FPNet you trained yourself) to measure the cost of prize compliance.

Expect all of 1–4 to land in 0.425–0.445. Treat differences below 0.015 as noise.

### 6.2 Nine-week plan to reach the top 10 (~0.46–0.50 private)
**Week 1: validation harness (the single most valuable asset).**
- Three masks:
  - Class 1: other libraries' spectra stay.
  - Class 2: purge every spectrum of the held-out metric key from all libraries.
  - Class 3: also purge the structure from all candidate DBs.
- Queries: np-examples (250) plus NP subsets of riken/gnps/mona/massbank (~1k), resampled to ~3 spectra per molecule and to np-examples adduct/CE mixes.
- Build a metric-exact scorer. Self-match check: best cosine must be 0.000 under the purge.
- Build a CV→LB translation from paired measurements, and require effects to hold on ≥3 panels (743254 methodology).

**Weeks 1–3: own, prize-clean core.**
- Train FPNet/MIST and a JESTR-style contrastive encoder on train.parquet + Enveda-180 Zenodo + MassSpecGym 1.5 + newer CC0/BY GNPS/MassBank/MoNA.
- Training choices: hard negatives drawn **from the same formula** (not just the 10 ppm window); intensity floor 0.1–1%; timsTOF-weighted batches; per-molecule multi-spectrum attention pooling (CE + polarity + adduct tokens).
- Correct the e180 positive-mode +0.5 mDa offset.
- Build the candidate DB: COCONUT ∪ train ∪ PubChem formula-filtered band (~150–1,200 Da) ∪ ChEBI/LIPID MAPS, with metric-key dedupe and popularity features.

**Weeks 2–6: isomer discrimination (95% of the class-2 headroom).**
- (a) **Cross-encoder reranker**: spectrum peaks × candidate graph with subformula-annotated peaks (peak→fragment-formula assignment constrained by the candidate formula), trained with listwise loss on same-formula panels.
- (b) **Forward simulators as features**: GLACIER, ICEBERG (MassSpecGym ckpt) and CFM-ID 4 on the top ~60. Fine-tune ICEBERG/GLACIER on timsTOF (e180 + np-examples) to close the CID-vs-HCD gap.
- (c) **Collision-energy-aware features**: fragment survival across the 20→80 eV ladder.
- (d) **Priors**: NPClassifier pathway/superclass predicted from the spectrum versus the candidate's class (Enveda's targets are plant NPs; 745029).
- (e) LambdaRank stacker optimising MRR directly, seed-averaged.

**Weeks 4–8: class 3.**
- Generators:
  - Analog-edit enumeration around the top library and analog hits, using biotransformation rules (hydroxylation, O-/N-methylation, glycosylation incl. hexose/deoxyhexose/pentose, acylation, prenylation, sulfation, reduction/oxidation) restricted to the exact predicted formula.
  - A FRIGID/DiffMS-type generator **retrained** on allowed data (FRIGID code is Apache), conditioned on the predicted formula and fingerprint.
- Gate: a "retrieval-failed" classifier (features: max library cosine, FPNet top-1 margin, analog support, simulator fit).
- Insert per the ΔE rule in §2. Validate only on real absent molecules: e.g. structures in the newest MassBank/GNPS releases that are absent from PubChem and COCONUT, and enveda-np-examples-like NPs. Do not rely on "delete from DB" simulations, which overstate reach ~3× (743254).

**Weeks 7–9: robustness and selection.**
- Stress-test the notebook on a synthetic hidden-like test: np-examples-style adducts incl. water losses and formate, masses 157–1,159, 1–16 spectra per molecule, ~1,500 spectra.
- Profile runtime with ≥30% headroom.
- Pin RDKit 2026.03.3.
- Merge teams before **Dec 7** if useful (you have only 4 submissions, so merging is easy).
- **Final two selections:** (i) best CV-validated prize-clean pipeline; (ii) best LB-consistent pipeline. If you are prize-chasing, make both licence-clean.

### 6.3 Risk list
1. **Shake-up.** Public is ~130 molecules. Paired SE ~0.016, ranker-seed noise ±0.006. Top-10 public spans 0.448–0.484 (~2 SE). Choose finals by CV plus LB jointly. Prefer seed-averaged, less public-tuned variants.
2. **Rerun failures.** Hidden adducts, masses and spectrum counts differ from the placeholder. A silent fallback can score baseline or 0. Do not key artifacts to visible-test masses. Watch the 9 h limit with ICEBERG plus the PubChem channel. P100 is gone.
3. **Licence and compliance.** Every top public notebook depends on `ahmedberatozer/*` NC-labelled assets and unlicensed code (745072 open). FRIGID weights are CC BY-NC. NPAtlas is NC. NIST is forbidden. The host says non-compliance can mean **removal from the LB** (745841), not just losing the prize. If you want prize or medal certainty, rebuild the core yourself by about week 4.
4. **Metric drift.** The metric already changed once (2026-10-07). The rerun data changed once (electron-mass fix). Re-verify locally with the exact v13/14 code after any announcement.
5. **Distribution shift.** The test is NP chemistry on timsTOF; the only big timsTOF library (e180) is drug-like. Fine-tuning FPNet on e180/timsTOF lowered LB in one report (0.337→0.328, 742055). Instrument match is not chemistry match.
6. **Over-expansion.** Blind PubChem expansion or generation without calibration lowers MRR (0.335→0.205 measured, 741597). Expand only together with a same-formula-aware ranker and gating.
7. **CV leakage.** The 250 np-examples structures have 55,599 spectra in other libraries. A random structure split is contaminated. Purge by metric key across all libraries.

---

## 7. Forum index (topic id: what matters)
- **Official:** 741359 Welcome / licence note (Healey); 740671 Discord; 741471 re-download train (water-loss adducts added); 746878 metric RemoveStereochemistry; 746954 rerun test updated (electron mass); 742265 grader = v13, P100s retired; 742274 empty entries dropped, Class 3 ∉ PubChem ∪ COCONUT; 744556 solutions stereo-stripped; 743368 and 743569 no CCS in test, Enveda-180 Zenodo OK.
- **Licence:** 741844, 741857 (PubChem OK), 741912 (CFM-ID/COCONUT OK), 742193 (open weights; NIST not OK), 742991 (MassSpecGym models OK), 743234 (COCONUT OK), 743236 and 745841 (train-derived artifacts OK; LB removal possible), 743774 (CFM-ID 4 OK), 744470 (NIST ⇒ disqualification), 745072 (NC Kaggle datasets: open), 745228 (CC BY-SA OK), 745604 (DreaMS OK), 745832 (GNPS NIST-matched exception), 746430 (FRIGID CC BY-NC: open), 745134 (ICEBERG atlas: open), 745615 (open).
- **Leak/data:** 741756, 742278 (placeholder = train), 745715 (id map: no overlap), 745148, 744446, 743395, 742087, 742042.
- **Methods and measurements:**
  - 741745 (haideptry 4-channel, 0.339; FPNet f·z)
  - 741597 (class shares from LB probing)
  - 742055 (failure taxonomy: ranking ≫ recall; 7 negative results)
  - 742088 (probes; top-25 ceiling ~0.40 at 0.324)
  - 743254 (30-submission lessons; MetFrag parsimony ablations; noise ~0.03 rule)
  - 743974 (ranker is the bottleneck; CV 0.846 / 0.584 for regimes A / B)
  - 744893 (DreaMS kNN demo)
  - 745952 (3-mask validation 0.933 / 0.630 / 0.347; locking engine picks hurt LB)
  - 741659 (MRR incentive gap for generation)
  - 744343 (forward model trick)
  - 745468 (formula right >95%, isomers wrong)
  - 746302 (ceiling ~0.55 realistic, 0.76 analytic: MarvinTMB #1)
  - 741404, 741423, 741641, 744011 (EDA)
- **Public notebook lineage** (from the 0.433 notebook):
  - prvsiyan analog propagation 0.335 → haideptry 4-channel 0.339 → megayak two-rankers 0.337 → llccqq624 W088 0.341
  - → dmitriigluzdov "From Spectra to Structures" 0.373 (160 features, PubChem channel, ICEBERG + GLACIER)
  - → ahmedberatozer v4m `fpnet_full1` (+0.013) → seyitkaangunes e5c / v4n fusion 0.386 / 0.399
  - → bobthebot369 popularity prior 0.401 → imranarif536 V44 PairTail 0.417 → V1 0.425 → obstacledeveloper V1.1 metric-safe tail **0.433**

## 8. Web sources
- Enveda press release (Business Wire, 2026-09-15): https://secure.businesswire.com/news/home/20260915334355/en/
- Enveda-180 Zenodo 21346580 (CC BY 4.0, 2026-07-13): https://zenodo.org/records/21346580
- DreaMS weights (CC BY 4.0): https://zenodo.org/records/10997887 ; code (MIT): https://github.com/pluskal-lab/DreaMS
- MassSpecGym (MIT, v1.5): https://huggingface.co/datasets/roman-bushuiev/MassSpecGym ; paper https://arxiv.org/abs/2410.23326 ; pitfalls paper https://arxiv.org/abs/2606.19624
- FRIGID (ICML 2026; code Apache-2.0; ckpt Zenodo 19685145): https://github.com/coleygroup/FRIGID , https://proceedings.mlr.press/v306/bohde26a.html
- GLACIER (Coley 2026): https://arxiv.org/abs/2606.29161
- JESTR: https://arxiv.org/abs/2411.14464 ; retrieval-objective trade-off paper: https://arxiv.org/abs/2602.16507
- MS2Mol (Enveda): https://chemrxiv.org/engage/chemrxiv/article-details/6492507524989702c2b082fc ; PRISM press: https://www.businesswire.com/news/home/20240529298878/en/
- COCONUT 2.0 (data CC0): https://coconut.naturalproducts.net/about ; NPAtlas (CC BY-NC): https://npatlas.org/download ; LOTUS: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9135406/
- CASMI history: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3901296/ ; CASMI 2016: https://hal.inrae.fr/hal-02626233

## 9. Local artifacts
In `/tmp/claude-0/-home-user-kraggle-3-in-one/5b333432-508e-5f10-be96-1dd09fd2558d/scratchpad/research/casmi/`:
- Forum and pages: `topics/full_*.txt` (76 topics, full text), `pages.txt` (all competition pages).
- Metric: `metric_nb/` (metric notebook).
- Leaderboard and notebooks: `lb/` (public LB CSV), `nbs/` (3 top public notebooks + metadata), `kernels_score.txt`.
- Data: `data_d/` (train/test/sample), `train_meta.parquet`, `structures.parquet` (per-structure libs and masses).
- EDA: `eda1.txt`, `eda2.txt`, `scripts/`.
