# RSNA Knee Abnormality Detection: discussions, rules, metric, data and top-team intelligence

*Compiled 2026-10-09 (about 15:40 UTC) by "Dr. Patella". Sources: all 173 discussion topics, with bodies and every comment, fetched through the Kaggle API (`topics/full/topic_<id>.json`); the competition pages (`pages/*.md`); the public leaderboard CSV downloaded 2026-10-09 14:57 UTC (`lb/`); Ryan Holbrook's efficiency-LB notebook output (`efflb/full_leaderboard.csv`); and web sources for prior art. Each claim is tagged [topic-id, author]. All forum content is treated as unverified participant claims unless a host wrote it.*

Hosts and staff: **Po-Hao "Howard" Chen** (@javaduke95, RSNA host; posts the rulings), **Ryan Holbrook** (Kaggle; efficiency LB, data fixes), **Jason Sho** (Kaggle; corrupt-file triage), **María Cruz** (Kaggle; welcome/Discord).

---

## 0. The ten facts that matter most

1. **Labels decide this competition, but in a specific way.** The test labels come from **two MSK radiologists reading the images**, with a third adjudicating, under a strict rubric where borderline cases count as negative. They were **not** derived from the reports [733826, 733491, 733343, Chen]. Report-derived labels top out at about **0.88–0.90 macro AUC vs gold-58**, whoever extracts them. The best image models already beat their teacher. The lever that people at **0.95+ without external data** describe is **out-of-fold image-model pseudo-labels blended into the report labels** (50/50 or heavier on the model) [735304 Archit Konde, Raymond Yuen, colum2131; 743148 Raymond Yuen].
2. **Small 2.5D CNNs at 224–320 px are enough.** Single architectures with 5-fold CV and pseudo-labels report public scores of **ConvNeXt-T 0.959, EfficientNetV2-S 0.956, CoAtNet 0.952** [735304 azamat1ch, 2026-10-09]. Others report ConvNeXt-T 0.953 / DINOv2-S 0.951 / both 0.955 [735304 colum2131], ResNet-50 @224 5-fold 0.954 [735304 CoolinLai], 0.955 single model 5-fold [735304 Tanuki_boosting], and a single-fold ResNet at 0.949 with about 5-minute scoring [735304 Scott Willis]. Larger encoders give nothing measurable [735154 stevenleehans: DINOv2 S→B +0.0011 against a 0.002 noise floor; 738096 Komil Parmar].
3. **External OAI data is the other lever, and its legal status is grey.** Danial Zakaria (nartaa) added 2,399 OAI knees with masked labels, supervising PF OA, Lateral OA and Synovitis. That took one CoAtNet from 0.942 to **0.945**, then 0.948 at 384 px, then **0.949** with mirror TTA [746792]. Host ruling: OAI is allowed *if accessible without institutional review/sign-off and free* [741819, Chen 09-21]. NDA reports "Institutional Approval Required: No" [743416 Eesh saxena], but users in China cannot register [746792 xun1993, fishcat]. **There is no explicit yes/no, and the questions on OAI-trained public weights are unanswered** [747455, 747660, 743416 Taeyang]. The whole public 0.950 tier (455 teams) runs on these OAI-trained weights.
4. **Metric:** an unweighted mean of 12 per-finding ROC-AUCs [Evaluation page]. It is rank-based, so calibration and prevalence shift do not matter, but every finding counts 1/12. The weak findings (Synovitis, Lateral OA, PF OA, Lateral Meniscus, Fracture) are where the macro score moves.
5. **Leaderboard (2026-10-09 14:57 UTC, 5,586 teams):** #1 = 0.964 (3 teams). The **top-10 cut is 0.962–0.963**, top-50 is **0.957**, top-100 is **0.954**, and rank 266 is 0.951. **455 teams are tied at 0.950** and 699 at 0.943 (public-notebook blocks). We are at **0.942, rank 1954**, 5 submissions, the last one on 09-21.
6. **Hard limits:** ≤9 h runtime for CPU or GPU notebooks; internet off; 5 submissions/day; **2 final selections**; team size ≤5; **entry and merger deadline Oct 15**; final deadline **Oct 22 23:59 UTC**; about **1,300 hidden test studies** (1,322 according to [744230]); no reports at test time.
7. **Efficiency prize ($7k/$6k/$5k)** = AUC/(0.5 − maxAUC) + RuntimeSeconds/32400. **0.001 AUC ≈ 70 s of runtime.** Runtime is the whole notebook, including pip, model load and DICOM decode [733475 Holbrook]. Both final picks are scored on both tracks [743356 Chris Deotte]. Efficiency #2 is a 0.962-public team (Pa3฿aJluHа), so a top-10 model can run in minutes.
8. **Shake-up risk is moderate to high around the medal lines and low at the top.** The public split is about 30% (roughly 400 studies, per the repo's prior audit; Kaggle does not state the fraction). A 0.001 difference is below resolution: one team spent 13 submissions that all read 0.937 [740439]. Historically, teams just inside versus just outside the bronze line kept or won a medal **24% of the time either way** [742327 Georgy Mamarin].
9. **Notebook failure modes that cost others submissions:** P100 fallback (torch needs sm_70) [735854]; **bf16 on T4** (emulated, about 50× slower, so a time guard filled half the test with 0.5 and scored 0.710) [744230]; missing JPEG-Lossless/J2K codecs [735854]; **2 truncated DICOMs** in train, with the test set not guaranteed clean [737163, 738708, 740094]; `/kaggle/input/competitions/<slug>/` path quirks [735854, 745984]; Kaggle-side `ERRORED_MOUNTING_DATASET` errors in early October (resubmitting did not consume a slot) [746542]; T4 queue stalls [743746].
10. **Host warning about a test-set trap:** `Fluid_Sensitive` and `Fat_Suppression` are identical in train, but "*should not be interpreted as… they must be equal in another dataset*" [737312 Chen]. Any slot logic keyed on one column must tolerate them differing. Whether every test study has all three planes is **unanswered** [744519].

---

## 1. Rules digest (pages/rules.md, Code Requirements, Timeline, Prizes, plus host rulings)

| Item | Rule / fact | Source |
|---|---|---|
| Submission mechanism | Notebook only; `submission.csv`; CPU ≤ 9 h or GPU ≤ 9 h; internet disabled | Code Requirements page |
| External data / models | "Freely & publicly available external data is allowed, including pre-trained models" (code page). Rule 2.6: must be "publicly available and equally accessible… at no cost" **or** meet the Reasonableness standard (cost/geo-restrictions); "acceptable unless specifically prohibited by the Host" | rules §2.6 |
| Host's external-data principle | Non-commercial licences alone do **not** exclude a dataset; "straightforward registration or click-through" is generally fine; institution-specific approvals, negotiated agreements, IRB or lengthy credentialing may exclude it | 733965 Chen 08-27 |
| **KneeCoT** | **Prohibited** (requires a formal agreement with the hospital) | 734109 Chen 08-25 |
| **OAI** | "does not violate… if generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge." No direct yes/no. NDA page says "Institutional Approval Required: No" [743416 Eesh saxena]; access is blocked from China (LG22 error) [746792 xun1993, fishcat] | 741819 Chen 09-21 |
| MRNet / fastMRI(+) / SKM-TEA / KneeMRI / KneeXNet | **No ruling.** MRNet's RUA forbids derivative works; KneeMRI is CC BY-NC-ND; fastMRI needs an emailed DSA. Tucker Arrants argues ND licences conflict with the winner's weight-release duty | 743416, 737950, 743420 (all unanswered) |
| Public weights trained by other participants (incl. OAI-trained) | **No ruling** on prize eligibility or the training-code duty | 744056, 741212, 747455, 747660 |
| Hosted LLM APIs for report labelling | **Allowed** ("You can use LLM API, such as those from OpenAI…"); not counted as private sharing; must be of minimal cost and reasonably accessible. LLM-derived labels or embeddings may be used as model inputs at train and inference (but there are no reports at inference) | 733965 Chen 08-09, 08-27 |
| DINOv3 (gated Meta licence) | **No ruling**; some applicants were refused access; public notebooks use a CC0 re-upload of DINOv3-derived fold weights | 733313 |
| RadImageNet ResNet-50 (CC-BY-NC-SA vs MIT mirror) | **No ruling** | 735121, 745283 |
| Hand-labelling | Banned only for validation/test records; manually relabelling *training* studies is treated as derived data, not external (community reading, no host ruling) | rules §3.4.b; 740718 |
| Team | Max 5. Merger allowed if combined submissions ≤ 5 × days running (≈385 by Oct 15) | rules §2.1 |
| Daily submissions | **5/day**; failed scoring runs ("Notebook Threw Exception", "unhandled error while rerunning") **do** count [734062]; Kaggle-infrastructure errors did not [746542 Quan Vu] | rules §2.2 |
| Final selection | **2** submissions; both are considered for the main *and* efficiency tracks | rules §2.2.b; 743356 |
| Timeline | Start 07-30 · **entry and merger deadline 10-15** · **final 10-22 23:59 UTC** · winners' deliverables 11-05 · RSNA 2026 Nov 29–Dec 3 | Timeline page |
| Prizes | Main: $9k, 7k, 6.5k, 6k, 5.5k, then 5k for 6th–10th. Efficiency: $7k/6k/5k. Total $77k | Prizes page |
| Winner licence | **CC-BY-NC 4.0** (the rules text also says "in no event limits commercial use", a contradiction raised in [739905], unanswered). Upstream incompatible licences are exempt (§2.5.a.4). Winners must release code **and weights publicly**, make a video, and post links in the forum | rules §2.5; Prizes page |
| Data licence | MIRA licence; no redistribution of competition data | rules §2.4 |
| Hidden test size | "about 1300 studies"; 1,322 per one participant's runtime arithmetic | data page; 744230 |
| Public/private split | Not stated officially. Site stratification was asked and never answered [734681]. Prior repo audit: about 30% public / 70% private | — |
| Efficiency prize | Score = AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400 (lower is better). Benchmark = sample_submission (0.5). Must beat the benchmark on private. GPU allowed; RuntimeSeconds covers the full notebook execution | Efficiency page; 733475 Holbrook |
| Efficiency LB | `ryanholbrook/rsna-knee-abnormalities-efficiency-lb`, updated daily, ranks only. #1 Scott Willis 0.959, #2 Pa3฿aJluHа 0.962, #4 Wassim Dobbi 0.960, #28 Zhengru Li 0.962. **We are #2389** | efflb CSV |

**Efficiency arithmetic:** with maxAUC ≈ 0.965, d(score)/d(AUC) = −1/0.465 = −2.15, so 0.001 AUC = 0.00215 = **70 s of runtime**. One hour of runtime costs the same as 0.052 AUC. A 0.955 model that scores in 6 min beats a 0.963 model that takes 15 min.

---

## 2. Metric and data digest

### 2.1 Metric
- Final = (1/12) Σ AUC_i over ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture. **No weighting** [Evaluation page].
- Edge case: a single-class column in private would make that AUC undefined. Asked in [740094], never answered; with about 900 private studies this is very unlikely to happen.
- Submission: header `StudyInstanceUID,ACL,…,Fracture`; any float works (decimal places are irrelevant [739200]). NaN predictions are a common cause of hidden-run failures [743779 Chris Deotte].
- Because AUC is rank-based: rank-average blends are safe; test-time calibration is pointless; **prevalence shift between train, public and private is explicitly possible** [data page], and it does not matter to AUC.

### 2.2 Label definitions (host, [733343]); borderline = negative
- **ACL**: high-grade partial (>50% of fibres) or complete tear. Signal change or degeneration without discontinuity = 0.
- **MCL**: high-grade partial or complete *acute* tear with fibre disruption and oedema. Low-grade sprain or chronic change = 0.
- **Med/Lat Meniscus**: signal that definitely contacts the surface on ≥2 images, or a morphological abnormality (truncated, diminutive, displaced fragment). Intrasubstance degeneration = 0.
- **Med/Lat/PF OA**: a moderate or large area (≥1 cm) of high-grade cartilage loss (>50% thickness), with or without marrow changes. Chondromalacia or chondropathy words alone do not map to this [733491 Chen].
- **Effusion**: a *moderate or large* amount. "Small" or "mild" effusion in a report is often labelled 0 [733826 Cho Royou; 735826 Tom Aindow; 738339].
- **Synovitis**: inflammation and thickening of the synovial lining. The criterion on non-contrast MRI was asked and never answered [737155].
- **Baker's**: a *moderate or large* popliteal cyst.
- **Contusion**: bone-marrow-oedema-like signal from impact without a fracture line. Degenerative or reactive oedema is not contusion [733491].
- **Fracture**: an *acute* cortical break or fracture line. An osteochondral fracture can be 0 [733826].
- The two readers looked **at the images only**. When report and image disagree, the image label is authoritative. Negatives mean "annotated absent", not "not annotated". The **same process was used for the test set** [733826, 733491 Chen].

### 2.3 Gold-58 and report labels
- Exactly 58 of 4,407 training studies carry all 12 labels; none are partially labelled [734106, 734055]. **The gold-58 sample is enriched:** every gold study has ≥1 positive, with a mean of 4.14 positives per study [733932]. Its ACL rate is 41% against roughly 20% in the corpus [734106].
- Gold-58 prevalence: Effusion 60%, Synovitis 47%, Medial Meniscus 45%, ACL 41%, Lateral Meniscus 40%, PF OA 36%, Contusion 33%, Fracture 31%, Medial OA 26%, Baker's 21%, Lateral OA 19%, MCL 16% (9 positives) [734055, 735855].
- A human strict-text reading of 20 gold studies agreed with the labels on **82.5%** of cells [733826]. Report-only labels **over-call** (clinical thresholds are looser), and they **under-report** synovitis and fracture: fracture appears in 7% of reports but 31% of gold labels [741310 gchauhan]; only about 16% of reports mention synovitis at all [737566 starkhushi].
- **Label-extractor ceilings vs gold-58:** regex/lexicon 0.814 against LLM 0.878 [733932 stevenleehans]; transparent rules 0.727 [734095]; best public tables steven v4 0.893 / v2 0.887 / pilkwang 0.870 [737454]. The 8 public LLM label sets are **statistically indistinguishable on 11 of 12 findings**; their CC0 consensus scores 0.889 [747540 karttikjangid05]. Practical ceiling ≈0.89–0.92 (Myo Min Htet 0.908, Cà Rốt Production 0.916) [743148, 738172].
- **What "not mentioned" means differs by finding** [733932]: when the report is silent, P(gold+) is 0.34 for Synovitis, 0.21 for PF OA, 0.03 for Baker's and 0.00 for Medial OA. Silence means *absent* for Baker's, OA and fracture (a radiologist confirms this for fracture and tears [737650 Amil Gentili]) but means *unknown* for synovitis. Filling only silent synovitis cells from the Effusion field moved that column from **0.678 to 0.790**. A blanket imputation across all 12 findings was **worse** (0.8805 against 0.8873 for the targeted version).
- Known contaminated tables: some public label sets copy gold-58 verbatim (yunusgmsoy v5, barun2104) and are useless for validation [repo audit, EXT_SUBMISSION_PLAN]. A gold+16 set ("gold-74", from JEV) exists [747540].

### 2.4 Images and data quirks
- 4,407 train studies; 24,371 series; median 5 series per study (3–14); every train study has sagittal, coronal and axial planes; 820k DICOMs, about 570 GB [734106, 739693]. Series usually have 20–45 slices (median 30) [data page].
- **Slice order:** filenames are random SOP UIDs. Filename order matches anatomy on only about 5% of series [735154], and InstanceNumber runs against position in more than a third of series [741310]. **Sort by ImagePositionPatient projected on the slice normal.** Fixing this was worth **+0.028** on one fold [745759 willyp74].
- Mixed transfer syntaxes (JPEG Lossless, JPEG 2000) need pylibjpeg/gdcm wheels offline [735854]. Dimensions vary (640–960 px, some 640×1280) [735855]. In-plane spacing is ~0.3 mm against ~3.5 mm between slices, which favours 2.5D [741310].
- **Two truncated DICOMs** in train (half-length PixelData): study …37833587… and …34685905… [737163, 738708]. Ryan Holbrook tried to patch the dataset on 09-18; the update "may have failed" [741916]. Test cleanliness is not guaranteed, so wrap every decode.
- `Fluid_Sensitive == Fat_Suppression` on all train series, **but the host warns they may differ** in other data [737312]. The non-FS group mixes T1, PD and T2 [741310].
- **Laterality tag missing on about 50% of studies** [741310; repo]. About 7 train studies contain **both knees** [735639 Tim Krige]. For bilateral studies the host "adjusted report text or DICOM metadata… to disambiguate" [733826].
- PatientSex is in the DICOM header (it was dropped from the CSV on purpose) [733423 Chen]. ACL ≈54% M / 32% F; Medial OA ≈12% M / 45% F [734004]. Metadata alone gives 0.652 macro (random folds) and **0.598 site-grouped**, i.e. site memorisation and no real shortcut [733517, 734004]. There are 265 scanner fingerprints.
- Data come from 16 sites on five continents, with reports in 12 languages [AuntMinnie]. A crude count found Dutch ~1,684, German ~715, Spanish ~681, English ~504 and others (Greek, Cyrillic, Turkish) [repo]. Turkish negation comes after the term [734106].
- FrameOfReferenceUID differs between series of the same study, so do not assume a shared coordinate frame across series (question unanswered) [736457].

### 2.5 Test-distribution notes
- No reports at test time [733592, 734118].
- Prevalence may differ across train, public and private [data page].
- The public LB scores **higher** than local OOF on weak labels; the gap is often about +0.02 to +0.05 [736635; 735304 Tucker Arrants: CV 0.87–0.90 maps to LB ~0.94]. The ordering is preserved: "extremely strong correlation" between OOF on extracted labels and LB [735304 Nicolai Karcher; Tucker Arrants], and OOF pseudo-label CV is "pretty well correlated with LB" [735304 tennogh].

---

## 3. Insight digest

### 3.1 What medal-zone and top people revealed (all public claims)
| Who (score) | Revealed | Topic |
|---|---|---|
| **azamat1ch** (public 0.95x) | Single architectures, 5-fold: **ConvNeXt-T 0.959**, EffNetV2-S 0.956, CoAtNet 0.952 | 735304 (10-09) |
| **colum2131** (0.956, rank ~60) | ConvNeXt-T 2.5D 0.953, DINOv2-S 2.5D 0.951, both 0.955. **Pseudo-labels**, no external data. Labels from a multimodal LLM reading reports *plus MRI slices*: Effusion 0.91–0.92, Synovitis 0.84–0.86 vs gold. The image model trained on those pseudo-labels reaches Effusion 0.986 and Synovitis 0.859 on gold. Train on **soft** probabilities | 735304, 745949 |
| **Tanuki_boosting** | 0.955 single model, 5-fold | 735304 |
| **CoolinLai** | ResNet-50 @224, 5-fold: 0.954 | 735304 |
| **Scott Willis** (0.959, efficiency #1) | Small ResNet/EfficientNet; labels from local **Gemma 4**, iterated to about 0.89 gold; a single-fold model hit 0.949 with ~5 min scoring; ~900 models trained; 60–70% of runtime is DICOM reading | 735304, 740474 |
| **Tucker Arrants** (strong; "low 0.95s") | Simple Qwen extraction (0.89 gold). "Labels are not the main lever". 224/288 is the sweet spot. Start from ResNet34/EffNet-B0 with **simple pooling**. Validate on **report-label OOF** (not gold, not LB); OOF tracks LB. The image student beats the teacher; correcting thousands of extraction failures moved the LB by 0 | 735304, 740610, 745214 |
| **Archit Konde** | Single-fold CoAtNet @224 at 0.950 (0.930 OOF on gold). Key: **mix OOF predictions into the extracted labels** (not replace; not mask). The model-heavy mix beat 50/50. Fractions beat 0/1. In-fold predictions just parrot the labels. Check *calibration* on gold-58 | 735304 |
| **Raymond Yuen** (0.958 team "tanuki and raymond") | Multiple weak-label sources (public, API, local) → several teacher models → pseudo-labels → final model. **0.5/0.5 blend** with the original labels. Different teachers help different targets. Total API cost under $5. 384→288 px barely matters | 735304, 743148 |
| **Danial Zakaria / nartaa** (0.949 single) | CoAtNet + finding-specific MIL attention; 96 slices / 94 three-slice windows, 5 slots; trained at 384, **native 320 centre crop** at inference (304: −0.001, 288: −0.002); **anatomical mirror TTA** (sagittal: reverse slice order; coronal/axial: flip width) +0.001; SWA of 3 epochs (+0.001 vs epoch 14); 94→24 windows −0.006; all-plane width flip −0.001; **OAI +0.005/+0.003**; 224 crop efficiency model 0.945 at ~7 min | 746792 |
| **Dread Development** (Raptor CoAtNet author) | Slice **density, not span**, is the lever: 44 slices at 6–94% = 0.926, 44 at 2–98% = 0.917, 64 = 0.928, 80 = 0.932; windows 42→62 +0.003; SWA ≈ no-op; changing pooling at inference costs ~0.02; RadImageNet negative in his pipeline; CoAtNet+ConvNeXt 0.944 in 13 min; "models that disagree outperform" | 737696, 742050, 742926 |
| **Ziad Ahmed** | The windows-per-study curve saturates at k≈31 (k=16 0.8867, k=31 0.8932, k=62 0.8948 on the weak-label ruler); density did not transfer to a slot-attention head | 737696 |
| **Chris Deotte** (NVIDIA GM) | Pseudo-labelled "single models" match ensembles; one L40S, dozens of experiments per day; predicts **gold ≈ 0.96, 1st ≈ 0.975** | 735304, 735767, 743303 |
| **hengck23** | "model is not the key, it is your labels"; OAI "changes the competition"; minimal-change label denoising ranked by gold+LB; active labelling with a multimodal LLM on ROI slices | 735304, 745861, 747658 |
| **NguyenThanhNhan** | Fine-tuned **Qwen-3.5-2B VLM** (LoRA + vision encoder) @384, single fold **0.950**, ~3.5 h | 735304 |
| **YYama** (top team) | Uses **no external data**; gains not disclosed | 743416 |
| SpeedSci | DINOv2 0.942/0.945, EffNet-B4 0.940, ResNet-50 0.940 (5-fold); ensemble 0.947–0.950. Changing labels alone moved DINOv2 0.931→0.942. EffNet with gold 0.945 scored only 0.910 LB (gold overfit) | 745214, 744511 |
| Yann Majewski | 0.936 single-fold small ResNet @224; combining his labels with a public set gave +0.015 | 735304 |
| Prateek Grover | CoAtNet inference ~40 → <10 min (details in §3.6) | 743374 |

### 3.2 Labels: what works and what does not
**Works**
- An LLM over regex (+0.06 gold); cheap models are enough (Gemma/Qwen locally; under $5 API) [733932, 735304].
- Soft targets over hard: SOFT beat HARD in 3/3 seeds, +0.014 [734105 FHZ982]; rounding hurt [735304 Archit].
- An explicit "not addressed" state, handled per finding: silence = negative for Baker's, OA and fracture; Synovitis silence → fill from Effusion [733932].
- Down-grade "small/mild" effusion and minor findings to match the rubric (Effusion 0.628→0.726 on gold) [738339]; sharpening soft labels that sit near 0.5 [738339].
- An OA vocabulary built from *consequences* (osteophytes, joint-space narrowing, chondral loss, "tricompartmental" fires all three) revived Lateral OA from 0.47 to 0.83 in a rule extractor [734095].
- **OOF-teacher pseudo-label blending**, 50/50 up to model-heavy, using only out-of-fold predictions, from several teachers [735304; 743148].
- Averaging two independent label tables beats either alone on several findings [737454]. A consensus of public tables ≈ the best single table [747540].
- Gold-58 is used in training at high weight by some; others hold it out for selection. Either way it is about 1.3% of the loss mass [735304 Tucker].

**Does not work / caution**
- Fine-tuning extraction prompts on gold-58 overfits (gold 0.945 → LB 0.910) [745214]; gold-58 rankings were overturned by the LB three times [733932]; one team's gold ranking ran *opposite* to the LB [740439].
- Masking uncertain cells did nothing [735304 Tom Aindow, Archit; 745576]. Replacing labels with predictions was worse than mixing [735304 Archit].
- Soft-bootstrapping that improved CV and gold sometimes did **not** move the LB [735304 Nicolai Karcher; 743148 Myo Min Htet, who used the same encoder for teacher and student].
- Blanket imputation of all findings from co-occurrence: −0.007 [733932].
- Reports as an auxiliary loss or embeddings: hurt [740162 Rasoul]. Image → generated report → labels is circular [743148 Sanjib Biswas].

### 3.3 Image pipeline consensus
- Physical **140 mm** FOV crop (100 mm cuts anatomy; 160 mm is similar) [735304 Parag, Cody_Null]; 130–150 mm used by others.
- **224–320 px** are all fine; 384→288 barely matters (Raymond Yuen) [735304, 743148]. CoAtNet benefits from a 384 cache with a native 320 crop [746792].
- 2.5D: 3 adjacent slices as channels (triplets), **adjacent beats spread** (centre-9 adjacent +0.018 over 9 equally spaced) [737597]. Per-series 2–98% percentile normalisation [repo; 737696].
- Coverage: 5 series slots (sag-FS, sag, cor-FS, cor, ax-FS). More windows help up to ~30–60 per study, then saturate [737696, 746792].
- Head: finding-specific attention-MIL over the window bag (CoAtNet "Raptor"), or simple pooling for CNNs [740610 Tucker]. GRU/LSTM over slices is untested publicly [738495].
- **Laterality:** canonicalising right→left knees was harmful in the repo's own A/B (−0.097). Berat found left/right not a problem [737597]. The useful form is the **anatomical mirror as TTA/augmentation** (sagittal = reverse slice order; coronal/axial = flip width) [746792]. If you flip, the medial/lateral labels must *not* swap: a width flip of a left knee looks like a right knee, and the medial compartment stays medial.
- Backbones: DINOv3 ≤ DINOv2 for several people [733313, 740162]; RadImageNet helps Synovitis (0.78 vs 0.62–0.72) [737566] but was negative for Dread [737696]; Pillar-0 breast-MRI fine-tune only 0.797 [738036]; size does not matter [735154, 738096].
- Training tricks that did nothing: EMA, mixup, 60 epochs (all within noise); **ASL loss −0.139** [737597]; bigger encoder [735154]; 2×2 region tokens [740610]. Longer schedules helped one DINO pipeline (25→50 epochs +0.004) [740610]. Heavy augmentation and regularisation help small CNNs [743148 Myo Min Htet].
- Preprocessing contract: normalisation constants must match the checkpoint (RAD-DINO mean/std bug) [735154]. Crop/geometry mismatches silently degrade arms [repo].

### 3.4 Per-finding headroom (gold-58 OOF diagnostics)
Will's 20-model DINO/CoAtNet stack [740610]: MCL 0.98, Baker's 0.96, ACL 0.96, Medial OA 0.95, Medial Meniscus 0.95, Contusion 0.92, Effusion 0.90, Fracture 0.88, **Lateral Meniscus 0.86, PF OA 0.82, Lateral OA 0.78, Synovitis 0.69**. Archit Konde: Synovitis 0.797, Lateral OA 0.816, PF OA 0.874 [735304]. The weak four (Synovitis, Lateral OA, PF OA, Lateral Meniscus) are exactly where OAI supervision was added by nartaa, and hengck23 hints that the classes with "more error" have "external data with external label" [747613].

### 3.5 CV/LB correlation and validation
- Use report-label OOF on 4,349 studies (resolution ~±0.001–0.002) as the main ruler; gold-58 (SE 0.02–0.03) only as a regression guard; the LB only for big moves [735304 Tucker; 745214; 733932; repo].
- The LB is quantised to 3 decimals, and differences under ~0.0005 are invisible [740439].
- Site-grouped CV costs ~0.05 on metadata-only models [733517, 734004]; it is unknown whether the split is by site [734681].
- Validate a pseudo-label pipeline against the **original** labels, never against the repaired ones [735304 Tucker].

### 3.6 Runtime, efficiency and notebook pitfalls
- Fast inference recipe [743374 Prateek Grover]: one model copy per GPU with its own shard (no DataParallel); read each DICOM once with 16 threads (59 → 372 MB/s); parse headers directly (0.307 → 0.058 ms/file); `np.frombuffer` on PixelData (1.22 → 0.33 ms/slice, bit-identical); percentiles on a strided subsample (the sort took 60% of decode time); fp16 weights and inputs without autocast (2.08×); uint8 to GPU with on-device normalisation.
- Slice-ordering header pass can be the wall (12 min of a 32-min run) [736678 Berat]. Do not read headers for all 820k files [745759].
- **bf16 on T4 is emulated, ~50× slower** → time guard → constant 0.5 → 0.710 instead of 0.851 [744230]. Use `torch.cuda.is_bf16_supported(including_emulation=False)` [744230 Roman Vuskov].
- P100 is incompatible with the current torch, so set `machine_shape: NvidiaTeslaT4` [735854].
- Hidden-run failures ("Notebook Threw Exception", "unhandled error while rerunning") pass on the 3 visible studies. Diagnose by running the exact inference path on ~1,300 *train* studies in a commit [743779 Chris Deotte; 734676; 734062 PC Jimmmy]. Common causes: NaN predictions, malformed DICOMs, memory/time, paths [743779; 745984; 735854]. These failures cost a submission slot [734062].
- Kaggle `ERRORED_MOUNTING_DATASET` errors (Oct 4–7) [746542]; GPU queues stuck for hours [743746]. Leave margin before Oct 22.
- An automated account ban hit Jiwei Liu over a TorchInductor-cache notebook (flagged as "resource abuse"); it was reinstated on appeal [737820]. Avoid unusual compute-caching notebooks.
- Our own history: v2–v5 crashed on the hidden rerun because fall-backs were turned into fatal asserts [EXT_SUBMISSION_PLAN]. Keep everything non-fatal, with prior fills and logged counts.

### 3.7 Leaks and shortcuts
No leak has been found. DICOM metadata reaches only 0.65 (random folds) or 0.60 (site-grouped) [733517, 734004]. PatientSex is a legitimate weak prior that is available at test time [733423, 734004]. Site fingerprints let a model memorise reporting style rather than disease.

### 3.8 Public notebook landscape (context; the sibling agent covers code)
- The 0.950 block (455 teams): "Apex Grandmaster Stack" (sujanmajhisuzan; nartaa's OAI CoAtNet + goodpjw2008's 2.5D ConvNeXt-MIL reader), plus re-blends of it (karttikjangid05 Apex+nartaa-0945 = 0.950; 60/40 with another family dropped to 0.948), heliosli blend-gold (nartaa 0949 at 0.7 + a ConvNeXt-small at 0.3) [txt/ md files]. **All of them depend on `nartaa/rsna-knee-publication-swa-weights-20261007`, which was trained with OAI** [737950 Ooi Zhee Chen].
- The 0.943 block (699 teams): Speedy Raptors / DINOsaur stacks. The 0.941 block came from per-label LB-probed weights [742050].
- The "0.954+" and "0.957" titles are self-reported or local numbers, not verified LB scores [747289 −24 votes; EXT plan on kminsher].
- Mattia Angeli: blend-weight tweaking is "overfitting the public LB 0.001 at a time" [742050, 736268].

### 3.9 Shake-up assessment
- Top-10 teams (100–300 submissions each) mostly have their own models; at the top the private order should mostly hold, give or take 0.002–0.004. PC Jimmmy expects a large shake-up because of label noise [735767, 742327].
- Public-notebook blocks (0.950, 0.943) will move together. A team with its own decorrelated model that sits at 0.950+ is likely to rise relative to the block [742327 CoreyJamesLevinson, PC Jimmmy: "jumped 1000 places when a shared notebook failed"].
- Per-label weights tuned on the public LB are the biggest overfit risk (Lateral Meniscus weight = 1.00 in the 0.941 recipe) [742050].
- **Rule risk:** if the host rules OAI or OAI-trained weights out, prize-eligible teams using them are DQ'd. Non-prize teams are rarely audited [743416 Tucker Arrants], but medals could still be affected. The DFDC precedent: the top two were removed for disallowed external data [743416 Eugene Khvedchenya].

---

## 4. Prior art with transferable tricks

**RSNA 2024 Lumbar Spine Degenerative Classification** (1,874 teams; winners announced Nov 2024: 1. Avengers, 2. IanPan-Kevin-Yuji-Bartley, 3. SonySpine & tkmn & Moyashii; RSNA press release, zenn summaries):
- 1st: a 3D ConvNeXt predicts the best slice (instance), a 2D ConvNeXt regresses disc coordinates on it, then ConvNeXt-S / EfficientNetV2-S on crops with a **bi-LSTM + attention-MIL** over slices. Shift augmentation is tuned to the localisation error. L1 + CE losses.
- 2nd: YOLOX or a CNN-transformer for localisation → MaxViT / **CoAtNet** / NFNet / CSN crops with 3/5-channel neighbours, 27 crop-offset patterns as TTA. **Dropping samples with large prediction error (noisy labels) "improved results substantially"**. **Labelled + pseudo-labelled data mixed 50:50**. One member used only the central 24 sagittal frames with LSTM + attention, sequence reversal and manifold-mixup augmentation, and 9-rotation TTA.
- 3rd: a CenterNet detector, a simple 2D encoder with attention, manual label fixes plus pseudo-labels.
- **Transfer here:** (a) noisy-sample dropping or down-weighting using OOF loss; (b) pseudo-label mixing; (c) anatomy-localised crops for the small structures (ACL/MCL/menisci/compartments) as a second view; (d) slice-sequence heads with reversal augmentation (reversal = our sagittal mirror); (e) offset-crop TTA.

**RSNA 2022 Cervical Spine 1st:** 3D segmentation → per-vertebra crops → 2.5D (±2 slices = 5 channels) CNN + **2-layer bi-LSTM**; mask-as-channel; 7.5 h ensemble. Transfer: crop around predicted anatomy and add a mask or ROI channel.

**RSNA 2023 Abdominal Trauma 1st** (recalled, not re-verified here): TotalSegmentator organ masks for crops, 2.5D CNN + GRU, auxiliary segmentation loss. Transfer: auxiliary dense supervision (even rough femur/tibia/patella masks) gives the model localisation priors for small structures.

**RSNA 2025 Intracranial Aneurysm:** the 2nd place used external TopCoW/TopBrain segmentation data [743416 YYama]. The pattern is that external segmentation-style supervision is what top medical teams add when it is allowed.

**MRNet (Bien et al. 2018, Stanford, 1,370 exams):** AlexNet per plane, max-pooled over slices, logistic regression across 3 planes. AUC: abnormal 0.937, **ACL 0.965, meniscus 0.847** (meniscus is the hard one). Azcona et al. reached 0.934 mean through transfer learning and careful augmentation; a BiGRU over slices beat attention in a 2026 follow-up. Transfer: **per-plane experts with late fusion**, sequence models over slices. MRNet itself is legally unusable here (RUA forbids derivatives; no ruling).

**fastMRI+ (Zhao et al. 2022, Sci Data):** 16,154 bounding boxes and 13 study-level labels for 22 knee pathology categories (coronal PD). Ideal as **localisation pretraining** for menisci, cartilage, marrow oedema and effusion, but access goes through the fastMRI DSA (application/agreement), so it is likely not "equally accessible". No ruling.

**OAI (NDA):** MOAKS semi-quantitative scores (cartilage per subregion, BMLs, meniscal tears, effusion-synovitis, Hoffa-synovitis, Baker's cyst) on thousands of knees, with sagittal IW-FS TSE, coronal IW, sagittal DESS and axial reformats. It covers exactly the weak findings; nartaa's masked-label recipe is the template [746792]. A dense network trained on 1,628 OAI knees detects MOAKS effusion [Sci Rep 2022]. Download through nda-tools with a file list of MRI-only gz files (about 450 GB in full) [746792 hengck23, minhtu]. **Legal status unresolved; see §5 risk R1.**

**Astuto et al. 2021 (Radiology: AI):** 3D CNN ROI detection then lesion grading for cartilage, BML, meniscus and ACL; AUC 0.83–0.93; 1,435 studies. Transfer: two-stage ROI → grade.

**Public-checkpoint options already on Kaggle:** the Raptor CoAtNet family (dreaddevelopment; finespacing v9 0.932 solo, widedense v4 0.927), nartaa 0949/0945 (OAI-trained), goodpjw2008's 2.5D ConvNeXt-MIL reader, the 20-member DINOv2 stack and DINOv3 folds (CC0 re-upload), RadImageNet heads.

---

## 5. Strategy: from 0.942 to the medal zone and a top-50 attempt in 13 days

### 5.1 Honest targets
- Today's cut-offs: top-10 ≥ 0.963, top-50 ≥ 0.957, top-100 ≥ 0.954, silver line ≈ 0.950/0.951 (top 5% = rank 279), bronze ≈ 0.950 (top 10% = rank 558; inside the 455-team tie block). These will rise about 0.002–0.004 by Oct 22 (Chris Deotte expects gold ≈ 0.96).
- **Top-10 (0.963+) is not realistic** on 60 T4 hours without OAI-scale external supervision and a mature pseudo-label loop. Probability is below 5%; I would not plan for it.
- **Top-50 (≈0.958–0.960 by deadline)** is a stretch, maybe 15–25%. It needs your own ConvNeXt-T/EffNetV2-S 5-fold with a pseudo-label round (people report 0.953–0.959 for that alone) plus a rank blend with the public 0.950 stack.
- **Silver or bronze (≈0.951–0.954)** is likely (≥60%) if your own 2.5D CNN reaches ≥0.948 and it blends with the public stack.

### 5.2 Calendar and compute
- GPU quota resets **Sat Oct 10 00:00 UTC** and **Sat Oct 17 00:00 UTC**, giving about 60 T4-hours plus whatever is left this week. **CPU notebooks do not consume GPU quota.** Run all DICOM decoding and cache building on CPU sessions (12 h each, several in parallel) and publish the cache as a private dataset. A 224 px, 6×9-slice uint8 cache is 11 GiB [735154]; at 256 px with 5 slots × 16–24 slices it is roughly 25–40 GiB.
- T4 training speed: ~0.23 s/step for DINOv2-S at 224 with fp16; one fold ≈ 76 min including cache build, ~20 min of GPU without it [735154]. A ConvNeXt-T 2.5D at 224 with ~48–96 images per study should take about 1–1.5 h/fold. With a T4×2 session, run **two folds in parallel, one process per GPU** (not DataParallel).
- Team merger deadline is **Oct 15**. Several solo or small teams with their own 0.945–0.955 models are openly recruiting (SpeedSci 0.947–0.950 [744511]; ringbearer 0.948 [744511]; Tanuki_boosting 0.955 single [735304]; Pand 0.945 [745091]; KalyanG17 0.942 [745091]). **A merger with a team that already has an independent 0.95x model is the single highest-expected-value move for top-50.** Decide by Oct 14.

### 5.3 What to do TODAY (Oct 9, 4 submissions left)
Each submission is a notebook version. The commit run on the 3 visible studies costs only minutes of GPU, and the hidden rerun does not use your quota.
1. **S1 – anchor at 0.950:** fork the strongest *reproducible* public stack (the Apex Grandmaster Stack family, or heliosli blend-gold / karttikjangid Apex+nartaa-0945) unchanged, on T4. Verify it finishes and record the hidden runtime. Expected 0.950, which moves you from rank ~1954 to ~700. Ties are broken by submission time, so earlier is better.
2. **S2 – diversity probe:** the same stack plus one decorrelated public family with **equal rank weights, no per-label weights** (for example the Raptor finespacing v9 or the goodpjw ConvNeXt-MIL reader at 0.2–0.3). Expected 0.949–0.952. This tells you whether diversity pays at this level.
3. **S3 – non-OAI hedge:** the best public stack that uses **no OAI-trained weights** (the Speedy Raptors 0.943 graph + Raptor v9/v4 views, i.e. the repo's `kaggle/ext` notebook, or Dread's latest non-OAI CoAtNet). It prices the hedge you may need if the host rules against OAI.
4. **S4:** keep it for a rerun if S1–S3 hit Kaggle errors (`ERRORED_MOUNTING_DATASET` reruns were free [746542]). Otherwise use it for an anatomical-mirror-TTA variant of S1, if S1 lacks one.
- **CPU today (free):** start the training cache: IPP-sorted, 140 mm crop, 2–98% normalisation, 256 px stored, 5 slots (sag-FS, sag-nonFS, cor-FS, cor-nonFS, ax-FS) × 24 slices uint8. Build the label table: the CC0 consensus of public LLM labels [747540] + steven v4 + pilkwang + JEV, kept soft. Apply the per-finding silence rules [733932]: Synovitis silent → fill from Effusion; Baker's/OA/Fracture silent → low prior (≈0.03–0.05); "small/mild" effusion and Baker's → ~0.3. Make 5 study-level folds, multilabel-stratified.

### 5.4 Plan Oct 10–22
| Days | Work | GPU-h | Submissions |
|---|---|---|---|
| Oct 10–11 | **Teacher A:** ConvNeXt-T 2.5D @224–256, 3-slice triplets, 16–24 windows, finding-wise attention-MIL (or mean+max pooling), soft BCE, fp16, AdamW cosine, flips and anatomical-mirror augmentation, 5 folds (two per T4×2 session). Track OOF macro on the weak labels and on gold-58 | ~8–10 | Teacher A alone (calibrates CV→LB; expect 0.940–0.950); Teacher A rank-blended 50/50 with the S1 stack |
| Oct 12–13 | **Pseudo-label round:** target = 0.4·report-soft + 0.6·OOF(Teacher A); keep report labels on explicit statements, use model-heavy weights on silent cells (synovitis, effusion, contusion). **Student B** = EffNetV2-S or ConvNeXt-T at another resolution (288) for diversity. Optionally drop or down-weight the top ~3% of cells by OOF loss (the lumbar-2nd trick) | ~8–10 | Student B; B+A; B+A+S1 |
| Oct 14 | Merger decision. Inference speed-up (threaded decode, fp16, one model per GPU) | ~2 | 2 probes (e.g. mirror TTA, window count) |
| Oct 15–16 | Optional **Student C** (DINOv2-S 2.5D, or CoAtNet-0 @224 if quota allows), or a second seed of B. Optionally a per-finding specialist only for the weak four if OOF shows ≥ +0.003 | ~8 | blends evaluated on OOF first; ≤2 LB checks/day |
| Oct 17–19 (quota reset) | Final pseudo-label round from the A+B(+C) ensemble OOF → **Student D** (the best backbone retrained). Full test-path rehearsal on 1,300 *train* studies in a commit to catch crashes and time-outs | ~15–20 | D; D+ensemble |
| Oct 20–21 | Freeze. Equal-weight rank blends only. Build two final candidates with verified runtimes (efficiency: one fast candidate under 15 min if it is within 0.003 of the best) | ~4 | final candidates |
| Oct 22 | Buffer for Kaggle outages; **select 2 finals by ~18:00 UTC** | — | — |

Design notes, each backed by evidence:
- Do not change the inference pooling of a trained head (−0.02 [737696]). Keep the training and inference windows consistent and use as many windows as the time budget allows (saturation ~30–60).
- Validate only on report-label OOF plus gold-58 as a guard. Promote a change only if OOF moves ≥ +0.002 and gold does not drop by more than its SE. Use the LB for blend sanity only, never for per-label weights.
- Do not canonicalise laterality; use mirror TTA (sagittal slice-order reversal, coronal/axial width flip). It was +0.001 for nartaa.
- RadImageNet-style medical pretraining may help Synovitis [737566], but its licence is unclear and it was negative for Dread. Treat it as optional diversity, not the core model.
- Wrap every decode, never assume `Fluid_Sensitive == Fat_Suppression`, handle missing planes with masks, no bf16, T4 only, no NaNs (clip, then `nan_to_num`), guard the time per study, and log fallback counts.

### 5.5 Private-LB-robust final pick
- **Pick 1 (main):** the best **OOF-supported** blend: your own models (A, B, D…) + the public 0.950 stack, equal-weight average ranks, no per-label LB-tuned weights. This should have a public score within 0.001–0.002 of your best.
- **Pick 2 (hedge):** depends on the OAI ruling.
  - If the host has **not** clearly allowed OAI by Oct 21: your **own models + non-OAI public members only**. This protects medal or prize eligibility if OAI-trained weights are ruled out, and it is decorrelated from the 455-team block.
  - If OAI is clearly allowed: the most *different* strong candidate (for example the fast efficiency-oriented variant, which also competes on the efficiency track).
- Avoid picking two near-identical blends. Avoid any blend whose public gain comes only from per-label weights (the Lateral Meniscus = 1.00 trap [742050]).

### 5.6 Risk list
| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| R1 | OAI or OAI-trained public weights ruled out (or prize DQ) | Medium / high for the 0.950 block | Pick 2 without OAI; your own models as the core; watch [743416], [747455], [747660] |
| R2 | Hidden-rerun crash (NaN, corrupt DICOM, codec, path, P100, bf16, missing plane, FS≠FatSat) | Medium / a lost day's slots | Train-set 1,300-study rehearsal; non-fatal fallbacks with logging; T4 pin |
| R3 | Timeout (> 9 h) or a time guard silently filling 0.5 | Low–medium | Log seconds per study ×1,322 + 30% headroom; report the fallback count |
| R4 | Kaggle platform errors or queues near the deadline | Medium (seen Oct 4–7) | Finish candidates by Oct 21; resubmit on infra errors |
| R5 | GPU quota exhausted mid-week | High | CPU-only caching; two folds per T4×2 session; ensure training never decodes DICOM; stop early per fold |
| R6 | Pseudo-label loop improves CV but not LB [735304 Karcher; 743148 Myo] | Medium | Strictly OOF teachers; different encoder for the student; validate against *original* labels |
| R7 | Public-LB overfitting by probing | Medium | ≤2 LB checks/day for blends; OOF-first decisions |
| R8 | Shake-up near medal lines (24% survival near bronze [742327]) | Medium | Own decorrelated models; equal-weight blends |
| R9 | Licence issues for prize eligibility (CC-BY-NC-SA RadImageNet, DINOv3 gated) | Low for non-prize | Prefer timm ImageNet weights (ConvNeXt/EffNetV2/CoAtNet) |
| R10 | Automated account ban for unusual notebooks [737820] | Low | Avoid compile-cache tricks and mass dataset pushes |

---

## 6. Topic index (most useful threads)
735304 Best single-model score (125 comments) · 746792 nartaa 0.949 recipe (OAI) · 733932 "Not addressed" is a label (steven) · 743416 External datasets clarification · 741819 OAI ruling · 734109 KneeCoT banned · 733965 LLM APIs allowed · 733826 / 733491 host on image-based labels · 733343 label definitions · 737312 FS vs FatSat warning · 737696 Raptor weights / density lever · 742050 why forks sit at 0.941 · 743374 fast inference · 744230 bf16 PSA · 735854 submission gotchas · 745759 slice-order bug · 741310 data traps · 747540 8 label sets vs gold · 737454 label tables benchmark · 734095 rule-based labeller · 737566 synovitis · 745949 effusion/synovitis · 740610 0.903 plateau plus Tucker's advice · 743148 pseudo-label 50/50 · 745214 label optimisation · 735154 encoder size null · 737597 ablations at 0.79 · 733517 / 734004 metadata probes · 736678 efficiency LB · 733475 RuntimeSeconds definition · 743356 two finals cover both tracks · 742327 shake-up stats · 740439 3-decimal LB · 735767 ceiling predictions · 744056 / 741212 / 747455 / 747660 public-weights eligibility (unanswered) · 744519 test planes (unanswered) · 740094 metric edge cases (unanswered).
