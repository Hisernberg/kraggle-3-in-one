# Attribution and scope — Enhanced v2

This is a derivative agent, not a claim of wholly original production planning.
The entire September 17 release is embedded without changing its source, including
its original 2802/V43/V46 lineage and notices. New portions use Apache-2.0.

- **sdy623 / jaxa623**, 2802 / K0006, “Two Identical Agents, 90 Points Apart”.
- **Ahmed Berat Ozer**, V43 production/recovery core and V46 opening/sale timing;
  the inherited credits to earlier contributors remain in the embedded source.
- **Seyit Kaan Gunes**, `kaggle.com/code/seyitkaangunes/kaggriculture-2820-score`:
  pre-overflow warehouse guard, clone-ordering and shop-aware herd mechanisms.
- **Ahmed Berat Ozer**, V47 transfer/optimization and V48 queue compaction.
  The two new user-supplied source files are preserved in `reference_20260918/`.
  V48 source SHA-256:
  `4b5402888feeb4170dce38f34bebe56788b62ca287139fce7db72df8eb89bb96`.
- **Kaggle**, official `kaggle-environments` game logic, Apache-2.0: used only
  for local evaluation, not imported by the runtime submission.
- **COK-ZhangZiliang/Kaggriculture**, Apache-2.0: evaluation opponent only.
  None of its production routes are copied into this new layer.

New work: official-settlement unit tests, exact product/slot assignment search,
isolated experimental configurations, optional public-state horizon gating,
paired live-opponent tournaments, independent holdout protocol, integration,
retry-safe last-callable entrypoint and release verification. The selected release
contains only enabled upgrades; rejected herd/production options remain in local
experiment files and are not enabled in the submission.

See the retained September 17 notes at `research/v26/THIRD_PARTY_NOTICES.md`
for original hashes and the full inherited lineage. The optional ZIP includes
the Apache-2.0 license, these notices and the prior notices.

Paper references are conceptual sources, not copied implementations, endorsements
or theoretical guarantees for this heuristic. No pretrained neural model, private
opponent information, future shop draws, identity routing, cross-game files or
network service is used for decisions.

The competition Rules/Evaluation pages were not readable through the available
reader. Local engineering validation is not certification of every competition
condition. Review current rules and upstream licensing before public redistribution.


# Attribution and scope

The submission is a derivative work, not a claim of wholly original production
planning. It preserves the supplied 2802 source and its embedded parent verbatim.

- **sdy623 (jaxa623), K0006 / “Two Identical Agents, 90 Points Apart”**, supplied
  by the user as `2802-two-identical-agents-90-points-apart.ipynb`.
  Decoded source SHA-256:
  `944aa64c5ae1296a9a17eb7961ed8d5a96d0b80183d41e954389983d73420a56`.
- **Ahmed Berat Ozer (ahmedberatozer), V43 “Recovering Lost Harvests”**,
  Apache-2.0. Embedded parent SHA-256:
  `919fc1d61050cd96f799979e49177ae3ac7bce98ec9835724238bea73f4a08ed`.
  Earlier authors and notices listed within this parent remain embedded.
- **Ahmed Berat Ozer, V46 “First-Turn Microstructure and Sale Timing”**,
  supplied by the user. The new opening and projected sale implementation are
  adaptations of its attributed Apache-2.0 mechanisms. Source SHA-256:
  `735c370383b70d3bf3aac792f2c147e0afc99166fc9f253ede10e8a030acedb6`.
  V46 credits Nathan Jacob for early feed-purchase insight and sdy623/jaxa623
  for the public sale-advance analysis; those conceptual credits are retained.
- **Kaggle, kaggle-environments**, Apache-2.0. Only local evaluation imports
  the pinned official game engine; the submitted agent does not import it.
- **COK-ZhangZiliang/Kaggriculture V10**, Apache-2.0, downloaded 2026-09-17.
  Used only as a local opponent. Its source and complete notices are in
  `reference_20260917/cok/`. None of its recovery routes are copied into the
  enhanced submission. GitHub source blob: `736577e3810f018f1844b8ac9824a4e69b4c37d3`.

New work in this session: test harness with version pinning and raw records;
composable market ablations; executable sale-prefix ranking by counterfactual
price impact; cash-buffer-controlled sale protection; integration, retry cache,
last-callable entrypoint and action-by-action packaging verification. The new
portions are offered under Apache-2.0, consistent with the supplied lineage.

Paper references are conceptual research references, not copied implementations
and not endorsements. No paper's Nash-equilibrium or safe-improvement guarantee
is asserted for this heuristic agent.

The Kaggle rules and evaluation pages did not expose their full text through
the available reader. This is an engineering audit against the public game
source, **not certification of every competition/legal condition**. The agent
uses documented observations and own planned continuations only. No private
rival state, environment seed, player identity, future shop draws, cross-game
files, external services or extra datasets are used for decisions. Review the
current competition rules and retained third-party licenses before publishing.
