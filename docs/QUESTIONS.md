# Thesis questions — settled, open, and queued

Working map of research questions for the diploma. The findings themselves live in
`docs/CHANGELOG.md` and `reports/executing-graph-algorithms.md`; this file tracks what
is being *asked*. Updated 2026-07-06.

---

## The spine (settled, 07-03 → 07-06)

**"What does it take for a fixed-depth transformer to actually execute a graph
algorithm?"** — answered for BFS/connectivity: five ingredients (supervised steps, leak
audit, local trace targets, zero weight decay, data over depth), depth-2 model flat
across diameters 2–18, adversarial dataset at 0.9925, mechanism verified per
sub-circuit, matched controls at chance. Novelty per claim assessed
(`reports/executing-graph-algorithms.md` §7). Remaining work is production: converged
reruns, depth-1@32k grid row, length-OOD (`cot_pos: none`), optional AdamW-vs-Adam probe.

---

## Next chapter candidate: isomorphism (proposed 2026-07-06)

Working title: **"Trace supervision as a general mechanism: from BFS to
Weisfeiler–Leman."** Theory anchor: message-passing GNNs are provably 1-WL-bounded
(Xu et al. 2019) — the isomorphism analog of what fixed depth was for connectivity.

### Q1 — recipe generality (recommended core question)

*Does the five-ingredient recipe generalize beyond BFS — can a fixed-depth transformer
execute color refinement (1-WL), with accuracy flat in the number of refinement rounds?*

- Direct transfer of the connectivity methodology: WL rounds ↔ BFS levels; "rounds
  needed to distinguish the pair" is the difficulty knob, exactly like diameter →
  build a `wl_round_controlled` pair generator the way `diameter_controlled` was built.
- Answer-only baseline should fail at high round counts; trace model should be flat.
- Known trap to design around: WL's per-round op is "hash the *multiset* of neighbor
  colors" — a global set operation, i.e. the Act-III globality pitfall. The trace must
  decompose it locally (per-node sorted neighbor-color list before the new color — the
  `bfs_expand` trick again). This makes Q1 a second, independent test of the
  trace-locality finding.
- Either outcome informative: success = the recipe is algorithm-general (external
  validity); failure localizes which ingredient was BFS-specific via role-stratified
  diagnostics.

### Q2 — past the WL ceiling (stretch; attempt only after Q1 lands)

*Can trace supervision push a transformer past 1-WL — beyond the provable ceiling of
every message-passing GNN?*

- Traces of individualization-refinement (nauty-style: individualize a node, refine,
  backtrack) on WL-equivalent non-isomorphic pairs (CFI gadgets, strongly regular
  graphs). Positive result = an empirical expressivity separation between GNNs and
  trace-trained transformers.
- Risk: backtracking traces are the *lookahead* structure where Bachmann & Nagarajan's
  pitfall bites hardest; traces are long and branchy. Scope as future work unless Q1
  goes unusually smoothly.

### Q3 — the shortcut ladder (dataset foundation for Q1/Q2)

*A leak-audited difficulty ladder for isomorphism benchmarks.*

- **Discovered shortcut in the current dataset**: `make_isomorphism_dataset` guarantees
  negative pairs have *different degree sequences* — every negative is solvable by
  degree-histogram comparison. Same family as the degree-0 shortcut (06-19), the
  Laplacian leak, and the sequence-length leak.
- Ladder: (a) degree-distinguishable (current; trivial) → (b) same-degree-sequence
  pairs distinguishable at WL round k (the honest middle; the Q1 dataset) →
  (c) WL-equivalent pairs (the wall; the Q2 dataset).

---

## Earlier candidate questions (2026-07-06 discussion) and their dispositions

1. **"Can we accelerate / minimize cost on graph tasks?"** → sharpened into *"parallel
   depth vs sequential tokens: which is the cheaper way to buy reachability?"* — a cost
   accounting (FLOPs, wall-clock, samples-to-grok) over the depth ladder vs the trace
   ladder. ~70% of the data already exists; `reports/gat-vs-local-attention.*` is its
   microscopic sibling (structural vs post-hoc sparsity). Candidate analytical chapter.
2. **"Cheap permutation-invariant graph embeddings?"** → ceiling warning: a cheap
   *complete* invariant would solve GI. Honest version = tradeoff study over the
   existing identity-scheme ladder (degree / adj_rows / random / LPE / learned ids /
   permuted ids): invariance vs expressivity vs sample cost. Well-trodden literature;
   supporting chapter at most.
3. **"An architecture better than transformers yet as fast?"** → unfalsifiable as
   stated; our own data (BDH/GAT/global all in one band in-distribution, all collapse
   OOD) says architecture mattered less than supervision structure. Folded into the
   discussion, not a chapter.

## Standing open problems

- **OOD / distribution-general algorithm execution**: grokked models are
  distribution-bound (ER 0.32 caterpillar-trained vs 0.54 blob-trained; isolated lookup
  transfers at 0.55). Strongest future-work thread: mixed-generator training sets.
  (CHANGELOG 07-05.)
- **Length/size OOD**: train diam ≤ 10, eval 11–18; `cot_pos: learned` vs `none`.
- **AdamW (decoupled) vs Adam (L2-in-gradient) at the same λ** — sharpens the
  weight-decay mechanism claim (CHANGELOG 07-06).
- **KV cache for `generate()`** — engineering, unblocks larger eval sets.
