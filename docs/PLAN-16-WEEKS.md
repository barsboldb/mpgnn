# 16-week plan — thesis A

Settled 2026-09-17. Title: **What it costs to make a graph algorithm learnable by a
transformer** / *Трансформерт графын алгоритм сургахад юу шаардагдах вэ*.
Spine: Q2 → Q1 → Q5 → Q6 (`docs/THESIS-CANDIDATES.md` §A).

Fixed dates: **week 13 = pre-defense**, **week 15 = criticism week**, **week 16 = defense**.
Everything research-side must be frozen by the end of week 11; week 12 is slides and
advisor review only. Week numbers are authoritative; calendar dates below assume week 1
starts Mon 2026-09-21 — adjust to the department calendar.

| week | dates (approx) | T1 dataset | T2 grid | T3 traces | T4 silent-op | T5 writing | T6 pre-def | T7 defense |
|---|---|---|---|---|---|---|---|---|
| 1 | Sep 21 | ██ | | | | intro/related | | |
| 2 | Sep 28 | ██ | spec | | | intro/related | | |
| 3 | Oct 5 | audit table | ██ | | | ch.1 | | |
| 4 | Oct 12 | | ██ | | | ch.1 | | |
| 5 | Oct 19 | | ██ | ██ | | | | |
| 6 | Oct 26 | | figures | ██ | | ch.2 | | |
| 7 | Nov 2 | | | ██ | ██ | ch.2 | | |
| 8 | Nov 9 | | | figures | ██ | ch.3 | | |
| 9 | Nov 16 | | | | **stop** | ch.3 | | |
| 10 | Nov 23 | | | | | ch.4 + buffer | | |
| 11 | Nov 30 | | | | | **full draft** | | |
| 12 | Dec 7 | | | | | | slides, review | |
| 13 | Dec 14 | | | | | | **PRE-DEFENSE** | |
| 14 | Dec 21 | | | | | revisions | | |
| 15 | Dec 28 | | | | | | | **criticism** |
| 16 | Jan 4 | | | | | | | **DEFENSE** |

---

## T1 — Dataset hardening and the leak audit (weeks 1–3)

The gating task. Every negative result in the thesis rests on `connectedness_hard`, which
still has a local bridge-detection shortcut (3-layer GIN with degree features: 0.975 at
depth < diameter).

- Kill the shortcut: bridges whose endpoints are degree-typical and whose k-hop
  neighbourhoods are indistinguishable from non-bridge neighbourhoods for k up to the
  diameter. Apply to `connectedness_hard` and `connectedness_hard_diam`.
- Re-measure every rung back to chance: min-degree, mean-degree, edge-count, GIN-degree at
  depth 3, sequence length. Regenerate caches; mark pre-fix results non-comparable.
- Half-day side measurement: the min-degree rule on GraphQA connectivity (Sanford's
  benchmark, ER 5–20 nodes). Goes in chapter 1 as the "reference setups" row.
- **Deliverable (end of week 3):** the leak-audit table — four leak classes, each with its
  falsifying baseline and the number it scores before and after the fix.
- **Kill criterion:** if the GIN-degree baseline is not at chance by end of week 2, ship
  the best construction achieved and narrow chapter 2's claim to "at depth ≥ diameter".
  Do not let this slip into week 4.

## T2 — The pre-registered grid (weeks 2–6)

- Week 2: write the sweep spec *before running anything* — configs, seeds, stopping rule,
  success threshold — and commit it. That commit is the pre-registration.
- Grid: depth {1,2,4,8} × width {64,128,256} × encoding {edge_list, adj_rows} × 3 seeds
  on the hardened `connectedness_hard` and `connectedness_hard_diam`; trained to
  saturation; log samples-to-learn, not just final accuracy.
- Overfit-N ladder alongside (N ∈ {10, 100, 1000, full}) at the 4×64 reference config —
  the capacity-vs-generalization separator that is the chapter's actual novelty.
- Sanity row: the same grid on Yehudai's data, where it should hit ~1.00 fast. Proves the
  pipeline, not the dataset, is what changed.
- **Deliverable (end of week 6):** chapter 2 figures — accuracy heatmap over the grid,
  samples-to-learn curves, the overfit ladder.

## T3 — Trace supervision replicates (weeks 5–8)

Mostly done; what is missing is what makes it publishable.

- Seed replicates (3+) on `diameter_controlled` and `connectedness_hard_diam` for error
  bars on the flatness plot.
- The **answer-only control on `connectedness_hard`** — currently missing; the other two
  datasets have it.
- Depth-1 row at matched data (32k), so the depth claim has its lower bound.
- Re-run the AR-CoT headline on the *hardened* dataset from T1 so chapters 2 and 3 share
  one dataset.
- **Deliverable (end of week 8):** chapter 3 figures — accuracy vs diameter with error
  bars, trace vs answer-only, depth 1 vs 2.

## T4 — The silent-op chapter, time-boxed (weeks 7–9)

Upside, not load-bearing. Hard stop at the end of week 9 regardless of outcome.

- Write up the three clean instances from existing logs and diagnostics (this needs no
  compute): predicted-then-fixed instance 3 as the centrepiece.
- Decide instance 2's status (fixed by 4× data — off-mechanism); footnote or bring under
  the branching-factor statement.
- `wl_gather` attempt, ≤ 2 weeks: unstall `iso_wl`, run the fix. If it lands, chapter 4
  becomes the headline and the abstract changes. If not, chapter 4 stays as three
  instances plus a stated prediction.
- **Deliverable (end of week 9):** chapter 4 draft in whichever of the two forms applies.

## T5 — Writing (weeks 1–11, continuous)

Write in parallel with experiments, never after them. Order is deliberate: the intro is
first because its positioning against Sanford / Ye / Abbe shapes how every result is
phrased.

- Weeks 1–2: introduction + related work. Position against Ye 2026 on page one ("data is
  one lever on the same axis"); Abbe 2024 on the representability/learnability gap;
  Sanford 2024 for the representability half.
- Weeks 3–4: chapter 1 (leak audit) — the taxonomy, the measured rungs, the reference
  setups, the clique probe.
- Weeks 6–7: chapter 2 (the grid) as figures arrive.
- Weeks 8–9: chapter 3 (traces).
- Week 10: chapter 4, discussion, the model-organism framing, the cost-accounting
  corollary; buffer for anything that slipped.
- **Week 11: complete draft to the advisor.** Research freeze. No new runs after this.

## T6 — Pre-defense preparation (weeks 12–13)

- Week 12: incorporate advisor comments; build the slide deck from the chapter figures
  (reuse `reports/progress-presentation.typ`); one dry run with a timer; write the
  one-page summary the committee reads.
- Week 13: **pre-defense.** Record every question asked — they are the criticism-week
  question list.

## T7 — Revisions, criticism, defense (weeks 14–16)

- Week 14: revisions from the pre-defense. Prioritize by what a reviewer can attack; this
  is where the "you didn't train long enough" rebuttal gets its samples-to-learn figure
  pointed at explicitly.
- Week 15: **criticism week.** Written responses to every reviewer point, tracked in a
  table (point / response / change made / where in the text).
- Week 16: **defense.** Final deck, final dry run, the anticipated-questions sheet from
  weeks 13 and 15.

---

## Milestones

| end of week | must be true |
|---|---|
| 2 | sweep spec committed; shortcut fix in progress |
| 3 | leak-audit table complete; GIN-degree at chance |
| 6 | grid + overfit ladder figures done |
| 8 | trace replicates + answer-only control done |
| 9 | `wl_gather` resolved either way; chapter 4 drafted |
| 11 | **full draft to advisor; research freeze** |
| 13 | pre-defense delivered |
| 15 | every criticism answered in writing |
| 16 | defense |

## Slack and what gives first

Two weeks of slack are built in: week 10 (writing buffer) and the T4 time-box. If more is
needed, cut in this order: (1) `wl_gather` attempt, (2) the Yehudai sanity row in T2,
(3) neural-compilation hardening (already optional, not scheduled), (4) depth-1 row in T3.
Never cut: the shortcut fix, the overfit ladder, the answer-only control, the week-11
freeze.
