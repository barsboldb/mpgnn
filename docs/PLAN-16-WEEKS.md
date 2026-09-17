# 16-week plan — thesis A

Settled 2026-09-17. Title: **What it costs to make a graph algorithm learnable by a
transformer** / *Трансформерт графын алгоритм сургахад юу шаардагдах вэ*.
Spine: Q2 → Q1 → Q5 → Q6 (`docs/THESIS-CANDIDATES.md` §A).

Calendar is the department's (from the submitted form *2026 намрын үечилсэн
төлөвлөгөө*): **week 1 = Mon 2026-09-07**, period ends 2026-12-28. Fixed checkpoints:
**week 6 = Явц 1** (progress review 1), **week 9 = Явц 2**, **week 13 = pre-defense**,
**week 15 = criticism**, **week 16 = defense**. The plan was drafted in week 2, so week 1
is already spent on literature.

Research freeze: **end of week 11**. Week 12 is the full draft and slides only.

| wk | starts | Явц | T1 dataset | T2 grid | T3 traces | T4 silent-op | T5 writing | T6/T7 defense |
|---|---|---|---|---|---|---|---|---|
| 1 | Sep 7 | | | | | | lit review | |
| 2 | Sep 14 | | ██ | | | | lit + intro | |
| 3 | Sep 21 | | ██ | spec | | | intro/related | |
| 4 | Sep 28 | | ██ | ██ | | | ch.1 | |
| 5 | Oct 5 | | audit table | ██ | | | ch.1 | |
| 6 | Oct 12 | **Явц 1** | | ██ | | | | |
| 7 | Oct 19 | | | ██ | ██ | | ch.2 | |
| 8 | Oct 26 | | | figures | ██ | | ch.2 | |
| 9 | Nov 2 | **Явц 2** | | | ██ | ██ | ch.3 | |
| 10 | Nov 9 | | | | figures | **stop** | ch.3 | |
| 11 | Nov 16 | | | | | | ch.4, discussion | |
| 12 | Nov 23 | | | | | | **full draft** | slides |
| 13 | Nov 30 | **pre-defense** | | | | | | ██ |
| 14 | Dec 7 | | | | | | revisions | ██ |
| 15 | Dec 14 | **criticism** | | | | | | ██ |
| 16 | Dec 21 | **defense** | | | | | | ██ |

The form groups this into six rows with two sub-tasks each (research / data analysis /
experimental setup / main experiments / results + writing / defense); the T-tasks below
are the same work at the granularity the repo works in.

---

## T1 — Dataset hardening and the leak audit (weeks 2–5)

The gating task. Every negative result rests on `connectedness_hard`, which still has a
local bridge-detection shortcut (3-layer GIN with degree features: 0.975 at depth <
diameter).

- Kill the shortcut: bridges whose endpoints are degree-typical and whose k-hop
  neighbourhoods are indistinguishable from non-bridge neighbourhoods for k up to the
  diameter. Apply to `connectedness_hard` and `connectedness_hard_diam`.
- Re-measure every rung back to chance: min-degree, mean-degree, edge-count, GIN-degree at
  depth 3, sequence length. Regenerate caches; mark pre-fix results non-comparable.
- Half-day side measurement: the min-degree rule on GraphQA connectivity (Sanford's
  benchmark, ER 5–20 nodes). Chapter 1's "reference setups" row.
- **Deliverable (end of week 5, for Явц 1):** the leak-audit table — four leak classes,
  each with its falsifying baseline, before and after the fix.
- **Kill criterion:** if GIN-degree is not at chance by end of week 4, ship the best
  construction achieved and narrow chapter 2's claim to "at depth ≥ diameter".

## T2 — The pre-registered grid (weeks 3–8)

- Week 3: write and commit the sweep spec *before running* — configs, seeds, stopping
  rule, success threshold. That commit is the pre-registration.
- Grid: depth {1,2,4,8} × width {64,128,256} × encoding {edge_list, adj_rows} × 3 seeds
  on the hardened datasets; trained to saturation; log samples-to-learn.
- Overfit-N ladder alongside (N ∈ {10, 100, 1000, full}) at the 4×64 reference config —
  the capacity-vs-generalization separator that is the chapter's novelty.
- Sanity row: the same grid on Yehudai's data (should hit ~1.00 fast).
- **Deliverable (end of week 8, for Явц 2):** chapter 2 figures — grid heatmap,
  samples-to-learn curves, overfit ladder.

## T3 — Trace supervision replicates (weeks 7–10)

- Seed replicates (3+) on `diameter_controlled` and `connectedness_hard_diam`.
- The **answer-only control on `connectedness_hard`** — currently missing.
- Depth-1 row at matched data (32k).
- Re-run the AR-CoT headline on the *hardened* dataset so chapters 2 and 3 share one
  dataset.
- **Deliverable (end of week 10):** chapter 3 figures — accuracy vs diameter with error
  bars, trace vs answer-only, depth 1 vs 2.

## T4 — The silent-op chapter, time-boxed (weeks 9–10)

Upside, not load-bearing. Hard stop end of week 10.

- Write up the three clean instances from existing logs (no compute needed).
- Decide instance 2's status (fixed by 4× data — off-mechanism).
- `wl_gather` attempt, ≤ 2 weeks including unstalling `iso_wl`. Lands → chapter 4 is
  the headline. Doesn't → three instances plus a stated prediction.

## T5 — Writing (weeks 1–12, continuous)

- Weeks 1–3: literature, introduction, related work. Position against Ye 2026 on page
  one ("data is one lever on the same axis"), Abbe 2024 on the gap, Sanford 2024 for
  representability.
- Weeks 4–5: chapter 1 (leak audit).
- Weeks 7–8: chapter 2 (grid).
- Weeks 9–10: chapter 3 (traces).
- Week 11: chapter 4, discussion, model-organism framing, cost corollary. **Freeze.**
- **Week 12: complete draft to the advisor + slides.**

## T6 — Pre-defense (weeks 12–13)

- Week 12: slides from chapter figures (reuse `reports/progress-presentation.typ`), one
  timed dry run, one-page summary.
- Week 13: **pre-defense.** Record every question — it is the criticism-week list.

## T7 — Revisions, criticism, defense (weeks 14–16)

- Week 14: revisions from the pre-defense; point the samples-to-learn figure at the
  "didn't train long enough" rebuttal explicitly.
- Week 15: **criticism week.** Written response table (point / response / change / where).
- Week 16: **defense.**

---

## Milestones

| end of week | must be true |
|---|---|
| 3 | sweep spec committed; shortcut fix in progress |
| 5 | leak-audit table complete; GIN-degree at chance |
| 6 | **Явц 1**: literature + hardened dataset + audit table |
| 8 | grid + overfit ladder figures done |
| 9 | **Явц 2**: main experiments reported |
| 10 | trace replicates + answer-only control; `wl_gather` resolved either way |
| 11 | research freeze |
| 12 | full draft to advisor; slides |
| 13 | pre-defense |
| 15 | every criticism answered in writing |
| 16 | defense |

## Slack and what gives first

The schedule is two weeks tighter than first drafted (the form's calendar started Sep 7).
Slack is now the T4 time-box and the writing overlap in weeks 7–10. If more is needed,
cut in this order: (1) `wl_gather`, (2) the Yehudai sanity row, (3) the depth-1 row.
Never cut: the shortcut fix, the overfit ladder, the answer-only control, the week-11
freeze.
