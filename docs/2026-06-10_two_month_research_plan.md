# Two-month research plan for DCC main-conference push

Date: 2026-06-10
Author: Okyumi
Horizon: 2026-06-10 → 2026-08-10 (≈ 9 weeks)

## 1. Where we stand today

What is already in the bag:

- **Sawyer / sgcrl workshop result.** 10-task Continual World V2 sequence,
  3 seeds. Headline: avg best-during-training success 0.87 for DCC vs
  the best non-decomposed baseline at 0.84 (R/C and C/R), with +0.16 on
  handle-press-side (k=5) and +0.27 on window-close (k=8). Full
  table + per-task line plot + bar chart in the appendix of the RLC
  2026 workshop paper.
- **Representation diagnostics on Sawyer.** Actor feature rank, weight
  norm, NRC1/NRC2 on critic, entropy across the curriculum. The story
  is: persistent critic's weight norm climbs monotonically (~490 → ~3830),
  rank crashes on the hard tasks for every baseline, NRC1 collapses
  hardest on persistent + CKA-mix.
- **Code:** sgcrl (`section3_done` + `clean`) has the decomposed critic,
  CKA grid, dynamic loss, ablation flags `--combine_mode` /
  `--goal_encoder_mode`, and the rl-metrics shim that now logs the
  full metric suite under DCC.
- **BuilderBench port (yesterday).** `continual_crl_dcc.py` driver,
  ablation grid (9 ablations × 3 seeds = 27 configs), SLURM launcher.
  Smoke-tested only.
- **OGBench (RA-owned).** Independent track; we coordinate but do not
  block on it.

## 2. The story we want to tell at main-conference level

One main claim and three supporting claims.

Main claim. A decomposed contrastive critic — shared body + per-task
encoder + a dynamics auxiliary on the shared body — recovers the
performance of an oracle "reset" baseline on hard tasks while keeping
the cross-task transfer of a persistent critic on easy tasks. This is
the per-task trade-off the workshop paper already shows on Sawyer; for
a main conference it must hold across at least two benchmarks and
survive ablations.

Supporting claims:

1. **Generality.** The same gains appear on a second manipulation
   benchmark (BuilderBench), without changing the algorithm.
2. **Where the gains come from.** The dynamic loss matters, the
   additive combination beats concat, and the shared goal encoder is
   the right default; we can show this with ablations.
3. **Representation health.** DCC keeps feature rank up, weight norm
   bounded, and NRC1 from collapsing — the persistent baseline doesn't.
   These metrics co-vary with the success rates in a way that explains
   the per-task trade-off mechanistically.

If we can land these four claims with 3 seeds each on two benchmarks,
the paper is at a main-conference bar. Anything beyond that (OGBench,
plasticity stress test, order robustness) is icing.

## 3. Budget

Assumptions taken from the BuilderBench launcher:

- 2 jobs per A100-80G GPU (`TASKS_PER_GPU=2`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`).
- Sawyer single run ≈ 30–36h on one GPU at 8e6 steps × 10 tasks.
- BuilderBench single run ≈ 36–48h on one GPU at 50e6 steps × 12 tasks.
- Assume ~8 GPUs available steadily, occasional bursts to 12.

Two-month gross budget: 8 GPUs × 60 days × 24h ≈ 11,500 GPU-hours.
After overhead (queue waits, failed jobs, eval re-runs, OGBench
parallel track, 30% overhead total), call it 8000 effective GPU-hours.

That's roughly:

- BuilderBench full ablation grid, 3 seeds, 9 configs ≈ 1300 GPU-h.
- Sawyer follow-up sweeps, 3 seeds, 12 configs ≈ 1100 GPU-h.
- Plasticity stress test (Sawyer extended to 20 tasks), 3 seeds, 4 configs ≈ 1200 GPU-h.
- Order-robustness scan (BuilderBench, 3 shuffles × 2 variants × 3 seeds) ≈ 900 GPU-h.
- OGBench (RA), independent ≈ 0 from this budget.
- Headroom for re-runs and exploratory dev: ~3500 GPU-h.

We are not GPU-bound. We are time-bound on the human side: data
analysis, paper writing, and figure iteration eat the back half of the
plan.

## 4. Priorities, by week

Phasing notes:

- Week 1 protects against late-stage rework: nail the experiment
  protocol so every later run is comparable.
- Weeks 2–4 are the meat: BuilderBench headline + dynamic-loss + combine-mode.
- Weeks 5–6 are deepening: goal-encoder scan + representation analyses.
- Weeks 7–8 are paper writing + risk hedges; week 9 is reserved buffer.

### Week 1 (2026-06-10 → 2026-06-16) — protocol + smoke

Goal: lock the experimental protocol before launching the expensive sweep.

- [ ] Re-run **one Sawyer DCC seed end-to-end** with the just-fixed
  rl_metrics path to confirm we get the full metric suite this time.
  This is a 30h job; start it Tuesday so it lands by Friday.
- [ ] Run **one BuilderBench DCC smoke** (1M steps × 2 tasks, 1 seed)
  through the new `continual_crl_dcc.py` driver to confirm: it survives
  the task boundary, dyn loss numbers look sane, wandb logs come
  through, checkpoints reload after preemption.
- [ ] **Lock the BuilderBench evaluation protocol.** Decide once: which
  metric is the headline (avg best-during-training success vs.
  end-of-training-on-task), how forgetting is computed, how many eval
  rollouts, whether the per-cube-success breakdown goes in the paper.
- [ ] Set up wandb groups: `dcc_main`, `dcc_ablations_combine`,
  `dcc_ablations_dyn`, `dcc_ablations_goal`, `dcc_plasticity`,
  `dcc_order_robust`. Standardise tag conventions.

Deliverable end of week: a one-pager `docs/2026-06-15_protocol.md`
locking metrics, seed count, and figure layout. Once written, do not
modify until results force a revision.

### Week 2 (2026-06-17 → 2026-06-23) — BuilderBench headline launch

Goal: get the headline number running on BuilderBench.

- [ ] Launch **headline runs** on BuilderBench: 3 seeds × 3 variants
  (baseline_persistent, baseline_reset, dcc_add_shared). 9 jobs ≈ 4
  GPU-days each at 2 jobs/GPU ≈ done in 2 calendar days.
- [ ] In parallel: **dynamic-loss ablation**. 3 seeds × 1 variant
  (dcc_no_dyn) on BuilderBench. 3 jobs, parallel to above.
- [ ] **Cross-validate** Sawyer DCC numbers from week-1 metric-fixed run
  against the workshop submission; if any diverge by > 0.02 in avg
  success, flag and investigate before week 3.

Deliverable: BuilderBench preliminary headline plot (mean ± stderr
across 3 seeds, 12 tasks).

### Week 3 (2026-06-24 → 2026-06-30) — combine-mode + first analysis

Goal: settle combine_mode and start representation analysis.

- [ ] Launch **combine-mode ablation** on BuilderBench: dcc_concat_shared,
  3 seeds, 3 jobs.
- [ ] Mirror combine-mode on Sawyer: 3 seeds, 1 variant (combine=concat
  with shared goal). Comparable run length to the existing 0.87 number.
- [ ] **Representation analysis** of the headline runs: rank trajectories,
  weight-norm trajectories, NRC1/NRC2 trajectories on BuilderBench. Reuse
  the sgcrl analysis scripts (`results/scripts/render_appendix_figs.py`)
  and adapt to BuilderBench's wandb tags.
- [ ] **Snapshot the figures.** Even if numbers will move, lock in
  figure templates: 1 per-task bar, 1 trajectories, 1 representation
  panel per benchmark. We will only update data into these templates.

Deliverable: draft Figure 1 (per-task bar across both benchmarks) and
draft Figure 4 (representation panel for BuilderBench).

### Week 4 (2026-07-01 → 2026-07-07) — goal-encoder scan

Goal: pick the right goal-encoder default.

- [ ] Launch **goal-encoder ablation** on BuilderBench: dcc_goal_{task,
  partial, decomposed, projected}. 4 variants × 3 seeds = 12 jobs.
  ~2 weeks of GPU at 2 jobs/GPU on 4 GPUs.
- [ ] **First write-up pass.** Draft the experiments section as
  bullet points (numbers + claims), no figures yet. This catches
  storytelling gaps before they cost a re-run.

Deliverable: draft of §5 Experiments (text only) with concrete numbers
filled in where available, placeholders where not.

### Week 5 (2026-07-08 → 2026-07-14) — analysis week

Goal: synthesise.

- [ ] **Synthesise** combine-mode + dynamic-loss + goal-encoder
  results. Decide whether `dcc_add_shared` is the canonical config or
  whether a different variant wins on BuilderBench.
- [ ] **Cross-benchmark representation analysis.** Are the rank /
  weight-norm / NRC1 trajectories qualitatively the same on both
  benchmarks? If yes, that is the strongest evidence the mechanism is
  general.
- [ ] **First full-paper draft.** Method section is mostly written
  already; experiments section moves from bullets to prose. Submit to
  Yann + RA for review by Friday.

Deliverable: full-paper v1.

### Week 6 (2026-07-15 → 2026-07-21) — plasticity + order robustness

Goal: cheap-but-convincing extras.

- [ ] **Plasticity stress test.** Sawyer 20-task curriculum (repeat the
  10-task sequence twice). 3 seeds × 3 variants (dcc_add_shared,
  baseline_persistent, baseline_reset). This is the strongest possible
  test of the "DCC keeps the network plastic past 10 tasks" claim and
  is unique to our setup. ~9 jobs at ~60h each.
- [ ] **Order robustness on BuilderBench.** 3 shuffled task orderings
  × 2 variants × 3 seeds = 18 jobs. Risks: variance might swamp the
  signal at 3 seeds; if so, fall back to 2 orderings × 5 seeds.

Deliverable: appendix-quality plasticity figure + per-order success
plot.

### Week 7 (2026-07-22 → 2026-07-28) — figure polish + paper draft v2

- [ ] **Lock figures.** No new figure templates; only data updates.
- [ ] **Paper draft v2.** Address week-5 review comments. Tighten the
  related-work section using the corrected bib (see
  `2026-05-26_bib_fixes`).
- [ ] **Internal review with Yann.** Schedule a 1-hour session by
  Friday.

Deliverable: full-paper v2.

### Week 8 (2026-07-29 → 2026-08-04) — risk hedges + RA coordination

- [ ] **Integrate OGBench results from RA.** If they are ready, include
  as an appendix section. If they are partial, mention as future work.
  Either way, do not block the paper on it.
- [ ] **Re-run any flagged seed** that came in below expectation in
  weeks 2–4. Use the existing checkpoints; do not start from scratch
  if avoidable.
- [ ] **Author-list, acknowledgements, supplementary material.**

Deliverable: paper v3 (submission-ready candidate).

### Week 9 (2026-08-05 → 2026-08-10) — buffer

This week is intentionally unscheduled. It absorbs the inevitable
overrun. If it is empty, use it for: extra seeds on the headline, a
final pass on figures, or an appendix experiment we punted.

## 5. Concrete experiment table

Notation: `name (benchmark, variants, seeds) — purpose`.

| #  | Name                       | Bench        | Variants                                         | Seeds | Priority | Est. GPU-h |
|----|----------------------------|--------------|--------------------------------------------------|-------|----------|------------|
| 1  | Headline                   | BB           | persistent, reset, dcc_add_shared                | 3     | P0       | 350        |
| 2  | Dynamic loss               | BB + Sawyer  | dcc_no_dyn (vs dcc_add_shared from #1)           | 3     | P0       | 240        |
| 3  | Combine mode               | BB + Sawyer  | dcc_concat_shared                                | 3     | P0       | 240        |
| 4  | Goal-encoder scan          | BB           | task, partial, decomposed, projected             | 3     | P1       | 480        |
| 5  | Sawyer metric re-run       | Sawyer       | dcc_add_shared with fixed rl_metrics             | 3     | P0       | 90         |
| 6  | Plasticity stress test     | Sawyer (20)  | dcc_add_shared, persistent, reset                | 3     | P1       | 540        |
| 7  | Order robustness           | BB           | dcc_add_shared, persistent × 3 shuffles          | 3     | P2       | 480        |
| 8  | phi_task size              | BB           | width × depth grid {64, 256} × {2, 8}            | 1     | P2       | 160        |
| 9  | Wall-clock & memory        | BB + Sawyer  | all DCC variants                                 | n/a   | P2       | 0 (instrument) |
| 10 | OGBench (RA-owned)         | OG           | DCC vs CRL                                       | 3     | P1*      | 0 (RA)     |

Total estimated GPU-h on our side: ~2600 (out of 8000 effective). The
slack absorbs failed jobs, exploratory dev, and one full re-launch of
the headline if something subtle is off (e.g., a metric-logging bug
discovered late).

\* OGBench priority is high but the work is RA-owned and on a parallel
track; we do not depend on it for the main result.

## 6. Risks and fallbacks

| Risk                                                | Likelihood | Mitigation / fallback |
|------|------|------|
| BuilderBench DCC does not beat persistent baseline   | M | If gain < 0.02 avg, lead with combined-benchmark story instead of cross-benchmark; emphasize sgcrl numbers; investigate whether the dyn target (cube positions) is appropriate. |
| Goal-encoder scan is inconclusive (variance > effect)| M | Drop from main body, keep in appendix as "we tried these". This is what most ablation tables look like anyway. |
| Plasticity stress test does not show DCC advantage at 20 tasks | L | Demote from main story to appendix; reframe as "performance degrades gracefully" rather than "preserves plasticity". |
| Order robustness shows DCC sensitive to ordering     | L | Keep result honest; report mean ± stderr across orderings, frame as "robust to ordering within stderr". |
| RA's OGBench result is not ready by week 8           | M | Do not block. Mention as future work. |
| HPC outage / quota cut                               | L | Buffer week 9 + 30% overhead in budget. Smaller seeds (2 instead of 3) on lowest-priority experiments. |
| Reviewer asks for SAC or Dreamer baseline on BuilderBench | M | We already have SAC R/R and P/P on Sawyer. If asked, run Brax SAC on BuilderBench as a 2-week followup. |
| Subtle bug found in dyn loss late in writing         | L | The plasticity + order-robustness experiments give us a check on this; if both show no DCC benefit, that's a real signal of a bug. |
| Concat combine accidentally beats additive           | M | Genuinely useful finding: lead with whichever wins; either way the decomposition story holds. Have the figure template ready for both variants. |

## 7. What we are NOT doing

Discipline list. These are tempting but cut to keep the plan
executable:

- No new benchmarks beyond BuilderBench + Sawyer + (RA's) OGBench. No
  Meta-World ML10 from scratch, no D4RL, no Atari.
- No new auxiliary losses beyond `L_dyn`. The CKA-style mixture
  baselines stay as the comparison cell, not as a new method.
- No replay strategies (HER variants, prioritised sampling). Tempting,
  off-scope.
- No theoretical analysis section. We may add a one-page sketch in the
  appendix if natural, but we do not budget time for it.
- No exhaustive seed counts (5+). 3 seeds across the board, with the
  option to add 2 more on a single P0 cell if a reviewer or rebuttal
  demands it.

## 8. Decision points

These are the moments where we stop and decide instead of pushing
through:

- **End of week 2.** Does the BuilderBench headline show a clear gain?
  If yes, full speed ahead. If no, week 3 is spent debugging instead
  of running the combine-mode ablation, and the timeline shifts by 1
  week.
- **End of week 4.** Is the experiments section coherent in bullet
  form? If we can not list the claims and the supporting numbers
  cleanly, more experiments will not help; we need to think harder.
- **End of week 6.** Is the v1 draft good enough to send to Yann? If
  not, week 7 is spent rewriting instead of running plasticity + order
  experiments.

## 9. Tracking

- All experiment commits go to `Okyumi/sgcrl` (`section3_done` + `clean`)
  or `Okyumi/builderbench` (`main`) per the workflow rules.
- Documentation lives in `docs/2026-MM-DD_<topic>.md` files on
  `section3_done`. The implementation-tracking doc gets a short entry
  per commit cluster.
- Wandb groups follow the `dcc_<scope>` naming above; one Slack
  notification to Yann at the end of each week with a one-sentence
  status.
