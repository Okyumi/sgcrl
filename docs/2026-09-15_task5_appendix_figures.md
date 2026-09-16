# Task-5 appendix figures (state vs action, feature shuffle, inject)

Date: 2026-09-15
Status: rendered from existing seed-6 measurements. No new training.

## Motivation

Appendix figures for the Task-5 failure mechanism, using only already
measured probes (jobs `17901349` and `17901867`). No invented
uncertainty: one seed, point estimates, \(n\) reported in the CSV.

A \(\phi(s,a)\) vs \(\psi(g)\) embedding plot is omitted. Those
checkpoints do not store per-transition embeddings, and a decorative
t-SNE of newly scored rollouts would not add a quantitative claim
beyond the shuffle drops and \(\mathrm{std}_a/\mathrm{std}_s\).

## Figures

Okabe–Ito colors, same spine/grid language as
`scripts/plot_action_informative_mass.py`.

| File | Claim |
|---|---|
| `results/img/paper/fig_task5_state_vs_action.{pdf,png}` | At a fixed hover state, Task 5 does not prefer press; score variation is across states, not actions. Push does the opposite. |
| `results/img/paper/fig_task5_feature_shuffle.{pdf,png}` | InfoNCE retrieval uses mechanism \(xy\) (and hand, on push), not \(z\) and not \(a\). Handle \(xy\) is a fixture (\(\Delta xy=0\)). |
| `results/img/paper/fig_task5_success_inject.{pdf,png}` | Cloning success to 20% of replay at 250.5k raises HER success briefly; eval does not retain. |

CSV of plotted values: `results/data/task5_appendix/fig_task5_appendix.csv`.

## Sources (measured)

- Same-state advice: `logs/jubail_task5_action_advice/runs/action_advice_handle_press_side_s6_step100200.json`, `action_advice_push_s6_step250500.json`, handle-late dump in `17901349_0.out` (inject overwrote the late JSON).
- Feature shuffle / one-step \(\Delta\): `logs/jubail_task5_feature_shortcut/runs/feature_shortcut_*_s6_step{100200,250500}.json`.
- Inject time series: parsed from `17901349_{0_0,1_1,2_2}.out` (`[eval @ …]`, `[her phase @ …]`, post-task eval). Inject is a **separate** 1M cell, not a fork of handle_measure.

## Launch

```bash
python tests/test_task5_appendix_figs.py
python scripts/plot_task5_appendix_figs.py
```

## Captions (for later paper use)

**State vs action.** Same-state critic scores at hover under \(g_{\mathrm{task}}\)
(seed 6). **A.** Mean score of a press-biased action minus \(\pi\).
**B.** Rank of that action among \(\{\pi,\text{press},32\times\mathrm{Unif}\}\)
(1 = best; dashed = chance). **C.** Cosine of \(\nabla_a s\) at \(a_\pi\)
with the press/push axis. **D.** Score std. across hover states versus
across actions in one state (log scale). Handle: 113 / 32 hover audits
at 100k / 952k. Push: 320 audits at 250k.

**Feature shuffle.** **A.** Drop in 256-way HER categorical accuracy after
shuffling one block of \(s\) or \(a\). Chance is \(1/256\). **B.** Mean
absolute one-step change of mechanism \(xy\) and \(z\) from a frozen hover,
\(\pi\) vs press (mm). Handle \(\Delta xy=0\) for every action.

**Inject.** Separate 1M DCC cells, seed 6, no Success-BC. **A.** 10-episode
eval every 50k, plus post-task eval at 990k. Vertical line: clone 50,100
transitions (8 eval episodes \(\times\) 326) into replay, 20% of the
250.5k buffer. **B.** HER-goal EMA success mass (printed every 10
episodes). Handle has no progress band (`prog=0`). The 0.20 dotted line
is the inject target, not a measured occupancy after the clone.

## Limitations

- One seed.
- Handle-late JSON recovered from stdout.
- Inject cell’s pre-250k eval is not the same trajectory as handle_measure.
- No embedding figure (embeddings were not serialized).
