# Task-5 appendix visuals (state similarity, hover advice, inject)

Date: 2026-09-25
Status: restyled from existing seed-6 measurements. No new training.
Paper text was not edited.

## Why these figures

The last appendix (`Task 5 as an example of sparse action information`)
already has a table. The previous bar charts duplicated that table.
These figures encode the same measured quantities in a form that is
easier to read next to the prose.

## Figures

| File | Appendix claim |
|---|---|
| `results/img/paper/fig_task5_state_similarity.{pdf,png}` (also `fig_task5_feature_shuffle.*`) | Matching uses state identity, not the press action. |
| `results/img/paper/fig_task5_state_vs_action.{pdf,png}` | At a frozen hover state, press is not preferred; score variation is across states. |
| `results/img/paper/fig_task5_success_inject.{pdf,png}` | Cloning success to 20% of replay does not retain Task 5. |

CSV: `results/data/task5_appendix/fig_task5_appendix.csv`.

## What is plotted (all measured)

**State similarity (A–C).** From
`feature_shortcut_*_s6_step{100200,250500}.json`, 256-way HER
categorical accuracy. Chance \(=1/256\).

- A. Drop after shuffling one block of \((s,a)\): mechanism \(xy\),
  \(z\), hand, action. Task 5: \(xy\) drop \(0.17\), action drop
  \(0.016\).
- B. Accuracy trajectory: intact \(\to\) shuffle \(xy\) \(\to\) \(z\)
  \(\to\) action. Both tasks collapse only when \(xy\) is shuffled.
- C. On-policy occupancy of the success band (`frac_in_band`) and, for
  Push, the progress band (`frac_in_progress_band`). Task 5: 3.3% in
  the \(z\)-band, no progress band. Push: 27% success + 44% progress.

**Hover advice (A–C).** From `action_advice_*` at handle 100k and
push 250k (the appendix table). Handle 952k is stored in the CSV and
not drawn, matching the table.

- A. Mean rank of the press/push-biased action among
  \(\{\pi,\text{press},32\times\mathrm{Unif}\}\).
- B. \(\cos(\nabla_a f, d_{\mathrm{press}})\) as a compass: press axis
  along \(+x\), gradient in the upper half-plane at
  \(\arccos(\cos)\).
- C. \(\mathrm{std}_s(f)\) vs \(\mathrm{std}_a(f)\) on a log axis, with
  the ratio annotated.

**Inject.** Parsed from `17901349_{0_0,1_1,2_2}.out`. Callouts 20% /
10% / 0% are the inject cell’s eval at 250.5k, 300.6k, and 990k.

## Captions (for the appendix, not pasted into the TeX yet)

**State similarity.** 256-way HER retrieval after shuffling one block of
the pair \((s,a)\), seed 6. **A.** Accuracy drop. **B.** Accuracy
itself; dashed line is chance \(1/256\). **C.** Fraction of on-policy
states in the environment success band and, for Push, the dense
progress band. Task 5 matching depends on fixture \(xy\) (episode
identity), which is shared by a state and its HER future; shuffling the
action barely matters.

**Hover advice.** Same hover state, commanded goal. **A.** Rank of the
task-progress action among 34 candidates. **B.** Local critic gradient
versus the press/push axis. **C.** Score variation across hover states
versus across actions in one state. Handle \(n=113\); Push \(n=320\).

**Inject.** Separate 1M DCC cells, seed 6, no Success-BC. Vertical line:
50,100 successful transitions cloned into a \(\approx\)250k buffer
(20% of replay). Eval on the inject cell goes \(20\%\to 10\%\to 0\%\);
HER success mass rises from \(\approx 0.03\) to \(0.15\) then decays.

## Launch

```bash
python tests/test_task5_appendix_figs.py
python scripts/plot_task5_appendix_figs.py
```

## Limitations

Same as `docs/2026-09-15_task5_appendix_figures.md`: one seed, on-policy
occupancy not the serialized Reverb buffer, inject cell is not a fork
of handle_measure. Compass plots \(\arccos(\cos)\) in the upper
half-plane; the sign of the out-of-plane component is not identified.
