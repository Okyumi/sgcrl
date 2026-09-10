# Results: actor-retention v3 + HER-phase geometry

Date: 2026-09-08
Status: analyzed (peg cells still finishing)

Sources: W&B groups `TASK58-ACTOR-RETENTION-1M-V3`,
`TASK58-HER-PHASE-GEOMETRY-1M`; stdout probes / `[her phase]` lines
(probe and HER-phase scalars often missing from W&B history).

## Actor-retention v3 (handle-press, 1M)

| Variant | peak μ | late μ | final μ |
|---|---:|---:|---:|
| dcc_control | 0.47 | 0.08 | 0.03 |
| actor_goal_task | 0.43 | 0.03 | 0.07 |
| actor_goal_mix | 0.40 | 0.03 | 0.00 |
| actor_success_score | 0.40 | 0.02 | 0.03 |
| success_bc_terminal | **0.93** | **0.58** | **0.63** |

## HER-phase geometry (selected)

At eval peak vs late, HER future fractions (stdout EMA):

| Task | peak eval | HER succ@peak | hover@peak | succ\|prog@peak | late eval | succ\|prog@late |
|---|---:|---:|---:|---:|---:|---:|
| push s5 | 1.00 | 0.34 | 0.09 | 0.61 | 0.10 | 0.72 |
| push s6 | 1.00 | 0.26 | 0.10 | 0.58 | 1.00 | 0.71 |
| push s7 | 0.80 | 0.16 | 0.09 | 0.57 | 0.60 | 0.62 |
| handle s5 | 0.40 | 0.03 | 0.13 | 0.03 | 0.40 | 0.10 |
| handle s6 | 0.40 | 0.07 | 0.13 | 0.07 | 0.00 | 0.05 |
| handle s7 | 0.40 | 0.03 | 0.08 | 0.03 | 0.00 | 0.04 |

Handle probe gaps under \(g_{\text{task}}\) at peak are typically **positive**
(success ≫ hover) with high `psi_cos` (~0.9).

## Verdict table

See chat summary for conjecture-by-conjecture status.
