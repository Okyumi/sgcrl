# Per-cell summary (averaged across 10 tasks)

Source: ``results/data/processed/per_seed_per_task.csv``, re-pooled under the project's group-routing rule.

Group routing: {reset, persistent}^2 cells use ``for_real``; any-CKA cells pool ``real1`` and ``real2``.

Each column averages the per-task mean of the named metric across the 10 tasks. n_seeds is the median seed count over the 10 tasks; n_pairs is the total (task, seed) pair count for the cell.

| cell | avg_best | avg_end | avg_mean_during | avg_auc | stability | groups | n_seeds | n_pairs |
|---|---|---|---|---|---|---|---|---|
| actor=reset-critic=reset | 0.832 | 0.469 | 0.433 | 0.434 | 0.486 | for_real | 5 | 46 |
| actor=reset-critic=persistent | 0.774 | 0.435 | 0.391 | 0.392 | 0.450 | for_real | 5 | 46 |
| actor=persistent-critic=reset | 0.827 | 0.463 | 0.444 | 0.445 | 0.480 | for_real | 5 | 46 |
| actor=persistent-critic=persistent | 0.791 | 0.430 | 0.380 | 0.381 | 0.449 | for_real | 5 | 46 |
| actor=reset-critic=cka | 0.840 | 0.464 | 0.446 | 0.447 | 0.476 | for_real | 5 | 50 |
| actor=cka-critic=reset | 0.841 | 0.469 | 0.428 | 0.428 | 0.481 | real1+real2 | 5 | 48 |
| actor=persistent-critic=cka | 0.761 | 0.446 | 0.394 | 0.395 | 0.492 | for_real | 5 | 46 |
| actor=cka-critic=persistent | 0.797 | 0.458 | 0.374 | 0.374 | 0.488 | real1+real2 | 4 | 41 |
| actor=cka-critic=cka | 0.781 | 0.409 | 0.379 | 0.380 | 0.428 | real1+real2 | 4 | 39 |
