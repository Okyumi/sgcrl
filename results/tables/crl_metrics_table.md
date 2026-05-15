# Continual-RL metrics per cell

Source: ``results/data/processed/per_seed_per_task.csv``, re-pooled under the project's group-routing rule.

Group routing: {reset, persistent}^2 cells use ``for_real``; any-CKA cells pool ``real1`` and ``real2``.

Forward-transfer reference cell: ``actor=reset-critic=reset`` (per-task ``best_success`` of each cell minus reference's ``best_success`` on the same task & seed, averaged over k=1..9). The reference is sourced from its own routed group(s): ('for_real',).

BWT / forgetting is NOT computable from these runs because ``intra_eval_previous_tasks=False`` on every run; ``stability = end / best`` within a task is the closest in-data proxy.

| cell | avg_best | forward_transfer (vs ref) | stability | groups | n_seeds | n_pairs |
|---|---|---|---|---|---|---|
| actor=reset-critic=reset | 0.832 | +0.000 ± 0.000 (n=41) | 0.486 | for_real | 5 | 46 |
| actor=reset-critic=persistent | 0.774 | -0.066 ± 0.025 (n=41) | 0.450 | for_real | 5 | 46 |
| actor=persistent-critic=reset | 0.827 | -0.007 ± 0.012 (n=41) | 0.480 | for_real | 5 | 46 |
| actor=persistent-critic=persistent | 0.791 | -0.049 ± 0.028 (n=41) | 0.449 | for_real | 5 | 46 |
| actor=reset-critic=cka | 0.840 | +0.005 ± 0.014 (n=41) | 0.476 | for_real | 5 | 50 |
| actor=cka-critic=reset | 0.841 | +0.007 ± 0.009 (n=41) | 0.481 | real1+real2 | 5 | 48 |
| actor=persistent-critic=cka | 0.761 | -0.078 ± 0.026 (n=41) | 0.492 | for_real | 5 | 46 |
| actor=cka-critic=persistent | 0.797 | -0.041 ± 0.024 (n=34) | 0.488 | real1+real2 | 5 | 41 |
| actor=cka-critic=cka | 0.781 | -0.068 ± 0.029 (n=34) | 0.428 | real1+real2 | 5 | 39 |
| actor=reset-critic=decomposed | 0.873 | -- | 0.479 | c2_decomposed | 3 | 30 |
