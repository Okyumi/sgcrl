# Per-task best mean success (± std across seeds)

Source: ``results/data/processed/per_seed_per_task.csv``, re-pooled here under the project's group-routing rule.

Group routing:
* GCRL baseline cells (actor and critic both in {reset, persistent}) -> ``for_real`` only.
* Any-CKA cells -> pooled across ``real1`` and ``real2``.

Metric: ``best_success`` = max of ``evaluator/success_rate`` over the per-step evaluation trajectory during training of that task. The cell value is the mean ± sample standard deviation across seeds (n shown in the bottom row).

| task_idx | env | actor=reset-critic=reset | actor=reset-critic=persistent | actor=persistent-critic=reset | actor=persistent-critic=persistent | actor=reset-critic=cka | actor=cka-critic=reset | actor=persistent-critic=cka | actor=cka-critic=persistent | actor=cka-critic=cka | actor=reset-critic=decomposed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | sawyer_hammer | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 1 | sawyer_push_wall | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 2 | sawyer_faucet_close | 0.920 ± 0.084 | 0.840 ± 0.055 | 0.860 ± 0.089 | 0.880 ± 0.084 | 0.840 ± 0.055 | 0.920 ± 0.045 | 0.820 ± 0.045 | 0.900 ± 0.082 | 0.900 ± 0.082 | 0.867 ± 0.058 |
| 3 | sawyer_push_back | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 4 | sawyer_stick_pull | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.980 ± 0.045 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.900 ± 0.141 | 1.000 ± 0.000 |
| 5 | sawyer_handle_press_side | 0.240 ± 0.089 | 0.120 ± 0.045 | 0.240 ± 0.055 | 0.200 ± 0.122 | 0.200 ± 0.000 | 0.260 ± 0.055 | 0.120 ± 0.045 | 0.200 ± 0.082 | 0.175 ± 0.050 | 0.400 ± 0.100 |
| 6 | sawyer_push | 0.980 ± 0.045 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.980 ± 0.045 | 1.000 ± 0.000 | 0.950 ± 0.100 | 1.000 ± 0.000 |
| 7 | sawyer_shelf_place | 0.980 ± 0.045 | 0.680 ± 0.319 | 0.940 ± 0.089 | 0.660 ± 0.385 | 0.980 ± 0.045 | 0.980 ± 0.045 | 0.660 ± 0.391 | 0.775 ± 0.320 | 0.750 ± 0.436 | 0.967 ± 0.058 |
| 8 | sawyer_window_close | 0.367 ± 0.058 | 0.433 ± 0.058 | 0.400 ± 0.100 | 0.367 ± 0.153 | 0.480 ± 0.110 | 0.400 ± 0.000 | 0.367 ± 0.153 | 0.375 ± 0.096 | 0.367 ± 0.058 | 0.633 ± 0.058 |
| 9 | sawyer_peg_unplug_side | 0.833 ± 0.058 | 0.667 ± 0.153 | 0.833 ± 0.058 | 0.800 ± 0.100 | 0.900 ± 0.071 | 0.867 ± 0.153 | 0.667 ± 0.058 | 0.725 ± 0.096 | 0.767 ± 0.058 | 0.867 ± 0.153 |
| **avg** | -- | **0.832** | **0.774** | **0.827** | **0.791** | **0.840** | **0.841** | **0.761** | **0.797** | **0.781** | **0.873** |
| _seeds (median)_ | -- | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 3 |
