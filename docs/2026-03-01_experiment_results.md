# Final evaluation success – all experiments

Success scores are taken from the **last row** of each run’s evaluator log (`logs/<run>/<uuid>/logs/evaluator/logs.csv`). The **final evaluation success** column uses the `success_1000` metric (rolling mean success over the last 1000 evaluation episodes) when available and in [0, 1]; otherwise the last episode’s `success` is used.

| Environment | Seed | Final evaluation success |
|-------------|------|--------------------------|
| sawyer_bin | 0 | 1.10% |
| sawyer_bin | 2 | 68.70% |
| sawyer_bin | 42 | 0.00% |
| sawyer_box | 2 | 29.30% |
| sawyer_faucet_close | 2 | 47.00% |
| sawyer_hammer | 2 | 78.70% |
| sawyer_handle_press_side | 2 | 0.30% |
| sawyer_peg_unplug_side | 2 | 0.60% |
| sawyer_push | 2 | 59.60% |
| sawyer_push_back | 1 | 0.20% |
| sawyer_push_back | 2 | 0.00% |
| sawyer_push_wall | 2 | 41.40% |
| sawyer_shelf_place | 2 | 0.00% |
| sawyer_stick_pull | 2 | 0.00% |
| sawyer_window_close | 2 | 2.70% |

To regenerate this table from the current logs, run from the repo root:

```bash
python3 scripts/get_final_eval_success.py
```
