# 2026-11-13: Jubail vanilla CRL smoke test tolerance

## Cause and motivation

Smoke job 19339713 authenticated to W&B successfully, then failed
test_gate_extremes_and_stopped_gradient before learner integration or Sawyer
training. The state-only critic test expected gate=1 with atol=1e-6, but the
reported float32 maximum absolute difference was 2.7418137e-6. Production was
blocked by the smoke dependency. No training metrics were expected in W&B.

Float32 standard-deviation reductions on identical probe scores can leave a
tiny nonzero result, yielding a gate slightly below one. The reported values
are consistent with numerical roundoff, not an appreciable gate error.

## Exact change

Only tests/test_vanilla_success_bc.py changes:
- Gate extreme comparison uses explicit rtol=0 and atol=1e-5.
- Gate values must still be finite and inside [0,1].
- The stopped-gradient test still requires exactly zero.
- All other numerical, configuration and integration checks remain enabled.

No training objective, network, success labeling, hyperparameter, seed,
checkpoint or launcher behavior changes. The objective and metrics remain as
documented in docs/2026-11-10_vanilla_crl_gate_bc.md.

## Validation and limitation

Reviewed the test and gate implementation at commit 05195ea. The observed
maximum error is below the new absolute tolerance, which is 0.001 percentage
points on the gate scale. Static checks confirmed the intended single assertion
was replaced and the exact gradient assertion was retained. The Python execution
service was unavailable in this session, so no fresh JAX test pass is claimed.
The resubmitted smoke job reruns all checks in the actual Jubail environment.

## Relaunch

The user reported an empty squeue after the failed smoke job.

```bash
cd /scratch/yd2247/sgcrl
git pull --ff-only origin section3_done &&
bash scripts/submit_vanilla_crl_gate_bc.sh
```

This submits a new smoke job and a production array dependent on its success.
Use the newly printed job IDs to inspect logs. Existing W&B credentials worked;
no new login or API key is needed for this failure.
