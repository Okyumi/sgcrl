# Vendored SAC reference (not imported at runtime)

`jaxgcrl_sac.py` and `jaxgcrl_sac_networks.py` are third-party reference code,
kept verbatim so the port in `sac/learning.py` and `sac/networks.py` can be
diffed against its source. **Nothing in the repository imports them** — they
are documentation, not a dependency (they need `brax`/`flax`, which are not in
`requirements.txt`).

| File | Provenance | Licence |
|---|---|---|
| `jaxgcrl_sac.py` | [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) SAC agent, itself derived from `brax.training.agents.sac` | Apache-2.0, "Copyright 2024 The Brax Authors" (header intact) |
| `jaxgcrl_sac_networks.py` | JaxGCRL SAC networks (`make_sac_networks` / `make_q_network` / `make_policy_network`) | Apache-2.0, "Copyright 2024 The Brax Authors" (header intact) |

What the port keeps identical to these files:

* the three SAC losses (`alpha_loss`, `critic_loss`, `actor_loss`),
* the per-step update order α → critic → actor → soft target update, using the
  *pre-update* α and Q,
* twin scalar Q with `min(Q1, Q2)` bootstrapping,
* `target_entropy = -0.5 * |A|` with a learned α.

What the port deliberately changes (see the module docstrings for the
reasoning): the residual body from `contrastive.networks.ResidualMLP` instead of
a flat MLP, the `[state, goal]` HER observation, the reverb/`tf.data` pipeline
instead of brax's vectorised rollouts, and the CKA-RL actor decomposition.
