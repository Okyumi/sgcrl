# Critic-gate implementation

For a sampled success state \((s,g)\), query the existing critic
\(f(s,a,g)=\varphi(s,a)^\top\psi(g)\). No extra head. The stored
successful action is the matching target only. Do not use it as a
probe.

## Action sensitivity \(\sigma_a\)

Keep \((s,g)\) fixed. Draw **8** actions uniformly from \([-1,1]\).
Score each one. \(\sigma_a\) is the standard deviation of those 8
scores.

\[
\sigma_a(s,g)
=
\mathrm{std}_{i=1}^{8}\,
f(s,a_i,g),
\qquad
a_i\sim\mathrm{Unif}[-1,1].
\]

## Local state sensitivity \(\sigma_s\)

Keep \(g\) fixed. Draw **one** policy action \(a_\pi\sim\pi(\cdot\mid s,g)\)
and reuse it for all 8 probes. Perturb only the state:

\[
\tilde s_i
=
s + \varepsilon\,\hat\sigma\odot z_i,
\qquad
z_i\sim\mathcal N(0,I),\quad \varepsilon=0.1.
\]

\(\hat\sigma\) is the per-coordinate standard deviation of states in
the current success minibatch. Treat it as a constant (no gradient).
Floor tiny coordinates so they are not zero. Score
\(f(\tilde s_i, a_\pi, g)\). \(\sigma_s\) is the standard deviation of
those 8 scores.

Do **not** take the std of scores across different states in the
batch. That is not local.

## Gate

\[
w(s,g)
=
\frac{\sigma_s}{\sigma_s+\sigma_a}.
\]

Stop gradients through \(w\). If both stds are ~0, set \(w=1\).

Multiply the success-matching term by \(w\):

\[
\lambda_{\mathrm{succ}}\,w(s,g)\,\log\pi(a\mid s,g_{\mathrm{task}}),
\qquad
\lambda_{\mathrm{succ}}=0.1.
\]

\(w\to 1\) when the critic is flat in action: match the successful
trajectory. \(w\to 0\) when the critic already ranks actions: leave
the contrastive actor term in charge.
