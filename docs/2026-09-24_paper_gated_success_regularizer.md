# Paper briefing: critic-gated success matching

For the model rewriting the algorithm section of the current draft.
Do **not** call this behaviour cloning. The old Eq. 7 term is *success
matching*: maximize \(\log\pi(a\mid s,g_{\mathrm{task}})\) on
trajectories that already satisfied the env success predicate.
The new paper variant is that same term with a **critic gate**.

## What to change

The previous draft has two actor cells: plain DCC, and DCC plus ungated
success matching (Eq. 7, \(\lambda_{\mathrm{succ}}=0.1\)). Keep both.
Add a third cell as the proposed method:

**DCC + critic-gated success matching.**

Ungated matching retains latch tasks (handle / window) and can stall
path tasks (stick-pull / shelf-place) by locking the first lucky
episode. The gate is the one regularizer for both families. Do not
introduce task lists, hover radii, or other reverse-engineered
geometry.

## Success buffer (unchanged)

The only extra signal is the env 0/1 success predicate already in
Section 2. After each replay episode \(\tau\),

\[
Y(\tau)=\mathbf{1}\bigl[\exists_t\, r_t=1\bigr].
\]

If \(Y(\tau)=1\), **every** transition of \(\tau\) is written into a
ring \(\mathcal{D}_{\mathrm{succ}}\) (capacity 4096) as
\((s,a,g_{\mathrm{task}})\). If \(Y(\tau)=0\), nothing is written.
Eval is unchanged (any in-episode hit still counts). Do not store only
the \(r_t=1\) frames: that ring is linger/hold and does not retain
latch tasks.

The stored action \(a\) is a **matching target only**. It is never
used as a ranking label or as an extra critic input.

## Ungated actor (old Eq. 7)

\[
J_{\pi}
=
J_{\mathrm{CRL}}(\pi)
+
\lambda_{\mathrm{succ}}
\,\mathbf{1}\bigl[|\mathcal{D}_{\mathrm{succ}}|>0\bigr]
\,\mathbb{E}_{(s,a)\sim\mathcal{D}_{\mathrm{succ}}}
\bigl[\log\pi(a\mid s,g_{\mathrm{task}})\bigr],
\]

with \(\lambda_{\mathrm{succ}}=0.1\). Plain DCC is
\(\lambda_{\mathrm{succ}}=0\).

## Critic gate (new)

At a sampled buffer state \((s,g)\), query the **existing** DCC score
\(f(s,a,g)=\varphi(s,a)^\top\psi(g)\). No extra head.

- **Action sensitivity** \(\sigma_a(s,g)\): std of \(f(s,a',g)\) over
  8 actions \(a'\sim\mathrm{Unif}[-1,1]\).
- **Local state sensitivity** \(\sigma_s(s,g)\): std of
  \(f(s+\varepsilon\hat\sigma\odot z,\,a_\pi,\,g)\) over 8 draws
  \(z\sim\mathcal{N}(0,I)\). Here \(a_\pi\sim\pi(\cdot\mid s,g)\),
  \(\varepsilon=0.1\), and \(\hat\sigma\) is the per-coordinate std of
  \(s\) in the current buffer minibatch (scale only; stop-gradient).

\[
w(s,g)
=
\frac{\sigma_s}{\sigma_s+\sigma_a}
\in[0,1],
\qquad
w \text{ is stop-gradient}.
\]

\(w\to 1\) when the critic is **flat in action** (cannot rank \(a\) at
this \(s\)): match the successful trajectory. \(w\to 0\) when the
critic already ranks actions: leave the DCC actor term in charge.

Gated objective:

\[
J_{\pi}
=
J_{\mathrm{CRL}}(\pi)
+
\lambda_{\mathrm{succ}}
\,\mathbf{1}\bigl[|\mathcal{D}_{\mathrm{succ}}|>0\bigr]
\,\mathbb{E}_{(s,a)\sim\mathcal{D}_{\mathrm{succ}}}
\bigl[w(s,g)\,\log\pi(a\mid s,g_{\mathrm{task}})\bigr].
\]

Same \(\lambda_{\mathrm{succ}}=0.1\). The gate is computed from critic
queries at the observed \((s,g)\) only. Nothing hindsight-ranked,
nothing from eval, nothing task-specific.

## Suggested names in the paper

| Cell | Name | \(\lambda_{\mathrm{succ}}\) | gate |
|---|---|---|---|
| DCC | decomposed contrastive critic, reset actor | 0 | — |
| DCC+SM | success matching | 0.1 | \(w\equiv 1\) |
| DCC+GSM | **gated success matching** (proposed) | 0.1 | local \(w(s,g)\) |

Prefer **gated success matching** (GSM) or **critic-gated success
matching**. If you use “predicate-gated”, make clear that the
*predicate* for the buffer is the env success bit, and the *gate* is
the critic weight \(w\). Never write “behaviour cloning”, “BC”, or
“imitation from an expert”.

## What the other model must not invent

- Do not gate on \(f(s,a_{\mathrm{succ}},g)\) vs \(f(s,a_\pi,g)\).
  \(a_{\mathrm{succ}}\) is hindsight; it is not available as a ranking
  oracle while interacting.
- Do not use batch \(\sigma_s\) (std of scores across the minibatch).
  That mostly measures \(s\)-\(g\) match. The paper method is **local**
  jitter of the same \(s\).
- Do not switch the buffer rule per task. One algorithm for all tasks.
