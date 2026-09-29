# Critic-gate appendix for the ICLR draft

Date: 2026-09-25  
Status: drafted locally; not yet inserted into `Okyumi/DCC---ICLR2027`.

## Motivation

The main text defines the critic gate \(w=\sigma_s/(\sigma_s+\sigma_a)\)
and defers the sampling of \(\sigma_a\) and \(\sigma_s\) to
`Appendix~\ref{app:critic-gate}`, which currently renders as `??`.

## Placement

Insert as a new appendix **section after C and before D**.

Current order: A results, B algorithm, C EM view of the success
regularizer, D training details, E benchmark, F baselines, G
representation diagnostics, H action information. The gate is the
missing operational piece of C/Eqs. 7–8, not a hyperparameter of D.

If the paper uses sequential `\section{}` after `\appendix`, this
becomes the new Appendix D and later letters shift. Cross-refs via
`\ref` update; the main-text `\label{app:critic-gate}` resolves the
`??`. Do not put this at the end (after H) and do not bury it in D.3
Infrastructure.

## Source

Drop-in LaTeX: `docs/paper/appendix_critic_gate.tex`.

In the paper repo, either `\input{appendix_critic_gate}` immediately
after the EM appendix, or paste the file body at that point. The
file uses an unnumbered display so later appendix equation numbers
do not shift.

## What it specifies

Local \(\sigma_a\): eight actions uniform on the action box, \((s,g)\)
fixed. Local \(\sigma_s\): eight jitters
\(s+\varepsilon\hat\sigma\odot z\) with \(\varepsilon=0.1\), \(z\sim
\mathcal N(0,I)\), same \(g\) and \(a_\pi\sim\pi(\cdot\mid s,g)\).
\(\hat\sigma\) is the minibatch coordinate-wise state std, treated as
a constant. Gate is stop-gradient; defaults to 1 if both stds vanish.
Stored successful action is not a probe.

## Known limitations

Could not push to `Okyumi/DCC---ICLR2027` from this host: the HPC SSH
key is not on GitHub, and the PAT pasted in chat was rejected as
invalid. The Overleaf project will pick this up once the file is on
`main` of that repo.
