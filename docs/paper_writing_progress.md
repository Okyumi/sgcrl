# Paper Writing Progress

Progress log for the NeurIPS 2026 paper. The LaTeX source lives in the
Overleaf-synced GitHub repo
[Okyumi/NeurIPS-2026---RL](https://github.com/Okyumi/NeurIPS-2026---RL);
this file tracks what has been drafted, what has been pushed, and what
remains. The planning documents in this directory
(`paper_plan.md`, `paper_planning_tracking.md`, `citations.md`) are the
upstream source of truth for narrative, citations, and decisions.

## Repository layout (paper)

| Path | Purpose |
|---|---|
| `neurips_2026.tex` | Main LaTeX entry point. Loads `appendix.tex` via `\input` and `references.bib` via `\bibliography`. |
| `appendix.tex`     | Experimental Setup appendix (Environments, Architecture, Training, Baselines, Evaluation, Hyperparameters, Implementation Details). |
| `references.bib`   | BibTeX file for all citations. Mirrors `docs/citations.md`. |
| `checklist.tex`    | NeurIPS submission checklist (unchanged template). |
| `neurips_2026.sty` | NeurIPS 2026 style file (unchanged). |
| `main.tex`          | Overleaf build root. Replaces `neurips_2026.tex` as the entry point; `\input{}`s introduction, problem_setup, discussion, references, and appendix. |
| `introduction.tex`  | Section 1 (Introduction). Motivational paragraph on the continual-RL / sparse-reward gap. Method preview and contributions pending. |
| `problem_setup.tex` | Section 2 (Problem Setup). |
| `discussion.tex`    | Discussion and Future Directions (closing paragraph + bulleted future directions). Empirical take-aways pending. |

## Branch policy (paper repo)

- **All writing commits go directly to `main` as user `Okyumi`.**
- No feature branches, no pull requests. Overleaf is synced with `main`
  and expects changes there.
- This file (the progress tracker) lives on `section3_done` of the
  sgcrl repo, per the user's instruction.

## Current status (2026-04-22)

**Draft 1 of the appendix and bibliography pushed.**

### Done

- [x] `neurips_2026.tex` wired for a modular appendix (`\input{appendix}`)
  and a BibTeX bibliography (`\bibliography{references}`); `amsmath`,
  `multirow` added for equations and tables.
- [x] `problem_setup.tex` drafted as Section 2 (current at commit
  `0e73bd0` on paper `main`). Pure mathematical definition: sequence
  of $N$ episodic MDPs on shared state and action spaces, each with a
  goal $g^{(k)}$ and a success predicate, terminal sparse reward
  $r^{(k)}_{T} = \mathbf{1}_{\mathrm{success}}^{(k)}(s_{T})$ and
  $r^{(k)}_t = 0$ for $t < T$, fixed-order continual protocol with
  per-task transition budget $T_k$ and no re-interaction with earlier
  MDPs. Four short paragraphs with display equations for the MDP
  tuple, success predicate, and reward.
- [x] Goal-conditioned policy, expected-return objective, and
  benchmark numbers deliberately kept out of Section 2 and reserved
  for the solver/method section and the appendix respectively.
- [x] `appendix.tex` with full sections for
  - **Environments and Tasks**: Meta-World / Continual World V2 ten-task
    Sawyer sequence, sparse goal-reaching reward, observations, actions,
    episode length, wrappers and preprocessing.
  - **Model Architecture**: residual-MLP backbone (width 1024, depth 4,
    LayerNorm, Swish), dual-encoder contrastive critic, NormalTanh actor,
    policy decomposition with the softmax blend, bounded knowledge pool
    with cosine-similarity merging, negative-bank module.
  - **Evaluation Protocol**: frequency, episodes, seeds, metrics (success
    rate, return, forward transfer), averaging. Written under the
    assumption that backward transfer / forgetting is out of scope for
    the current draft.
  - **Hyperparameters**: single table covering architecture, optimisation,
    replay, SAC-style adaptive entropy, continual protocol, negative bank,
    evaluation.
  - **Implementation Details**: JAX / Haiku / Brax / JaxGCRL stack,
    NVIDIA A100 hardware, SLURM launcher, memory-fraction scheme,
    reproducibility and licensing notes.
- [x] `appendix.tex` subsections `\subsection{Training Details}` and
  `\subsection{Baselines}` deliberately left empty with inline LaTeX
  comments, per the user's instruction.
- [x] `references.bib` populated with all 32 entries cross-referenced with
  `docs/citations.md` (contrastive GCRL, continual RL, plasticity,
  representation drift, SAC, HER, Meta-World, JAX/Haiku/Brax).
- [x] Structural sanity check: braces balance in both `.tex` files, every
  `\cite{...}` key used in `appendix.tex` resolves to a `references.bib`
  entry.

### Not yet done

- [ ] Introduction: method preview paragraph and contributions list
  (waiting for experiment results to inform the wording).
- [ ] Related Work (likely Section 3, still pending).
- [ ] Method (likely Section 4).
- [ ] Experiments (likely Section 5).
- [ ] Discussion: empirical take-aways paragraph, to be written
  against the final numbers.
- [ ] Training Details subsection (explicit training-loop pseudocode,
  optimiser schedule, replay and target-update details, continual
  protocol transitions).
- [ ] Baselines subsection (sparse-SAC, dense-SAC, policy-decomp + SAC,
  their hyperparameters and matching conventions).
- [ ] Forgetting / backward-transfer evaluation protocol (deferred).
- [ ] Main body: introduction, problem setting, related work, method,
  experiments, discussion.
- [ ] Figures and tables for results.

## Commit log (paper repo `main`)

| Date | Commit | Files | Note |
|------|--------|-------|------|
| 2026-04-22 | `f388c66` | `neurips_2026.tex`, `appendix.tex`, `references.bib` | Initial appendix and BibTeX draft. Merged directly into `main` as user `Okyumi`. Feature branch removed. |
| 2026-04-22 | `a13c2c2` | `problem_setup.tex`, `neurips_2026.tex` | Draft of Section 2 (Problem Setup). Linear narrative from MDP tuple $\to$ continual protocol $\to$ return $\to$ goal-conditioning $\to$ sparse reward $\to$ benchmark instantiation. One idea per sentence. |
| 2026-04-22 | `f63ba4a` | `problem_setup.tex` | Rewrite of Section 2. Tightened to three short paragraphs + one equation. Reward is the **terminal 0/1 success indicator** at episode end (no dense shaping anywhere). Goal-conditioned policy and per-task expected-return objective removed from this section and reserved for the solver discussion. |
| 2026-04-22 | `0e73bd0` | `problem_setup.tex` | Refine Section 2: drop the sparse-reward prose sentence (equation is the definition), remove the benchmark-instantiation paragraph (belongs elsewhere), split into four short paragraphs with display equations for the MDP tuple, success predicate, and terminal reward so the page breathes. |
| 2026-04-22 | `c41c139`, `457af2e` | `main.tex` (new) | Overleaf-side edits: build root renamed to `main.tex`, placeholder template sections removed, small grammar fixes. Pushed by user directly from Overleaf. |
| 2026-04-22 | `e758796`, `1191077` | `introduction.tex`, `discussion.tex`, `main.tex` | Draft Introduction motivational paragraph (continual RL is almost always dense, sparse-reward RL is almost always single-task, the combination is what matches deployment) and Discussion + Future Directions (testbed framing, limitations, bulleted list of future directions: representation drift, scalability, offline-to-online reuse, longer sequences, fixed-goal and state-conditioned-bandit variants, extensions beyond RL). Wired into `main.tex` after the rebase against the Overleaf edits. |
| 2026-04-22 | `4bfb308` | `introduction.tex` | Rewrite Introduction to follow the user's preferred preliminary draft. Two short paragraphs: (1) gap statement (continual RL almost always dense; sparse-reward RL almost always single-task; we want an agent that learns continuously without dense engineering, seeing only the per-task goal; this is a gap), (2) natural solver class (self-supervised / goal-conditioned RL) with contrastive GCRL named as a proper candidate and two inline placeholders for the motivation and methodology content. |
| 2026-04-22 | `6332ec4` | `discussion.tex` | Tighten discussion. Drop the standalone limitations paragraph (content already conveyed by the framing paragraph and future-directions list). Remove the 'we list below open questions\ldots' filler. Trim each future-direction bullet to a single clear thought. |

## Conventions

- LaTeX: two-space indentation, hard-wrap around 75 columns, one idea per
  paragraph. Tables use `booktabs` + `multirow`.
- BibTeX: plainnat style, arXiv identifiers in `eprint`, venues in
  `booktitle` / `journal`. Keys match those used in `docs/citations.md`.
- Citation commands: `\citep{...}` for parenthetical, `\citet{...}` for
  textual. `natbib` is loaded by the NeurIPS 2026 style file by default.
- No defensive meta-commentary in the prose. When a framing is rejected, the rejected framing is silently dropped; only the positive claim survives. Full rule in `docs/paper_planning_tracking.md`.
- Golden rules for research writing (stored to agent memory on 2026-04-22): one clear idea per sentence; define terms precisely; remove anything that does not help the reader understand question, method, result, or claim; clarity before cleverness. One sharp idea per method; minimal but meaningful novelty; isolate the effect with strong ablations and fair, matched baselines; robustness across seeds, settings, and failure cases; insight into what property matters, when it helps, when it fails, what it teaches.
- Appendix-only content: anything that is "how we did it" rather than
  "what we found". Claims that are load-bearing for the headline results
  stay in the main body.

## Next actions

1. Draft Training Details and Baselines subsections once experiments are
   launched (these benefit from being written against actually-running
   configurations). Commit directly to `main`.
2. Begin drafting the Introduction and Problem Setting sections,
   mirroring the bulleted narrative in `docs/paper_plan.md`. Commit
   directly to `main`.
