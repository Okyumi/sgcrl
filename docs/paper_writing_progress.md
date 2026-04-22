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

## Branch policy

- `main` on the paper repo is the live version synced with Overleaf.
- Work goes on a feature branch (`appendix-bibliography`,
  `experiments-draft`, ...), is reviewed, and is merged into `main`.
- This file lives on `section3_done` of the sgcrl repo, as requested.

## Current status (2026-04-22)

**Draft 1 of the appendix and bibliography pushed.**

### Done

- [x] `neurips_2026.tex` wired for a modular appendix (`\input{appendix}`)
  and a BibTeX bibliography (`\bibliography{references}`); `amsmath`,
  `multirow` added for equations and tables.
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

- [ ] Training Details subsection (explicit training-loop pseudocode,
  optimiser schedule, replay and target-update details, continual
  protocol transitions).
- [ ] Baselines subsection (sparse-SAC, dense-SAC, policy-decomp + SAC,
  their hyperparameters and matching conventions).
- [ ] Forgetting / backward-transfer evaluation protocol (deferred).
- [ ] Main body: introduction, problem setting, related work, method,
  experiments, discussion.
- [ ] Figures and tables for results.

## Files changed per push (running log)

| Date | Branch | Files | Note |
|------|--------|-------|------|
| 2026-04-22 | `appendix-bibliography` | `neurips_2026.tex`, `appendix.tex`, `references.bib` | Initial appendix and BibTeX draft. Pushed for review; not yet merged into `main`. |

## Conventions

- LaTeX: two-space indentation, hard-wrap around 75 columns, one idea per
  paragraph. Tables use `booktabs` + `multirow`.
- BibTeX: plainnat style, arXiv identifiers in `eprint`, venues in
  `booktitle` / `journal`. Keys match those used in `docs/citations.md`.
- Citation commands: `\citep{...}` for parenthetical, `\citet{...}` for
  textual. `natbib` is loaded by the NeurIPS 2026 style file by default.
- No defensive meta-commentary in the prose (see
  `docs/paper_planning_tracking.md`).
- Appendix-only content: anything that is "how we did it" rather than
  "what we found". Claims that are load-bearing for the headline results
  stay in the main body.

## Next actions

1. Open a pull request on the paper repo merging
   `appendix-bibliography` into `main` once the user reviews the draft.
2. Draft Training Details and Baselines subsections once experiments are
   launched (these benefit from being written against actually-running
   configurations).
3. Begin drafting the Introduction and Problem Setting sections in a new
   branch on the paper repo, mirroring the bulleted narrative in
   `docs/paper_plan.md`.
