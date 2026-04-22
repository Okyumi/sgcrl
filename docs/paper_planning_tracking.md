# Paper Planning — Progress Tracker

Working document for the NeurIPS 2026 submission planning process. Captures decisions, open threads, and evidence gathered in support of the paper plan in `docs/paper_plan.md`.

## Current state

- `docs/paper_plan.md` — narrative outline in bulleted story form, matching the register of the OT-reward-paper outline the user shared.
- `docs/citations.md` — verified citation list with annotations.
- `docs/algorithm_pseudocode.md` — reference pseudocode for the training loop.
- `docs/negative_bank.md` — design note for the offline-to-online negative bank.
- `docs/section3_continual_rl.md` — implementation specification for the `section3_done` branch.
- Branch of record: `section3_done`. Last commit pushed: `11fb574`.

## Decisions locked in

- The paper is about sparse-reward continual goal-conditioned RL. This setting is a first-class contribution.
- Framework language: **contrastive GCRL + knowledge decomposition (policy decomposition + knowledge pool)** in a sparse-reward continual setting.
- "CKA-RL" is treated as a convenient shorthand for policy decomposition and a knowledge pool applied to the actor network, and is cited as prior work studying those ideas (Hu et al., NeurIPS 2025). Any statement about the paper's relation to CKA-RL stays out of the writing; this is an internal working note only.
- Contrastive GCRL is the underlying RL algorithm, not a contribution. No InfoNCE exposition in the main body.
- Actor auto-reset is disabled by default.
- Adaptive entropy is enabled (SAC dual gradient, `target_entropy=-2.0`).
- 9-cell ablation: actor $\in \{\text{reset},\text{persistent},\text{decomposed}\}\times$ critic $\in \{\text{reset},\text{persistent},\text{decomposed}\}$, 5 seeds each, 10 tasks, 8M env steps per task.
- Future-work section will explicitly acknowledge that future methods — RL or otherwise — can likely achieve better results on this setting than ours, and will invite the community to expand research on it.

## Outline register (style reference)

- Bulleted story: one claim per bullet, plain language, no in-narrative sub-headers.
- Inline math where needed (e.g. $r_g$, $\theta'$), no proof-style derivations.
- **No padding. Rule of thumb.**
  - When asked to write part of the paper, do not add unnecessary sentences just to make the piece look long. Two sentences can beat a paragraph. Length is never the target; narrative fit is.
  - Before writing any paper piece, recall the golden rules above: one clear idea per sentence, remove anything that does not serve the reader, clarity before cleverness, sentences must flow organically.
  - The user's own preliminary drafts tend to be short, direct, and organic. My writing should read the same way.
- **No defensive meta-commentary. Rule number one.**
  - The rule is narrow: once the user has told me to drop a framing, I must not then produce a sentence in the writing that explicitly rejects that framing (e.g., "we do not frame this as an extension of CKA-RL" after being told not to frame as CKA-RL). Such a sentence is a facade that demonstrates compliance while naming the rejected framing, and it reinforces the rejected image more strongly than the intended one.
  - The correct response is to silently drop the rejected framing and write the most natural positive claim for the narrative flow, without mentioning the rejected framing at all.
  - Sentences that begin with "this is not X" for legitimate contrastive reasons — where no prior instruction has ruled out X and the negation genuinely serves the reader — remain acceptable. The rule is *not* a blanket ban on "this is not X" phrasing.
  - Avoid hedges whose only purpose is to demonstrate compliance with a constraint, such as "to the best of our knowledge".
  - This rule applies to every artifact: planning docs, bullet points, paper prose, code comments, commit messages.
- Logistics, schedule, and open questions sit below the narrative as short reference sections.

## Evidence gathered on novelty of the setting

Searches run (Apr 21, 2026) for: "sparse reward continual reinforcement learning benchmark", "goal-conditioned continual reinforcement learning benchmark", "continual reinforcement learning without dense reward".

Findings:

- [Wołczyk et al., NeurIPS 2021] Continual World uses the Meta-World V2 **dense** reward functions on the 10-task Sawyer sequence.
- [Hu et al., NeurIPS 2025] CKA-RL benchmarks on Meta-World, SpaceInvaders, and Freeway all use dense rewards.
- [Yu et al., CoRL 2020] Meta-World ships two dense-reward versions, V1 and V2. The benchmark is defined around the dense rewards; sparse success is reported only as a binary evaluation metric.
- [Sparse Meta-World / LatCo, ICML 2021] A visual sparse-reward variant of Meta-World exists (7 tasks). It targets single-task model-based RL, not continual RL.
- [Ni et al., NeurIPS 2021] Hindsight relabelling for sparse-reward meta-RL — meta-RL, not continual RL.
- [Hu et al., 2025] DISCOVER — directed sparse-reward goal-conditioned long-horizon RL. Single task, not continual.
- [Hu et al., NeurIPS 2024] GCPO — goal-conditioned policy optimisation, single task.
- [Park et al., ICLR 2025] OGBench — offline goal-conditioned RL benchmark, not continual.

**Conclusion.** No prior work, to the best of current search, proposes a continual-learning benchmark specifically for sparse-reward, goal-conditioned continual RL on a multi-task manipulation sequence. Existing continual-RL benchmarks use dense rewards; existing sparse-reward GCRL work is single-task or meta-RL. The paper can therefore introduce **sparse-reward continual goal-conditioned RL** as a setting in its own right, explicitly noting this gap in the literature.

## Evidence gathered on representation drift / encoder plasticity in continual RL

Searches run (Apr 21, 2026) for: "representation drift continual reinforcement learning", "encoder plasticity continual RL empirical", "feature representation forgetting continual RL".

Relevant prior work to cite when the paper discusses how $\phi, \psi$ evolve across a task sequence:

- **[Zhang, Dou, Wu, 2022]** *Feature Forgetting in Continual Representation Learning*, arXiv:2205.13359. Devises a representation-level evaluation protocol for continual learning; shows that "feature forgetting" exists beyond output-level catastrophic forgetting; proposes gating adapters as a mitigation.
- **[Caccia et al., 2021]** Embedding drift in continual learning; shows that representations of previously seen data drift as training proceeds, degrading retrieval and transfer even with replay. Foundational reference for the "representation changes under new-task updates" claim.
- **[Lesort et al., 2023]** *Continual Representation Learning* — knowledge accumulation and feature forgetting in representation-based continual learning.
- **[TeLAPA, arXiv:2604.15414, 2026]** *Preserving Plasticity in Continual Reinforcement Learning*. Reframes continual RL around latent manifold dynamics, explicitly uses anchor sets + replay + periodic re-embedding to stabilise encoder drift. Directly adjacent to our "how do $\phi, \psi$ evolve" question.
- **[C-CHAIN, ICML 2025]** *Mitigating Plasticity Loss in Continual RL by Reducing Churn*. Shows plasticity loss correlates with rank decrease of the NTK; proposes a churn-reducing regulariser. Relevant to the actor-plasticity diagnostics in our paper.
- **[Anthes et al., 2024]** Empirically verifies that with orthogonal-optimisation continual learning, representations of past tasks drift while performance stays stable — the canonical demonstration that representation drift and stable behaviour can co-exist.
- **[BFP — Gu et al., 2023]** *Backward Feature Projection* allows new features to change up to a learnable linear transformation of old features; integrated with experience replay.
- **[Dohare et al., Nature 2024]** Loss of plasticity in deep continual learning (already in citations).
- **[Sokar et al., ICML 2023]** ReDo; dormancy threshold $\tau=0.025$ (already in citations).
- **[Papyan et al., PNAS 2020]** Neural Collapse, NRC1/NRC2 (already in citations).
- **[Lyle et al., 2022]** *Understanding and Preventing Capacity Loss in Reinforcement Learning* — feature rank collapse in RL representations.
- **[Kumar et al., ICLR 2021]** *Implicit Under-Parameterization Inhibits Data-Efficient Deep RL* — rank-collapse diagnostic for value-function representations.

**Conclusion.** Representation drift and encoder plasticity in continual RL are actively studied empirically. The paper's discussion of how $\phi, \psi$ adapt can be grounded in this body of work rather than presented as speculative.

## Motivations embedded in the current plan

- **Deployment.** In most real-world and simulator settings we ultimately only care about goal completion; dense reward engineering is human-heuristic and largely determines perceived algorithm quality.
- **Scalability.** Continual skill acquisition requires enough expressivity and effective computation in the policy (Myers et al., *On Computation and Reinforcement Learning*, arXiv:2602.05999). Contrastive GCRL scales cleanly to very deep networks (Wang et al., NeurIPS 2025), making it a strong substrate.
- **Offline-to-online nature.** By task $k$, the agent owns replay buffers from tasks $0,\ldots,k-1$. Continual learning therefore has a natural offline-to-online structure that prior work has not exploited; the negative bank follows from this observation.
- **Contrastive RL as a lens.** The continual setting is also an interesting probe on contrastive RL itself — how $\phi, \psi$ adapt, how the representation space evolves, when additional compute or expressivity is needed.

## Open questions for the user / supervisor

- Should the 9-cell grid be the headline figure, or should we lead with cell (h) alone and appendix the rest?
- Dense-reward SAC and dense-reward policy-decomposition baselines — main body for calibration, or appendix?
- Twenty-task stress test — main experiments or appendix robustness study?
- Negative-bank analysis — standalone narrative slot, or subsection inside the experiments?

## Running history of revisions

- **Draft 1.** Academic-prose paper plan with abstract and section-by-section narrative. Rejected: too casual in places, wrong CKA-RL citation (Kaplanis), not what was wanted.
- **Draft 2.** Formal academic prose; CKA-RL citation corrected to Hu et al. (NeurIPS 2025). Rejected: does not match the bulleted-outline register.
- **Draft 3.** Bulleted narrative matching the OT-reward-paper reference. Flagged as catering/defensive in tone.
- **Draft 4.** Defensive framing removed. Baseline accepted; motivations expansion requested.
- **Draft 5.** Expanded motivations (reward engineering, scalability, offline-to-online, contrastive-RL-as-lens). Myers et al. 2026 added to citations.
- **Draft 6.** Added a paragraph on the setting being new and a representation-drift grounding. The "to the best of our knowledge / no prior benchmark" bullet was still defensive meta-commentary and was rejected as such.
- **Draft 7 (current).** Rewritten opening: state the setting first and describe it briefly; then contrast the continual-RL literature (dense, hand-engineered) with the sparse-reward RL literature (single-task) and name the gap as content; then motivate contrastive goal-conditioned RL as the natural solver and describe our method. Future-work section explicitly acknowledges room for better methods (RL or otherwise) and invites the community to expand research on the setting.

## Actionable next steps

- Confirm the Apr 22 launch of the full 9×5 grid on NYUAD HPC is ready.
- Iterate on paper plan with Prof. Ross in the Apr 24 check-in.
- Begin drafting figure placeholders against the paper_plan narrative.
- Keep all updates on `section3_done` with author `Okyumi`.
