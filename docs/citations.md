# Citations — Sparse-Reward Continual Goal-Conditioned RL

This document lists the references intended for the NeurIPS 2026 manuscript, each with a short annotation describing (i) what the cited work contributes and (ii) how our paper uses it. Citations are grouped thematically. BibTeX keys in brackets are our working conventions and will be normalised to the venue style in the final manuscript.

All citation details below have been verified against the primary sources (arXiv landing pages, conference proceedings, journal pages).

---

## 1. Primary methodological neighbours

### [hu2025cka]
**Hu, Y., Lian, J., Wen, J., Li, S., Chen, X., Wen, Q., Xiao, Y., Tan, M.** *Continual Knowledge Adaptation for Reinforcement Learning.* NeurIPS 2025. arXiv:2510.19314.
*What it contributes.* CKA-RL proposes a decomposition of the SAC actor into a frozen base plus additive per-task knowledge vectors, managed by a bounded knowledge pool with cosine-similarity merging. The critic is SAC, reset each task. Dense-reward Meta-World is the benchmark.
*How we use it.* Cited in related work as the closest prior instantiation of policy decomposition and bounded knowledge pools in continual reinforcement learning. We adopt the same mathematical form for the decomposition and the merge rule. The present work is **not** an extension of CKA-RL: our algorithm uses a contrastive goal-conditioned critic in place of SAC, operates in the sparse-reward regime, and integrates a cross-task negative bank. In the manuscript, CKA-RL appears as prior work that has studied these building blocks, and is reproduced as a baseline under matched decomposition machinery.

### [liu2025single]
**Liu, G., Tang, J., Eysenbach, B.** *A Single Goal is All You Need: Skills and Exploration Emerge from Contrastive RL.* ICLR 2025. arXiv:2408.05804.
*What it contributes.* Shows that contrastive goal-conditioned RL with a single commanded goal per episode induces skill acquisition and exploration without reward shaping.
*How we use it.* Direct predecessor of our contrastive critic pipeline. Our single-task sanity-check configuration reduces exactly to this algorithm.

### [wang2025thousand]
**Wang, K., Javali, A., Bortkiewicz, M., Trzciński, T., Eysenbach, B.** *1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities.* NeurIPS 2025 (Best Paper). arXiv:2503.14858.
*How we use it.* Source of the deep residual MLP backbone (1024 width, LayerNorm + Swish, residual connections) used throughout the actor and critic. Demonstrates that contrastive goal-conditioned RL scales to very deep networks, which is a precondition for the representational demands of continual skill acquisition.

### [eysenbach2022contrastive]
**Eysenbach, B., Zhang, T., Salakhutdinov, R., Levine, S.** *Contrastive Learning as Goal-Conditioned Reinforcement Learning.* NeurIPS 2022. arXiv:2206.07568.
*How we use it.* Formal foundation of contrastive goal-conditioned RL: the equivalence between InfoNCE training of `φ(s,a)^⊤ψ(g)` and the discounted visitation objective `r_g(s,a) = (1-γ)p(s'=g|s,a)`.

### [bortkiewicz2025jaxgcrl]
**Bortkiewicz, M., Pałucki, W., Myers, V., Dziarmaga, T., Arczewski, T., Kuciński, Ł., Eysenbach, B.** *Accelerating Goal-Conditioned RL Algorithms and Research.* ICLR 2025. arXiv:2408.11052.
*How we use it.* The JaxGCRL codebase is the upstream implementation from which our `section3_done` branch derives. Cited for reproducibility and for the JAX/Haiku infrastructure.

### [myers2025demystifying]
**Myers, V., et al.** *Demystifying the Mechanisms Behind Emergent Exploration in Goal-Conditioned RL.* arXiv:2510.14129.
*How we use it.* Analyses exploration mechanisms in single-goal contrastive RL; referenced in related work and in the discussion of why contrastive critics expose useful gradients during the early exploration phase.

---

## 2. Continual reinforcement learning

### [khetarpal2022towards]
**Khetarpal, K., Riemer, M., Rish, I., Precup, D.** *Towards Continual Reinforcement Learning: A Review and Perspectives.* JAIR, 2022. arXiv:2012.13490.
*How we use it.* Survey reference establishing the problem setting.

### [wolczyk2021cw]
**Wołczyk, M., Zając, M., Pascanu, R., Kuciński, Ł., Miłoś, P.** *Continual World: A Robotic Benchmark for Continual Reinforcement Learning.* NeurIPS 2021. arXiv:2105.10919.
*How we use it.* Source of the ten-task Meta-World Sawyer sequence used as the backbone of our benchmark.

### [wolczyk2022disentangling]
**Wołczyk, M., Zając, M., Pascanu, R., Kuciński, Ł., Miłoś, P.** *Disentangling Transfer in Continual Reinforcement Learning.* NeurIPS 2022. arXiv:2209.13900.
*How we use it.* Shows, in dense-reward SAC, that transferring the critic yields more forward transfer than transferring the actor. We revisit the same question in the sparse-reward regime with a contrastive critic, and our findings align with and extend theirs.

### [rolnick2019clear]
**Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., Wayne, G.** *Experience Replay for Continual Learning (CLEAR).* NeurIPS 2019.
*How we use it.* Canonical replay-based continual-RL baseline; referenced as orthogonal methodology in related work.

### [kirkpatrick2017ewc]
**Kirkpatrick, J., et al.** *Overcoming Catastrophic Forgetting in Neural Networks (EWC).* PNAS 2017.
*How we use it.* Canonical regularisation-based continual-learning baseline; referenced in related work.

---

## 3. Plasticity loss and representation diagnostics

### [dohare2024loss]
**Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., Sutton, R. S.** *Loss of Plasticity in Deep Continual Learning.* Nature, 2024.
*How we use it.* Primary reference for the plasticity-loss phenomenon that motivates our actor-side diagnostics.

### [nikishin2022primacy]
**Nikishin, E., Schwarzer, M., D'Oro, P., Bacon, P.-L., Courville, A.** *The Primacy Bias in Deep Reinforcement Learning.* ICML 2022. arXiv:2205.07802.
*How we use it.* Primacy-bias phenomenon; discussed in related work on plasticity.

### [sokar2023redo]
**Sokar, G., Agarwal, R., Castro, P. S., Evci, U.** *The Dormant Neuron Phenomenon in Deep Reinforcement Learning (ReDo).* ICML 2023. arXiv:2302.12902.
*How we use it.* Source of the dormancy threshold `τ = 0.025` used for Swish activations in our actor-side diagnostics, and of the dormant-neuron reset protocol that we implement as a diagnostic mechanism (disabled by default).

### [papyan2020prevalence]
**Papyan, V., Han, X. Y., Donoho, D. L.** *Prevalence of Neural Collapse During the Terminal Phase of Deep Learning Training.* PNAS, 2020.
*How we use it.* Provides the NRC1 and NRC2 neural-collapse statistics used as representational diagnostics on the actor encoder.

---

## 4. Contrastive representation learning

### [oord2018cpc]
**van den Oord, A., Li, Y., Vinyals, O.** *Representation Learning with Contrastive Predictive Coding.* arXiv:1807.03748.
*How we use it.* Introduces the InfoNCE objective that we use for critic training.

### [he2020moco]
**He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.** *Momentum Contrast for Unsupervised Visual Representation Learning (MoCo).* CVPR 2020.
*How we use it.* Analogue for the memory-bank-of-negatives design pattern; cited in the discussion of the negative bank.

### [robinson2021contrastive]
**Robinson, J., Chuang, C.-Y., Sra, S., Jegelka, S.** *Contrastive Learning with Hard Negative Samples.* ICLR 2021.
*How we use it.* Theoretical and empirical support for hard-negative mining; directly motivates the `hard_weighted` negative-bank variant.

---

## 5. Supporting algorithmic components

### [haarnoja2018sac]
**Haarnoja, T., Zhou, A., Abbeel, P., Levine, S.** *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.* ICML 2018.
*How we use it.* Source of the adaptive entropy coefficient mechanism (SAC dual gradient) that we use with `target_entropy = −2.0`.

### [andrychowicz2017her]
**Andrychowicz, M., et al.** *Hindsight Experience Replay.* NeurIPS 2017.
*How we use it.* Hindsight relabelling is the positive-pair mechanism for our contrastive objective.

### [yu2020metaworld]
**Yu, T., Quillen, D., He, Z., Julian, R., Hausman, K., Finn, C., Levine, S.** *Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning.* CoRL 2020.
*How we use it.* Source of the Sawyer manipulation tasks used in our ten-task sequence.

---

## 6. Notes on prior-version errors

An earlier draft of this file attributed the continual knowledge adaptation work to "Kaplanis et al., 2024." That attribution was incorrect. The correct citation is **Hu, Lian, Wen, Li, Chen, Wen, Xiao, Tan, *Continual Knowledge Adaptation for Reinforcement Learning*, NeurIPS 2025, arXiv:2510.19314**, and it has been propagated throughout the paper plan and this file.
