# NeurIPS 2026 Paper — Citation List

Each entry below lists the full reference, a short tag (for BibTeX), and every paper section where we expect to cite it. When a citation is the **primary** reference for a concept, it is marked with ★.

---

## 1. Cornerstone works (must-cite, readme-level)

### ★ Eysenbach et al., NeurIPS 2022
**Title:** "Contrastive Learning as Goal-Conditioned Reinforcement Learning"
**Authors:** Benjamin Eysenbach, Tianjun Zhang, Ruslan Salakhutdinov, Sergey Levine
**Tag:** `eysenbach2022contrastive`
**URL/DOI:** https://arxiv.org/abs/2206.07568
**Cite in sections:**
- §1 Introduction — introducing contrastive GCRL
- §2 Related Work — primary reference for CRL
- §3 Preliminaries — full derivation of InfoNCE → reachability critic
- §5 Method — our critic objective is their InfoNCE
**Notes:** This is *the* foundational paper for contrastive GCRL. Cite as "contrastive RL (Eysenbach et al., 2022)" on first use.

### ★ Liu et al., ICLR 2025 (SGCRL)
**Title:** "A Single Goal Is All You Need"
**Authors:** Anonymous / Liu et al. (in our context, the Keith-Ross-supervised predecessor paper)
**Tag:** `liu2025single`
**URL/DOI:** https://arxiv.org/abs/2409.15830 (tentative — verify final citation)
**Cite in sections:**
- §1 Introduction — single-goal variant context
- §2 Related Work
- §3 Preliminaries — for fixed-goal reduction and the specific sparse-reward Meta-World variant they analyse
- §5 Method — our single-task behaviour reduces exactly to their implementation (single-task sanity check)
**Notes:** The immediate predecessor of our work. Defines the sparse-goal Meta-World variant we build on.

### ★ Kaplanis et al., 2024 (CKA-RL)
**Title:** "Continual Reinforcement Learning with Knowledge Adaptation" (or verify exact title)
**Authors:** Christos Kaplanis et al.
**Tag:** `kaplanis2024continual`
**URL/DOI:** (TBD — check exact arXiv)
**Cite in sections:**
- §1 Introduction — as the method we extend
- §2 Related Work — as the closest methodological neighbour
- §3 Preliminaries — definition of the actor decomposition θ' = θ_base + Σ α_j v_j + v_k
- §5 Method — point out our divergences (contrastive critic, critic decomposition, neg bank)
- §7 Results — as a baseline
**Notes:** The paper we draw inspiration from for knowledge decomposition. Our contribution is lifting this from scalar-Q SAC continual learning to contrastive GCRL continual learning.

### ★ Wang et al., 2025 (Scaling CRL / 1000-layer paper)
**Title:** "Scaling Contrastive Reinforcement Learning" (or "1000-layer RL")
**Authors:** Kevin Wang et al.
**Tag:** `wang2025scaling`
**URL/DOI:** https://github.com/wang-kevin3290/scaling-crl (verify arXiv)
**Cite in sections:**
- §1 Introduction — scaling motivation for contrastive RL
- §2 Related Work
- §3 Preliminaries — our architecture (ResidualMLP + L2 energy + adaptive entropy) draws directly from here
- §5 Method — target_entropy = -0.5 × action_dim convention
- §6 Experimental Setup — hyperparameter choices
**Notes:** Our ResidualMLP + LayerNorm + Swish + L2 energy + adaptive entropy setup matches their conventions. Cite as "we follow the scaling-CRL pipeline of (Wang et al., 2025)" in §6.

---

## 2. Continual RL / lifelong RL foundations

### Wolczyk et al., NeurIPS 2021 — Continual World
**Title:** "Continual World: A Robotic Benchmark for Continual Reinforcement Learning"
**Tag:** `wolczyk2021continual`
**URL:** https://arxiv.org/abs/2105.10919
**Cite in sections:**
- §2 Related Work — establishes Meta-World as the continual RL benchmark
- §6 Experimental Setup — our task sequence derives from Continual World / Meta-World
**Notes:** Originator of the Meta-World-as-continual-benchmark convention. Our 10-task sequence is a subset of their CW10.

### ★ Wolczyk et al., NeurIPS 2022 — Disentangling transfer
**Title:** "Disentangling Transfer in Continual Reinforcement Learning"
**Tag:** `wolczyk2022disentangling`
**URL:** https://arxiv.org/abs/2209.13900
**Cite in sections:**
- §1 Introduction — motivating critic transfer > actor transfer
- §2 Related Work
- §7 Results — we replicate their finding (critic matters more) but in a contrastive setting
- §10 Discussion — key supporting result for our persistent-critic recommendation
**Notes:** Central empirical justification for our focus on critic transfer. They showed it for SAC; we extend to contrastive GCRL.

### Mendez and Eaton, 2022 — modular continual RL (likely cited)
**Title:** "Modular Lifelong Reinforcement Learning via Neural Composition"
**Tag:** `mendez2022modular`
**URL:** https://arxiv.org/abs/2207.00429
**Cite in sections:**
- §2 Related Work — predecessor to knowledge-pool ideas
**Notes:** Optional if space permits; they use neural composition, a related idea to our knowledge pool.

### Khetarpal et al., 2022 (lifelong RL survey)
**Title:** "Towards Continual Reinforcement Learning: A Review and Perspectives"
**Tag:** `khetarpal2022towards`
**URL:** https://arxiv.org/abs/2012.13490
**Cite in sections:**
- §1 Introduction (optional) — for general framing
- §2 Related Work — comprehensive review
**Notes:** Good general reference for the continual RL problem.

---

## 3. Continual learning in supervised and general deep learning

### Kirkpatrick et al., PNAS 2017 — EWC
**Title:** "Overcoming catastrophic forgetting in neural networks"
**Tag:** `kirkpatrick2017overcoming`
**URL:** https://arxiv.org/abs/1612.00796
**Cite in sections:**
- §2 Related Work — catastrophic forgetting
**Notes:** Classic reference for catastrophic forgetting. Mention alongside L2/EWC-based regularization approaches as alternatives to our pool-based approach.

### Rusu et al., 2016 — Progressive Networks
**Title:** "Progressive Neural Networks"
**Tag:** `rusu2016progressive`
**URL:** https://arxiv.org/abs/1606.04671
**Cite in sections:**
- §2 Related Work — architectural expansion approach, contrast with our bounded pool
**Notes:** Contrast: Progressive networks grow without bound; our pool has K_max.

### Mallya and Lazebnik, CVPR 2018 — PackNet
**Title:** "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning"
**Tag:** `mallya2018packnet`
**URL:** https://arxiv.org/abs/1711.05769
**Cite in sections:**
- §2 Related Work — parameter-efficient continual learning
**Notes:** Predecessor to LoRA-style per-task deltas.

### Rolnick et al., NeurIPS 2019 — CLEAR (experience replay for CRL)
**Title:** "Experience Replay for Continual Learning"
**Tag:** `rolnick2019experience`
**URL:** https://arxiv.org/abs/1811.11682
**Cite in sections:**
- §2 Related Work — rehearsal-based approaches
- §4 Problem Setup — "unlike rehearsal methods, we do not replay past transitions during training; we use past goals only as InfoNCE negatives"
**Notes:** Important contrast. Cite when introducing the offline-to-online bank — CLEAR uses cross-task replay for rehearsal; we use it for contrast.

---

## 4. Goal-conditioned RL and hindsight

### ★ Andrychowicz et al., NeurIPS 2017 — HER
**Title:** "Hindsight Experience Replay"
**Tag:** `andrychowicz2017hindsight`
**URL:** https://arxiv.org/abs/1707.01495
**Cite in sections:**
- §3 Preliminaries — HER is how we relabel goals for InfoNCE positives
- §5 Method — our flatten_fn implements future-state relabeling
- §5.4 Negative bank — the bank stores HER-relabeled goals, reusing the same relabeling procedure
**Notes:** Primary reference for future-state relabeling. Essential for our data pipeline.

### Ghosh et al., 2021 — GCSL
**Title:** "Learning to Reach Goals via Iterated Supervised Learning"
**Tag:** `ghosh2021learning`
**URL:** https://arxiv.org/abs/1912.06088
**Cite in sections:**
- §2 Related Work — alternative GCRL training (supervised)
**Notes:** Optional, for comprehensive GCRL coverage.

### Ma et al., 2022 — VIP
**Title:** "VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training"
**Tag:** `ma2022vip`
**URL:** https://arxiv.org/abs/2210.00030
**Cite in sections:**
- §2 Related Work — contrastive value learning in a pretraining context
**Notes:** Alternative contrastive value approach.

---

## 5. Contrastive representation learning (non-RL)

### van den Oord et al., 2018 — CPC / InfoNCE
**Title:** "Representation Learning with Contrastive Predictive Coding"
**Tag:** `oord2018representation`
**URL:** https://arxiv.org/abs/1807.03748
**Cite in sections:**
- §3 Preliminaries — origin of the InfoNCE objective
**Notes:** Foundational for the loss function we use. Must-cite in Preliminaries.

### Chen et al., ICML 2020 — SimCLR (negatives & hard-mining context)
**Title:** "A Simple Framework for Contrastive Learning of Visual Representations"
**Tag:** `chen2020simple`
**URL:** https://arxiv.org/abs/2002.05709
**Cite in sections:**
- §5.4 Negative bank — for "the importance of negative sampling" framing
**Notes:** Canonical reference for the role of negatives in contrastive learning.

### Robinson et al., ICLR 2021 — Hard negative mining in contrastive learning
**Title:** "Contrastive Learning with Hard Negative Samples"
**Tag:** `robinson2021contrastive`
**URL:** https://arxiv.org/abs/2010.04592
**Cite in sections:**
- §5.4 Negative bank — direct inspiration for our hard-negative mining strategy
**Notes:** Important. Our per-anchor top-M hard mining is closely related to their framework.

### Kalantidis et al., NeurIPS 2020 — MoCHi (mixing hard negatives)
**Title:** "Hard Negative Mixing for Contrastive Learning"
**Tag:** `kalantidis2020hard`
**URL:** https://arxiv.org/abs/2010.01028
**Cite in sections:**
- §5.4 Negative bank — alternative hard-negative strategy
**Notes:** Optional context citation.

### He et al., CVPR 2020 — MoCo (memory bank analogy)
**Title:** "Momentum Contrast for Unsupervised Visual Representation Learning"
**Tag:** `he2020momentum`
**URL:** https://arxiv.org/abs/1911.05722
**Cite in sections:**
- §5.4 Negative bank — closest conceptual analogue in vision (cross-batch negatives)
**Notes:** Direct analogue of our offline-to-online bank — just across tasks instead of mini-batches.

---

## 6. Plasticity loss and network resets in deep RL

### ★ Nikishin et al., ICML 2022 — Primacy Bias
**Title:** "The Primacy Bias in Deep Reinforcement Learning"
**Tag:** `nikishin2022primacy`
**URL:** https://arxiv.org/abs/2205.07802
**Cite in sections:**
- §2 Related Work — plasticity loss and resets
- §8 Results / §10 Discussion — we found seed-dependent plasticity-loss traps, matching their observations
**Notes:** Key reference for network-reset as a plasticity-loss mitigation. We disable periodic resets in our method (they would interfere with ablations) but acknowledge the phenomenon is real.

### ★ Sokar et al., ICML 2023 — ReDo
**Title:** "The Dormant Neuron Phenomenon in Deep Reinforcement Learning"
**Tag:** `sokar2023dormant`
**URL:** https://arxiv.org/abs/2302.12902
**Cite in sections:**
- §2 Related Work — dormant neurons in RL
- §5.2 Representation diagnostics — our dormant-ratio threshold τ = 0.025 is calibrated following their discussion
- §8 Results — our Fig 4 plots dormant ratio in their framework
**Notes:** Primary reference for our dormant-ratio metric. Cite their τ ∈ {0, 0.025, 0.1} sweep and justify our choice for Swish.

### ★ Dohare et al., Nature 2024 — Loss of plasticity
**Title:** "Loss of plasticity in deep continual learning"
**Tag:** `dohare2024loss`
**URL:** https://www.nature.com/articles/s41586-024-07711-7
**Cite in sections:**
- §1 Introduction — primary evidence that plasticity loss is a real phenomenon in continual deep learning
- §2 Related Work
- §8 Results — our actor-plasticity finding supports theirs
**Notes:** High-profile recent paper. Use liberally in motivation.

### Lyle et al., 2022 — Feature rank and plasticity in DQN
**Title:** "Understanding and Preventing Capacity Loss in Reinforcement Learning"
**Tag:** `lyle2022understanding`
**URL:** https://arxiv.org/abs/2204.09560
**Cite in sections:**
- §2 Related Work — feature rank / NRC in RL
- §5.2 Diagnostics — we import their feature_rank metric
**Notes:** Important precedent for using feature rank as an RL diagnostic.

### Abbas et al., 2026 — activation design for plasticity
**Title:** (TBD — likely related to "Swish / SiLU in continual RL")
**Tag:** `abbas2026activation`
**URL:** (TBD)
**Cite in sections:**
- §5.2 Diagnostics — justifies our τ = 0.025 threshold for Swish
**Notes:** Verify citation; if unavailable, replace with a Lyle paper.

---

## 7. Neural Rank Collapse (NRC)

### Papyan et al., PNAS 2020 — Neural collapse
**Title:** "Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training"
**Tag:** `papyan2020prevalence`
**URL:** https://www.pnas.org/content/117/40/24652
**Cite in sections:**
- §2 Related Work — origin of neural collapse
- §5.2 Diagnostics — NRC1 and NRC2 derive from this work
**Notes:** Foundational reference for NRC metrics.

### Zhu et al., NeurIPS 2021 — Neural collapse geometry
**Title:** "A Geometric Analysis of Neural Collapse with Unconstrained Features"
**Tag:** `zhu2021geometric`
**URL:** https://arxiv.org/abs/2105.02375
**Cite in sections:**
- §5.2 Diagnostics — NRC2 definition
**Notes:** Used by the He / Kumar RL-NRC papers.

---

## 8. Specific techniques we reuse

### Haarnoja et al., ICML 2018 — SAC
**Title:** "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"
**Tag:** `haarnoja2018soft`
**URL:** https://arxiv.org/abs/1801.01290
**Cite in sections:**
- §2 Related Work — the baseline continual-RL method
- §5 Method — adaptive entropy / dual-gradient alpha autotuning
- §6 Experimental Setup — SAC baselines
**Notes:** Our adaptive entropy exactly follows SAC's formulation. Must-cite.

### Haarnoja et al., 2018b — SAC v2 / auto-alpha
**Title:** "Soft Actor-Critic Algorithms and Applications"
**Tag:** `haarnoja2018algorithms`
**URL:** https://arxiv.org/abs/1812.05905
**Cite in sections:**
- §5 Method — auto-α dual-gradient mechanism
**Notes:** The version that introduces adaptive α. Our target_entropy = -½ |A| is the "softened" heuristic from here.

### Yu et al., CoRL 2020 — Meta-World
**Title:** "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning"
**Tag:** `yu2020metaworld`
**URL:** https://arxiv.org/abs/1910.10897
**Cite in sections:**
- §6 Experimental Setup — our benchmark
**Notes:** Must-cite in §6.

### Hoffer et al., NeurIPS 2017 — triplet loss (optional)
**Tag:** `hoffer2017triplet`
**Cite in sections:**
- §5.4 Negative bank — conceptually related
**Notes:** Optional.

---

## 9. Recent goal-conditioned & offline-to-online work (for Discussion/Related)

### Park et al., 2024 — Discovery of Surprising RL (optional)
**Title:** "A Surprising Failure Mode of Goal-Conditioned RL"
**Tag:** `park2024surprising`
**URL:** (TBD)
**Cite in sections:**
- §2 Related Work
- §10 Discussion — failure modes in GCRL
**Notes:** Check recent arXiv. Optional.

### Nair et al., 2020 — RPL / RIS / offline-to-online discussion
**Title:** (various offline-to-online papers)
**Tag:** `nair2020offline`
**Cite in sections:**
- §2 Related Work — offline-to-online transition in RL
- §4 Problem Setup — our framing of continual RL as offline-to-online
**Notes:** Useful for framing. Pick the strongest single citation.

### Zheng et al., 2023 — Hybrid RL / offline-to-online bridges
**Tag:** `zheng2023hybrid`
**Cite in sections:**
- §2 Related Work
- §4 Problem Setup
**Notes:** Optional.

---

## 10. JAX / Implementation references

### Bradbury et al., 2018 — JAX
**Tag:** `bradbury2018jax`
**URL:** https://github.com/google/jax
**Cite in:** §6 Experimental Setup (infrastructure)

### Hennigan et al., 2020 — Haiku
**Tag:** `hennigan2020haiku`
**URL:** https://github.com/deepmind/dm-haiku
**Cite in:** §6 Experimental Setup

### Hoffman et al., 2020 — Acme
**Tag:** `hoffman2020acme`
**URL:** https://arxiv.org/abs/2006.00979
**Cite in:** §6 Experimental Setup

---

## 11. Summary table — citations per section

| Section | Minimum citations | Flavour |
|---|---|---|
| 1. Introduction | Eysenbach 2022, Liu 2025, Kaplanis 2024, Wolczyk 2022, Dohare 2024 | frame & motivate |
| 2. Related Work | 15–20 across all buckets above | comprehensive |
| 3. Preliminaries | Eysenbach 2022, Oord 2018, Andrychowicz 2017, Haarnoja 2018, Kaplanis 2024 | technical definitions |
| 4. Problem Setup | Rolnick 2019, Wolczyk 2021, Khetarpal 2022 | continual-RL context |
| 5. Method | Eysenbach 2022, Kaplanis 2024, Wang 2025, Andrychowicz 2017, Haarnoja 2018 | algorithmic lineage |
| 5.4 Neg bank | He 2020 (MoCo), Robinson 2021, Chen 2020 | hard negatives & memory bank |
| 5.2 Diagnostics | Papyan 2020, Zhu 2021, Sokar 2023, Lyle 2022, Abbas 2026 | representation metrics |
| 6. Experimental Setup | Yu 2020 (MetaWorld), Wang 2025 (hyperparams), Haarnoja 2018b (α) | setup & hyperparameters |
| 7. Results (main grid) | Kaplanis 2024, Wolczyk 2022 | baselines and comparison |
| 8. Results (representations) | Dohare 2024, Nikishin 2022, Sokar 2023, Lyle 2022 | interpretation |
| 9. Results (neg bank) | He 2020 (MoCo), Robinson 2021 | comparison context |
| 10. Discussion | Wolczyk 2022, Dohare 2024, Nikishin 2022 | implications |

**Estimated total references:** 40–50 (strong for a 10-page NeurIPS paper).

---

## 12. Actionable to-dos on citations

- [ ] Verify **Liu et al. 2025 (SGCRL)** exact arXiv / publication venue.
- [ ] Verify **Kaplanis et al. 2024 (CKA-RL)** exact title and venue.
- [ ] Verify **Wang et al. 2025 (scaling-CRL)** arXiv ID and title.
- [ ] Verify **Abbas et al. 2026** exists or replace with a 2024/2025 Swish-plasticity paper.
- [ ] Find a concrete citation for **offline-to-online RL** framing (Nair 2020, Zheng 2023, or similar).
- [ ] Double-check all Andrychowicz / HER citations for venue (NeurIPS vs ICLR).
- [ ] Run through the top-10 recent NeurIPS papers on "continual RL" and pick up any 2025-era work we may be missing.
- [ ] If space permits, add a **PackNet / Progressive Networks / LoRA** citation trail for the "knowledge decomposition" lineage.
