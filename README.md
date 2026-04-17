# Continual Goal-Conditioned RL with Contrastive Critics

This repository implements a framework for continual reinforcement learning that combines **goal-conditioned contrastive RL** (SGCRL) with **CKA-RL**-style knowledge adaptation.

The core idea: replace SAC in the CKA-RL pipeline with a contrastive critic that learns state-goal reachability structure, then study whether this structure transfers across tasks.

## Architecture

| Component | Description |
|---|---|
| `run_continual_contrastive.py` | Main experiment driver (sequential, single-process) |
| `contrastive/` | Core algorithm: networks, learner, config, knowledge pool |
| `env_utils.py` | MetaWorld Sawyer environment wrappers |
| `draft_3.sh` | SLURM launcher for HPC experiments |
| `doc/` | Documentation, experiment guides, audit logs |

## Configuration

Two orthogonal axes control transfer across tasks:

**Actor mode** (`--actor_mode`):
- `cka` — CKA-RL decomposition: θ' = θ_base + Σ α_j v_j + v_k
- `reset` — reinitialize from scratch each task
- `persistent` — single network, continuously trained

**Critic mode** (`--critic_mode`):
- `persistent` — carry forward φ(s,a)ᵀψ(g) across tasks
- `reset` — reinitialize each task
- `cka` — CKA-RL decomposition for the critic

This gives 9 configurations (3 × 3). See `doc/audit_apr17.md` for a detailed trace of each.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single-task test (2 tasks, 10K steps)
python run_continual_contrastive.py \
    --seed=42 --num_tasks=2 --steps_per_task=10000 --base_steps=10000 \
    --alg=contrastive_cpc

# Full experiment (SLURM)
ACTOR_MODE=cka CRITIC_MODE=persistent USE_TASK_ID=false SEED=6 sbatch draft_3.sh
```

## Task Sequence

10 MetaWorld Sawyer manipulation tasks:
1. hammer
2. push_wall
3. faucet_close
4. push_back
5. stick_pull
6. handle_press_side
7. push
8. shelf_place
9. window_close
10. peg_unplug_side

## References

- Liu et al., "A Single Goal Is All You Need," ICLR 2025
- Hu et al., "Continual Knowledge Adaptation for Reinforcement Learning," NeurIPS 2025
- Eysenbach et al., "Contrastive Learning as Goal-Conditioned RL," NeurIPS 2022
- Wang et al., "1000 Layer Networks for Self-Supervised RL," 2025
