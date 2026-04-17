# Continual Goal-Conditioned RL with Contrastive Critics

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run (SLURM)
ACTOR_MODE=cka CRITIC_MODE=persistent USE_TASK_ID=false SEED=6 sbatch draft_3.sh

# Local test (2 tasks, 10K steps)
python run_continual_contrastive.py \
    --seed=42 --num_tasks=2 --steps_per_task=10000 --base_steps=10000 \
    --alg=contrastive_cpc --nouse_wandb
```

## Configurations

Two axes — actor mode and critic mode — give 9 experiment configurations:

|  | reset critic | persistent critic | cka critic |
|---|---|---|---|
| **reset actor** | A1: fully independent | A2: critic-only transfer | A3: reset + CKA critic |
| **cka actor** | B3: CKA-RL baseline | B1: main hypothesis | B2: full CKA |
| **persistent actor** | C2: actor-only persistence | C1: both persistent | C3: persistent + CKA critic |

Example commands:
```bash
ACTOR_MODE=reset CRITIC_MODE=reset USE_TASK_ID=false SEED=6 sbatch draft_3.sh      # A1
ACTOR_MODE=cka CRITIC_MODE=persistent USE_TASK_ID=false SEED=6 sbatch draft_3.sh   # B1
ACTOR_MODE=persistent CRITIC_MODE=persistent USE_TASK_ID=false SEED=6 sbatch draft_3.sh  # C1
```

## Documentation

| Document | Contents |
|---|---|
| [`doc/EVERYTHING_ABOUT_CODE_STRUCTURE.md`](doc/EVERYTHING_ABOUT_CODE_STRUCTURE.md) | Code layout, where to modify each component |
| [`doc/EVERYTHING_ABOUT_EXPERIMENT_SETUP.md`](doc/EVERYTHING_ABOUT_EXPERIMENT_SETUP.md) | All configs, flags, commands, metrics |
| [`doc/Metaworld_documentation/`](doc/Metaworld_documentation/) | MetaWorld environment details |
| [`doc/archive/`](doc/archive/) | Historical audits and implementation notes |

## References

- Liu et al., "A Single Goal Is All You Need," ICLR 2025
- Hu et al., "Continual Knowledge Adaptation for Reinforcement Learning," NeurIPS 2025
- Eysenbach et al., "Contrastive Learning as Goal-Conditioned RL," NeurIPS 2022
- Wang et al., "1000 Layer Networks for Self-Supervised RL," 2025
