# Task 7: GIFs of the stored success-buffer ring

Date: 2026-09-22  
Status: renderer added. GIFs from
`scripts/record_success_buffer_gifs.py` /
`sbatch DRAFT_jubail_task7_success_buffer_gifs.sh`.

## What "r0…r9" was

The Task-7 shelf-place GIF canvas labeled `r0`–`r9` is **not**
\(\mathcal{D}_{\mathrm{succ}}\). `record_checkpoint_rollout_gifs.py`
rolls the **current policy** for `--rollouts 10` with eval seeds
`6+1000+i` (1006–1015). Return is the sparse env sum; "success" is any
in-episode hit. Those clips show what the frozen actor does, not what
the BC ring cloned.

## What this dump is

Mid-task checkpoints store
`decomposed_training_state.success_buffer_{observation,action,size,index}`.
Final `task_7.pkl` does **not**. Paper plain DCC and episode-wide
Success-BC therefore have **no ring on disk**. The dump uses:

| snapshot | label | ring |
|---|---|---|
| terminal seed 6 @ 1.50M | terminal-episode | 750 = first 5 cloned episodes |
| warmup seed 6 @ 2.30M | warmup, still contiguous | 2624 |
| first-success-window seed 6, first fill | window \(K=64\) | first non-empty mid-ckpt |
| terminal / warmup late @ 7.90M | same labels, wrapped ring | 4096 mixed \((s,a)\) |

Playback writes stored hand / object xyz. Default GIFs are **one
150-step MetaWorld episode** in FIFO order (`--episode-len 150`), so
each clip is a full cloned attempt. `--episode-len 0` restores the
old teleport split.

## Labeling vs geometry

Insertions use `outcome_task_success` (`native_info` last-bit or window),
not a post-hoc object-to-shelf check. Jump-split of the first-fill rings
(object teleport \(>0.12\) m):

| snapshot | stored runs | last pose in \(0.07\) ball |
|---|---:|---:|
| terminal seed 6 @ 1.50M | 5 episodes × 150 | **1/5** (all 5 touched the ball) |
| warmup seed 6 @ 2.30M | 17×150 + 74 | **4/18** |
| first-success-window @ 1.40M | 27×150 + 46 | **0/28** |

The misses are cloned because the sparse bit said success, then BC NLL
pulls \(\pi\) toward those \((s,a)\). That is not a DCC policy-gradient
sign error. Late 8M rings are a shuffled FIFO (median object jump
\(0.19\) m) and are not watchable as episodes.

Login-node EGL cannot render MuJoCo; the canvas uses 3D object/hand
paths. GPU job `18069285` dumps camera GIFs of the same rings.

```bash
sbatch DRAFT_jubail_task7_success_buffer_gifs.sh
python tests/test_success_buffer_gifs.py
```

Output: `logs/jubail_task7_success_buffer_gifs/gifs/`
