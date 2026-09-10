# What “stages” mean, in plain numbers — and the new mass-inject tests

Date: 2026-09-09
Status: implemented + submitted (SLURM `17270536`, W&B `TASK58-MASS-INJECT-DWELL-1M`)

## Plain language (no jargon)

Every episode has **150 steps**. Think of a timeline of where the hand / object
is during those 150 steps.

### Push (a task that retains)

On a *successful* push episode the cube must end within **5 cm** of the target.
Along the way the cube is often **between** start and target. We count:

- **far**: hand still far from the cube  
- **near / hover**: hand close to the cube, cube still far from target  
- **part-way** (`object_progress`): cube is already within **15 cm** of the
  target but not yet within 5 cm — this is the “middle strip”  
- **done** (`success`): cube within 5 cm  

That middle strip matters because HER turns those mid-way states into training
goals. So the buffer fills with “keep moving the cube closer” goals, not only
“hover near the cube” goals.

### Handle press side

Success = handle height within **2 cm** of the pressed height (`z≈0.07`).
There is **no middle strip** for the handle the way there is for the cube:
either the handle is pressed enough, or it is not. We only count:

- **far / near**: approaching the handle  
- **hover**: touching the handle but **not** pressed enough  
- **success (for HER)**: touching **and** pressed  

Important: after the handle is pressed, the env can keep giving reward even if
the hand leaves. HER does **not** treat “handle down, hand away” as a success
goal. So a successful handle episode can contribute only a **short** run of
HER-success goals (the frames where the hand is still on the pressed handle).

We now log every eval:

```
[stage dwell] far=.. near=.. hover=.. prog=.. succ=.. (mean steps / 150)
```

on successful episodes only. That answers: “how many of the 150 steps does the
policy spend in each place?”

## Why OOD / hover HER goals dominate on handle

Not mysterious: if successful episodes only put a few steps of HER-success into
the buffer, and the actor then stops succeeding, almost every new episode is
approach/hover/far. HER goals are sampled from those futures → hover/far
dominate. On push, successful (and part-way) episodes keep dumping many
success/progress futures into the buffer, so those goals stay common.

## New experiments (what you asked to run)

| Cell | What |
|---|---|
| handle_measure | Log stage dwell + press−π score; no inject |
| push_measure | Same (comparison) |
| handle_inject_10pct | At first ≥20% eval success, **clone** successful episodes until ≈**10%** of current buffer size |
| handle_inject_20pct | Same to ≈**20%** |

Still **no Success-BC**. If 10–20% mass makes the actor keep succeeding and HER
stay healthy → scarcity was the bottleneck. If not → actor optimization /
landscape problem remains.

Also logs `[press vs pi] gap=score(press)-score(π)` at hover under \(g_{\text{task}}\):
- gap ≫ 0 and π still does not press → actor is not following the critic  
- gap ≤ 0 → critic landscape itself does not prefer press at those states  

## Launch

```bash
sbatch DRAFT_task58_mass_inject_dwell.sh
```
