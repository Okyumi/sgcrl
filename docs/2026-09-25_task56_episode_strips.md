# Task-5 / Task-6 episode filmstrip (appendix occupancy visual)

Date: 2026-09-25
Status: recorded. Native EGL 640×480 from job `18217413` (same
deterministic episodes as the audited GIFs: handle return 97 after 6
resets, push return 91). Paper TeX was **not** edited. No new training.

## Why a filmstrip, not another chart

The Task-5 appendix already has a table (press−π, rank, cos, std_a/std_s,
xy-shuffle vs action-shuffle, inject). Restyling those numbers as bars
or compasses repeats the same measurement. The claim that needs pixels
is geometric:

- Task 5 (`sawyer_handle_press_side`) is a **latch**: handle \(z\) is
  almost constant until a brief press, then sits in the success band.
- Task 6 (`sawyer_push`) is a **path**: the cube translates over a large
  fraction of the 150-step horizon.

That is the occupancy story behind sparse \(\mathcal{S}_{\mathrm{act}}^{\varepsilon}\)
on Task 5 vs dense action-informative mass on Task 6. A Dreamer-style
chain of RGB frames makes the difference visible; a sparkline of the
task coordinate stops the reader from having to squint at a 2 cm handle
drop.

## Figure

`results/img/paper/fig_task56_episode_strip.{pdf,png}`

Two rows, eight frames each, tabletop crop of the default Sawyer camera
(floor tiles removed so the fixture / cube are large enough at ICLR
width). Frame borders and the bar under each strip are HER phase
(far / hover / progress / success). Dots on the sparkline mark the
displayed frames. Shaded band is the environment success region
(handle \(z\in[0.05,0.09]\); push object–target \(\le 5\) cm).

| Row | Episode | Checkpoint | Why this snapshot |
|---|---|---|---|
| a | one eval episode of Task 5 | `task_0_step_150300.pkl`, seed 6 | 100.2k eval was 0; this is the audited first-success hunt (6 resets, return 97) |
| b | one eval episode of Task 6 | `task_0_step_200400.pkl`, seed 6 | first paced eval GIF on the same run (return 91). The probe table uses 250.5k; both are solved push snapshots |

Matched algorithm: DCC decomposed, reset actor, corrected wrapper, no
Success-BC, seed 6. Hunt up to 20 deterministic eval resets and keep the
first success so the latch is actually in the strip.

NPZ: `results/data/task56_episode_strips/{handle,push}_strip.npz`
plus `manifest.json`.

## What the pixels are supposed to show

Even sampling of a successful handle episode spends most frames *after*
the press: the arm reaches the fixture and stays. That is the occupancy
point, not a failure of the camera. Push frames should show the cube
travel. The sparkline is the scalar the critic would have to rank
actions *for*: handle \(z\) (step) vs object–target distance (ramp).

Do **not** read this figure as \(D_{\mathrm{TV}}\) or as \(\delta_\varepsilon\).
It is one on-policy eval episode per task, not the serialized Reverb
buffer (those were not saved).

## Launch

```bash
python tests/test_task56_episode_strips.py

# GPU node, EGL offscreen (cn270 / nvidia partition)
export MUJOCO_GL=egl
python scripts/record_task56_episode_strips.py

# re-compose from saved NPZ
python scripts/plot_task56_episode_strips.py
```

SLURM: `sbatch DRAFT_jubail_task56_episode_strips.sh`

## Caption (for the appendix; not pasted into the TeX)

**Occupancy, not scores.** One eval episode from matched seed-6 DCC
checkpoints. **(a)** Task 5 at the 150k first-success snapshot: handle
\(z\) is a latch (drops once around step 50, then sits in the 2 cm
success band for the rest of the horizon). **(b)** Task 6 at 200k:
object–target distance falls over most of the horizon. Frames include
the first hover / progress / success step and are otherwise spaced
across the episode, cropped to the table. Colour of the frame border
and of the bar under the strip is HER phase. Shaded band: success
region of the wrapper. This is the geometry behind sparse action
information on Task 5; it does not restate the probe table.

## Limitations

- One seed, one episode per task, on-policy eval (mode of \(\pi\)).
- Handle is shown at 150k rather than the 100k probe used in the
  numbers, because 100k had no successful eval episode. The 150k
  snapshot is the same run.
- Camera is the default Sawyer view; a 2 cm handle drop is small in
  RGB, which is why the sparkline is part of the figure.
- `--from-gifs` attaches the already-audited 320×240 GIFs (handle 150k
  first-success, push 200k paced) to a states-only re-roll when EGL is
  unavailable. Native EGL recording of the **same** checkpoints
  overwrites that fallback with 640×480 frames. Both paths are the
  same deterministic eval episodes (handle return 97 after 6 resets;
  push return 91 on the first reset).


