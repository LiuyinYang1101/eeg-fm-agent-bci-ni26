# Running the code yourself

This is the code as it ran during the competition, not a polished library. It was written by agents for their own use, so expect long scripts and terse names. Paths that pointed into my machine were replaced with `~`.

## What you need

- **The organisers' kit and data.** The competition repository `neural-interfaces26/2026-competition` provides the evaluation kit and the instructions for obtaining the Track 2 data (the PROTEUS export). Nothing from the dataset is included here.
- **The public datasets used for continued pre-training** (Dreyer 2023, Scherer 2015) and for one member's pre-fine-tune (see [`PROVENANCE.md`](PROVENANCE.md)); the scripts fetch them through MOABB.
- **The published ST-EEGFormer-large checkpoint.**
- **Python 3.12 with PyTorch**, and the kit's own pins (`env/requirements_kit.txt`, `env/setup_local.sh`).
- **A GPU.** Fine-tuning the five replaced FM members of the last package took about 4.6 H200 GPU-hours; building and checking a package took about 5 more.

Not included:

- **Trained weights** of any kind.
- **The foundation-model adapter module and the ST-EEGFormer model code that the FM scripts import.** `t2p_fm.py`, `t2p_cpt.py` and `t2p_preft.py` import `t1_fm` and `fm/steeg_mae.py`, `fm/steeg_vit.py`, `fm/steeg_channels.json` from the Track 1 tree of the private repository, which is not part of this export. ST-EEGFormer's own code is available from its repository; the adapter module also holds the in-house MOR architecture and is not released yet; I have not decided whether it will be. Until then the FM scripts document what was done but do not run from this repository alone.
- **The in-house MOR model.** It is unpublished. Its classes in the submission templates and in `t2p_fm.py` / `t2p_preft.py` are stubs that raise `NotImplementedError`, its packaging and self-test scripts are left out, and imports of `inhouse_fm` do not resolve. The final packages do not use it.
- **The organisers' kit** (`benchmark_utils`), which the submission templates and replay checks import.

Placeholders in the code and logs: `$S1_ROOT` stands for the project directory of the cluster allocation the jobs ran on, `$FM_STORE_ROOT` for a data store of another project, and `inhouse_fm` (also `INHOUSE_SRC`, `INHOUSE_FM_SRC`) for the package of the in-house model. They replace the original names everywhere, so a script that used them needs your own values before it runs.

## Where things are

| Path | Role |
|---|---|
| `src/ni26/` | Shared library: the scorer replica (`metrics.py`), windowing, streaming, small baselines, the pure-PyTorch ST-EEGFormer forward (`fm/steeg_pt.py`) |
| `tracks/t2_bci/scripts/t2p_bench.py` | The benchmark harness: trains members on a fold, scores them under the lab protocol |
| `tracks/t2_bci/scripts/t2p_fm.py`, `t2p_cpt.py`, `t2p_preft.py` | Foundation-model members: fine-tuning, continued pre-training, pre-fine-tuning |
| `tracks/t2_bci/scripts/t2p_final.py` | Final fits on all released sessions |
| `tracks/t2_bci/scripts/t2p_repack.py` | Builds and re-configures submission packages, with their consistency checks |
| `tracks/t2_bci/submission/` | The submission templates (`t2p_submission_w21lp.py` is the final one) |
| `tracks/t2_bci/sofia/` | Cluster launchers and the arm definitions of every experiment, as submitted |
| `scripts/local/` | One-off analysis and read scripts, named after the decision they served (`t2_d642_*.py` belongs to T2-D642) |
| `tests/` | Script-style checks, run before each build |

To find the code behind a result, take its decision id from [`IDEAS_TRIED.md`](IDEAS_TRIED.md) or the log and search for it: most scripts and arm files carry the id in their name or header.

## What cannot be reproduced

- **The sealed scores.** The hidden test set is closed.
- **Bit-identical members.** Fine-tunes are not deterministic across hardware; our own retrains of the same recipe differed by about 0.0025 in lab score from draw to draw.
