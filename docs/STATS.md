# The numbers, and how each was counted

Every figure on the front page is listed here with its source and its caveat. The machine-readable copy is [`stats/stats.json`](../stats/stats.json).

## Result

| Quantity | Value | Source |
|---|---|---|
| Final sealed score (balanced accuracy averaged over subject × session × context cells) | **0.844** | The Codabench submission panel, as I saw it on 2026-11-22 |
| Final rank | **13th of 302 ranked entries** | My reading of the final live board; the public page shows only the top ten and reports 302 ranked entries |
| Best entries | 0.872, 0.867, 0.865 | Public leaderboard page, stamped 2026-11-21 23:27 UTC |
| Top-ten line | 0.844 | Same page (rows nine and ten both read 0.844) |
| Runtime of the final package on the organisers' worker | 2,795 s of a 3,600 s budget | Codabench panel |

### Sealed score history

| Upload | Date | Sealed score | Note |
|---|---|---|---|
| Three first-day packages | Nov 5 | 0.805 / 0.806 / 0.808, later 0.798 / 0.798 / 0.799 | The organisers re-ran every entry on Nov 10 |
| M7 | Nov 10 | 0.825, later 0.819 | Same re-run |
| W14-lite | Nov 15 | 0.827 | |
| W18 | Nov 17 | 0.835 | |
| **W25-ALL6** | Nov 21 | **0.844** | The final entry |
| W30 | Nov 21 | 0.838 | Uploaded on the last evening; below the final entry |

## Work

| Quantity | Value | How counted |
|---|---|---|
| Duration | 48 days | First commit 2026-10-05; sealed phase closed 2026-11-21 23:59 CET |
| Registered decision entries, this track | 572 | Table rows T2-D99 … T2-D659 in `docs/log/DECISIONS_T2.md` (2026-10-14 → 2026-11-22). The first nine days used a log shared by all tracks, which is not exported here; a side log of the FM agent adds seven entries (`docs/log/DECISIONS_T2FM.md`) |
| Outcome of those 572 entries | 43 shipped in a package; 51 adopted and later superseded; 302 closed; 70 killed early; 5 withdrawn; 89 descriptive or diagnostic; 12 launched or planned but never read | Classified by reading each entry; see [`IDEAS_TRIED.md`](IDEAS_TRIED.md) and `ideas_index.json`. The log itself is the authority |
| Entries catalogued in all | 603 | The 572 above, 24 Track 2 entries of the first nine days from the shared early log, and 7 from the FM side log |
| Commits touching this track | 2,122 | `git log` over the track's code, tests, scripts and decision log in the private challenge repository (6,647 commits in total, on 49 distinct days) |
| Python | 517 files, 86,766 lines | This export |
| Job scripts | 413 shell + 66 Slurm files, 18,297 lines | This export |
| Experiment-arm definitions | 1,542 files | `tracks/t2_bci/sofia/arms/`: one small text file per launched arm |
| Tests | 112 files | `tests/` (script-style checks, not unit-test functions) |
| Submission templates | 16 | `tracks/t2_bci/submission/` |

## Compute

| Quantity | Value | How counted |
|---|---|---|
| Cluster jobs for this track | 2,293 | Slurm accounting ledger, job names of this track, to 2026-11-20 |
| GPU-hours on NVIDIA H200 | ≈ 2,400 (2,395.6 in the ledger) | Same ledger, all job states (2,007 completed, 177 cancelled, 86 failed, 3 timed out). The last day's jobs (about 10 to 20 GPU-hours) are not in it |
| Local GPU | one RTX 5090, shared with other projects | Not metered. Used for lab reads and package checks, not for any shipped weight |

## Agent usage

Counted from the local session logs of the coding agent (Claude Code): each model call's reported usage, de-duplicated by message and request id.

| Scope | Model calls | Output tokens | Cache-read tokens | Cache-write tokens | Fresh input tokens |
|---|---|---|---|---|---|
| Whole challenge: all four tracks, the manager and review sessions, 23 sessions plus their sub-agents | 84,228 | 69.3 million | 27.80 billion | 0.40 billion | 0.25 million |
| Track 2 worker sessions only (five sessions across the 48 days) | 21,473 | 20.7 million | 7.97 billion | not separated | not separated |

Caveats:

- The Track 2 figure is a lower bound for the track. It leaves out the first project session (which began on Track 2 before the tracks were split), the manager sessions, and the review session, all of which spent part of their effort on Track 2.
- Most tokens are cache reads: an agent session re-reads its own context on every call. Output tokens are the better measure of work produced.
- Outside-review calls to another vendor's model are not counted.
- Models, by share of calls: mostly Claude Opus 5.5, with Claude Fable 5.1 and 5.2 in the last two weeks.

## The human side

| Quantity | Value | How counted |
|---|---|---|
| Messages I typed or pasted, all sessions | 725 | Session logs, human turns only (tool results, system notices and automatic check-ins removed by a filter; a few pasted briefs are included) |
| Of which in the Track 2 worker sessions | 71 | Same |
| Of which in the manager sessions | 466 | Same |
| Uploads | all made by me | By rule, the agents never uploaded |
