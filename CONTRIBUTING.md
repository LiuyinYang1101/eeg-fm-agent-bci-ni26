# Proposing an idea

This project stays open for two questions. Can EEG foundation models do better on this task than we managed? And how should AI agents be organised and supervised for work like this? I test the most interesting model proposals with my spare time and compute and record the outcome here, failures included. Ideas about the agent set-up and the workflow are just as welcome; they have their own section below.

## What makes an idea testable here

Open an issue with the "Idea" template and answer four things.

1. **Hypothesis.** One sentence: what will improve, for whom (new users, returning users, both), and by roughly how much.
2. **Mechanism.** Why it should work on this task: cued, class-balanced recordings of about 190 four-second windows per session, 41 channels, one window at a time, no labels at test time.
3. **Cheapest test.** The smallest experiment that could show it is wrong. Zero-GPU checks on stored outputs are welcome.
4. **What is already known.** Search [`docs/IDEAS_TRIED.md`](docs/IDEAS_TRIED.md) first. If something close was tried, say what is different.

## How ideas are judged

The same way the agents judged their own, as written down in [`docs/LESSONS.md`](docs/LESSONS.md):

- the decision rule is written before the number exists;
- comparisons are paired (same seeds, same folds);
- held-out people, not held-out windows, and the number of people behind every delta is stated;
- anything that uses stream history is also read under shuffled window orders;
- a result is reported with its probability of being noise, not just its sign.

## What will not be accepted

- Anything that uses the label order or packaging regularities of a test set.
- Data whose licence does not allow this use, or whose origin is not stated.

## Ideas about the agent set-up and the workflow

The experiment was also a test of a way of working: a manager agent, one worker agent per track, short-lived helper agents, a reviewer in the last days, expert panels for important decisions, and a human supervisor who typed goals and approvals. [`README.md`](README.md) describes it and [`docs/MILESTONES.md`](docs/MILESTONES.md) shows where it helped and where it failed.

Open an issue with the "Agent set-up or workflow" template and say:

1. **What you would change:** a role, a rule, the panels, the reviews, the way decisions were logged, or how I supervised.
2. **Where the record shows the problem:** a milestone, a lesson or a decision id, if you can point to one.
3. **What you expect it to change:** fewer wasted runs, earlier diagnosis, better use of the human's time, and how one could tell.

I cannot re-run the competition, so these will not be scored the way model ideas are. I will answer each one and try the ones I can in my next project.

## Code

Pull requests are welcome for fixes and for self-contained experiments. Note that the cluster launchers under `tracks/t2_bci/sofia/` are tied to the allocation the project ran on; see [`docs/REPRODUCE.md`](docs/REPRODUCE.md).
