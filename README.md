# How far can EEG foundation models go when AI agents do the work?

**I supervised Claude agents through Track 2 (BCI decoding) of the NeurIPS 2026 EEG/EMG Foundation Challenge. This repository is the record of that experiment.**

**A one-page summary with charts: <https://liuyinyang1101.github.io/eeg-fm-agent-bci-ni26/>** (the same page is [`docs/index.html`](docs/index.html) in this repository).

For 48 days (2026-10-05 → 2026-11-21) I gave direction in plain language and Claude agents did the rest: they read the rules, wrote the code, launched the cluster jobs, logged each decision (in almost every case with the pass/fail rule written before the result), reviewed each other's work and built the submission files. The question I wanted answered: *how far can EEG foundation models (FMs) go on BCI decoding across sessions and across people?* The foundation model is ST-EEGFormer, which I co-authored.

**The result: 0.844 on the hidden (sealed) test set, 13th of 302 ranked entries. The leaders finished at 0.872, 0.867 and 0.865.**

That is not a win. I am publishing the record because I think it is more useful than the rank: what was tried, what each attempt measured, where the agents were wrong, and what my own steering changed. I cannot tell from one run whether the gap to the leaders comes from the foundation models, the agents or my guidance. My guess is my guidance. The material to argue about it is here.

> The rank is my reading of the final live leaderboard. The public leaderboard page lists only the top ten.

## Headline numbers

| | |
|---|---|
| Duration | 48 days, 2026-10-05 → 2026-11-21 |
| Final sealed score (cell-mean balanced accuracy, 3 classes) | **0.844** (13th of 302); best entry 0.872 |
| Sealed score path | 0.798 → 0.819 → 0.827 → 0.835 → **0.844** (a last extra upload scored 0.838) |
| Decisions logged for this track | 572 entries (T2-D99 … T2-D659), almost all with the pass/fail rule written before the result: 43 shipped in a package, 51 adopted and later superseded, 372 closed or killed, 5 withdrawn, 89 diagnostic, 12 never read |
| Code | 517 Python files (86,766 lines), 479 job scripts, 1,542 experiment-arm definitions, 112 test files, 16 submission templates |
| Commits touching this track | 2,122 (of 6,647 in the private challenge repository) |
| Compute | ≈ 2,400 H200 GPU-hours over 2,293 cluster jobs, plus one local RTX 5090 |
| Agent usage, whole challenge (all four tracks and the manager) | 84,228 model calls, 69.3 million output tokens, 28.3 billion tokens processed including cache reads |
| Agent usage, the Track 2 worker agents alone (a lower bound for the track) | 21,473 model calls, 20.7 million output tokens |
| Messages I typed | 725 across all sessions; 71 of them to the Track 2 worker agents |

How each number was counted, and its caveats, is in [`docs/STATS.md`](docs/STATS.md).

## The task

Decode which of three cued mental commands (kinesthetic motor imagery, mental calculation, word association) a person is performing, from 4-second EEG windows (41 channels, 120 Hz). The hidden test set mixes later sessions of ten people whose early sessions were released with labels and ten people never seen before. The model receives one window at a time, with no user or session identity, and may adapt to the stream as long as it uses no labels and nothing from the future.

## What the final system is

In short: an ensemble of six fine-tuned foundation-model networks and three families of small convolutional networks, with a correction step that uses the fact that every recording contains the three tasks equally often. The full description is in [`docs/METHOD.md`](docs/METHOD.md).

- **Six fine-tuned ST-EEGFormer-large members** (the foundation-model family, 35 % of the blend), with a continued-pretraining step on unlabelled in-domain data and an augmentation recipe that was the largest single gain of the last week.
- **Three convolutional families** (FBCNet, ShallowNet, IFNet), each with per-user copies for returning users and test-time views.
- **A small tree model on window means.**
- **Causal, label-free adaptation inside each stream:** a running class-balance correction (a causal Sinkhorn decode), in-stream refits of small heads and prototypes, and self-training students.
- It runs in about 2,800 s of the 3,600 s the organisers allow.

## Which models were better alone

On the one test setup where most models were scored alone (four people, early November), the fine-tuned foundation model was level with the best small convolutional network, or slightly ahead: FBCNet 0.742; one ST-EEGFormer-large network 0.739 after 30 epochs and 0.757 after 100; three of them voting 0.774. The forms differ (FBCNet is scored with its per-person copies, the ST-EEGFormer networks have none), and the foundation model's extra pre-training had seen unlabelled recordings of three of the four people. IFNet (0.731) and ShallowNet (0.709) followed. The other foundation models we tried were behind (MOR, an in-house model, 0.727 with three networks voting; REVE 0.584; CBraMod 0.488; LaBraM 0.484), and so were the classical pipelines (best 0.585). This is a reading of a decision log, not a benchmark: the test setups changed over the weeks, and most of the weak models got one configuration and little tuning. Tables, setups and caveats are in [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md).

## What I think we learned

This is one entry in one competition. Read these as observations, not findings. Each points to its record.

1. **The foundation model helped, but not enough.** In early November the convolutional-only fallback was about 0.036 behind the stack with FM members in our lab, and the largest single gain of the last week was a training-augmentation recipe for the FM members (T2-D557: +0.005 to +0.010 over three reads). A smaller ST-EEGFormer pre-trained from scratch on freely redistributable data only (about 19 GPU-hours) was 0.12 behind the published checkpoint as a member and unusable (T2-D356).
2. **The biggest gain came from a fact about the test, not from a model.** Every recording contains the three tasks equally often, and correcting the predictions as they come in so that they stay balanced was worth about 0.12 in our lab (T2-D319, T2-D445). Refitting the FM members' small output heads on our own predictions inside a recording helped a little (about +0.003, T2-D553); changing the network's weights without labels never helped where we tried it (T2-D113, T2-D555), and selecting among members per window lost to the plain blend (T2-D622).
3. **The features look good enough; what is missing is knowledge of the person.** A simple readout on frozen FM features, fitted causally on the same user's earlier windows with their true labels, beat our whole system by 0.040 on new users; the same readout fitted on our own pseudo-labels lost 0.064 (the first-principles diagnostics of Nov 19, T2-D622 to T2-D629).
4. **The one lab gain we could measure reached the hidden set at 37 to 48 % of its size (T2-D657).** The one candidate we shipped against a warning in our own numbers (better on new users, worse on returning users at single-model level) scored 0.006 lower on the hidden set than the package it was built on, after showing +0.0024 in the lab (T2-D652, T2-D659).
5. **The expensive mistakes were errors of judgement under time pressure, not bugs.** They are listed in [`docs/MILESTONES.md`](docs/MILESTONES.md) and in [`docs/LESSONS.md`](docs/LESSONS.md).

## How the experiment was run

- **I** set the goals, asked questions, and approved or refused what the agents proposed. I made every upload; the agents were never allowed to.
- **A manager agent** ran no experiments. It kept the plan and the list of open items, checked in every hour, read the leaderboard, passed findings between tracks and called the panels.
- **Worker agents, one per competition track.** Track 1 (decoding images from EEG), Track 2 (BCI decoding, the subject of this repository) and Track 3 (predicting sleep onset) each had their own Claude session and their own decision log. The Track 2 worker wrote and ran all the code here.
- **Helper agents** were short-lived: a worker or the manager started one for a single bounded job, such as reading a long log, checking a build or reviewing a change with fresh eyes. About 300 over the whole challenge.
- **Expert panels** were called for important decisions and code. Each worker wrote a short brief, a model from another company and agents with no memory of our plan were asked to attack it, and the manager wrote up the conclusion. 25 panel rounds and 140 written reviews over the whole challenge.
- **A review agent**, added for the last four days, recomputed the results independently and read each change before a file went out.
- **Rules the agents held themselves to** (not always kept; the lapses are in the log)**:** write the decision rule before the number exists; pair every comparison; score every candidate by loading the packaged submission through the organisers' kit; never use the label order or packaging regularities of the test set; say plainly when a claim is withdrawn.

## What is in this repository

| Path | What it is |
|---|---|
| [`docs/MILESTONES.md`](docs/MILESTONES.md) | The dated path: for each milestone, the prompt I typed, what the agents reasoned, and the outcome |
| [`docs/IDEAS_TRIED.md`](docs/IDEAS_TRIED.md) | Every idea tested, by theme, with its number and verdict (machine-readable index: `docs/ideas_index.json`) |
| [`docs/log/DECISIONS_T2.md`](docs/log/DECISIONS_T2.md) | The raw decision log (572 entries). Dense, written by the agents for themselves; the authority for every number |
| [`docs/METHOD.md`](docs/METHOD.md) | The method description of the final packages and the score history |
| [`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md) | Convolutional networks, classical pipelines and foundation models scored alone, grouped by test setup (machine-readable: `docs/model_comparison.json`) |
| [`docs/LESSONS.md`](docs/LESSONS.md) | The rules of thumb the agents wrote down, with the evidence behind each |
| [`docs/STATS.md`](docs/STATS.md) | How the headline numbers were counted |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | What the foundation-model checkpoints were pre-trained on, as far as our records allow |
| [`docs/OPEN_IDEAS.md`](docs/OPEN_IDEAS.md) | What was never tested, and what we would try next |
| `tracks/t2_bci/`, `src/ni26/`, `scripts/local/`, `tests/` | The code as it ran: training, benchmarking, packaging, submission templates, cluster launchers, tests |

No data, no trained weights and no private conversation logs are included. A few modules that the foundation-model scripts import from other parts of the private repository are not exported yet; [`docs/REPRODUCE.md`](docs/REPRODUCE.md) lists them. In this public copy, personal names, account names, other teams' names and submission ids, the names of unrelated projects, the cluster allocation's id and file-system roots were replaced by neutral placeholders, also inside quoted text and in the code. Passages about upload logistics, guesses about other teams' methods, and descriptions of an unpublished in-house model and an in-house dataset were cut or reworded, and that model's code is replaced by stubs. The experiments, their numbers and their verdicts are unchanged.

## What would you have done differently?

The competition is closed, but two questions are still open for me. Can an EEG foundation model do better on this task than we managed? And was this a good way to run AI agents on a research problem: one manager, one worker per track, panels for the big decisions, rules written before results, and me typing a few lines a day?

- **Propose a model idea:** open an issue using the idea template (hypothesis, mechanism, cheapest test). [`CONTRIBUTING.md`](CONTRIBUTING.md) explains what makes an idea testable here, and [`docs/OPEN_IDEAS.md`](docs/OPEN_IDEAS.md) lists where we would start.
- **Tell me what you would change in the agent set-up or the workflow:** the roles, the panels, the rules, the reviews, or how I supervised. There is an issue template for that too. [`docs/MILESTONES.md`](docs/MILESTONES.md) shows where the set-up helped and where it failed.
- **Try it yourself:** [`docs/REPRODUCE.md`](docs/REPRODUCE.md) says what you need (the organisers' kit and data, a published checkpoint, a GPU) and where the entry points are.
- I will test the proposals that look most promising when I have spare time and GPUs, and log what happens here in the same format, including when it fails.

## Licence

Code written for this project is released under the MIT licence ([`LICENSE`](LICENSE)); documents under CC BY 4.0. Files derived from other projects keep their own licences and are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).
