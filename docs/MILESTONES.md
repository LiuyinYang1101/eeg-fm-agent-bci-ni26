# Milestones: the path to 0.844

This page is the dated path of one human–AI experiment. I was the human supervisor: I steered several Claude sessions through Track 2 (BCI decoding from EEG) of the NeurIPS 2026 EEG/EMG Foundation Challenge, from 2026-10-05 to 2026-11-21:

- a **manager** session, which coordinated the tracks, kept an hourly log and asked for outside reviews;
- a **Track 2 worker** session, which ran the experiments and built every submission file;
- in the last four days, a **review** session, which read every change and re-scored every result independently.

The task was to classify three cued mental commands (motor imagery, mental calculation, word association) from 4-second EEG windows. The sealed test set held later sessions of the ten public users and ten users absent from the public data. The score is balanced accuracy averaged over user × session × context cells, and only a team's best sealed upload counts.

**Result: 0.844, 13th of 302 ranked entries. The leaders finished at 0.872, 0.867 and 0.865.** The rank is my reading of the final live board. The public page lists only the top ten: rows nine and ten also read 0.844, and our entry is just outside the list.

Each milestone below gives what happened, my prompt, what the agents did and reasoned, and the outcome. The mistakes are included.

## How to read this page

- **My prompts are verbatim**, typos included. "…" marks a cut. "[…]" marks removed private detail or attachments. Names of people, teams and unrelated projects inside quotes were replaced by neutral placeholders.
- **Quoted log text** is verbatim too. The logs call me "the user" or "the owner".
- **Approval lines** were drafted by the agents and entered by me in the worker's session. No build could be promoted without one. They are marked as such.
- **Times are local:** CEST (UTC+2) until 24 October, CET (UTC+1) after.
- **Ids.** T2-D557 is a row of the Track 2 decision log ([`log/DECISIONS_T2.md`](log/DECISIONS_T2.md)). Each row states its rule before its numbers exist. D1 to D98 are rows of the log that all tracks shared in the first nine days; it is not exported here. "Manager log 11-20 16:52" is an entry of the manager's hourly log, also not exported. A58 is an entry of [`LESSONS.md`](LESSONS.md).
- **Terms.**
  - *Strangers:* users absent from the training data. *Returning:* later sessions of users seen in training.
  - *Composite:* the registered weighted sum of the two (0.57 × strangers + 0.38 × returning, plus a small penalty term), built to mirror the make-up of the sealed set.
  - *Warm-up:* the open practice phase before the sealed phase.
  - *Gated:* a submission file that passed every packaging and replay check.
  - *FM:* foundation model. *ST-L* is ST-EEGFormer-large. *MOR* is an in-house foundation model.
  - Deltas such as +0.0052 are changes in the lab's balanced accuracy unless they are marked "sealed".

## Sealed scores

| Date | File | Sealed score | Note |
|---|---|---|---|
| 11-05 | three first-day files | 0.805 / 0.806 / 0.808 | 0.798 / 0.798 / 0.799 after the organisers re-ran the board on 11-10 |
| 11-10 | M7 | 0.825 | 0.819 after the same re-run |
| 11-15 | W14-lite | 0.827 | +0.008 over M7 |
| 11-17 | W18 | 0.835 | expected 0.839 (0.833 to 0.845) |
| 11-21 | **W25-ALL6** | **0.844** | forecast 0.844 to 0.854; the final score |
| 11-21 | W30 | 0.838 | the lab had said +0.0024 over W25-ALL6 |

---

## 1. 2026-10-05 — Project start: Track 2 first, on proxy data

I opened the first session with a pasted project brief and asked for a first night of baselines. The competition's own Track 2 recordings were not yet in hand, so the night ran on two public BCI datasets as proxies (D1 to D8).

**The prompt**

> **Me, 10-05 20:53 (first session):** "1 and 2 I did already, 3 is okay, for 4, I actually want to start with simpler models like eegnet to start with. However, I would like you to maybe carefully check the data specs, model training pipelines, […] …"

> **Me, 10-05 21:24 (first session):** "yes, and please proceed for tonight. Also, please perform the most basic data quality check and feature analysis that should be critical for us to understand the data we are working with"

**What the agents did.** Each question was written down with its pass bar before the runs. A cheap one-run screen ranked models like the full cross-session protocol (D1, Spearman ρ 0.95). No compact CNN won on both proxy datasets (D2: "NOT MET for any model"). The scored users' calibration session was worth +0.074 for the EEGNet control (D4: "the sealed design's calibration data is a major lever").

**Outcome.** A screening protocol and an EEGNet control. No model was adopted.

## 2. 2026-10-06 — Proxy tuning stopped

**The prompt**

> **Me, 10-06 10:27 (first session):** "okay, set up different folders for track 1 and 4, I think it's not worth trying to advance on proxy datasets. Please first download data, do data quality checks and also run simple baselines first"

**What the agents did.** The first session moved to the two tracks whose own data was available. The shared log has no Track 2 row between D8 (10-06) and D81 (10-13).

**Outcome.** Track 2 paused for a week. Later reads agreed with the call: labelled external data, added whole or per user, hurt (T2-D219, T2-D453), and only unlabelled continued pretraining on such data paid (manager log 11-18 14:52).

## 3. 2026-10-13 — The organisers replace the Track 2 dataset; restart in a dedicated worker session

The organisers opened a new warm-up phase on a new dataset (PROTEUS) with a new scoring protocol, and said that earlier scores were no longer comparable. I started a Track 2 worker session and pasted the announcement.

**The prompt**

> **Me, 10-13 13:59 (Track 2 worker):** "for our case, we need to advance our fm, with harnessing and finetuning, etc. You can safely skip some unnecessary steps, like comparing with random init model"

> **Me, 10-13 22:51 (Track 2 worker):** "how about euclidian allignment to solve inter subject and inter session variance drift"

**What the agents did.**
- The worker ran a first screen of 520 runs the same day (D81). FBCNet became the CNN base (D85: +0.115 over the EEGNet control, 9 of 10 participants).
- It registered a causal output-prior correction (D83) and confirmed it on fresh seeds (D85). The correction divides each window's class probabilities by the session's running mean prediction and uses no labels.
- The random-init control arm of the FM screen was cancelled on my line (D82).
- I also told the worker to develop on the competition's own split only. D88 records it: the leave-one-subject-out protocols were dropped and 33 running jobs cancelled.
- Euclidean alignment did not help FBCNet (D93).
- Every first attempt to combine an FM with the CNN base lost (D90). The first that added was ST-L after continued pretraining, as a 4:1 probability ensemble with FBCNet: +0.025 on fresh seeds, 6 of 6 (D95, 10-14).

**Outcome.** A base recipe within a day and a half: FBCNet, ST-L and the output-prior correction, developed on the starter kit's split.

## 4. 2026-10-14/15 — A manager session, an FM-first direction, and the first warm-up score (0.549)

**The prompt**

> **Me, 10-14 09:07 (manager):** "I have three shells named:track1-image, track2-bci-decoding, and track3-sleep running on this competition's 3 different tracks, I want you to act like a project lead that can learn from each's progress and findings and come up with global valuable directions and summarize important lessons and you can improve the efficiency or help each track"

> **Me, 10-15 17:27 (manager):** "I feel at this moment, we should prioritize our tasks on higher level exploration, rather than small things like multi seed, ensembles, etc, those should be left in the end. For me high level and more interesting and important questions should be around our own foundation models: steegformer and the new […] model. …"

Three earlier directions, typed in the first session while it worked on another track, were carried to every track:

> **Me, 10-08 10:25 (first session):** "did you use lora to finetune fms? We have found that lora is necessary on small size datasets"

> **Me, 10-08 12:15 (first session):** "… of course, we should always start from simple stuff"

> **Me, 10-08 17:02 (first session):** "… I would like to develope on top of the SOTA model"

**What the agents did.** The manager started its cross-track log and turned the direction into an FM research programme (manager log, 10-15). On Track 2 this produced rows T2-D126 to T2-D134 and several later ones.
- Every continued or new pretraining of the MOR lost to the weights already in hand (T2-D123, T2-D129, T2-D141, T2-D144, T2-D151).
- A bounded LoRA screen (T2-D130) left one pass: role-specific LoRA on the MOR. That member entered the package at weight 0.1 (T2-D137, T2-D140).
- A second CNN family with a different inductive bias added +0.0145, 7 of 7 seeds (T2-D116).

**Outcome.** The three-family package (FBCNet 0.6, ShallowNet 0.2, ST-L 0.2) was the first warm-up upload on 10-15. It scored 0.549, equal to the local replay to within one window of 755.

## 5. 2026-10-19 — "I don't want any track stops working"

**The prompt**

> **Me, 10-19 13:17 (manager):** "I don't want any track stops working, I believe there are always new ideas and directions to try out. Can you make them start busy"

> **Me, 10-15 10:52 (manager), on Track 2:** "… HOw about subject specific LoRA module to increase the capacity to capture subject specific features? …"

**What the agents did.** A standing rule followed: each track keeps a backlog of at least three items at the top of its decision log and checks in hourly. Track 2 got a budget of 200 GPU-hours. In the next five days the worker registered about ninety rows (T2-D144 to T2-D234), and most closed.

The line that was adopted answered my subject-capacity question in another form. An oracle per-user fine-tune was worth +0.044 on returning users' cells (T2-D111). So each CNN family got per-user fine-tuned copies, chosen per stream by a fingerprint router that needs no user id (T2-D124c). Copies of both families together read +0.0141, 12 of 12 seed pairs (T2-D159, T2-D160).

**Outcome.** The late-October package: FBCNet 0.5, ShallowNet 0.2, ST-L 0.2, MOR 0.1, with per-user copies and the output-prior correction.

## 6. 2026-10-26 — The warm-up board stops meaning anything

**The prompt**

> **Me, 10-26 14:25 (manager):** "I think the warm-up score doesn't make sense …"

**What the agents did.** The manager recorded that "the warm-up board is not a ranking signal" (manager log 10-26 14:53). Warm-up uploads were kept for one purpose: to check that a file runs on the organisers' worker and reproduces the local replay.

**Outcome.** From then on the warm-up phase was a packaging check, not a score to select on.

## 7. 2026-10-27 — A causal class-balance decoder (SK10); warm-up scores of 0.57

**The prompt**

> **Me, 10-27 13:58 (Track 2 worker; approval line):** "I approve shipping SK10 (causal Sinkhorn, 10 iterations) as T2's decoder; QI is not shipped. Disclosure: it assumes only within-session class balance, as the shipped prior does; state it in the method write-up. …"

**What the agents did.** From the public label schedule the worker registered count-aware decoders (T2-D285).
- SK10 takes the package's probabilities for windows 1 to t of a session stream and rebalances them ten times, so that each class holds equal total mass. It then outputs the row for window t. One iteration is the earlier prior correction. It uses no labels and no future windows.
- The read was +0.0071 and +0.0068 on two independent seed sets, with 12 of 12 pairs up in both.
- A second decoder (QI) used the fixed 10 / 10 / 10 schedule of the online runs and also passed. Whether to ship it was left to me, and I declined.
- On the organisers' worker the SK10 file scored 0.574, against 0.571 for the same members with the prior correction. That is +0.0035 where the replay had said +0.009.
- On 10-30 a forgetting factor λ 0.95 added +0.0042 locally (T2-D304) and +0.003 to +0.004 on the worker (0.574 and 0.578).

**Outcome.** Eight warm-up files were scored between 10-15 and 10-30 (W1, W2, W5 to W10; W3 and W4 were never uploaded), from 0.549 to 0.578. The decoder shipped in every later file.

## 8. 2026-10-27/28 — Every session killed at once; restart from the repository

At 19:09 on 10-27 the operating system's out-of-memory daemon killed the terminal that held the manager and all three track sessions (manager log 10-28).

**The prompt**

> **Me, 10-28 09:00 (new manager session):** "you are the overall competition manager responsible for track 1,2,3 communication, arrangement, and report to me big things. Our previous manager and each track's worker ended due to crash oom, now please give me prompts for each track's worker to catch up and continue the work"

> **Me, 10-28 09:13 (manager):** "I think it's a bad idea to build all uploads for the final phase at this moment: to me the best strategy would be having two or three initial best uploads prepared in the begining and then keep iterating for improvements: we can use more training data and still improve our solutions with tricks and new directions."

**What the agents did.** The log's account: the sessions' memory had grown to 24.2, 14.1, 8.4 and 4.5 GB, and a local data read started 36 seconds earlier tipped the machine over. About 14 hours were lost. The dead sessions had left no handover, so the new manager wrote catch-up prompts from the repository, the cluster's job records and the compute ledger. It added hygiene rules: lean sessions, and heavy reads one at a time under a memory cap. My strategy line was recorded verbatim, and the pre-building of later uploads stopped.

**Outcome.** All sessions were running again on 10-28. The later restarts (10-31; 11-10 for a model update; 11-16 for a reboot) used written worker prompts or handover files.

## 9. 2026-10-29 to 11-01 — "ship our MOR if it's valuable or the best"; more compute

**The prompt**

> **Me, 10-29 09:57 (manager):** "measure, I just want to ship our MOR if it's valuable or the best"

> **Me, 10-31 13:17 (manager):** "I checked the budget usage, we still have many credits leeft, try to lift up the gpu boundary and seek heavily for more improvements"

**What the agents did.** The manager asked each track for a paired read with and without the MOR on its candidates (manager log 10-29 09:58). Track 2 built a MOR-free template and had both variants verified on the warm-up worker. The manager offered each track a further 1,000 GPU-hours inside a pool cap (manager log 10-31 13:19).

**Outcome.** On 11-05 the file with the MOR scored 0.806 and the one without it 0.805. The pre-set rule needed +0.004, so the MOR was left out of every later file (T2-D321; manager log 11-05 11:34).

## 10. 2026-11-05 — The sealed phase opens: 0.805 to 0.808, and a surprise

Three files went up on the first day and scored 0.805, 0.806 and 0.808. At 09:30 the team was sixth and the leader stood at 0.83.

**The prompt**

> **Me, 11-05 09:43 (manager):** "for T2, I would argue that in our development phase, the val is below 0.6, but the sealed phase is 0.8, so this difference may indicate that we can't draw the same conclusions just from our old numbers."

> **Me, 11-05 13:35 (manager):** "I think we should really rely on our local eval and only upload for real huge improvements"

**What the agents did.**
- T2-D319 answered the first question within the hour, from stored predictions. The warm-up test set had 8 cells, and 6 of them came from one participant, whose cells averaged 0.588. The other users' 16 cells averaged 0.800, so "0.81 is in line with the non-P04 users". Earlier reads "were diluted by P04". Selection moved to a typical-user read with a guard on that participant.
- The next upload was held. An early upload now needed a local gain of +0.025 on fresh seeds (T2-D331).

**Outcome.** The three first-day files scored within 0.003 of each other. The next upload was held for a large local gain.

## 11. 2026-11-05 to 11-09 — Outlier ideas, a stack that did not replicate, and a turn to risk

**The prompt**

> **Me, 11-05 20:07 (manager):** "I think we need some out of box (outlier) ideas to get huge improvements. Therefore, I think it's better to have a few creative and high risk high gain ideas tested andplease also schedule some panel discussion regularly between tracks time to time to motivate and share info."

> **Me, 11-06 07:23 (manager):** "the two-stage ST-EEGFormer recipe gives about +0.0005; the specialist ST-EEGFormer model gives about +0.001; the extra in-house model variants give up to about +0.001. Three things together reach 0.003"

> **Me, 11-09 15:42 (manager):** "launch a fable review on this matter and give a better strategies, especially when we are lagging behind, I think we need more risky plans rather than playing safe"

**What the agents did.**
- Daily cross-track panels and one high-risk bet per track started that evening (manager log 11-05 20:35).
- The three-part stack was registered and read at +0.0027 with 8 of 12 pairs up, just under its bar of +0.003 and 9 of 12. Its MOR-free form read +0.0022 (T2-D373). I chose a fresh-seed repeat of that form. It read +0.0005: "D373's +0.0022 was mostly the winner's curse" (T2-D376; A26).
- What passed in these days sat on the FM side: more ST-L members and a larger FM share (T2-D326, T2-D331). LoRA on ST-L failed (T2-D329: −0.019, 0 of 12).
- On 11-06 I had limited uploads to candidates with a chance at the top two. After the risk review of 11-09 that rule was replaced, and the next upload went to the largest stack, M7 (manager log 11-09 16:32).

**Outcome.** M7 was built: 15 networks per CNN family and five ST-L members.

## 12. 2026-11-10 — New sessions on a newer model; M7 scores 0.825, then 0.819 after the organisers' re-run

**The prompt**

> **Me, 11-10 11:36 (manager):** "I want to restart all claude session to update and use the latest model, is it safe to do it? any handover instructions?"

> **Me, 11-10 11:57 (new Track 2 worker):** "your are the track 2 worker, please check the latest status of the competition track 2 and critically think about our plan, we aim for top 3 and now is lagging behind, which means we need something different"

> **Me, 11-10 16:01 (manager):** "track2's results are surprisingly well[…]"

**What the agents did.**
- The new manager read the whole board first. Forty of the top 50 entries sat between 0.777 and 0.811, and our three files were inside one draw's standard deviation of each other (manager log 11-10 12:44). That morning it rated Track 2 "under 10 %, commodity level".
- M7 was uploaded at 14:44 and scored 0.825, sixth, in 2,250 s. The log says the morning call "was WRONG". It re-estimated the lab-to-sealed transfer at 0.66 ± 0.1, where 0.35 had been assumed (manager log 11-10 16:44).
- That evening the organisers re-ran the whole board, and the median entry fell by 0.006 (T2-D445; manager log 11-11 07:27). M7 became 0.819 and the three earlier files 0.798, 0.798 and 0.799.
- On 11-11 the manager found two changes in the kit repository: the re-run was an official backend fix, and the FAQ no longer promised chronological order inside a run (manager log 11-11 11:44).

**Outcome.** 0.819, fifth on the morning of 11-11.

## 13. 2026-11-11 to 11-13 — "think reversely"; the delivery-order question; a candidate held back

**The prompt**

> **Me, 11-11 12:17 (Track 2 worker):** "I think we need to think reversely, why our latest submission showed a huge improvements and what does this tell us about the sealed test data and how we can optimize and make further improvements"

> **Me, 11-12 17:57 (Track 2 worker):** "1. go 2. not go 3. clean useless files"

> **Me, 11-13 13:59 (Track 2 worker):** "I don't want to waste on 0.822, which is not so useful since it can't help us land on the top 3. …"

**What the agents did.**
- The reverse read found that about 80 % of M7's step over the earlier files was on the FM side: five ST-L members, a larger FM share and longer training (T2-D450).
- The order question: the decoder's forgetting factor assumes that windows arrive in time order. T2-D448 fitted the re-run drops, and the worker wrote down its prior (blocks intact: 0.80).
- A harder bet on order, balancing inside each run, lost even with the true run boundaries: −0.0064, 1 of 12 (T2-D449).
- One lever passed: a fixed start vector for the decoder, +0.0042 with 12 of 12 pairs on a third, fresh draw set (T2-D459). It gave the candidate M7-B3, expected at 0.821 to 0.822 and gated on 11-12.
- My three lines: "go" approved the start-vector template, "not go" turned down the worker's second proposal, and the 0.822 line held M7-B3 back.

**Outcome.** Nothing was uploaded from 11-11 to 11-14. The delivery order of the sealed set was never measured, and every later file kept λ 0.95.

## 14. 2026-11-13/14 — First principles: the reference montage

**The prompt**

> **Me, 11-13 13:59 (Track 2 worker; the same message):** "… And how about the groundtruth neurophisiology features? can you show the spectrotemporal features in topoplots as you did earlier on the open datasets […]? Can we start from the foundamentals and apply first principles"

> **Me, 11-13 16:07 (Track 2 worker):** "do the test"

> **Me, 11-14 08:37 (Track 2 worker; approval line):** "T2: template change approved: per-member input reference (W14, ref: car | lap; D475e / D480), 5 + 3; runtime bar 52 ms; build M7-W14"

**What the agents did.**
- The worker's first read compared the three tasks with each other in the delivered reference and found no clear motor signature. I doubted it, and the row was "VERIFIED AND CORRECTED after the user's doubt" (T2-D473).
- The corrected read used pre-cue rest as the baseline and a common average reference (CAR). Motor imagery then shows the expected bilateral sensorimotor desynchronisation, larger than for the two mental tasks (beta at C3: −1.90 dB in 10 of 10 users, against −1.24 and −1.20). The first read had cancelled the shared effect by contrasting tasks, and "the delivered reference masks it" (manager log 11-13 14:44).
- Tested as model input (T2-D475), CAR did not help FBCNet (−0.0025): its "learned spatial filters already re-reference its input". Three CAR-input ST-L members added to five plain ones gave +0.0052. A surface-Laplacian view failed (T2-D480).
- In the same days a fresh-context reviewer agent proposed decoding each model family before pooling: +0.0060, 11 of 12, replicated on a fresh draw set (T2-D476).

**Outcome.** Template W14: M7's line, the start vector, per-family decoding and three CAR members.

## 15. 2026-11-15 — W14-lite scores 0.827; "anything that can increase a little bit"; the window mean

**The prompt**

> **Me, 11-15 09:56 (manager):** "at this point, I feel anything that can increase a little bit should be used, since we are doing a competition, and with many small changes theremight be bigger improvements"

> **Me, 11-15 13:42 (manager):** "I also want to ask: have you tested using some libraries like TSFEL to extract millions of possible time-series, and eeg features and use a tree model to identify useful ones and test more classic ML approaches? …"

> **Me, 11-15 15:22 (manager):** "isn't it counter intuitive that you said our model may not aware of mean/abs, max, min in each window, yet you are demeaning again?"

**What the agents did.**
- W14-lite, a member subset of W14 chosen for runtime (T2-D496), was uploaded at 11:27 and scored 0.827. That is +0.008 over M7, about half of the local sum (manager log 11-15 13:44).
- My first line became a rule: sub-bar positives enter one stack, the stack is read on fresh seeds, and a leave-one-out column decides which parts stay (T2-D512). In-stream adaptation stayed (+0.0033). It refits per-stream copies of two CNN families inside `predict` on the package's own decoded labels.
- The feature sweep ran the same afternoon. Boosted trees on 16,454 library features added +0.0054 at a 10 % share (T2-D515).
- My de-meaning question then located the source. On de-meaned windows the gain vanished: it "lived in the per-channel window MEAN that every member's hard-wired _demean removes (never read as an arm)" (manager log 11-15 15:44).
- The window means became a small tree family (+0.0073, T2-D516) and a side input to two CNN families (+0.0074, T2-D517).
- The same day the manager's first answer about last year's challenge solutions rested on web summaries, and I rejected it. The manager then read the code. Four ideas taken from it were tested and closed (T2-D518 to T2-D521).

**Outcome.** W17 was gated by 20:44. The local ledger stood at +0.0274 over W14-lite's form, with an expected sealed score of about 0.835 (manager log 11-15 16:47).

## 16. 2026-11-16/17 — Closed rows re-read; W18 scores 0.835; final sprint on Track 2

**The prompt**

> **Me, 11-16 04:24 (manager):** "remember all little improvements should count. Now also revisit those closed ideas that do bring improvements"

> **Me, 11-16 15:02 (manager):** "for track 2, since we are still improving, do we still need to upload the one we planned? …"

> **Me, 11-17 16:32 (manager):** "if in the end we have not much hope on track 3, use all resources on track 2 to do a final sprint"

**What the agents did.**
- Within the hour each track listed its closed rows that had shown a positive delta. Track 2 had eight, none of them ever read on the current stack (manager log 11-16 04:44).
- Two survived a fresh-draw stack read with leave-one-out: a window-mean side head on the FM members (T2-D524) and one motor-imagery-expert ST-L member (T2-D529). Together they read +0.0044 over W17, 3 of 3 draws (T2-D532).
- On my upload question the manager recommended holding W17: "W18 contains W17" (manager log 11-16 15:52).
- The build exposed two weak checks. A rehearsal failed its parts check for a legitimate reason, and the check was changed after the failure. The manager therefore set three conditions before any build output counted. One of the manager's own new gates turned out to be vacuous and was replaced before its numbers (manager log 11-16 11:52, 15:52).
- The machine restarted twice that day. The first restart cost the sessions their timers and about 25 minutes of local GPU work. The second was at my request, with handover files (manager log 11-16 10:44, 18:24).

**Outcome.** W18 was uploaded on 11-17 at 09:08 and scored 0.835 in 2,782.57 s, against an expectation of 0.839 (0.833 to 0.845). The local GPU was given to Track 2 (manager log 11-17 12:52, 13:52).

## 17. 2026-11-17/18 — Old augmentations on the FM members; W20 and W21

**The prompt**

> **Me, 11-17 15:02 (manager):** "what about track 2, any insights from new exp? Maybe the data aug or other findings can transfer to our old recipe?"

> **Me, 11-17 19:58 (manager):** "but I don't want to uplaod tomorrow. I still want to improve and I believe we still could"

> **Me, 11-18 11:27 (manager):** "I don't like that all tracks are idle: we should take advantage of the compute and time and making more improvements"

**What the agents did.**
- The worker listed the augmentations that had been small positives on October's stacks and "never re-read since": gain jitter with a time shift, channel dropout and mixup. T2-D557 applied them together to the FM members. The reads were +0.0052 ("the largest single item in days"), +0.0084 on fresh fits and +0.0101 on top of W20, each on 3 of 3 draws.
- W20 itself added per-stream class prototypes on the FM features (T2-D560, +0.0012 on fresh fits). W20 with T2-D557 was gated at 06:52 on 11-18.
- The worker and the manager then both recommended stopping, and the log reads "both tracks idle by design" (manager log 11-18 09:52).
- After my line the sprint reopened. Label propagation over the FM features, as the teacher for the CNN-side refits, passed on three sets (+0.0025, +0.0022, +0.0016; T2-D569). It gave W21, gated at 14:52.
- An outside review that I requested (GPT-6-astra) named two gaps between the lab read and the package. Both checks came out in the default's favour (T2-D570, T2-D571).
- From 11-18 the review session reviewed every Track 2 change and re-scored every read.

**Outcome.** W21 was the default candidate, forecast at 0.844 with half transfer and 0.853 with full transfer (T2-D573; manager log 11-18 18:52).

## 18. 2026-11-19 — A first-principles panel finds that the lab did not train like the shipped build

**The prompt**

> **Me, 11-19 10:59 (manager):** "what are the most promising way towards 0.85? Why do you think the other teams could reach the number and why we can't"

> **Me, 11-19 11:06 (manager):** "can you launch fable 5.2 and gpt 6 to discuss and apply the first principle to see what else worth trying"

> **Me, 11-19 18:28 (manager):** "well, I would like to aim high and target at the first 0.859's score, and I give you permissions to prioritize this so you can use all resources and make efforts towards it"

**What the agents did.**
- That morning an audit I had asked for found that a decoder change the manager had recommended for the next build (T2-D603) reversed its sign under the two plausible delivery orders. The recommendation was withdrawn and the pre-build cancelled (manager log 11-19 10:56; A53).
- The panel's one new finding: the lab panels trained on one session per user, while the shipped build trained on all 14 public sessions, where two three-session users hold 43 % of the windows (T2-D630). Balancing the CNN training alone closed at +0.0017.
- The follow-up reads showed something worse. In the shipped composition the per-user copy router sent 8 of 8 held-out users to one three-session user's copy, and routing cost them −0.021 where the lab had measured −0.007 (T2-D630, T2-D631; A54).
- The fix read +0.0092 on strangers, but +0.00388 on the composite against a bar of +0.004, so it closed by its rule (T2-D632). I chose to build it anyway as an exception (W23).
- A full-pipeline 2 × 2 then showed that W23's gain was about zero through the whole stack: "the FM absorbs the CNN composition gain" (T2-D638; manager log 11-19 20:52).
- That pointed at the FM members themselves. T2-D642, user-balanced training of all six FM members, was registered at 18:47 after my all-resources line. An earlier "not feasible" had assumed that the chain must run serially.

**Outcome.** W21 stayed the default, and T2-D642 ran overnight. The manager's odds that evening: top five 40 to 50 %, top three 20 to 30 %, first place 2 to 5 % (manager log 11-19 20:52).

## 19. 2026-11-20 — User-balanced FM members give W25-ALL6; a wrong diagnosis is withdrawn; the runtime budget is re-read

**The prompt**

> **Me, 11-20 14:01 (manager):** "but we still have like 12 hours and around 24 hours before the ddl. Why can't we keep trying? I want you keep trying"

> **Me, 11-20 16:30 (manager):** "now, any small gain is plausible and should be included"

> **Me, 11-20 22:39 (manager):** "I don't want to submit anything today, I want our full confidence and possible improvements confirmed tomorrow and then upload"

**What the agents did.**

*The candidate.* T2-D642 retrained all six FM members with loss weights that make every user count equally. Its first full-chain read gave +0.0071 on the composite. Two replicates at fresh seeds gave +0.0030 and −0.0017: the discovery read was biased up (A56). A fourth read was run so that the decision would rest on unselected evidence only (T2-D648): strangers +0.0030, composite +0.0003. By the pinned rule W25-ALL6 "stays recommended", "but the expected sealed gain over W21 is ≈ zero" (manager log 11-20 13:52). A half dose kept about 90 % of the stranger gain and removed only about 20 % of the cost to the two heavy users (T2-D647; A55).

*The wrong diagnosis.* After my 14:01 line the manager moved the hard stop and launched four more items. One was W30: W25's members plus a virtual-participants augmentation (T2-D652). The manager justified them with a diagnosis: the "returning" term of the composite measured only the two three-session users, so a family of changes had been closed on a biased proxy. The review session derived a re-weighted composite, sealed\*, and rules were re-pinned on it (manager log 11-20 14:52). An outside reviewer asked for one check. The review session opened the fold definition and found that the term already trains those two users on a single session, which is the case it was said to miss. At 16:52 the log reads that the diagnosis "was WRONG and sealed\* is WITHDRAWN", before any number depended on it. The cost was about 2.5 hours of rule changes and about 150 GPU-hours of arms that became descriptive (A58).

*The inclusion rule.* My 16:30 line became a rule: a part is included if its two-read means, and its second read alone, are positive on strangers and on the composite (manager log 11-20 16:52). That rule carries no separate guard for returning users (T2-D659).

*Runtime and capacity.* A re-read of the kit's newest commit showed that the execution budget is 3,600 s. An earlier note that "the real worker limit was 80 min" had read a task timeout as slack (A59). By 19:54 none of the project's 110 queued tasks was running on the cluster. Among them was a test-time multi-view read of the FM members, from an idea of mine (T2-D654).

**Outcome.** I entered the W25-ALL6 approval line at about 18:40, and the file was promoted at 19:17 after a fresh review. Nothing was uploaded on 11-20.

## 20. 2026-11-21, morning — Nothing ran overnight; "why not testing w30 now?"

**The prompt**

> **Me, 11-21 10:51 (manager):** "why not testing w30 now?"

> **Me, 11-21 10:52 (manager):** "in the meantime test w30, I will upload w25"

**What the agents did.**
- The cluster ran none of the project's tasks all night. The 05:00 cut-off fired as registered and cancelled 20 job arrays unread: 6 of W30 and 14 of the multi-view read (manager log 11-21 05:52).
- The hourly log reads "Tracks idle by design" at 06:52, 08:52 and 09:52. A few GPUs had come free shortly after the cut-off (manager log 11-21 10:59).
- On my line the manager reopened W30, but only as an information read with "no donors, no build". It believed that a gated file by the evening was out of reach (A62).
- At 12:37 the member gate passed on strangers (+0.0106, positive on all five member slots). The single-session returning read was negative on all five (−0.0217) (manager log 11-21 12:52).

**Outcome.** W30's read was running, and W25-ALL6 stayed the file to upload.

## 21. 2026-11-21, 15:27 — W25-ALL6 uploaded: 0.844; the board had moved

**The prompt**

> **Me, 11-21 15:32 (manager):** "I just submitted the w25"

**What the agents did.**
- The public leaderboard page had not changed since 11-19, and the scripted board read was blocked. A live panel pasted by me at 15:35 showed the top-three line at 0.862, up from 0.848 on the stale page. The log notes that "the earlier 25 to 30 % odds are void" (manager log 11-21 15:52; A65).
- The score arrived at 17:19: 0.844 in 2,795.3 s, row 10 of the board, +0.009 over W18. The manager's note on the board snapshot calls it "the bottom edge of the forecast 0.844 to 0.854 (about half of the lab gain transferred)" (snapshot of 11-21 17:19).
- The runtime was 13.8 % below the line used for the timing gate.

**Outcome.** 0.844, the final score.

## 22. 2026-11-21, evening — Two risky options closed by their own rules; W30 through a one-hop build

**The prompt**

> **Me, 11-21 16:29 (manager):** "any risky attempt at this moment: e.g., assuming the trials are continuous and follow order"

> **Me, 11-21 18:48 (manager):** "high risk high gain, we need a risky solution"

> **Me, 11-21 21:59 (Track 2 worker; approval line):** "T2 W30 approved: W30 = W25-ALL6 with the five headed FM members replaced by the D652 vp-2 user-balanced all-user donors …"

**What the agents did.**

*The order hedge.* The manager answered that the shipped decoder already bets on chronological order. It registered the opposite hedge before any number (T2-D656, 16:37): a config-only variant with a larger λ, to ship only if it gained +0.002 under a session shuffle. Larger λ lost under every order (−0.0066 and −0.0069 under the shuffle), so no hedge file was built (17:17). A sentence in the log that the shipped λ "is not hurt by a decode-only session shuffle" was withdrawn 40 minutes later: levels are not comparable across orders, only paired contrasts are (manager log 11-21 17:58; A63).

*The build path.* By 15:58 W30 had passed the inclusion rule on two full-chain reads (strangers +0.0060, returning −0.0009, composite +0.0024). The manager had accepted that it could not ship, because it needed five new fits plus a five-hour build chain. Asked again, the worker and the review session established independently that the package was one repack of the W25-ALL6 file with the new members, about three hours in all. The manager logged its earlier acceptance as wrong and launched the fits at 17:31 (manager log 11-21 17:52; A62).

*The gamble.* T2-D658 raised the FM share from 0.35 to 0.55. It was registered at 18:50 with a harm bound: it would be offered only if its composite against plain W30 was at least −0.003. It read −0.0035, with per-user changes from +0.026 to −0.042: "the spread the gamble wanted, around a mean below the line". It was not offered (manager log 11-21 19:52). The manager had put the chance of reaching 0.85 at about 5 % (manager log 11-21 18:52).

*The build.* The one-hop build job failed at 20:14 on a parts assertion, after it had written the file. Both sessions traced the failure to the check and not to the file: the assertion compares against a static reference that lacks the in-stream FM refits, so it cannot pass on this base. The file was pinned by its hash. By 20:54 it had passed the kit replay, the package check, a paired timing run, the streaming check and a structural comparison (manager log 11-21 20:52, 21:52).

**Outcome.** I entered the W30 line at 21:59 and uploaded the file at 22:10.

## 23. 2026-11-22 — W30 scores 0.838; post-mortem

**The prompt**

> **Me, 11-22 12:03 (manager):** "[…] it's done, w30 didn't give us improvvements"

**What the agents did.**
- W30 came in at 0.838, 0.006 below W25-ALL6, where the lab had said +0.0024. The manager's estimate (about 60 % to beat 0.844, spread 0.003) was, in its own log, "too optimistic in mean and too narrow in spread" (manager log 11-22 12:04).
- The worker's numbers-only post-mortem (T2-D659): the composite had priced the returning group with a full-chain read on two people (−0.0009), while the member-level reads on every training user had said −0.0206. The same arithmetic with the member-level term gives −0.0050, against the sealed −0.006. The row labels this "Post hoc, one data point".
- The same recipe had been closed three days earlier by a returning guard (T2-D582). The inclusion rule of 11-20 had no such guard.
- A caveat stands. With one upload, a real cost to returning users cannot be separated from the fit variance of five freshly trained members.
- The review session's retrospective lists what review caught and what it missed. It notes that "The exact re-scores never caught anything": every catch came from reading a registration or the code. Lessons A62 to A66 record the day.

**Outcome.** The final score is 0.844. Under best-of scoring W30's miss cost nothing on the board. The final public page shows 302 ranked entries, the leaders at 0.872 / 0.867 / 0.865, and rows nine and ten at 0.844.

---

## What my steering changed

Cases where a prompt reversed or redirected the agents, with the result.

| Date | My line, in short | What changed | Result |
|---|---|---|---|
| 10-06 | proxy datasets are not worth advancing | Track 2 paused until the real data arrived | Later reads agreed that labelled external data hurts (T2-D219, T2-D453) |
| 10-13 | skip random-init controls; develop on the competition's split only | one arm and 33 jobs cancelled (D82, D88) | Faster first days. The kit split's test cells later proved to be dominated by one participant (T2-D319) |
| 10-15 | research the foundation models first, ensembles last | an FM programme (T2-D126 to T2-D134 and later rows) | Every MOR pretraining arm lost. ST-L became the FM family and held 0.35 of the final blend. The last two sealed steps came from FM-side changes (T2-D557, T2-D642) |
| 10-28 | do not pre-build every upload | later draws cancelled | The freed compute went to new experiments (T2-D291) |
| 10-29 | ship the MOR only if it is valuable | a MOR-free template built beside the one with the MOR | 0.806 with, 0.805 without on 11-05: the MOR was left out |
| 11-05 | sealed 0.8 against development 0.6: old conclusions may not hold | T2-D319; selection moved to a typical-user read | The 0.57 warm-up level was explained by one participant |
| 11-05 | rely on local evaluation; upload only for large gains | the next upload held | The next four uploads each set a new best: 0.825, 0.827, 0.835, 0.844 |
| 11-06 | three small gains add up to 0.003 | stack registered and repeated on fresh seeds (T2-D373, T2-D376) | +0.0005 on the repeat: no gain; lesson A26 |
| 11-09 | more risk when lagging behind | my own top-two-only upload rule of 11-06 replaced; M7 uploaded | 0.825, sixth, against the manager's "under 10 %" call that morning |
| 11-13 | hold the 0.822 candidate | M7-B3 was not uploaded | W14-lite (0.827) carried the start-vector lever two days later |
| 11-13 | start from the neurophysiology | first read corrected (T2-D473); CAR members (T2-D475) | +0.0052 in the lab; part of W14-lite |
| 11-15 | why de-mean again? | the window mean tested for the first time (T2-D516, T2-D517) | +0.0073 and +0.0074 in the lab; part of W18 (0.835) |
| 11-16 | revisit closed ideas that showed gains | closed-positive audit (T2-D524, T2-D529) | +0.0044 in the lab; part of W18 |
| 11-16 | do we still need today's upload? | W17 held; W18 was uploaded the next day | 0.835 |
| 11-17 | can the old augmentations transfer? | T2-D557 | +0.0052 to +0.0101 in the lab, the largest item of the last week |
| 11-18 | "I don't like that all tracks are idle" | sprint reopened after a stop recommendation (T2-D569) | W21 |
| 11-19 | first-principles panel; aim for first place with all resources | the composition mismatch (T2-D630 to T2-D632); T2-D642 run in parallel | W25-ALL6, 0.844. W21 was never uploaded, so the sealed step from W18 cannot be split between W21's parts and T2-D642 |
| 11-19 | build the router fix as an exception | W23 built and read through the full stack (T2-D638) | About zero; not shipped |
| 11-19 | search for better seeds | T2-D640 | At most +0.0004; closed (A57) |
| 11-20 | any small gain should be included | an inclusion rule with no returning guard | W30 passed it and scored 0.838 sealed, 0.006 below the file it was built on |
| 11-21 | "why not testing w30 now?" | W30 read, then built and uploaded | 0.838. No gain, and no loss under best-of scoring |
| 11-21 | "we need a risky solution" | T2-D658, registered with a harm bound | Closed on its bound (−0.0035 against −0.003); not uploaded |

## Where the agents were wrong

- **10-27.** Session memory grew to as much as 24 GB, and a local data read tipped the machine into an out-of-memory kill of all four sessions. About 14 hours were lost.
- **11-05 to 11-21.** The assumed lab-to-sealed transfer factor moved from 0.35 to 0.66 to about 0.5, and ended at about 0.4 on the last step (manager log 11-10 16:44, 11-15 13:44; A66).
- **11-10.** The manager rated Track 2 "under 10 %, commodity level" a few hours before M7 scored 0.825, sixth.
- **11-13.** The worker's first neurophysiology read contrasted the tasks with each other and missed the motor signature. It was corrected after I doubted it (T2-D473).
- **Until 11-15.** Every member removed the window mean in a hard-wired step that had never been tested as an arm (manager log 11-15 15:44).
- **11-15.** The manager's first answer about last year's solutions rested on web summaries. It was redone from the code after I rejected it.
- **11-16.** A parts check failed on a legitimate difference and was changed after the failure. A replacement gate written by the manager was vacuous (manager log 11-16 11:52, 15:52).
- **11-18.** The worker and the manager recommended stopping with three days left. The next item tried after my objection passed (T2-D569).
- **11-19.** The manager's recommendation of T2-D603 was withdrawn after an order check (A53).
- **Until 11-19.** The lab panels trained on one session per user and the shipped build on all sessions. The copy router's hub went unseen until then (A54).
- **11-20.** The manager's diagnosis of a biased returning proxy, and the re-weighted composite built on it, were wrong and were withdrawn the same afternoon (A58).
- **Until 11-20.** The runtime limit was taken as 80 minutes. The kit says 3,600 s (A59).
- **11-20.** The review session passed a code change that contained a key error. The worker caught it.
- **11-21.** W30 was reopened with no ship path, and a seven-hour build estimate was accepted unexamined. The real path took about three hours (A62).
- **11-21.** The top-three odds were quoted from a public page that was two days stale (A65).
- **11-21.** The statement that the shipped λ "is not hurt by a decode-only session shuffle" rested on levels that are not comparable across orders. It was retracted (A63).
- **11-21.** Nobody asked what the repack's parts assertion compares against until the build failed on it (A62).
- **11-21/22.** W30 was called positive on a lab composite of +0.0024. The member-level returning reads, negative in 10 of 10, were not in the rule. The sealed result was −0.006 (T2-D659; A66).
