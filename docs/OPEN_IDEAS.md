# Open ideas: what was never tested, and where we would start

The competition is closed, so nothing here can be scored on the hidden test set any more. Ideas are tested on the public sessions with the lab protocol described in [`METHOD.md`](METHOD.md): new users are emulated by holding people out, returning users by training on a person's first session and scoring the later ones. Section 4 of the front page says how far such lab numbers carried to the hidden set for us (about 40–50 %), so treat small lab gains with suspicion.

## Launched or designed, never read

| Idea | State | Where |
|---|---|---|
| **Subject-adversarial fine-tuning of the FM members** (a gradient-reversal head that tries to identify the user). The adversary was shown to engage: its held-out user accuracy fell from 0.986 to 0.091, with chance at 0.111. | Cut for cluster capacity on the last night; never scored | T2-D649 |
| **Several views of each test window through the FM** (small time shifts and per-channel gains, the same augmentations the members were trained with), optionally only for low-confidence windows | Template built, never scored | T2-D654 |
| **A back-end fitted on training users' FM features** (whitening the between-user directions before the head; or projecting them out and keeping the head) | Designed, judged too slow to ship; prior from a proxy read is poor | T2-D653 |
| **Resetting or re-scoring the decoder's history when a head is refit mid-stream** | Small consistent lab gains (+0.0003 to +0.0008); depends on the delivery order of test windows | T2-D595, T2-D603, T2-D607 |

## Questions the records leave open

1. **Per-user supervision is the missing ingredient. How can a model get closer to it at test time?** A readout fitted on a new user's own earlier windows with true labels beat the whole system by 0.040; with pseudo-labels it lost 0.064. An outside reviewer proposed training the FM to use reference windows from the same person as context ("episodic context conditioning"). It was not tried.
2. **Why did gains for new users carry over so poorly?** The one lab gain we could measure reached the hidden set at 37 to 48 % of its size, and the last candidate lost 0.006 after a lab gain of 0.0024. Two readings that our data cannot separate: plain retraining noise, or a real cost for returning users that a two-person lab read could not see.
3. **Is the foundation model the limit, or the way it was used?** The FM family held 35 % of the blend. Raising its share to 55 % helped four of eight held-out users by 0.009 to 0.026 and hurt the other four, two of them by 0.042 (T2-D658). What separates those people is unknown.
4. **What did the leaders do?** We do not know. The top six finished between 0.859 and 0.872, and the gap from them to the next entry (0.847) was about 0.012, as large as our best single lab gain of the last month.
5. **An unexplained reversal.** Removing the decoder's forgetting helped an early package under shuffled window order and hurt the final one (T2-D448 against T2-D656).

## What would change our mind

- A member, FM or not, that alone beats the fine-tuned ST-EEGFormer-large members on held-out users by a clear margin under the same protocol.
- A label-free adaptation of the FM that survives the checks in `LESSONS.md` (paired seeds, held-out users, both delivery orders).
- A lab protocol whose gains carry to an independent set of people at a rate near one. That alone would be worth more than any single model idea.
