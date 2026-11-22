# Which models were better? CNNs, classical pipelines and foundation models on Track 2

Track 2 of the NeurIPS 2026 EEG/EMG Foundation Challenge is a 3-class decoder of cued mental commands from
4-second EEG windows (41 channels, 120 Hz), one window at a time, for new and returning users. The team finished
13th of 302 with 0.844 ([`STATS.md`](STATS.md)). This page collects every place in the decision log where a
model was scored **alone**, or swapped like for like, and says what those records do and do not show.

It is a reading of an experiment log, not a benchmark. The log was written to decide what to ship next, so models
were measured under whatever protocol was in force that week. Scores are cell-mean balanced accuracy; chance is one
third.

- Machine-readable copy: [`model_comparison.json`](model_comparison.json)
- Every logged experiment: [`IDEAS_TRIED.md`](IDEAS_TRIED.md)
- The raw log: [`log/DECISIONS_T2.md`](log/DECISIONS_T2.md)

## Summary

1. **Best alone: FBCNet, a filter-bank CNN, and ST-EEGFormer large (ST-L), a public foundation model that was
   continued-pre-trained (CPT) and fully fine-tuned.** The CNN was ahead on the kit split in October (D88,
   T2-D112). On the four-user panel of early November one ST-L net at 30 epochs was level with FBCNet's family
   net (0.7393 against 0.7424), and the same net at 100 epochs and a vote of three ST-L nets were ahead (0.7573,
   0.774) (T2-D361, T2-D460, T2-D491). The forms differ: the ST-L nets are pooled nets without per-user copies,
   FBCNet's family net has them. Six ST-L members together were ahead on ten strangers (0.703 against 0.678,
   3 seeds, T2-D498).
2. **CNNs:** FBCNet, then IFNet and ShallowNet (0.7424 / 0.7312 / 0.7085, T2-D491), then a gap to EEGConformer,
   EEGNeX, ATCNet, Deep4, EEG-TCNet and EEGNet (0.657 down to 0.475; T2-D465, T2-D346, T2-D432).
3. **Classical pipelines:** all far below the CNNs alone. The best, a filter-bank tangent-space classifier, read
   0.585 where the CNN members read 0.708–0.742 (T2-D467, T2-D487).
4. **Other foundation models:** none reached FBCNet alone. Two came into the range of the weaker shipped CNNs as
   votes of three nets: MOR, an in-house foundation model, fully fine-tuned (0.727, T2-D463), and ST-EEGFormer
   base with CPT (0.713, T2-D460). ST-EEGFormer small, REVE, LaBraM, CBraMod, NeuroRVQ and MIRepNet were lower,
   some at chance (T2-D460, T2-D452, T2-D355, T2-D313, T2-D305).
5. **Caveats:** two to four scored users in most tables; a headline table assembled from 21 rows; an FM
   pre-training pool that held unlabelled windows of three of the four headline users (D89); and lab differences that reached
   the hidden set at 0.37 to about half of their size (T2-D657, T2-D514).

## How to read this page

**Terms, once.**

| Term | Meaning |
|---|---|
| strangers, new users | People absent from training |
| returning users | People with earlier labelled sessions in training; their later sessions are scored |
| kit split | The organisers' starter split of the ten public users: 6 training users, 2 validation users, 2 leaderboard users |
| alone | One model, or one family of the package, scored without the other families |
| family net | A CNN as it sits in the package: the pooled net blended with fine-tuned per-user copies, averaged over 8 test-time views |
| decode | A causal, label-free step that rebalances the three classes within a stream. October rows use a one-step prior correction, November rows a 10-iteration Sinkhorn. Rows marked "raw" use neither |
| CPT | Continued pre-training: more masked-reconstruction training of a foundation model (FM) on unlabelled EEG before the fine-tune |
| MOR | An in-house foundation model, still in development |
| mix | The October package objective: the mean of a returning-user score and a new-user score over 12 seed-paired indices |
| n/m | Seeds, draws or users up, out of m |

**Protocols.** Absolute scores are comparable only inside one protocol, and most safely inside one row.

| Short name | Dates | Who is scored | Seeds | What is scored |
|---|---|---|---|---|
| First screen | 13 Oct | Each of the 10 public users held out in turn (28 cells) | 3; 6 for the confirmation | One model, no decode (D81, D85) |
| Kit split | 13 Oct to 5 Nov | "Kit test": the 2 leaderboard users. "Kit val": the 2 validation users. All four are strangers; training uses the other 6 | 10 for CNNs and 6 for FMs in D88; 12 later | One model, last epoch, with the prior correction. Pairs such as 0.4865 / 0.7182 are kit test / kit val |
| Typical panel | 5 to 14 Nov | Four users, 16 cells: three strangers under the kit split and one returning user. The hardest of the five development users is left out | 12 in most rows | A member alone: one net, a family net, or a vote of three nets. Decoded in the rows that say so |
| Ten strangers | 14 to 17 Nov | Each of the 10 public users held out in turn; the other nine users' first sessions in training | 3 | A family, or its pooled nets, alone after its own decode |

**Rules of this page.** Every number is copied from the row whose id stands next to it. A difference is given only
where the row states it; none was computed here. Ids: "T2-D491" is a row of the Track 2 log; "D81" to "D97" are
rows of a shared early log, listed in `IDEAS_TRIED.md` under "Before the Track 2 log". "Final" in the last column
means the model family is in the final package, the file that scored 0.844 (T2-D657).

## Headline table: the typical panel, member alone

This is the one protocol under which most families were measured alone. It rests on **four users** and, in most
rows, **12 seeds**. It is not one experiment: the 38 lines come from 21 rows dated 5 to 14 November, and the
forms differ. Lines that share an id were scored in the same read.

| # | Model | Group | Form | Score | Reference in the same row | Difference stated | Id | Final? |
|---|---|---|---|---|---|---|---|---|
| 1 | ST-EEGFormer large + CPT, full fine-tune | FM | vote of three nets | 0.774 | | | T2-D460 | final |
| 2 | ST-EEGFormer large + CPT, full fine-tune, 100 epochs | FM | one net | 0.7573 | same member at 30 epochs 0.7393 | +0.018 | T2-D361 | final |
| 3 | FBCNet (13-band filter-bank CNN) | CNN | family net, decoded | 0.7424 | | | T2-D491 | final |
| 4 | ST-EEGFormer large + CPT, full fine-tune, 30 epochs | FM | one net | 0.7393 | | | T2-D361 | final |
| 5 | IFNet + causal Euclidean alignment | CNN | family net, decoded | 0.7312 | | | T2-D491 | final |
| 6 | MOR (C2L300 generation), full fine-tune | FM | vote of three nets, 3 draws | 0.727 | ST-L vote of three 0.780 | −0.052 (0/3) | T2-D463 | no |
| 7 | ST-EEGFormer large + CPT, Euclidean-aligned input | FM | one net | 0.721 | ST-L member 0.739 | | T2-D323 | no |
| 8 | ST-EEGFormer base + CPT | FM | vote of three nets | 0.713 | ST-L vote of three 0.774 | −0.0613 (0/12) | T2-D460 | no |
| 9 | ShallowNet + causal Euclidean alignment | CNN | family net, decoded | 0.7085 | | | T2-D491 | final |
| 10 | ST-EEGFormer base, released weights | FM | vote of three nets | 0.679 | ST-L vote of three 0.774 | −0.0947 (0/12) | T2-D460 | no |
| 11 | ST-EEGFormer small + CPT | FM | vote of three nets | 0.668 | ST-L vote of three 0.774 | −0.1059 (0/12) | T2-D460 | no |
| 12 | EEGConformer | CNN | one routed net, decoded | 0.657 | IFNet 0.727, ShallowNet 0.705 | | T2-D465 | no |
| 13 | ST-EEGFormer large + CPT, LoRA rank 16 | FM | one net | 0.652 | ST-L member, full fine-tune 0.739 | | T2-D329 | no |
| 14 | MOR (C2L300 generation), role-specific LoRA | FM | vote of three nets | 0.646 | ST-L vote of three 0.774 | −0.1284 (0/12) | T2-D463 | no |
| 15 | MOR (new generation with supervised post-training), role-specific LoRA | FM | one net | 0.641 | the package's MOR member | +0.0206 (8/12) | T2-D407 | no |
| 16 | ST-EEGFormer large + CPT, all-token head | FM | one net | 0.639 | ST-L member, mean-pooled head 0.739 | | T2-D349 | no |
| 17 | EEGNeX + causal Euclidean alignment | CNN | per-user copies at r 0.5, decoded; views not stated | 0.628 | | | T2-D346 | no |
| 18 | ST-EEGFormer base, pre-trained from scratch on open data only (90k steps) | FM | one net | 0.624 | ST-L member 0.739 | −0.116 | T2-FM-D2 | no |
| 19 | ShallowNet, stock library network, base recipe | CNN | pooled net, decoded | 0.619 | the team's ShallowNet member 0.709 | | T2-D432 | no |
| 20 | ATCNet + causal Euclidean alignment | CNN | per-user copies at r 0.5, decoded; views not stated | 0.617 | | | T2-D346 | no |
| 21 | ST-EEGFormer small, released weights | FM | vote of three nets | 0.616 | ST-L vote of three 0.774 | −0.1583 (0/12) | T2-D460 | no |
| 22 | Filter-bank tangent space + logistic regression (FBTS-LR) | classical | one deterministic fit | 0.585 | the package 0.813 | | T2-D328 | no |
| 23 | REVE (base), no CPT | FM | one net | 0.584 | ST-L member 0.75 | | T2-D452 | no |
| 24 | Deep4 + causal Euclidean alignment | CNN | per-user copies at r 0.5, decoded; views not stated | 0.568 | | | T2-D346 | no |
| 25 | Block of four classical members, mean | classical | deterministic, decoded | 0.563 | the package 0.829 | | T2-D467 | no |
| 26 | Deep SPD network (TSMNet form), aligned input, no SPD re-centring | CNN | pooled net, decoded | 0.5594 | FBCNet 0.742 | | T2-D491b | no |
| 27 | Random convolutions (MiniRocket style) + logistic regression | classical | one member | 0.5437 | FBTS-LR in the same read 0.5850 | | T2-D433 | no |
| 28 | Filter-bank CSP + LDA | classical | deterministic, decoded | 0.536 | the package 0.829 | | T2-D467 | no |
| 29 | Deep SPD network (TSMNet form) with per-stream SPD re-centring | CNN | pooled net, decoded | 0.5328 | FBCNet 0.7424 | | T2-D491 | no |
| 30 | EEG-TCNet + causal Euclidean alignment | CNN | per-user copies at r 0.5, decoded; views not stated | 0.499 | | | T2-D346 | no |
| 31 | MantisV2 (a time-series FM), fine-tuned end to end | FM | one net, decoded, 6 seeds | 0.494 | its random-initialised twin 0.475 | +0.019 | T2-D434 | no |
| 32 | CBraMod, token-grid head | FM | one net | 0.488 | | | T2-D355 | no |
| 33 | LaBraM (base), no CPT | FM | one net | 0.484 | ST-L member 0.75 | | T2-D452 | no |
| 34 | EEGNet, stock library network, base recipe | CNN | pooled net, decoded | 0.475 | the team's ShallowNet member 0.709 | | T2-D432 | no |
| 35 | Riemannian minimum distance to mean (MDM) | classical | deterministic, decoded | 0.403 | the package 0.829 | | T2-D467 | no |
| 36 | xDAWN-covariance tangent space + logistic regression | classical | deterministic, decoded | 0.386 | the package 0.829 | | T2-D467 | no |
| 37 | CBraMod, mean-pooled head | FM | one net | 0.386 | | | T2-D341 | no |
| 38 | Imaginary coherency + logistic regression | classical | one member | 0.373 | CNN members alone 0.708–0.742 | | T2-D487 | no |

**What the table supports.**

- The top nine lines are the three CNN families that shipped (lines 3, 5, 9), ST-L variants (lines 1, 2, 4, 7)
  and two votes of three nets from other FMs (lines 6 and 8). Compare votes with votes: those two are −0.052 and
  −0.0613 below the ST-L vote of their own read (T2-D463, T2-D460). Among single nets, nothing but the ST-L is
  above ShallowNet; the next model is EEGConformer at 0.657 (T2-D465).
- It does **not** separate FBCNet from one ST-L member. Their lines come from different rows, and one is a family
  net with per-user copies and test-time views, the other a single net. The level of one model also moves between
  reads: IFNet 0.7312, 0.729 and 0.727 (T2-D491, T2-D336, T2-D465); the ST-L vote 0.774 and 0.780 (T2-D460,
  T2-D463); the MOR adapter form 0.646 on 12 draws and 0.627 on 3 draws (T2-D463). This page treats lines with
  different ids that are closer than that last spread as ties.
- One same-read comparison of the families exists on this panel, as recall per class and not as the cell score:
  FBCNet 0.811 / 0.672 / 0.684, ShallowNet 0.737 / 0.692 / 0.679, IFNet 0.775 / 0.675 / 0.677, the five ST-L
  members 0.842 / 0.747 / 0.700 (T2-D439, on files whose seeds had been seen before). The ST-L family was highest
  on every class.
- **A bias toward the FMs with CPT.** The CPT pool of the kit split held the unlabelled windows of the kit's
  train and validation participants (D89; the pool lists in T2-D111 and T2-D123 say the same). This panel's users
  are the two validation users, one leaderboard user and one kit-train participant who is scored on sessions 2–3
  (T2-D432). So the pool held three of the four panel users, and only the leaderboard user was kept out. The CNNs
  never saw the two validation users. For the kit-train participant, D89 does not say whether the scored sessions
  were in the pool; the pool built for the returning-user split held that participant's session 1 only (T2-D111).
  The CPT step gained far more on the two validation users than on the two leaderboard users, who were kept out of the
  pool: +0.0787 against +0.0223 (D94). The ten-stranger read below removes this by using released weights.

## Convolutional and compact networks

| Model | Protocol | Score | Reference and its score | Difference stated | Id | Final? |
|---|---|---|---|---|---|---|
| FBCNet, per-window z-score | First screen, 3 seeds | 0.5657 | EEGNet, no alignment 0.4227 | +0.143 (10/10 users) | D81 | final |
| FBCNet, per-window z-score | First screen, 6 seeds | 0.5637 | EEGNet 0.4489 | +0.1148 (9/10 users) | D85 | final |
| ShallowNet + Euclidean alignment | First screen, 3 seeds | 0.5339 | EEGNet 0.4497 | | D81 | final |
| EEGNet, per-window z-score | First screen, 3 seeds | 0.4497 | | | D81 | no |
| ShallowNet + Euclidean alignment | Kit split, 10 seeds | 0.4931 / 0.6062 | FBCNet 0.4865 / 0.7182 | | D88 | final |
| FBCNet, no alignment | Kit split, 10 seeds | 0.4865 / 0.7182 | EEGNet 0.3446 / 0.4904 | | D88 | final |
| FBCNet + Euclidean alignment | Kit split, 10 seeds | 0.4881 / 0.7100 | FBCNet, no alignment 0.4865 / 0.7182 | | D88 | final |
| FBCNet + running per-channel scale | Kit split, 10 seeds | 0.4899 / 0.6839 | FBCNet, no alignment 0.4865 / 0.7182 | | D88 | final |
| EEGNet, per-window z-score | Kit split, 10 seeds | 0.3446 / 0.4904 | FBCNet 0.4865 / 0.7182 | | D88 | no |
| FBCNet member of the late-October package (13 bands, augmentations, per-user copies) | Kit split, kit test only, 12 seeds, decode not stated | 0.5164 | its half-window crop members 0.4761, 0.4599 | | T2-D275 | final |
| HEEGNet (hyperbolic EEG network) | Kit split, kit test only, 12 seeds, decode not stated | 0.3434 | its EEGNet control 0.3469; the FBCNet member 0.5164 | | T2-D318 | no |
| FBCNet | Typical panel, family net, decoded | 0.7424 | | | T2-D491 | final |
| IFNet + Euclidean alignment | Typical panel, family net, decoded | 0.7312 | | | T2-D491 | final |
| IFNet + Euclidean alignment (first read) | Typical panel, family net | 0.729 | ShallowNet 0.709 | | T2-D336 | final |
| ShallowNet + Euclidean alignment | Typical panel, family net, decoded | 0.7085 | | | T2-D491 | final |
| EEGConformer | Typical panel, one routed net, decoded | 0.657 | IFNet 0.727, ShallowNet 0.705 | | T2-D465 | no |
| EEGNeX + Euclidean alignment | Typical panel, per-user copies at r 0.5, decoded; views not stated | 0.628 | | | T2-D346 | no |
| ShallowNet, stock library network | Typical panel, pooled net, decoded | 0.619 | the team's ShallowNet member 0.709 | | T2-D432 | no |
| ATCNet + Euclidean alignment | Typical panel, per-user copies at r 0.5, decoded; views not stated | 0.617 | | | T2-D346 | no |
| Deep4 + Euclidean alignment | Typical panel, per-user copies at r 0.5, decoded; views not stated | 0.568 | | | T2-D346 | no |
| Deep SPD network (TSMNet form), no SPD re-centring | Typical panel, pooled net, decoded | 0.5594 | FBCNet 0.742, IFNet 0.731, ShallowNet 0.709 | | T2-D491b | no |
| Deep SPD network (TSMNet form), per-stream SPD re-centring | Typical panel, pooled net, decoded | 0.5328 | FBCNet 0.7424 | | T2-D491 | no |
| EEG-TCNet + Euclidean alignment | Typical panel, per-user copies at r 0.5, decoded; views not stated | 0.499 | | | T2-D346 | no |
| EEGNet, stock library network | Typical panel, pooled net, decoded | 0.475 | the team's ShallowNet member 0.709 | | T2-D432 | no |
| FBCNet family | Ten strangers, family, 3 seeds | 0.678 | ST-L family 0.703 | | T2-D498 | final |
| IFNet family | Ten strangers, family, 3 seeds | 0.648 | ST-L family 0.703 | | T2-D498 | final |
| ShallowNet family | Ten strangers, family, 3 seeds | 0.637 | ST-L family 0.703 | | T2-D498 | final |
| FBCNet | Ten strangers, pooled nets only, 3 draws | 0.659 | | | T2-D554 | final |
| IFNet | Ten strangers, pooled nets only, 3 draws | 0.631 | | | T2-D554 | final |
| ShallowNet | Ten strangers, pooled nets only, 3 draws | 0.626 | | | T2-D554 | final |
| Raw end-to-end net with a learned front end, no alignment | Ten strangers, 3 draws | 0.617 | aligned ShallowNet / IFNet side nets 0.670 | | T2-D521 | no |
| 2-D CNN on log-spectrograms, from scratch | Ten strangers, pooled nets only, 3 draws | 0.532 | FBCNet 0.659 | | T2-D554 | no |

Notes on this group:

- **FBCNet first, by a wide margin over EEGNet, under every protocol.** The gap to EEGNet was +0.1148 in the first
  screen (D85) and the two were 0.4865 against 0.3446 on the kit's leaderboard users (D88).
- **ShallowNet against FBCNet depends on the users.** On the two leaderboard users ShallowNet was ahead
  (0.4931 against 0.4865); on the two validation users it was far behind (0.6062 against 0.7182) (D88). On the
  typical panel and on ten strangers FBCNet was ahead in every read (T2-D491, T2-D498, T2-D554).
- **The recipe mattered as much as the architecture.** The stock ShallowNet read 0.619 and the team's ShallowNet
  member 0.709 on the same cells; the row states the difference as +0.090 (T2-D432). The members carry training
  augmentations, per-user copies and test-time views that the stock network does not.
- **Weaker families were not tuned.** EEGConformer ran at library defaults (T2-D465). EEGNeX, ATCNet, Deep4 and
  EEG-TCNet ran under the recipe used for IFNet (T2-D346). HEEGNet ran under one recipe of its own (T2-D318).
  Their levels are levels at one configuration.
- **Alignment.** Causal Euclidean alignment did not help FBCNet (0.4881 / 0.7100 against 0.4865 / 0.7182, D88).
  ShallowNet and IFNet shipped with it, and a net with a learned front end and no alignment was lower
  (0.617 against 0.670, T2-D521).
- **One difference without a level:** with nine training users, EEGNeX alone was −0.0559 against ShallowNet
  (T2-D485).
- An October screen of IFNet, EEGNet, EEG-TCNet, Deep4, EEGNeX, ATCNet and EEGConformer scored them only inside a
  package on the two validation users (best: IFNet 0.7502 against a base of 0.7428; every other candidate at most
  0.7435; T2-D118). It gives no alone numbers.

## Classical pipelines

| Model | Protocol | Score | Reference and its score | Difference stated | Id | Final? |
|---|---|---|---|---|---|---|
| Filter-bank tangent space + logistic regression (FBTS-LR), Euclidean alignment | First screen, 3 seeds | 0.4665 | EEGNet 0.4497; FBCNet 0.5657 | | D81 | no |
| FBTS-LR with causal re-centring | Typical panel, one deterministic fit | 0.585 | the package 0.813 | | T2-D328 | no |
| Block of four classical members, mean | Typical panel, deterministic, decoded | 0.563 | the package 0.829 | | T2-D467 | no |
| Filter-bank CSP + LDA | Typical panel, deterministic, decoded | 0.536 | the package 0.829 | | T2-D467 | no |
| Riemannian minimum distance to mean | Typical panel, deterministic, decoded | 0.403 | the package 0.829 | | T2-D467 | no |
| xDAWN-covariance tangent space + logistic regression | Typical panel, deterministic, decoded | 0.386 | the package 0.829 | | T2-D467 | no |
| Imaginary coherency + logistic regression | Typical panel | 0.373 | CNN members alone 0.708–0.742 | | T2-D487 | no |
| wPLI + logistic regression | Typical panel | 0.346 | CNN members alone 0.708–0.742 | | T2-D487 | no |
| Random convolutions (MiniRocket style) + logistic regression | Typical panel | 0.5437 | FBTS-LR in the same read 0.5850 | | T2-D433 | no |
| Gradient-boosted trees on 16,454 engineered features | Ten strangers, one tree seed | 0.579 | | | T2-D515 | no |
| The same tree bag on de-meaned windows | Ten strangers | 0.543 | the bag on raw windows 0.584 | | T2-D515d | no |
| Window-mean tree family: gradient-boosted trees on 82 per-channel window means | Ten strangers | 0.469 | | | T2-D516 | final |

Notes on this group:

- **Every classical pipeline was far below the CNN members alone.** The best, FBTS-LR, read 0.585 where the CNN
  members read 0.708–0.742 (T2-D467, T2-D487). Averaging four of them did not help (0.563, T2-D467).
- **One like-for-like change of a classical pipeline:** a label-free dispersion stretch on top of the causal
  re-centring cost FBTS-LR −0.0076 (0/12) (T2-D421).
- **The broad feature sweep was a feature finder, not a strong model.** The tree bag on 16,454 features read 0.579
  alone (T2-D515). Its value in the blend came from the per-window mean, which every network removed: on de-meaned
  windows it fell to 0.543 (T2-D515d).
- **The one classical family that shipped is the weakest alone of those that were kept:** the window-mean trees at
  0.469 (T2-D516). The next section says why it was kept.

## Foundation models

### The October screen and its follow-ups (kit split, one model, prior correction; kit test / kit val)

| Model | Protocol | Score | Reference and its score | Difference stated | Id | Final? |
|---|---|---|---|---|---|---|
| MOR (plain variant), fixed-gain input | Kit split, 6 seeds | 0.4476 / 0.5015 | FBCNet 0.4865 / 0.7182 | | D88 | no |
| MOR (post-trained variant), 12 epochs | Kit split, 6 seeds | 0.4451 / 0.5529 | FBCNet 0.4865 / 0.7182 | | D88 | early uploads only |
| MOR (control variant), fixed-gain input | Kit split, 6 seeds | 0.4272 / 0.5580 | FBCNet 0.4865 / 0.7182 | | D88 | no |
| ST-EEGFormer large, released weights | Kit split, 6 seeds | 0.4192 / 0.5551 | FBCNet 0.4865 / 0.7182 | | D88 | no |
| ST-EEGFormer base, released weights, mixup | Kit split, 6 seeds | 0.4125 / 0.6133 | FBCNet 0.4865 / 0.7182 | | D88 | no |
| REVE (base), fixed-gain input | Kit split, 6 seeds | 0.4078 / 0.5388 | FBCNet 0.4865 / 0.7182 | | D88 | no |
| LaBraM (base), fixed-gain input | Kit split, 6 seeds | 0.3440 / 0.4830 | EEGNet 0.3446 / 0.4904 | | D88 | no |
| LaBraM (base) | First screen, 3 seeds | 0.397 | EEGNet 0.449 | | D86 | no |
| ST-EEGFormer large + CPT + fixed-gain input | Kit split, 6 seeds | 0.4427 / 0.6687 | the same from released weights 0.4204 / 0.5900 | +0.0223 / +0.0787 | D94 | final |
| ST-EEGFormer large + CPT (fresh seeds) | Kit split, 6 seeds | 0.4379 / 0.6549 | FBCNet, same seeds 0.4864 / 0.7142 | | D95 | final |
| REVE (base) (fresh seeds) | Kit split, 6 seeds | 0.3938 / 0.5171 | FBCNet, same seeds 0.4864 / 0.7142 | | D96 | no |
| ST-EEGFormer large + CPT + mixup | Kit split, 12 seeds | 0.4630 / 0.6949 | the same without mixup 0.4403 / 0.6618 | +0.0227 (11/12) / +0.0331 (12/12) | T2-D112 | final |
| MOR (post-trained variant), 12 epochs | Kit split, 12 seeds | 0.4425 / 0.5507 | | | T2-D107 | early uploads only |
| MOR, role-specific LoRA rank 16 | Kit split, 12 seeds | 0.4612 / 0.5750 | MOR, full fine-tune 0.4425 / 0.5507 | +0.0188 (7/12) / +0.0243 (10/12) | T2-D130 | early uploads only |
| MOR architecture pre-trained from scratch (two arms) | Kit split, kit test only, 6 seeds | 0.3695, 0.3501 | MOR with its original pre-training 0.4681 | | T2-D141 | no |
| NeuroRVQ-EEG, full fine-tune | Kit split, raw, 12 seeds | 0.3764 / 0.4314 | ST-L member 0.4431 / 0.6610 | −0.067 on kit test | T2-D313 | no |
| NeuroRVQ-EEG, LoRA rank 16 | Kit split, raw, 12 seeds | 0.3849 / 0.4568 | ST-L member 0.4431 / 0.6610 | | T2-D313b | no |
| MOR, 568M parameters | Kit split, raw, 12 seeds | 0.3954 / 0.5949 | MOR, 100M parameters 0.4104 / 0.5692 | −0.015 on kit test | T2-D314 | no |
| MIRepNet, published fine-tune | Kit split, kit test only, 12 seeds, decode not stated | 0.3417 | | | T2-D305 | no |
| CBraMod, mean-pooled head | Kit split, kit test only, 12 seeds, decode not stated | 0.337 | | | T2-D341 | no |

- **In October no FM beat FBCNet alone on either pair of users.** D88 states it: none passed the bar against the
  CNN base; the best FM kit test was 0.4476 and the best FM kit val 0.6133, against 0.4865 / 0.7182. With CPT and
  mixup the ST-L reached 0.4630 / 0.6949 (T2-D112), still below on both.
- **On the two leaderboard users the ST-L did not move afterwards.** In November its kit-test curve was flat at
  ≈ 0.44 over the fine-tune (T2-D361). Its later gain sits on the other users.
- **LaBraM was at or below the EEGNet level** in both early protocols (0.3440 against 0.3446 in D88; 0.397
  against 0.449 in D86). NeuroRVQ, MIRepNet and CBraMod were near chance on the kit test (T2-D313, T2-D305,
  T2-D341). MIRepNet and NeuroRVQ were never read on the typical panel.

### The typical panel (5 to 14 November)

The FM lines of the headline table, with the rows that add a second number.

| Model | Protocol | Score | Reference and its score | Difference stated | Id | Final? |
|---|---|---|---|---|---|---|
| ST-EEGFormer large + CPT: vote of three | Typical panel, 12 draws | 0.774 | | | T2-D460 | final |
| ST-EEGFormer large + CPT, 100 / 60 / 30 epochs: one net | Typical panel, 12 seeds | 0.7573 / 0.7524 / 0.7393 | the 30-epoch member 0.7393 | +0.018 / +0.013 | T2-D361 | final |
| ST-EEGFormer large, CPT length 1k / 2k / 5k / 10k steps: one net | Typical panel, 12 seeds | 0.7305 / 0.7371 / 0.7393 / 0.7330 | | | T2-D348 | final (5k) |
| ST-EEGFormer base + CPT; released weights: vote of three | Typical panel, 12 draws | 0.713; 0.679 | ST-L vote of three 0.774 | −0.0613; −0.0947 | T2-D460 | no |
| ST-EEGFormer small + CPT; released weights: vote of three | Typical panel, 12 draws | 0.668; 0.616 | ST-L vote of three 0.774 | −0.1059; −0.1583 | T2-D460 | no |
| ST-EEGFormer base from scratch on open data only, 45k / 90k steps: one net | Typical panel, 12 seeds | 0.607 / 0.624 | ST-L member 0.739 | −0.132 / −0.116 | T2-FM-D2 | no |
| REVE (base), no CPT, two input scalings: one net | Typical panel, 12 seeds | 0.582, 0.584 | ST-L member 0.75 | | T2-D452 | no |
| LaBraM (base), no CPT: one net | Typical panel, 12 seeds | 0.484 | ST-L member 0.75 | | T2-D452 | no |
| REVE (base) with a masked-patch CPT: five-member family | Typical panel, 10 draws | 0.573 | the family without the CPT 0.605 | −0.0321 (0/10) | T2-D492 | no |
| LaBraM (base) with a masked-patch CPT: five-member family | Typical panel, 10 draws | 0.481 | the family without the CPT 0.504 | −0.0237 (0/10) | T2-D492 | no |
| CBraMod, mean-pooled head: one net | Typical panel, 12 seeds | 0.386 | | | T2-D341 | no |
| CBraMod, token-grid head, two recipes: one net | Typical panel, 12 seeds | 0.455, 0.488 | | | T2-D355 | no |
| MOR, the package's member against the plain variant | Typical panel, 12 seeds | no level in the row | MOR, plain variant | +0.0746 | T2-D386 | early uploads only |
| MOR, plain variant + CPT on the challenge pools, 5k / 10k / 20k steps | Typical panel, 12 seeds | no level in the row | MOR, plain variant | +0.0182 / +0.0201 / +0.0191 | T2-D386 | no |
| MOR, new generation with supervised post-training, two checkpoints: one net | Typical panel, 12 draws | 0.639, 0.641 | the package's MOR member; ST-L members 0.721–0.758 | +0.0186 (9/12), +0.0206 (8/12) | T2-D407 | no |
| MOR, C2L300 generation, role-specific LoRA; annealed: vote of three | Typical panel, 12 draws | 0.646; 0.640 | ST-L vote of three 0.774 | −0.1284; −0.1337 | T2-D463 | no |
| MOR, C2L300 generation, full fine-tune: vote of three | Typical panel, 3 draws | 0.727 | ST-L vote of three 0.780 | −0.052 (0/3) | T2-D463 | no |
| MantisV2 (time-series FM), fine-tuned | Typical panel, 6 seeds | 0.494 | random-initialised twin 0.475 | +0.019 | T2-D434 | no |
| MantisV2, frozen embeddings + logistic regression | All cells of the five development users | 0.5285 | FBTS-LR on the same cells 0.5526 | | T2-D433 | no |

### Ten strangers (14 to 17 November)

| Model | Protocol | Score | Reference and its score | Difference stated | Id | Final? |
|---|---|---|---|---|---|---|
| ST-EEGFormer large family: six members from the released weights, **no CPT** | Ten strangers, family, 3 seeds | 0.703 | FBCNet 0.678, IFNet 0.648, ShallowNet 0.637 | | T2-D498 | final (with CPT) |
| ST-L member interpolated with a word-task checkpoint | Ten strangers, one member, score as written by the training run | 0.6096 | the plain member of the same seed 0.6032 | +0.0065 | T2-D544 | no |

- **This is the only read with the FM family and the CNN families alone in one place.** The six ST-L members
  together were above each CNN family (T2-D498). It rests on 3 seeds, and the full stack in the same read was
  0.7440.
- The FM ran here without CPT, because every CPT pool held the held-out user; the row puts CPT at about +0.02
  alone (T2-D498). So the comparison does not favour the FM.
- No other FM was read on ten strangers.

### Fine-tuning mode: frozen, LoRA, full

| Model | Protocol | Score | Reference and its score | Difference stated | Id | Final? |
|---|---|---|---|---|---|---|
| ST-EEGFormer large, frozen, linear probe on last-layer features | Kit split, one deterministic probe | 0.391 / 0.500 | MOR, same probe 0.443 / 0.484 | | T2-D126 | no |
| ST-EEGFormer large + CPT, encoder frozen, head trained | Kit split, the three strangers of the typical panel, 12 draws, decoded | 0.5164 | the same member fully fine-tuned 0.7025 | | T2-D454 | no |
| ST-EEGFormer large, LoRA rank 16 | Kit split, 12 seeds | no level in the row | the full fine-tune | −0.0003 (5/12) / +0.0012 | T2-D130 | no |
| ST-EEGFormer large + CPT, LoRA rank 16 | Typical panel, 12 seeds | 0.652 | the full fine-tune 0.739 | | T2-D329 | no |
| MOR, role-specific LoRA rank 16 | Kit split, 12 seeds | 0.4612 / 0.5750 | MOR, full fine-tune 0.4425 / 0.5507 | +0.0188 / +0.0243 | T2-D130 | early uploads only |
| MOR (C2L300 generation), full fine-tune | Typical panel, vote of three, 3 draws | 0.727 | its adapter form in the same read 0.627 | | T2-D463 | no |
| NeuroRVQ-EEG, LoRA rank 16 | Kit split, raw, 12 seeds | 0.3849 | its full fine-tune 0.3764 | | T2-D313b | no |
| REVE (base), LoRA rank 8 | Kit split, 6 seeds | 0.3999 / 0.5283 | REVE, full fine-tune, default input 0.3721 / 0.4918 | | D88 | no |

- **Frozen features were far below a fine-tune** for the ST-L: 0.5164 against 0.7025 (T2-D454).
- **LoRA on the ST-L has two rows that disagree.** In October LoRA matched the full fine-tune on the kit split
  (T2-D130). In November, on the later recipe and the typical panel, it was 0.652 against 0.739 (T2-D329). A
  follow-up with a LoRA-sized learning rate fixed the under-fit and still read −0.0003 (4/6) as a swap in the
  package (T2-D372).
- **For MOR the direction changed with the generation:** an adapter recipe beat the full fine-tune for the first
  generation (T2-D130); for the C2L300 generation the full fine-tune was higher (T2-D463).

### Continued pre-training

| Model | Protocol | Score | Reference and its score | Difference stated | Id | Final? |
|---|---|---|---|---|---|---|
| ST-EEGFormer large + CPT | Kit split, 6 seeds | 0.4427 / 0.6687 | released weights 0.4204 / 0.5900 | +0.0223 / +0.0787 | D94 | final |
| ST-EEGFormer large + CPT on an 8× bigger pool | Kit split, 12 seeds | 0.4397 / 0.6789 | the first CPT 0.4403 / 0.6618 | −0.0006 (5/12) / +0.0171 (10/12) | T2-D109 | no |
| ST-EEGFormer large + CPT + supervised pre-fine-tune on 10 labelled corpora | Kit split, 12 seeds | 0.4323 / 0.6740 | CPT only 0.4403 / 0.6618 | −0.0080 (5/12) / +0.0122 (7/12) | T2-D103 | no |
| ST-EEGFormer base + CPT | Kit split, 6 seeds | 0.3771 / 0.5902 | released weights 0.3868 / 0.5706 | −0.0097 / +0.0196 | D89 | no |
| MOR + close-domain CPT | Kit split, 12 seeds | 0.4315 / 0.5652 | MOR without it 0.4425 / 0.5507 | −0.0110 (3/12) / +0.0145 (9/12) | T2-D123 | no |

- **CPT helped the ST-L on the kit split** (D94). For the base model the kit-test effect was mixed (−0.0097 in
  D89, +0.0078 on fresh seeds in D94) while kit val rose both times. On the typical panel the base and small
  models with CPT were above their released weights (0.713 against 0.679 and 0.668 against 0.616, T2-D460).
  Part of these gains may be exposure: see the bias noted under the headline table.
- **More or different pre-training did not add:** a bigger pool, a supervised stage, other lengths (T2-D109,
  T2-D103, T2-D348). A stand-in CPT made REVE and LaBraM worse (T2-D492). From-scratch pre-training on open data
  stayed far below the released ST-L (T2-FM-D2), and the same held for the MOR architecture (T2-D141).

## What the blend needed

A model that is weaker alone can still help a blend, and a model that is stronger alone can fail to. These rows
score a family being added to, removed from, or swapped inside the package of the day. The package changed
between rows, so the sizes are not comparable across rows.

| Change in the package | Panel | Effect | Id |
|---|---|---|---|
| Add the ST-L (with CPT) to FBCNet at 4 : 1 | Kit split, 6 fresh seeds | +0.0247 / +0.0090, 6/6 seeds up on kit test | D95 |
| Add REVE to FBCNet at 5 : 1 | Kit split, 6 fresh seeds | −0.0052 / −0.0055, 2/6 | D96 |
| Train an FM and FBCNet jointly at the feature level | Kit split, 6 seeds | ST-EEGFormer base 0.4159 / 0.6101 and large 0.4098 / 0.5805, against FBCNet alone 0.4888 / 0.7167 | D90 |
| Add ShallowNet at 0.2 to FBCNet + ST-L | Kit split, 7 fresh seeds | +0.0145 (7/7) / +0.0148 (7/7) | T2-D116 |
| Add IFNet at 0.2 as a further family (October) | Kit split, 6 fresh seeds | −0.0002 (2/6) / +0.0116 (4/6) | T2-D118 |
| The ST-L member with mixup instead of without | Kit split, 12 seeds | +0.0227 alone; −0.0006 / −0.0006 (6/12) in the two-family blend | T2-D112 |
| Remove the ST-L member from the October package | October "mix" objective, 12 indices | the member carries +0.0173 (11/12) | T2-D303 |
| Remove the MOR member from the October package | October "mix" objective, 12 indices | the member carries +0.0005 (6/12) | T2-D303 |
| Remove both FMs from the October package | October "mix" objective, 12 indices | together they carry +0.0225 (12/12) | T2-D303 |
| Add the ST-L member to the two-family CNN-only package | Typical panel, 12 draws | +0.0358 (12/12) | T2-FM-D2 |
| Add the from-scratch ST-EEGFormer base instead | Typical panel, 12 draws | −0.0024 (5/12) at 45k steps, +0.0038 (7/12) at 90k | T2-FM-D2 |
| Two ST-L members instead of one at the same share | Typical panel, 12 fresh seeds | +0.0060 (11/12) | T2-D326 |
| Five ST-L members instead of one (between the 5 Nov and 10 Nov packages) | Typical panel, 12 draws | +0.0109 (12/12); FM share 0.22 → 0.35: +0.0047; IFNet: +0.0038 | T2-D450 |
| FM share away from 0.35 | Typical panel | 0.25: −0.0091; 0.55: −0.0046; 0.65: −0.0112 | T2-D451 |
| Add IFNet at 0.1 (November) | Typical panel, 12 draws | +0.0026 (11/12) | T2-D336 |
| IFNet in place of ShallowNet | Typical panel, 12 draws | −0.0051 (2/12) | T2-D336 |
| IFNet at 0.10 together with a fifth ST-L member | Typical panel, 12 fresh seeds | +0.0047 (10/12) | T2-D345r |
| Add EEGNeX as a fourth CNN family | Typical panel | +0.0031 (10/12) on the screen, +0.0012 (8/12) on fresh seeds | T2-D346 |
| EEGNeX as a fourth family in the later stack | Typical panel, 10 draws | −0.0039 (0/10) | T2-D485 |
| EEGConformer in IFNet's slot | Typical panel, 12 draws | −0.0062 (2/12) | T2-D465 |
| Add HEEGNet at 0.1 | October "mix" objective, 12 indices | −0.0028 (2/12) | T2-D318 |
| NeuroRVQ in the FM slot; added at 0.1 | October "mix" objective, 12 indices | −0.0551 (0/12); −0.0085 (1/12) | T2-D313 |
| Add CBraMod at 0.10 | Typical panel, 12 draws | −0.0056 (1/12) | T2-D341 |
| REVE (two input scalings) or LaBraM as a sixth FM member | Typical panel, 12 draws | −0.0058 (0/12), −0.0043 (2/12), −0.0098 (0/12) | T2-D452 |
| Three ST-EEGFormer base nets with CPT added to the five ST-L | Typical panel, 12 draws | −0.0025 (3/12) | T2-D460 |
| A new-generation MOR in place of one ST-L member (two checkpoints) | Typical panel, 12 draws | −0.0050 (0/12), −0.0037 (0/12) | T2-D407 |
| The fully fine-tuned C2L300 MOR added; replacing an ST-L | Typical panel, 3 draws | −0.0003; −0.0030 | T2-D463 |
| LoRA ST-L in place of the fully fine-tuned one | Typical panel, 12 draws | −0.0190 (0/12) | T2-D329 |
| Add FBTS-LR at 0.05 | Typical panel, 12 draws | +0.0013 (8/12); the hardest user +0.0109 (11/12) | T2-D328 |
| Add FBTS-LR at 0.05 to a later package | Typical panel, 12 draws | −0.0024 (3/12); the hardest user +0.0129 (12/12) | T2-D408 |
| Add the block of four classical members at 0.10 | Typical panel, 12 draws | −0.0035 (1/12) | T2-D467 |
| Add the imaginary-coherency member at a cross-fitted weight | Typical panel, 10 draws | +0.0012 (7/10) | T2-D487 |
| Add the tree bag on 16,454 features as a fifth family at 0.10 | Ten strangers, 3 draws | +0.0054 | T2-D515 |
| Add the window-mean tree family as a fifth family at 0.075 | Ten strangers, 3 draws | +0.0118 | T2-D516 |
| The same on the stack that was shipped next | Ten strangers, 3 draws | +0.0073; returning users −0.0016 | T2-D516b |
| Add the log-spectrogram CNN at 0.10 | Ten strangers, 3 draws | +0.0005 | T2-D574 |
| The raw end-to-end net in place of the aligned side nets | Ten strangers, 3 draws | −0.0048 (0/3) | T2-D521 |

What these rows say:

- **The FM family was the largest single addition to the CNNs.** It was worth +0.0173 in the October package
  (T2-D303) and +0.0358 over a CNN-only pair in early November (T2-FM-D2). The team's own reverse
  read puts about 80 % of the step between two uploaded packages on the foundation-model side (T2-D450).
- **Only one FM added.** Every other FM cost the package or left it unchanged at the weights tried (D96, T2-D313,
  T2-D341, T2-D452, T2-D460, T2-D407, T2-D463). MOR, which was a member in October, carried +0.0005, the level
  of noise (T2-D303).
- **Complementarity beat strength among the CNNs.** IFNet was stronger than ShallowNet alone and still lost as its
  replacement (T2-D336); both were kept. A fourth CNN family did not hold: EEGNeX passed a screen, then failed on
  fresh seeds and in the later stack (T2-D346, T2-D485); EEGConformer was lower in IFNet's slot (T2-D465).
- **A weak member needs errors that are different and enough right answers.** FBTS-LR had an error correlation of
  0.16 with the package and was right on 42 % of its errors, and still cost the typical users while helping the
  hardest user (T2-D408). The window-mean trees were weaker alone (0.469) and added +0.0118, with an error
  correlation of 0.023: they read a signal that every network had removed (T2-D516).
- **A stronger member did not always make a stronger blend.** Mixup made the ST-L +0.0227 better alone and
  changed the two-family blend by −0.0006 (T2-D112). Across nine ST-L variants the package kept 0.29–0.42 of a
  member's gain (T2-D252).
- **The blend is well above its best part.** On ten strangers the full stack read 0.7440 where the best family
  alone read 0.703 (T2-D498).

## Model sizes

Parameters of ONE network, counted on 2026-11-22 by building it from this repository's code for the task's input (41 channels, 480 samples = 4 s at 120 Hz, 3 classes): the CNNs through `ni26.train.build_model`, the foundation-model encoders through `t2p_fm.make_encoder` (encoder only, without the classification head). The classical pipelines are not networks. A family in the package holds several such networks (seeds, per-user copies), and the lines scored as a vote of three hold three.

| Model | Group | Parameters | Name in the code |
|---|---|---|---|
| ST-EEGFormer large | FM | 302.5M | `st_large` |
| ST-EEGFormer base (also the open-data-only run) | FM | 85.2M | `st_base` |
| ST-EEGFormer small | FM | 25.3M | `st_small` |
| REVE (base) | FM | 69.3M | `reve` |
| LaBraM (base) | FM | 5.8M | `labram` |
| CBraMod | FM | 4.9M | `cbramod` |
| MOR | FM | about 100M (from its checkpoint record, not counted here) |  |
| EEGConformer | CNN | 461,539 | `conformer` |
| Deep4 | CNN | 289,753 | `deep4` |
| ShallowNet | CNN | 69,843 | `shallow` |
| ATCNet | CNN | 59,433 | `atcnet` |
| EEGNeX | CNN | 57,035 | `eegnex` |
| FBCNet, 13-band bank | CNN | 26,537 | `fbcnet_bboth` |
| IFNet | CNN | 13,911 | `ifnet` |
| EEG-TCNet | CNN | 4,595 | `eegtcnet` |
| EEGNet | CNN | 2,483 | `eegnet` |

FBCNet, ShallowNet and IFNet are the configurations of the final package (`fbcnet_bboth`, `shallow`, `ifnet`); the code also holds wider IFNet variants (`ifnet_b3` 37,873, `ifnet_b4` 58,111) that did not ship.

**The final package as a whole** (W25-ALL6, counted from the tensors in the submitted file): about 1.84 billion parameters in 521 networks, plus 3 gradient-boosted tree models. The six ST-EEGFormer-large members (303.0M each with their heads) hold 98.9 % of the parameters. The three CNN families hold 495 networks (15 pooled networks and 150 per-user copies each) and 18.4M parameters together; 20 small side networks hold another 0.85M.

## What was never measured alone

- **BIOT, BENDR, SignalJEPA:** do not occur in the log. **EEGPT:** named once as never run (T2-D452).
- **CSP + LDA without a filter bank, and log-band-power LDA as a member:** no row. Band-power LDA appears only as
  an in-stream adaptive decoder (T2-D292). Ridge or boosted trees on covariance and band-power features are
  mentioned as too weak (T2-D515) without a row of their own in this log.
- **HEEGNet, NeuroRVQ, MIRepNet:** October protocols only. Never on the typical panel or on ten strangers.
- **EEGNet, EEGConformer, EEGNeX, ATCNet, Deep4, EEG-TCNet, the SPD network, every classical pipeline except the
  two tree bags, and every FM except the ST-L:** never on the November ten-stranger panel. EEGNet and FBTS-LR
  were in the October first screen, which also holds out each user in turn (D81).
- **The ST-L with CPT on ten strangers:** not possible under the leak rule of that panel (T2-D498).
- **The first FM screen on new-user folds (D82):** stopped by a protocol change before it was read. One number
  survives, LaBraM 0.397 (D86).
- **Frozen-encoder reads** exist only for the ST-L and MOR (T2-D126, T2-D454) and for the time-series FM
  (T2-D433). **LoRA** has numbers for the ST-L, MOR, NeuroRVQ and, at rank 8 in the October screen, REVE (D88).
- **Any model alone on the hidden set.** The hidden set scored whole packages only.
- **Any family alone on returning users, as a level.** The typical panel mixes one returning user with three
  strangers; ten strangers has none. One row gives a difference instead: what a pooled family gains on the second
  session of each of the two returning users when that user's first session is in training. ST-L +0.160 and
  +0.160; FBCNet +0.060 and +0.201; IFNet +0.132 and +0.243; ShallowNet +0.092 and +0.168 (T2-D505).

## Where rows disagree

- **The CNN order on the kit split.** ShallowNet was above FBCNet on the two leaderboard users and far below it on
  the two validation users (D88). Each pair of users gives a different winner.
- **The FM order between October and November.** In D88 the MOR variants had the best FM kit test (0.4476,
  0.4451) and the ST-L 0.4192. On the typical panel the ST-L members were 0.721–0.758 and the MOR 0.639–0.641
  (T2-D407).
- **LoRA on the ST-L:** equal to the full fine-tune in T2-D130, far below it in T2-D329 (different recipe, panel
  and month).
- **The same fine-tunes, two numbers.** The 12 role-LoRA MOR runs read 0.4612 on the kit test with the prior
  correction (T2-D130) and 0.4104 as raw argmax (T2-D314). Decoded and raw rows cannot be mixed.
- **A statement that crosses protocols.** T2-D452 gives the ST-L's lift since 13 October as 0.41 → 0.75 alone.
  The second number is a typical-panel level. The ST-L levels on record for 13 October are kit-split numbers
  (0.4192 / 0.5551, D88). The two ends are not on one scale.
- **A kill line from another scale.** T2-D554 killed a network for being under 0.70 alone and then notes that
  every shipped family is under 0.70 in that decode. The verdict held on the gap to the weakest shipped family
  instead.

## What this does not show

- **It rests on very few people.** There are ten public users. The kit split scores two plus two of them, the
  typical panel four, and only the last protocol all ten. One user often decides a comparison: EEGConformer's
  whole gap to the other families is one user (0.450 against 0.631 / 0.703, T2-D465).
- **Protocols changed three times**, and with them the users, the decode, the number of training users and the
  form of a "member". Levels of 0.4865, 0.7424 and 0.678 for FBCNet are three protocols, not three models
  (D88, T2-D491, T2-D498).
- **Recipes were not equal.** FBCNet, ShallowNet, IFNet and the ST-L had weeks of recipe work. Most other models
  had one configuration and a kill line, and several rows say so (T2-D341, T2-D465, T2-D460). A low line in these
  tables means "not competitive at the configuration tried, on these users".
- **The FM with CPT had seen two of the headline users without labels** (D89). That is allowed for returning users
  on the hidden set, and it is not what "stranger" means.
- **Lab differences shrank on the hidden set.** The last package step reached the hidden set at about 0.37 of its
  lab size (T2-D657; [`METHOD.md`](METHOD.md) gives a range up to just under one half), an earlier step at about
  half (T2-D514). Differences of a few thousandths between families in the blend tables are inside that
  uncertainty.
- **Nothing here is a significance test.** The counts "n/m" are seeds or draws up, on shared users, not
  independent samples.
- **It is one task:** three cued mental commands, 4-second windows, one recording setup. The order of the models
  may differ elsewhere.
