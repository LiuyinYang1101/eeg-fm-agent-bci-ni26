# T2 method: PROTEUS BCI decoding (Track 02)

> **2026-11-22: §0 below is the final state (the two files uploaded on Nov 21). `docs/T2_FINAL_STACK.md` is the Nov 14 stack; §1–§6 are the
> October history (§2 describes the late-October package).**

## 0. Final state (2026-11-22)

### 0.1 The files uploaded on Nov 21
| file | sha256 | what | lab evidence | sealed |
|---|---|---|---|---|
| **W25-ALL6** `t2p_fin2_w25all6` | `2b03eb03…ff1a` | W21 with all six FM members re-trained with user-balanced loss weights (T2-D642) | four full-chain reads vs W21: strangers +0.0075 / +0.0049 / +0.0011 / +0.0032; the unselected reads 2–4 average OLD sealed +0.0003 (a tie, picked by D648's registered rule) | **0.844** / 0.843 / 0.843, rank 10, 2,795.3 s (submission 1063122, 15:27) |
| **W30** `t2p_w30oh_PREBUILD` | `82fcb63f…acf3` | W25-ALL6 with the five headed FM members replaced by the D652 all-user donors (W25's recipe + virtual participants vp 2); the MI slot unchanged | two full-chain reads vs W25-ALL6 (T2-D652): mean strangers ΔS +0.0060 (6 / 8 users), mean OLD sealed +0.0024, leave-one-out min +0.0036; returning −0.0009 | **0.838** (submission 1063517, uploaded 22:10; duration unknown): 0.006 BELOW W25-ALL6 |

**The stack both files share (template `t2p_submission_w21lp.py`, read from the shipped configs):**
- **Families:** three CNN families plus an FM family, mixed at fixed shares: FBCNet 0.3143, ShallowNet 0.2357, IFNet 0.10, ST-EEGFormer-L 0.35. The FM family is six ST-L members: a head-less MI-expert slot and five slots with window-mean side heads. A window-DC family enters at weight 0.075.
- **In-stream adaptation** (per stream, no labels; refit points 60 / 100 / 150 windows):
  - the per-user CNN copies are refit on the package's own decoded labels (confR);
  - the side heads are refit (D539b);
  - the FM side heads are refit (D553);
  - FM prototypes are built (D560);
  - the CNN-side teacher labels are cleaned by label propagation over the FM features (cnn_lp, D569).
- **Decoder:** a per-family causal Sinkhorn (10 iterations, n0 3, forgetting λ 0.95) with shipped start vectors.

### 0.2 Sealed-score history (T2, the board's cell-mean balanced accuracy)
| upload | date | sealed | note |
|---|---|---|---|
| A1, MOR, S1 | Nov 5 | 0.805 / 0.806 / 0.808 → **0.798 / 0.798 / 0.799** | after the organisers' Nov 10 re-run |
| M7 | Nov 10 | 0.825 → **0.819** | |
| W14-lite | Nov 15 | **0.827** | |
| W18 b1 | Nov 17 | **0.835** | 2,782.57 s |
| W25-ALL6 | Nov 21 | **0.844** | 2,795.3 s, rank 10 |
| W30 | Nov 21 | **0.838** | below W25-ALL6 |

**T2 FINAL = 0.844 (W25-ALL6).**

**W30's miss, in one line (T2-D659):** the lab's +0.0024 priced the returning stratum with a two-person full-chain read (−0.0009). The member-level returning reads on every training user had said −0.0206 (rt1) / −0.0114 (rt2). Post hoc, the same arithmetic with rt1 gives −0.0050 against the sealed −0.006. D582 had failed this strangers-up / returning-down pattern's returning guard on the earlier base.

### 0.3 Transfer and runtime (T2-D657)
- **Transfer of the last step:** the sealed gain is +0.009 over W18 (+0.008 to +0.010 after the board's rounding).
  - Our lab's W18 → W25-ALL6 gain:
    - W20+D557 vs W18's form: +0.0177 [+0.0121, +0.0233]: D573's participant bootstrap on D571's real-teacher forms (recomputed starts), recorded in the T2-D571 row. D573's own primary read (shipped starts) is +0.0171 [+0.0113, +0.0224] (strangers +0.0170, returning +0.0173);
    - W21's cnn_lp: about +0.0015 (sets C / D +0.0017 / +0.0012);
    - plus D642's step.
  - Valued at its registered decision mean (OLD sealed +0.0050), the total is +0.0242 (+0.0236 with D573's primary), so **τ ≈ 0.37 (0.38)**.
  - Valued at its unselected reads 2–4 (+0.0003), the total is +0.0195 (+0.0189), so **τ ≈ 0.46 (0.48)**. Range: τ ≈ 0.37–0.48.
  - W18's own step had realised about half.
- **Runtime:** the D517 line (62.0 × ms − 163, fitted on earlier sealed runtimes) overstated the RTX PRO 4500 worker by 13.8 % (W18: 12.4 %). Scaling W18's measured run by the H200 ms per window was within 1.4 %: use that.

### 0.4 The one-hop build path (W30, Nov 21) and its gates
W25-ALL6 and W21 differ in their configs only in `fm_members`. Every hop of the tested build chain after its first (`--fm-from`) is a template or `--set` config step, and the member-dependent refits run in-stream. So a swap of FM members can be built as ONE repack of the previous final ZIP:
`t2p_repack --zip <W25-ALL6> --template t2p_submission_w21lp.py --fm-from <merged donor>` (`tracks/t2_bci/sofia/launch_w30_onehop.sh`).
- **The one member-derived constant** (the FM decoder start `tta.per_family.fm`) is not recomputed by the tested chain either, so the one-hop is exactly as stale as W25-ALL6.
- **Gates used on Nov 21** (the ZIP pinned by its sha256 after the build):
  1. **K1-only on the pinned ZIP** (`t2p_k1_only.slurm`): the kit replay and the D553 / D560 / D569 line checks.
  2. **The FM-swap package check** (`t2p_w30_pkgcheck.py`, pkgcheck kind `w30`):
     - the config diff is `fm_members` only, and the non-FM files are byte-equal to the base's;
     - outputs are finite and rerun-equal on all 14 streams;
     - the paired ms per window vs the base is the timing gate (within ±2 % inherits the base's timing).
  3. **K2** (the kit's streaming harness == the check's predictions).
  4. **The structural zipcheck** (`t2p_w25_zipcheck.py --keep-slots 0 --d642-ids …`):
     - 4b: every member and side head byte-equal to the merged donor's slot files;
     - 4c: every `fm_members` entry equal to the donor's, key by key;
     - 5-lite: each slot's source job and seed;
     - plus Track3's independent structural check.
- **The repack's parts check**, fixed 2026-11-22 (a6113198, `tests/test_t2p_repack_parts_gate.py`):
  - On an FM-adapting base it used to refuse by construction. Its static reference never models fm_side_refit / fm_proto / cnn_lp, and the Nov 21 build 1649496 exited before K1 on it.
  - Now, for `--fm-from` onto such a base, the CNN parts p1 / p2 / p3 stay strict on every window. The FM part, the recombination and the side heads are strict on the windows before the first refit point, where the adaptation is inert.
  - Every other repack keeps the strict all-window form.
- **Named limitation:** NEW's adapted FM path after the first refit is exercised (K1, K2, the package check) but not compared with an independent reference.

### 0.5 Post-competition list
- **W29 (T2-D649), the FM subject-adversary:** engaged at λ 0.1 (the twin's adversary accuracy fell from 0.986 to 0.091; chance 0.111). It was cut for capacity and never read.
- **D654, the FM view-TTA:** cut for capacity. Its read needs the bw21 FM splices regenerated (they were pruned on 2026-11-22; `setread/bw21/PRUNED_fmspl_2026-11-22.md`).
- **The D658 spread finding:** a 4× FM-share step (0.35 → 0.55, the risky alternative for the last upload) gave the wanted per-user spread (+0.026 … −0.042) around a mean of −0.0035. A config knob bought variance at about the price of the mean (hub LESSONS A64).
- **The unexplained sign flip:**
  - Under a session shuffle, forgetting λ 1.0 beat 0.95 on the old stacks M7 / S1 (+0.0044 / +0.0041, D448).
  - It lost on W25-ALL6 (OLD sealed −0.0069, D656).
  - A candidate, untested: the in-stream adaptation runs in recording order even in a decode-only reorder, and M7 / S1 had none.
  - Related (A63): decode-only reorder LEVELS are not comparable across orders. Only within-order contrasts are.

T2-D178. This is a summary for the lead and the workshop report. Every number below comes from a pre-registered row in
`docs/DECISIONS_T2.md` (cited as Dxxx) or a row in `docs/SUBMISSIONS.md`. Where a number and its row disagree, the row wins.

## 1. Task and protocol
- Sealed task: 3 cued commands (kinesthetic MI / mental calculation / word association) from 4 s windows (41 EEG
  channels, 120 Hz after the kit pipeline).
  - Two groups of users: later sessions of the 10 public users, and 10 new users.
  - `predict()` sees one window at a time, with no user or session id.
- Metric: balanced accuracy, averaged over subject × session × context cells.
- Development split = the start kit's (D88; user 2026-10-13: "stick to the competition setting"):
  - **kit**: train P01 P02 P03 P06 P07 P10, val P05 P08, test P04 + P09 (the leaderboard users);
  - **seen1**: session 1 of P04 / P09 / P10 in training, their later sessions scored as returning users.
- Package-shape objective (D124c / D141): new = (kit test + seen1 val) / 2; mix = (seen + new) / 2, over 12 seed-paired
  indices.
- Adoption bar d141: mix Δ ≥ +0.005 with ≥ 9 / 12 indices up AND kit test Δ ≥ −0.005. Every bar was fixed before its
  numbers existed (hub LESSONS F1).

## 2. The shipped package (from T2-D159 / D160, template 19a7edb)
Output = 0.5 FBCNet + 0.2 ShallowNet + 0.2 ST-EEGFormer-L + 0.1 our MOR, then a causal output-prior correction
(α 1, n0 3) per stream.

| member | weight | what | rows |
|---|---|---|---|
| FBCNet (B3 bank: 13 bands, +1–4 and +40–58 Hz) ×10 seeds, align none | 0.5 | pooled net blended at r 1/3 toward the routed user's own fine-tuned copies (lr 1e-3, 80 epochs) | D115, D122, D124c, D156, D159 |
| ShallowNet ×10, causal Euclidean alignment | 0.2 | pooled net blended at r 1/2 toward the routed user's copies (lr 1e-3, 40 epochs) | D116, D158, D159, D160 |
| ST-EEGFormer-L ×1, CPT on PROTEUS + Dreyer + Scherer, gain calibration, mixup 0.4 | 0.2 | pooled | D94, D111, D112a |
| our MOR (mor_p1v2), role-specific input-dropout LoRA r16, merged | 0.1 | pooled | D130, D133, D137, D140 |

- **Routing:** each stream keeps a running mean of a session-stable fingerprint (D124b: IAF, 1/f slope, relative-band
  topography, covariance shape). It is projected to an LDA space of the training users; the nearest centroid picks the
  copies. There is no gate.
  - Returning users are routed to themselves (P04 / P10 later sessions: 1.000).
  - New users are routed to their most similar training user.
- **Runtime:** ≈ 12.5 ms per window on one H200 (D162, D173). That projects to ≈ 21 min per sealed run against the
  60 min limit.
- **Output decoder: causal Sinkhorn, 10 iterations (SK10; T2-D285 / D286; the user approved it 2026-10-27; it ships iff its
  warm-up twin T2-W7 Finishes, else the prior correction above stays).**
  - **What it does:** within each session stream, at window t it takes the package's output probabilities for windows 1..t
    (plus 3 uniform pseudo-windows). It rebalances them, 10 times over, so that each class has equal total mass. It then
    outputs the rebalanced row of window t. One iteration is exactly the prior correction above.
  - **What it assumes (disclosure):** only that the three classes are balanced within a session, which is what the
    shipped prior correction already assumes. It uses no labels, no future windows, nothing across sessions and no run
    lengths or cue schedule.
  - **Not shipped:** a run-quota decoder (QI) that used the Online runs' fixed 30-window, 10 / 10 / 10 schedule.
  - **Read:** package-shape mix +0.0071 / +0.0068 over the prior on two independent seed sets, 12 / 12 indices up in both.

## 3. What moved the package (adopted, in order)
| step | change | read | rows |
|---|---|---|---|
| 1 | three families: FBCNet 0.6 + ShallowNet-EA 0.2 + ST-L 0.2 | test +0.0145 (7 / 7), val +0.0148 | D116 |
| 2 | FBCNet B3 13-band bank | test +0.0121 (12 / 12) | D122 |
| 3 | the D124c user blend + role-LoRA MOR 0.1 = package (c) | mix +0.0117 (12 / 12); seen +0.0184, new +0.0049 | D137 |
| 4 | per-user copies of BOTH CNN families, ShallowNet blend r 1/2 | mix **+0.0141 (12 / 12)**, kit test +0.0141, seen +0.0167, new +0.0115 (fresh seeds) | D159, D160 |

- Step 4 decomposes into two parts:
  - the FBCNet copies lift **returning** users (seen +0.010);
  - the ShallowNet copies lift **new** users through their nearest training user (kit test +0.008).
- Each part missed the bar alone (D156 +0.0049, D158 +0.0026). They add (D159), which was confirmed on seeds that
  neither single read used.
- Official-kit check of step 4 (D167): the kit-split twin replays the warm-up at 0.5677, against 0.5606 for the step-3
  kit package (T2-W3, same seeds, one draw). The last official warm-up score, T2-W2, was 0.54354.

## 4. Closed levers (pre-registered, not adopted)
- **Pooling and calibration:**
  - geometric pooling (D104);
  - prior re-picks (D152; D170 with copies);
  - weight re-picks (D145; D163 with copies);
  - blend-r re-picks (D145, D157, D171, D172).
- **Test-time adaptation:** BN statistics (D100), self-training (D113), MAE inside predict (D125).
- **CNN recipe:** augmentation (D112b), mixup (D146), weight EMA (D153), a fourth family (D118).
  - Per-user copy dose is closed both ways: FBCNet 160 epochs (D168); ShallowNet 20 / 80 epochs (D164, D169).
- **Routing:** distance-gated r (D124b), soft top-2 (D165).
- **FM pretraining (LESSONS C29):** every continued or new pretraining of the MOR lost to the pinned weights as T2's
  member:
  - MAE CPT through the task front end (D123, D129);
  - TP-1 from scratch (D141);
  - TP-2 to 50k steps (D144, killed at −0.084 / −0.107);
  - TP-W warm start through its own front end (D151).
- **FM harnessing:**
  - LoRA, regularisation, readout, SupCon, joint retention (D125–D134; role-LoRA is the one pass);
  - per-user heads on the frozen MOR (D166, +0.0004: the user signal sits in the encoder).
- **Infeasible on time per window (D173):** ST-L per-user copies (D174), 15-seed CNN families (D175).

## 5. Sealed candidates (none uploaded by T2; the user uploads)
- Six ZIPs on the step-4 template pass the D140 bars (D162):
  - golden 1.000, worker guard, kit Simulated + warm-up replays, clean replay 0 differences, robustness all_pass;
  - 12.0–12.6 ms per window.
- They are the big-twin line (seeds 0–9), four seed draws (20–29 … 80–89), and the D140 structural line.
- SHA-256 and paths: `docs/SUBMISSIONS.md`.
- They are proposed to replace `f403527d`, D150's four draws and `94ccbbc1` one for one (PLAN §6b item 8). The user
  decides.
- T2-W4 (candidate): the kit-split twin `b012b1a1…`, for an official warm-up number of the new template.

## 6. Compute
- The challenge's cluster allocation, tag ni26. The T2 keep-busy GO was 200 GPU-h from 2026-10-19; ≈ 110 used by 2026-10-21.
- Per job: `challenge_hub/ledger/BUDGET.md` (every submission carries an `S1_PURPOSE` naming its row).
- Local development: one RTX 5090.

## External unlabelled data for continued pretraining (declared 2026-11-05)

- **Shipped (CPT v1, D94 / D111):** PROTEUS (NEMAR nm000290, CC BY 4.0), Dreyer2023 (wide windows), Scherer2015 (MOABB / NEMAR nm000144, BNCI2015_004, **CC-BY-NC-ND-4.0**). Labels are not used.
- **J-CPT arm A (T2-D350; only if one of its checkpoints ships):**
  - Arm A's pool only (NOT the shipped CPT v1 above; Shin2017B and the others below are in no shipped or shot-list T2 model): all T2 CPT pools: PROTEUS + its session crops (P01 P02 P03 P06 P07 + P10 session 1 only), Dreyer2023, Cho2017, Lee2019_MI, PhysionetMI, Schirrmeister2017, Shin2017B and Stieger2021.
  - The CC-BY-NC-ND sets, back in on the user's ruling of 2026-11-05: BNCI2014_001 (= BCI Competition IV 2a), BNCI2015_001 and Scherer2015. They are public and redistributable verbatim; they are used unlabelled, and no data is redistributed.
  - T1's image-EEG pools (Alljoined stim_train, THINGS-EEG1 / EEG2, NOD) and T3's Muse / PSG-Muse pools, each declared by its own track.
  - Licences of the other MOABB-style sets: verify on each source page before a J-CPT checkpoint ships.
  - **Out:** Weibo2014 (licence unverified), and the non-redistributable TUH and SEED.
