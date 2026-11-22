# T2 closed-with-positive audit (2026-11-16 04:30)

**What this is.** The manager relayed the owner's line at about 04:2x: "remember all little improvements should count. Now also revisit those closed ideas that do bring improvements".

**Method.** I swept every result in `docs/DECISIONS_T2.md` from D400 to D524. Before D400 I used only the backlog table's "near miss" entries.

A row is eligible when all of these hold:
- its registered stack read had a positive mean;
- it was closed for a reason other than losing;
- no later base re-read it as negative;
- it is not a form of the window mean;
- it is not a leak.

The panel and the base are named in every row, because most of these were never read on K = 9 or W17.

**Key to the "why closed" codes.** These are the manager's reasons:
- (a): the gain was under the size bar.
- (b): it was read on an older base and never re-read on W17.
- (c): it missed a guard by a small margin.
- (d): it was judged "not additive" on one draw.
- (e): it was dropped for runtime or complexity.

**Legend for the table.** Δ is the stack-level difference on the row's own typical-user mean, with the count of draws up. P is my probability that the gain is still additive on W17. The W17 runtime headroom is 3,400 − 3,186 s ≈ 214 s, which is ≈ 3.4 ms at 62 s per ms.

## Eligible rows, ranked by Δ × P

| rank | row | mechanism | Δ (draws up) | panel / base | why closed | ms / window | form | P | Δ × P |
|---|---|---|---|---|---|---|---|---|---|
| 1 | D490 / D490r | ShallowNet trained on out-of-fold cross-user soft targets | stack +0.0025 (7 / 10), replicate +0.0004 (4 / 10). Alone it is +0.0205 (10 / 10) and +0.0196 (9 / 10), and the alone gain replicated. | kit panel, W14 | (a), (b) | 0 | weights only: a ShallowNet pooled-net and copies retrain | 0.35 | ≈ 5e-4 |
| 2 | D455 / D488 | MI-expert ST-L FM member: an MI first stage on outside data, then the PROTEUS fine-tune | Five reads, all positive (detail below the table) | kit panel, M7 then W14 | (a), (c): P05 −0.010 … −0.014 broke the old −0.005 floor; the final-week floor is −0.020 | 0 (swaps one of the six FM members) | weights only; sofia FM training before the Nov 18 12:00 cut-off | 0.30 | ≈ 4e-4 |
| 3 | D487 | Imaginary-coherency family: 18 nodes × 153 pairs × 3 bands = 459 features, causal per-stream re-centring, L2 logistic, w 0.03 | +0.0012 (7 / 10). P05 +0.0028: the only addition that did not cost P05. Error correlation with the stack is 0.03. | kit panel, W14 | (a): "a size a template change cannot justify" | ≈ 0.3–0.5 (estimate, not measured) | template change | 0.35 | ≈ 4e-4 |
| 4 | D481 / D451 | Family shares re-fitted, cross-fitted | W14 +0.0009 (6 / 10), replicate −0.0000 (7 / 10); M7 fresh FM 0.45 +0.0004 (6 / 12) | kit panel, W14 / M7 | (b): never re-fitted since P1 / P2 / DC / side changed the families | 0 | config only: repack `--set` | 0.25 | ≈ 2e-4 |
| 5 | D497b | Per-stream gate between the joint and the per-family decode (g2: family class-marginal L1, from window 21) | +0.0014 over the better fixed choice; min −0.0053 (P07) | ten strangers, CNN mix, W13 decoder | (c): under the −0.005 floor (the final-week floor is −0.020) | ≈ 0.05 | template change | 0.20 | ≈ 3e-4 |
| 6 | D437 / D438 | ShallowNet user-share bags / user-level boosting | uw1 +0.0022 (8 / 12), runs +0.0016 (8 / 12), ub5 +0.0023 (10 / 12). All of it is P05 (+0.018 … +0.023); without P05 it is −0.003. | kit panel, M7 | (a) | 0 | weights only | 0.15 | ≈ 3e-4 |
| 7 | D495 | Stranger's copy chosen by causal MI on the stream prefix instead of the fingerprint route | CNN mix +0.0088 on the kit strangers (P05 / P08 / P09). Returning users were never scored under it. | kit panel, M7 / W14 CNN mix | (a): +0.02 bar | **large**: every training user's copy runs on the prefix (unpriced) | template change | 0.15 | ≈ 3e-4 (unpriced runtime) |
| 8 | D244 | FM temperature in the blend (T 2) | +0.0023 (10 / 12) | Oct-24 package (FM 0.2, before the per-family decode) | (a), (b) | 0 | template field | 0.10 | ≈ 2e-4 |

Detail for rank 2, the five D455 / D488 reads:
- M7: MI2 +0.0025 (8 / 12) and MI3 +0.0008 (8 / 12).
- W14: mi1 +0.0004 (5 / 10) and its replicate +0.0010 (5 / 10).
- W14: mi2 +0.0020 (8 / 10), descriptive.

## Not eligible: the nearest misses, with the read that closed them

| row | first read | what closed it |
|---|---|---|
| D300 / D358 FBCNet spectral tilt | +0.0024 / +0.0023 (9 / 12) | D415 on M7: −0.0011 (6 / 12) |
| D428 / D429 forgetting EA (ShallowNet) | member +0.0053 / +0.0042 | M7 shape −0.0001 / −0.0006 |
| D485 EEGNeX 4th CNN family | pooled K = 9 mix +0.0058 (5 / 5) | in the shipped shape −0.0039, P05 −0.0227 |
| D512 P3 CNN seed bag ×2 | +0.0006 | the ZIP already averages 15 seeds per family (D446e) |
| D446 top-10 wide add | +0.0067 (5 / 6) | the seed-count effect (D446e: variants − seeds −0.0006) |
| D400 / D442 ST-L tilt; D361 longer ST-L fine-tunes | — | adopted later (P2; M7's 100-epoch members) |
| D518 C5 config-diverse side nets | +0.0002 | the W17 read is noise |
| D516b DC share 0.10 | +0.0098 vs FINAL | D523 in W17: −0.0004 vs 0.075 |

**The K = 9 / W17-era rows (D498–D523) hold no closed positive** except variants that were already re-read in W17's context. Every eligible row above was read on the kit panel and an older base.

## The top three, and what each needs

1. **D487 imaginary coherency** (template change).
   - The re-read is CPU only. Its feature script `scripts/local/t2_d487_wpli.py` runs on every window, then an L2 LR per lc9 fold, then the W17 reader with the family at w 0.03.
   - If it is a part, it needs a template W18 (+ an imcoh family: numpy FFT and LR weights), the owner's line, and a runtime measurement.
2. **D490 ShallowNet soft targets** (weights only).
   - The K = 9 read needs nested OOF teachers: for each of the 10 folds × 9 training users, a teacher trained without both users. That is ≈ 90 short ShallowNet fits plus the student draws.
   - All of it fits on the local 5090 in ≈ half a day. The build is 10 teachers + 15 seeds + copies on all users.
3. **D455 / D488 MI-expert FM member** (weights only).
   - On sofia, the MI first stage plus tilt members on lc9 folds, swapping one of the six. ≈ 20–30 GPU-h.
   - It must start by about Nov 16 midday to meet the FM cut-off (Nov 18 12:00).

**Zero-GPU extras.** D481's shares re-fit and D497b's gate can be re-read on the stored D512 / W17 dumps within the hour.

**What "fresh draws" costs.** Every zero-GPU read uses draws 810–812, which already selected P1 / P2 / DC / side. The stack rule's fresh-draw read needs new member draws (GPU) for any part that passes.
