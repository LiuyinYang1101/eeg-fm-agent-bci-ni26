# T2 — the arms cancelled or never read, 2026-11-20 (the audit's unread-arms list)

None of these arms were ever read: no number from them exists anywhere, and none fed any decision. Each was cancelled (or never submitted) before
its first task started, and its dirs were deleted unread where any existed.

- **The final:** W25-ALL6 (the owner's item 167, typed 18:44; Track3's fresh review PASS; PROMOTED 19:17, sha256
  2b03eb037d3ac4962ec5d54e146364c1a5fc6bacb57fcf976b55614b9714ff1a). W21 is the fallback.
- **The one item read tonight after the promotion:** T2-D655 (fm_weight 0.40), local, NOT included.
- **Sources:** DECISIONS_T2 rows T2-D649 / D651 / D652 / D654 / D655 and outputs/t2_d649/reads.log, t2_cutoff.log, scratch_triage/.

## 1. The engagement pick (T2-D649, rule 3, 15:36): the unpicked λ arms of the screen
The smoke picked λmax = 0.1 by the adversary's held-out accuracy alone. The screen's other two λ arms were s1-cancelled and deleted unread by
outputs/t2_d649_reads.sh.

| arm | arrays (job ids) |
|---|---|
| G03 (λ 0.3), s0–s5 | 1646536 1646537 1646538 1646539 1646540 1646541 |
| G1 (λ 1), s0–s5 | 1646542 1646543 1646544 1646545 1646546 1646547 |

## 2. The CAPACITY TRIAGE (the manager's GO 17:2x): cut, not tested, not failed
At 17:16, zen4_h200 was fully allocated by others (21 / 22 nodes; 1 drained). We had 17 tasks running and 356 pending (reason Priority). These were
s1-cancelled before any task started (0 dirs to delete):

| item | what | arrays (job ids) |
|---|---|---|
| **W29 (T2-D649), entirely** | the member screen C + G01, six slots | 1646521 1646522 1646526 1646527 1646528 1646529 1646530 1646531 1646532 1646533 1646534 1646535 |
| | the replicate C + G01 at 1300 + slot | 1646713 1646714 1646715 1646716 1646719 1646721 1646723 1646724 1646726 1646727 1646729 1646731 |
| | the rtl arm G01 at 1400 + slot | 1646705 1646706 1646707 1646708 1646709 1646711 |
| | the stack's G arms (s0, s1) | 1647474 1647482 |
| every rtl array (descriptive since 16:29) | W30's d652rl | 1646513 1646514 1646515 1646516 1646517 |
| | D654's d654rl | 1646828 1646829 1646830 1646831 1646832 1646833 |

**W29 → the post-competition list:** "engaged at λ 0.1 (the twin's adversary accuracy 0.986 → 0.091, chance 0.111), never read (queue)".

## 3. Never submitted (no ids)
- D654's 1700 set (d654b): 0 / 6, refused at the cap, then replaced by the queued C@1800 arrays.
- The stack fits G s2–s5, V s1–s5 and J s1–s5 (paused under the cap, then dead with the W29 cut).
- The W29 donors, the J donors and every W29 / J PREBUILD step (W29 cut before its screen read).

## 4. At the cut-off (t2_cutoff_sched.sh, with the owner's +3 h shift: 05:00 both if nothing runs; 06:00 D654; 09:00 W30)
**Executed 2026-11-21 05:00:43–05:00:49** (outputs/t2_cutoff.log). The 05:00 check found 0 of the arrays running and 0 ended since 20:00, so
BOTH parts were cut at once (`t2_cutoff.sh GO ALL`). Only ids from the launch logs were touched, intersected with the pending set: 20 arrays,
all s1-cancelled and verified gone from squeue. None had started a task, so there were no dirs to delete. The parts are recorded as
'cut for capacity (never read)' (setread/bw21/cut_W30.json, cut_D654.json), and `--final` derived **final_pick = W25-ALL6** by the rule.

| part | arrays (job ids) | status |
|---|---|---|
| W30 (T2-D652) | 1646511 (d652 s5) · 1646548 1646549 1646550 1646551 1646552 (d652b) | **cancelled 05:00**, pending, never started |
| D654 | 1646821–1646826 (d654a) · 1647465 1647467 1647468 1647469 1647470 1647471 (stk C) · 1646864 (H200 probe) · 1647160 (W31v golden) | **cancelled 05:00**, pending, never started |

The waiting chains (t2_d652_chain_v3.sh, t2_d654_chain_v3.sh, t2_w30_spec.sh) were stopped at 05:01. They submitted and read nothing after
the cut (LESSONS E9).

W30's member arrays s1–s4 (1646490 1646491 1646492 1646505) ran to completion on sofia, but were never fetched or read. (Reopened on 2026-11-21: see §5.)

## 5. W30 REOPENED, 2026-11-21 11:02: a post-deadline INFORMATION read (the owner's instruction)
The owner at 11:0x, relayed verbatim by the manager: "in the meantime test w30, I will upload w25". The read cannot change the Nov 21 upload
(W25-ALL6). The final record stays W25-ALL6: cut_W30.json is kept, and `--final` is unchanged. Registered in the T2-D652 row before any number.

- **No longer unread:** W30's member arrays s1–s4 (1646490 1646491 1646492 1646505) are fetched and read with the relaunched s5.
- **Still cancelled unread:** the arrays cut at 05:00 (1646511, 1646548–1646552) stay cancelled and were never read. Their arms run again under
  new job ids, listed below once submitted. S1_PURPOSE on each: "reopened on the owner's instruction 2026-11-21, post-deadline information read".
- **D654** stays cut and unread; it is not reopened.

| what | arrays (job ids) | status |
|---|---|---|
| s5 member array (t2d652_s5_stlcarsd_V, 6 tasks) | 1649143 (11:03, snapshot 8dac51f7) | relaunched for the member sign gate; 3 tasks running 11:04 |
| read-2 fits d652b (5 × 6 tasks) | 1649148 1649149 1649150 1649151 1649152 (11:05; --dependency=after:1649143 --nice=1000) | behind s5; cancelled on a member-gate FAIL, or at Nov 22 08:00 if any task has not started |

Chain: outputs/t2_w30_reopen_chain.sh (log outputs/t2_d652/reopen.log; set reads via t2_setread_chain_v2.sh, deadline Nov 22 12:00).

**The result (2026-11-21 15:59:44):** the member gate PASSED (pooled dS +0.0106); set read 1 dS +0.0076 / OLD sealed +0.0029 (Track3's re-score: exact);
set read 2 dS +0.0045 / OLD sealed +0.0019 (Track3's re-score: exact); `--decide652` = **INCLUDED** (confirmed by Track3) (mean dS +0.0060, mean OLD sealed +0.0024, both read-2 legs > 0).
Information only. **W30 could not ship today under any choice:** it needs five NEW all-user donor fits (the set reads used the fold-pair member
fits), then W25's build chain, about 7 h at best plus queue waits. The manager (16:1x): no donors, no build (E9). The final stays W25-ALL6.
The fold-pair member fits and dumps (sofia_s1/ni26-t2p-d652-s*_V-*, ni26-t2p-d652b-s*_V-*) are KEPT for the audit record; never prune them.

**Why the pass could not ship (the manager's registration):** The reopen was registered at 11:02 with 'no donors, no build' on the MANAGER's instruction (the manager's 11:0x message: "No donors and no build: […]"), so a pass could not ship. The manager's own statement (16:2x): had the all-user donors and the W30 PREBUILD been started speculatively at 11:02, a gated file might have existed by about 19:00-21:00 (about 9 GPU-h); 'that is my error' (the manager's registration, not Track2's). For the record: at 11:02 neither the member gate (12:37) nor the inclusion (15:58) was known […].

## 6. The post-competition list (T2)

| item | state | what it would take |
|---|---|---|
| **W29** (T2-D649, the FM subject-adversary) | engaged at lambda 0.1 (the twin's adversary accuracy 0.986 -> 0.091, chance 0.111), never read (queue; §2) | the member screen + two set reads |
| **W30** (T2-D652, W25's C lines + vp 2) | INCLUDED by the inclusion rule on two reads (mean dS +0.0060, mean OLD sealed +0.0024; §5); not built | five all-user donors (launch_w30_donor.sh GO all) + launch_w30.sh (W25's chain), about 7 h; then the owner's typed W30 line |

**To fix before any W30 GO (unfixed, noted 2026-11-21 15:59:44):** `tracks/t2_bci/sofia/launch_w30.sh` GO greps `decide_d652.json` for `"adopt": true`,
but `t2_d642_set_read.py --decide652` writes `"included": true`. GO would refuse on a pass. Its header also still quotes the old D652 rule
(dS >= +0.003, OLD sealed >= -0.001, sealed* >= +0.002), not the owner's 16:31:32 inclusion rule. PREBUILD is unaffected.

