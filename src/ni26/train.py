"""Compact-CNN training for Track 2 — one transparent loop for every braindecode model.

Design choices (each one is a lesson, see LESSONS.md):
* The whole fold lives on the GPU as float32 (windows are small); batches are index gathers → no loader noise.
* Selection is NEVER on test (A1/A2): an INNER VALIDATION set made of other training users' LATER units (the
  sealed situation in miniature) picks the epoch ("best"); "last" = fixed-epoch model. Both are reported.
* Test metrics are logged per epoch for analysis of the dynamics only (flagged ``test_*``), never used to choose.
* Per-window test probabilities are saved so late fusion (TSF two-stream) can be evaluated without retraining.
* Deterministic seeds; cudnn.benchmark off.
"""

import copy
import inspect
import math
import os
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics


@dataclass
class Recipe:
    name: str = "r1"
    epochs: int = 60
    lr: float = 1e-3
    wd: float = 1e-2
    bs: int = 64
    label_smoothing: float = 0.1
    warmup_frac: float = 0.1
    mixup: float = 0.0          # alpha of Beta(alpha, alpha); 0 = off
    max_norm: float = 0.0       # gradient clip; 0 = off
    # T2-D112 session-shift augmentation (all off = r1 exactly; they draw from the torch RNG only when on)
    gain_jit: float = 0.0       # per-window per-channel gain x exp(U(-g, g))
    tshift: int = 0             # per-window circular time shift, U{-s..s} samples
    chdrop: float = 0.0         # per-window channel dropout probability (whole channel zeroed)
    # T2-D111 user-aware heads (fbcnet_uid / fbcnet_moe): weight of the auxiliary user-id CE
    uid_w: float = 0.5
    # T2-D153: EMA decay of the parameters AND buffers per optimizer step (0 = off = r1 exactly); fit_eval's last-epoch
    # predictions and state_last (the packaged / per-user-FT starting weights) are the EMA's; the curves and `best` stay raw
    ema: float = 0.0
    # T2-D154: lr / epochs of the per-user fine-tune copies (t2p_bench --peruser, t2p_final); the defaults = PERUSER = r1 exactly
    pu_lr: float = 3e-4
    pu_epochs: int = 20
    # T2-D193: "virtual participants": every training window gets a random draw from VP_BANK (a spherical-spline remap of jittered
    # electrodes x per-channel gains x a global shift; set by the caller, t2p_bench.vp_bank); 0 = off = r1 exactly; k = electrode jitter k x 5 mm (T2-D195)
    vp: int = 0
    # T2-D200: strength of the bank's draw, M -> I + a (M - I), gains -> 1 + a (g - 1) (1.0 = the bank as drawn, bitwise); pu_vp > 0 = the
    # per-user copies also train through the bank at strength pu_vp (0 = unaugmented copies, T2-D193..D199)
    vp_a: float = 1.0
    pu_vp: float = 0.0
    # T2-D202: the number of pre-drawn virtual participants in the caller's bank (t2p_bench / t2p_final vp_bank K; 64 = T2-D193..D201)
    vp_k: int = 64
    # T2-D214: the bank's per-channel gain half-width (U(1 - g, 1 + g)) and global shift range (U{-s..s} samples); 0.1 / 1 = T2-D193..D213
    vp_g: float = 0.1
    vp_s: int = 1
    # T2-D215: "virtual references": each bank draw's map is followed by a random partial re-reference, x -> x - a 1 (w . x), with w a
    # Dirichlet(1) weighting of 1-4 random electrodes and a ~ U(0, vp_r) (its own generator; 0 = off = the bank of T2-D193..D214, bitwise)
    vp_r: float = 0.0
    # T2-D224: sharpness-aware minimisation of the pooled net (Foret et al. 2021): each step takes the gradient at w + sam * g / ||g||
    # (the same augmented batch, BatchNorm running statistics not updated by the second pass); 0 = off = r1 exactly; plain CE only
    sam: float = 0.0
    # T2-D226: label smoothing of the per-user copies (peruser_for); 0.1 = PERUSER's = every copy before T2-D226, bitwise
    pu_ls: float = 0.1
    # T2-D227: weight averaging of the per-user copies (SWA, Izmailov et al. 2018): the copy = the uniform average of its weights and float buffers
    # at the end of each of its last round(swa x epochs) epochs; pu_swa = the pooled recipe's value passed to the copies; 0 = off = bitwise unchanged
    swa: float = 0.0
    pu_swa: float = 0.0
    # T2-D228: which part of the per-user copy trains: "all" (every copy before T2-D228, bitwise), "head" (the classifier `final_layer.*` only),
    # "feat" (everything but the classifier); frozen parameters keep the pooled net's values (BatchNorm running statistics still follow the
    # user's windows, as in "all"); pu_train = the pooled recipe's value passed to the copies
    train_part: str = "all"
    pu_train: str = "all"
    # T2-D229: "virtual IAF": every pooled training window is time-warped by a random factor a (log-uniform over [1 / (1 + fw), 1 + fw], FW_K
    # values), which maps a frequency f to f / a (T2-D191's FFT resample + reflect pad / centre crop, as a fixed L x L matrix per factor);
    # simulates users whose rhythms sit at other frequencies (between-user IAF sd 1.08 Hz, T2-D187); 0 = off = bitwise unchanged; copies never warp
    fw: float = 0.0
    # T2-D230: "virtual users from real covariances": every pooled training window of training user u is re-coloured toward another training
    # user v (uniform, v != u) by M = I + rc (T_uv - I), T_uv = the optimal-transport (Bures) map from N(0, C_u) to N(0, C_v), C = the users'
    # mean spatial covariances of the training windows (rc 1: the window's covariance becomes v's); before the vp draw; 0 = off = bitwise unchanged
    rc: float = 0.0
    # T2-D236: rc_lo > 0 -> each window's strength is drawn U(rc_lo, rc) (M = I + s (T_uv - I)) instead of the fixed rc; 0 = off = bitwise unchanged
    rc_lo: float = 0.0
    # T2-D237: rc_mix 1 -> each window's target = the Bures midpoint of two other training users' covariances (a virtual user, not a real one);
    # 2 -> the real targets and the midpoints, drawn uniformly over both; 0 = off = bitwise unchanged
    rc_mix: int = 0
    # T2-D412: rc_cls 1 -> CLASS-CONDITIONAL re-colouring: a window of user u, class c moves toward user v's class-c mean covariance
    # (M[c, u, v] = I + rc (T_{uc,vc} - I), the same Bures map per class) instead of v's class-pooled one; the GA panel's "learned generator"
    # at its small-data optimum (a class x user coloured Gaussian), applied to real windows; fixed strength, real targets only; 0 = off = bitwise unchanged
    rc_cls: int = 0
    # T2-D413: syn > 0 -> every pooled-net batch of B real windows gets round(syn x B) SYNTHETIC windows appended, drawn uniformly (their own
    # torch.Generator, seed + 4241) from SYN_BANK (the protocol's MeanFlow bank of interpolated new-people windows, t2p_meanflow.py); they take
    # the same augment (rc from their dominant parent user); plain CE over the joined batch; the copies never see them; 0 = off = bitwise unchanged
    syn: float = 0.0
    side_mix: float = 0.0       # T2-D580: alpha of Beta(alpha, alpha) for a JOINT mixup of (x, the side vector, y) in the side-head branch; 0 = off (bitwise)
    # T2-D589 (the in-stream P1 students trained on noisy pseudo-labels): pu_mix = the pooled recipe's alpha of Beta(alpha, alpha) mixup for the
    # COPIES' fine-tune (peruser_for passes it on as ft_mix; finetune() reads ft_mix only, never mixup); 0 = off = bitwise unchanged
    pu_mix: float = 0.0
    ft_mix: float = 0.0
    # T2-D239: the per-user copies' data: user k's own windows + round(pu_rcx x n_k) windows of the other training users (drawn without
    # replacement, uniform over their windows), each mapped onto k's covariance by T2-D230's full map (rc_bank at 1.0, M[v, k]: T C_v T = C_k);
    # read by the copy loops (t2p_bench), not passed to the copy recipe; 0 = off = the copies' data bitwise unchanged
    pu_rcx: float = 0.0
    # T2-D293: user-adversarial (DANN, Ganin et al. 2016) pooled training: a linear user classifier on the input of `final_layer`, behind a
    # gradient-reversal layer of fixed strength user_adv, CE on the window's source training user (`utr`); the classifier is discarded after
    # training (state_dict, packages and per-user copies unchanged); plain CE branch only; 0 = off = bitwise unchanged
    user_adv: float = 0.0
    # T2-D374: TUNED DANN (all 0 = T2-D293's fixed linear adversary, bitwise): adv_ramp 1 -> lambda(p) = user_adv (2 / (1 + e^(-10 p)) - 1),
    # p = the training progress; adv_hidden > 0 -> a 2-layer MLP adversary (d -> adv_hidden -> n_users); adv_lr_mult = the adversary's lr
    # multiple (its own AdamW param group); adv_cond 1 -> CLASS-CONDITIONAL (CDAN-style): one adversary per class, each on its class's windows
    adv_ramp: int = 0
    adv_hidden: int = 0
    adv_lr_mult: float = 1.0
    adv_cond: int = 0
    # T2-D315: group DRO over the TRAINING USERS (Sagawa et al. 2020, online): per-user batch-mean CE, user weights q_u <- q_u exp(gdro * loss_u)
    # kept across steps, loss = sum_u q_u loss_u over the users in the batch (q renormalised over them); 0 = off (plain CE, bitwise)
    gdro: float = 0.0
    # backlog 70: MixStyle (Zhou et al. 2021) across training users: with probability mixstyle per batch, each window's per-channel time mean / std
    # at one map (MIXSTYLE_AT: FBCNet spatial_conv's output, ShallowNet bnorm's) are mixed, Beta(0.1, 0.1), with those of a window of ANOTHER
    # training user AND THE SAME CLASS in the batch (the per-channel std after spatial filtering is these nets' class signal, so an any-class
    # partner is label noise: one-seed smoke 10-28, ShallowNet −0.07 / −0.09); training only (eval = the plain net; packages / copies
    # unchanged); plain CE branch; 0 = off = bitwise unchanged
    mixstyle: float = 0.0
    # T2-D365 (outlier B6-1): user-level feature re-styling on final_layer's input (training only; ustyle_hook): per training user, running
    # per-element mean / variance of that input (EMA 0.1, detached); with probability ustyle per window, f <- (f - mu_u) / sd_u * sd_t + mu_t,
    # mu_t = (1 - ustyle_s) mu_u + ustyle_s mu_v (sd likewise), v != u a uniform other training user. User-level statistics average over
    # the user's classes, so no window's label moves (unlike backlog 70's per-window MixStyle, T2-D297); copies unchanged; plain CE branch;
    # 0 = off = bitwise unchanged
    ustyle: float = 0.0
    ustyle_s: float = 0.75
    # backlog 73: same-class cross-user input mixing (after augment): with probability xmix per window, x <- (lam x + (1 - lam) x_b) /
    # sqrt(lam^2 + (1 - lam)^2), x_b a random in-batch window of the SAME class from ANOTHER training user, lam ~ U(0, 1); a window with no such
    # partner is unchanged; the label is kept; plain CE branch; 0 = off = bitwise unchanged
    xmix: float = 0.0
    # backlog 74: a random spectral tilt per window (the aperiodic 1/f slope differs between users): rfft x ((f + 1) / 11)^beta, beta ~ N(0, tilt^2)
    # (amplitude; = 1 at 10 Hz; zero phase, linear, one window), at TILT_FS; 0 = off = bitwise unchanged
    tilt: float = 0.0
    # T2-D308: virtual sessions: with probability vs per window, x <- M x, M a random same-user drift map from VS_BANK (Bures maps between the
    # caller's training sessions of one user, and between the halves of each training session; t2p_bench.vs_bank); 0 = off = bitwise unchanged
    vs: float = 0.0
    # backlog 69 (FM -> CNN distillation, Hinton et al. 2015): kd = alpha > 0 -> the plain CE becomes (1 - kd) CE + kd T^2 KL(p_T || q_T) on
    # the PROTEUS training windows, p = data["ptr"] (the protocol's FM teacher's probabilities of each training window before augmentation),
    # p_T ~ p^(1 / T), q_T = softmax(logits / T), T = kd_t; kd_ext 1 -> + kd T^2 KL on a same-size batch of unlabelled external windows per step
    # (data["Xkd"], teacher data["pkd"]; drawn by their own generator, no augmentation, forwarded with BatchNorm / dropout in eval mode so the
    # running statistics stay PROTEUS's); plain CE branch only; kd 0 = off = bitwise unchanged
    kd: float = 0.0
    kd_t: float = 2.0
    kd_ext: int = 0
    # backlog 65 (a): rc_ext 1 -> the re-colouring targets are the external users of RC_EXT_BANK (t2p_covbank: per-subject mean covariances on the
    # PROTEUS montage, set by the caller) instead of the other training users; each target is trace-matched to the source user (the window's
    # spatial shape changes, its power does not); one target drawn per window; 0 = off = bitwise unchanged
    rc_ext: int = 0
    # backlog 65 (b): a rigid cap shift per virtual participant: each bank draw's jittered electrodes are also rotated together about the
    # head centre by a small random rotation whose common scalp displacement has RMS vp_cap metres (t2p_bench.vp_bank cap, its own
    # generator); 0 = off = the bank of T2-D195..D215, bitwise
    vp_cap: float = 0.0
    # T2-D430 "stranger-safe copies" (panel P7 seed M1): pu_kl = beta > 0 -> each per-user copy step adds beta x KL(p_pooled || p_copy) on a
    # same-size batch of the OTHER training users' windows (the caller's X_other; p_pooled precomputed once from the pooled net in eval mode;
    # drawn by its own generator, no augmentation; the copy forwarded with BatchNorm / dropout in EVAL mode, as kd_ext, so its running statistics
    # stay its user's): the copy specialises on its user and stays pooled-like on strangers. pu_kl = the pooled recipe's value passed to the copies
    # as kl_other; 0 = off = bitwise unchanged
    pu_kl: float = 0.0
    kl_other: float = 0.0
    # T2-D478 (seat G10 item 1, "train as we decode"): sbal = k > 0 -> stream-balanced batches (bs // k streams x k windows, each stream's windows
    # shuffled and cut into chunks of k, the chunks shuffled) and the loss CE(z - log m_s), m_s = the batch's per-stream mean softmax (detached):
    # the loss is blind to each stream's class prior, as the per-stream Sinkhorn decoder is. Pooled nets and copies (peruser_for passes it on).
    # Needs data["str"] (the training windows' sessions; stream = user x session). 0 = off = bitwise unchanged
    sbal: int = 0


RECIPES = {
    "r0": Recipe("r0", epochs=40, lr=1e-3, wd=1e-2, label_smoothing=0.0),   # = the first-day EEGNetClf
    "r1": Recipe("r1"),
    "r1_mix": Recipe("r1_mix", mixup=0.2),
    "r1_mx4": Recipe("r1_mx4", mixup=0.4),                                  # T2-D146: the D112a / T1-D122 / D142 mixup alpha
    "r1_ema": Recipe("r1_ema", ema=0.995),                                  # T2-D153: weight EMA, horizon ~200 steps
    "r1_pul": Recipe("r1_pul", pu_lr=1e-3),                                 # T2-D154 U1: per-user FT lr 1e-3
    "r1_pue": Recipe("r1_pue", pu_epochs=40),                               # U2: per-user FT 40 epochs
    "r1_pus": Recipe("r1_pus", pu_epochs=10),                               # U3: per-user FT 10 epochs
    "r1_pulx": Recipe("r1_pulx", pu_lr=1e-3, pu_epochs=40),                # T2-D155: per-user FT lr 1e-3 + 40 epochs
    "r1_pulxx": Recipe("r1_pulxx", pu_lr=1e-3, pu_epochs=80),              # T2-D156: per-user FT lr 1e-3 + 80 epochs
    "r1_pulxxx": Recipe("r1_pulxxx", pu_lr=1e-3, pu_epochs=160),           # T2-D168: per-user FT lr 1e-3 + 160 epochs
    "r1_pulxx_vp": Recipe("r1_pulxx_vp", pu_lr=1e-3, pu_epochs=80, vp=1),  # T2-D193: FBCNet r1_pulxx + virtual participants
    "r1_pulx_vp": Recipe("r1_pulx_vp", pu_lr=1e-3, pu_epochs=40, vp=1),    # T2-D193: ShallowNet r1_pulx + virtual participants
    "r1_pulxx_vp2": Recipe("r1_pulxx_vp2", pu_lr=1e-3, pu_epochs=80, vp=2),  # T2-D195: FBCNet, electrode jitter 10 mm
    "r1_pulx_vp2": Recipe("r1_pulx_vp2", pu_lr=1e-3, pu_epochs=40, vp=2),    # T2-D195: ShallowNet, electrode jitter 10 mm
    "r1_pulxx_vp3": Recipe("r1_pulxx_vp3", pu_lr=1e-3, pu_epochs=80, vp=3),  # T2-D197: 15 mm
    "r1_pulx_vp3": Recipe("r1_pulx_vp3", pu_lr=1e-3, pu_epochs=40, vp=3),
    "r1_pulxx_vp4": Recipe("r1_pulxx_vp4", pu_lr=1e-3, pu_epochs=80, vp=4),  # T2-D197: 20 mm
    "r1_pulx_vp4": Recipe("r1_pulx_vp4", pu_lr=1e-3, pu_epochs=40, vp=4),
    "r1_pulxx_vp2_pv5": Recipe("r1_pulxx_vp2_pv5", pu_lr=1e-3, pu_epochs=80, vp=2, pu_vp=0.5),    # T2-D200: copies at half the 10 mm bank
    "r1_pulx_vp2_pv5": Recipe("r1_pulx_vp2_pv5", pu_lr=1e-3, pu_epochs=40, vp=2, pu_vp=0.5),
    "r1_pulxx_vp2_pv10": Recipe("r1_pulxx_vp2_pv10", pu_lr=1e-3, pu_epochs=80, vp=2, pu_vp=1.0),  # T2-D200: copies at the full bank
    "r1_pulx_vp2_pv10": Recipe("r1_pulx_vp2_pv10", pu_lr=1e-3, pu_epochs=40, vp=2, pu_vp=1.0),
    "r1_pulxx_vp2_e90": Recipe("r1_pulxx_vp2_e90", epochs=90, pu_lr=1e-3, pu_epochs=80, vp=2),    # T2-D202: pooled 90 epochs
    "r1_pulx_vp2_e90": Recipe("r1_pulx_vp2_e90", epochs=90, pu_lr=1e-3, pu_epochs=40, vp=2),
    "r1_pulxx_vp2_e120": Recipe("r1_pulxx_vp2_e120", epochs=120, pu_lr=1e-3, pu_epochs=80, vp=2),  # T2-D202: pooled 120 epochs
    "r1_pulx_vp2_e120": Recipe("r1_pulx_vp2_e120", epochs=120, pu_lr=1e-3, pu_epochs=40, vp=2),
    "r1_pulxx_vp2_e180": Recipe("r1_pulxx_vp2_e180", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2),  # T2-D203: pooled 180 epochs
    "r1_pulx_vp2_e180": Recipe("r1_pulx_vp2_e180", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2),
    "r1_pulxx_vp2_e240": Recipe("r1_pulxx_vp2_e240", epochs=240, pu_lr=1e-3, pu_epochs=80, vp=2),  # T2-D204: pooled 240 / 360 epochs
    "r1_pulx_vp2_e240": Recipe("r1_pulx_vp2_e240", epochs=240, pu_lr=1e-3, pu_epochs=40, vp=2),
    "r1_pulxx_vp2_e360": Recipe("r1_pulxx_vp2_e360", epochs=360, pu_lr=1e-3, pu_epochs=80, vp=2),
    "r1_pulx_vp2_e360": Recipe("r1_pulx_vp2_e360", epochs=360, pu_lr=1e-3, pu_epochs=40, vp=2),
    "r1_pulxx_vp2_e180_pu40": Recipe("r1_pulxx_vp2_e180_pu40", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2),    # T2-D213: copies x 0.5
    "r1_pulxx_vp2_e180_pu160": Recipe("r1_pulxx_vp2_e180_pu160", epochs=180, pu_lr=1e-3, pu_epochs=160, vp=2),  # T2-D213: copies x 2
    "r1_pulx_vp2_e180_pu20": Recipe("r1_pulx_vp2_e180_pu20", epochs=180, pu_lr=1e-3, pu_epochs=20, vp=2),
    "r1_pulx_vp2_e180_pu80": Recipe("r1_pulx_vp2_e180_pu80", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2),
    "r1_pulxx_vp2_e180_g2": Recipe("r1_pulxx_vp2_e180_g2", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_g=0.2),  # T2-D214: gain U(0.8, 1.2)
    "r1_pulx_vp2_e180_g2": Recipe("r1_pulx_vp2_e180_g2", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_g=0.2),
    "r1_pulxx_vp2_e180_s6": Recipe("r1_pulxx_vp2_e180_s6", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_s=6),    # T2-D214: shift +-6 samples
    "r1_pulx_vp2_e180_s6": Recipe("r1_pulx_vp2_e180_s6", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_s=6),
    "r1_pulxx_vp2_e180_lr2": Recipe("r1_pulxx_vp2_e180_lr2", epochs=180, lr=2e-3, pu_lr=1e-3, pu_epochs=80, vp=2),   # T2-D216: pooled lr x 2
    "r1_pulx_vp2_e180_lr2": Recipe("r1_pulx_vp2_e180_lr2", epochs=180, lr=2e-3, pu_lr=1e-3, pu_epochs=40, vp=2),
    "r1_pulxx_vp2_e180_wd5": Recipe("r1_pulxx_vp2_e180_wd5", epochs=180, wd=5e-2, pu_lr=1e-3, pu_epochs=80, vp=2),  # T2-D216: pooled wd x 5
    "r1_pulx_vp2_e180_wd5": Recipe("r1_pulx_vp2_e180_wd5", epochs=180, wd=5e-2, pu_lr=1e-3, pu_epochs=40, vp=2),
    "r1_pulxx_vp2_e180_vr": Recipe("r1_pulxx_vp2_e180_vr", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5),  # T2-D215: virtual references
    "r1_pulx_vp2_e180_vr": Recipe("r1_pulx_vp2_e180_vr", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5),
    # T2-D589: the same pooled line, its in-stream / per-user COPIES fine-tuned with mixup Beta(0.4, 0.4) (pu_mix; the pooled fit is unchanged)
    "r1_pulx_vp2_e180_vr_pmx4": Recipe("r1_pulx_vp2_e180_vr_pmx4", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, pu_mix=0.4),
    "r1_pulx_vp2_e180_vr_gd3": Recipe("r1_pulx_vp2_e180_vr_gd3", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, gdro=0.003),  # T2-D315
    "r1_pulxx_vp2_e180_vr10": Recipe("r1_pulxx_vp2_e180_vr10", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=1.0),  # T2-D220: vr dose a ~ U(0, 1)
    "r1_pulx_vp2_e180_vr10": Recipe("r1_pulx_vp2_e180_vr10", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=1.0),
    "r1_pulxx_vp2_e180_vr_mx4": Recipe("r1_pulxx_vp2_e180_vr_mx4", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, mixup=0.4),  # T2-D220: + pooled mixup 0.4
    "r1_pulx_vp2_e180_vr_mx4": Recipe("r1_pulx_vp2_e180_vr_mx4", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, mixup=0.4),
    "r1_spec_e180_mx4": Recipe("r1_spec_e180_mx4", epochs=180, mixup=0.4),   # T2-D554: the speech recipe (SpecAugment inside specnet), long cosine
    "r1_pulxx_vp2_e180_vr_ls0": Recipe("r1_pulxx_vp2_e180_vr_ls0", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, label_smoothing=0.0),  # T2-D223: pooled LS 0
    "r1_pulx_vp2_e180_vr_ls0": Recipe("r1_pulx_vp2_e180_vr_ls0", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, label_smoothing=0.0),
    "r1_pulxx_vp2_e180_vr_ls2": Recipe("r1_pulxx_vp2_e180_vr_ls2", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, label_smoothing=0.2),  # T2-D223: pooled LS 0.2
    "r1_pulx_vp2_e180_vr_ls2": Recipe("r1_pulx_vp2_e180_vr_ls2", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, label_smoothing=0.2),
    "r1_pulxx_vp2_e180_vr_sam05": Recipe("r1_pulxx_vp2_e180_vr_sam05", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, sam=0.05),  # T2-D224: pooled SAM
    "r1_pulx_vp2_e180_vr_sam05": Recipe("r1_pulx_vp2_e180_vr_sam05", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, sam=0.05),
    "r1_pulxx_vp2_e180_vr_sam2": Recipe("r1_pulxx_vp2_e180_vr_sam2", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, sam=0.2),
    "r1_pulx_vp2_e180_vr_sam2": Recipe("r1_pulx_vp2_e180_vr_sam2", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, sam=0.2),
    "r1_pulxx_vp2_e180_vr_bs32": Recipe("r1_pulxx_vp2_e180_vr_bs32", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, bs=32),  # T2-D225: pooled batch 32 / 128
    "r1_pulx_vp2_e180_vr_bs32": Recipe("r1_pulx_vp2_e180_vr_bs32", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, bs=32),
    "r1_pulxx_vp2_e180_vr_bs128": Recipe("r1_pulxx_vp2_e180_vr_bs128", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, bs=128),
    "r1_pulx_vp2_e180_vr_bs128": Recipe("r1_pulx_vp2_e180_vr_bs128", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, bs=128),
    "r1_pulxx_vp2_e180_vr_pls2": Recipe("r1_pulxx_vp2_e180_vr_pls2", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, pu_ls=0.2),  # T2-D226: copies LS 0.2 / 0.3
    "r1_pulx_vp2_e180_vr_pls2": Recipe("r1_pulx_vp2_e180_vr_pls2", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, pu_ls=0.2),
    "r1_pulxx_vp2_e180_vr_pls3": Recipe("r1_pulxx_vp2_e180_vr_pls3", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, pu_ls=0.3),
    "r1_pulx_vp2_e180_vr_pls3": Recipe("r1_pulx_vp2_e180_vr_pls3", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, pu_ls=0.3),
    "r1_pulxx_vp2_e180_vr_psw25": Recipe("r1_pulxx_vp2_e180_vr_psw25", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, pu_swa=0.25),  # T2-D227: copies SWA
    "r1_pulx_vp2_e180_vr_psw25": Recipe("r1_pulx_vp2_e180_vr_psw25", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, pu_swa=0.25),
    "r1_pulxx_vp2_e180_vr_psw50": Recipe("r1_pulxx_vp2_e180_vr_psw50", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, pu_swa=0.5),
    "r1_pulx_vp2_e180_vr_psw50": Recipe("r1_pulx_vp2_e180_vr_psw50", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, pu_swa=0.5),
    "r1_pulxx_vp2_e180_vr_phead": Recipe("r1_pulxx_vp2_e180_vr_phead", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, pu_train="head"),  # T2-D228
    "r1_pulx_vp2_e180_vr_phead": Recipe("r1_pulx_vp2_e180_vr_phead", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, pu_train="head"),
    "r1_pulxx_vp2_e180_vr_pfeat": Recipe("r1_pulxx_vp2_e180_vr_pfeat", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, pu_train="feat"),
    "r1_pulx_vp2_e180_vr_pfeat": Recipe("r1_pulx_vp2_e180_vr_pfeat", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, pu_train="feat"),
    "r1_pulxx_vp2_e180_vr_fw05": Recipe("r1_pulxx_vp2_e180_vr_fw05", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, fw=0.05),  # T2-D229
    "r1_pulx_vp2_e180_vr_fw05": Recipe("r1_pulx_vp2_e180_vr_fw05", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, fw=0.05),
    "r1_pulxx_vp2_e180_vr_fw10": Recipe("r1_pulxx_vp2_e180_vr_fw10", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, fw=0.10),
    "r1_pulx_vp2_e180_vr_fw10": Recipe("r1_pulx_vp2_e180_vr_fw10", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, fw=0.10),
    "r1_pulxx_vp2_e180_vr_rc5": Recipe("r1_pulxx_vp2_e180_vr_rc5", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.5),  # T2-D230 (FBCNet)
    "r1_pulxx_vp2_e180_vr_rc10": Recipe("r1_pulxx_vp2_e180_vr_rc10", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=1.0),
    "r1_pulxx_vp2_e180_vr_rc25": Recipe("r1_pulxx_vp2_e180_vr_rc25", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.25),  # T2-D231 (the rc dose)
    "r1_pulxx_vp2_e180_vr_rc75": Recipe("r1_pulxx_vp2_e180_vr_rc75", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75),
    "r1_pulx_vp2_e65_vr": Recipe("r1_pulx_vp2_e65_vr", epochs=65, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5),   # T2-D547: the side ShallowNet nets at 65 epochs (the curve diagnosis)
    "r1_pulxx_vp2_e240_vr_rc75": Recipe("r1_pulxx_vp2_e240_vr_rc75", epochs=240, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75),   # T2-D548: FBCNet at 240 epochs (diagnosis only)
    # T2-D430 (panel P7 M1, stranger-safe copies): the shipped FBCNet / ShallowNet-IFNet recipes with pu_kl 1 (the copies stay pooled-like on the other users' windows)
    "r1_pulxx_vp2_e180_vr_rc75_kl1": Recipe("r1_pulxx_vp2_e180_vr_rc75_kl1", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, pu_kl=1.0),
    "r1_pulx_vp2_e180_vr_kl1": Recipe("r1_pulx_vp2_e180_vr_kl1", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, pu_kl=1.0),
    # T2-D478: the shipped FBCNet / ShallowNet-IFNet recipes with stream-balanced batches + the per-stream prior-blind loss (k = 16)
    "r1_pulxx_vp2_e180_vr_rc75_sb16": Recipe("r1_pulxx_vp2_e180_vr_rc75_sb16", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, sbal=16),
    "r1_pulx_vp2_e180_vr_sb16": Recipe("r1_pulx_vp2_e180_vr_sb16", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, sbal=16),
    "r1_pulxx_vp2_e180_vr_rc75_gd3": Recipe("r1_pulxx_vp2_e180_vr_rc75_gd3", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, gdro=0.003),  # T2-D315
    "r1_pulx_vp2_e180_vr_rc75": Recipe("r1_pulx_vp2_e180_vr_rc75", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, rc=0.75),  # T2-D233 (ShallowNet: its EA inputs still differ between users)
    "r1_pulx_vp2_e180_vr_rc10": Recipe("r1_pulx_vp2_e180_vr_rc10", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, rc=1.0),
    "r1_pulxx_vp2_e180_vr_rcu25": Recipe("r1_pulxx_vp2_e180_vr_rcu25", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=1.0, rc_lo=0.25),  # T2-D236 (a random strength per window)
    "r1_pulxx_vp2_e180_vr_rcu50": Recipe("r1_pulxx_vp2_e180_vr_rcu50", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=1.0, rc_lo=0.5),
    "r1_pulxx_vp2_e180_vr_rc75m1": Recipe("r1_pulxx_vp2_e180_vr_rc75m1", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, rc_mix=1),  # T2-D237 (midpoint targets)
    "r1_pulxx_vp2_e180_vr_rc75m2": Recipe("r1_pulxx_vp2_e180_vr_rc75m2", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, rc_mix=2),
    # T2-D239: the copies see the other users' windows mapped onto their user's covariance; pu_epochs scaled so each copy takes rc75's steps
    "r1_pulxx_vp2_e180_vr_rc75_pux1": Recipe("r1_pulxx_vp2_e180_vr_rc75_pux1", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, rc=0.75, pu_rcx=1.0),
    "r1_pulxx_vp2_e180_vr_rc75_pux3": Recipe("r1_pulxx_vp2_e180_vr_rc75_pux3", epochs=180, pu_lr=1e-3, pu_epochs=20, vp=2, vp_r=0.5, rc=0.75, pu_rcx=3.0),
    # T2-D293: user-adversarial pooled nets on the L_V CNN recipes (FBCNet rc75, ShallowNet vr), lambda 0.1 / 0.3
    "r1_pulxx_vp2_e180_vr_rc75_ua1": Recipe("r1_pulxx_vp2_e180_vr_rc75_ua1", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, user_adv=0.1),
    "r1_pulxx_vp2_e180_vr_rc75_ua3": Recipe("r1_pulxx_vp2_e180_vr_rc75_ua3", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, user_adv=0.3),
    # T2-D374: TUNED DANN on the FBCNet rc75 line (ramp, MLP adversary at 10x lr; marginal dm / class-conditional dc; lambda_max 0.03 / 0.1)
    "r1_pulxx_vp2_e180_vr_rc75_dm03": Recipe("r1_pulxx_vp2_e180_vr_rc75_dm03", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, user_adv=0.03, adv_ramp=1, adv_hidden=128, adv_lr_mult=10.0, adv_cond=0),
    "r1_pulxx_vp2_e180_vr_rc75_dm1": Recipe("r1_pulxx_vp2_e180_vr_rc75_dm1", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, user_adv=0.1, adv_ramp=1, adv_hidden=128, adv_lr_mult=10.0, adv_cond=0),
    "r1_pulxx_vp2_e180_vr_rc75_dc03": Recipe("r1_pulxx_vp2_e180_vr_rc75_dc03", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, user_adv=0.03, adv_ramp=1, adv_hidden=128, adv_lr_mult=10.0, adv_cond=1),
    "r1_pulxx_vp2_e180_vr_rc75_dc1": Recipe("r1_pulxx_vp2_e180_vr_rc75_dc1", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, user_adv=0.1, adv_ramp=1, adv_hidden=128, adv_lr_mult=10.0, adv_cond=1),
    "r1_pulx_vp2_e180_vr_ua1": Recipe("r1_pulx_vp2_e180_vr_ua1", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, user_adv=0.1),
    "r1_pulx_vp2_e180_vr_ua3": Recipe("r1_pulx_vp2_e180_vr_ua3", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, user_adv=0.3),
    # backlog 70: the two lines + MixStyle p 0.5 (the paper's default; one setting)
    "r1_pulxx_vp2_e180_vr_rc75_ms5": Recipe("r1_pulxx_vp2_e180_vr_rc75_ms5", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, mixstyle=0.5),
    "r1_pulx_vp2_e180_vr_ms5": Recipe("r1_pulx_vp2_e180_vr_ms5", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, mixstyle=0.5),
    "r1_pulxx_vp2_e180_vr_rc75_us5": Recipe("r1_pulxx_vp2_e180_vr_rc75_us5", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, ustyle=0.5),  # T2-D365
    "r1_pulx_vp2_e180_vr_us5": Recipe("r1_pulx_vp2_e180_vr_us5", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, ustyle=0.5),
    # backlog 72: the two lines with their copies through the vp bank at full strength (T2-D200 pv10 on the current line; the pooled nets unchanged)
    "r1_pulxx_vp2_e180_vr_rc75_pv10": Recipe("r1_pulxx_vp2_e180_vr_rc75_pv10", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, pu_vp=1.0),
    "r1_pulx_vp2_e180_vr_pv10": Recipe("r1_pulx_vp2_e180_vr_pv10", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, pu_vp=1.0),
    # T2-D339: the 15 mm spatial bank (vp 3, T2-D197) on the two shipped lines (everything else as r1_pulxx_vp2_e180_vr_rc75 / r1_pulx_vp2_e180_vr)
    "r1_pulxx_vp3_e180_vr_rc75": Recipe("r1_pulxx_vp3_e180_vr_rc75", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=3, vp_r=0.5, rc=0.75),
    "r1_pulx_vp3_e180_vr": Recipe("r1_pulx_vp3_e180_vr", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=3, vp_r=0.5),
    # backlog 73: the two lines + same-class cross-user input mixing at p 0.5 (one setting)
    "r1_pulxx_vp2_e180_vr_rc75_xm5": Recipe("r1_pulxx_vp2_e180_vr_rc75_xm5", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, xmix=0.5),
    "r1_pulx_vp2_e180_vr_xm5": Recipe("r1_pulx_vp2_e180_vr_xm5", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, xmix=0.5),
    # backlog 74: the two lines + a random spectral tilt per window, amplitude sd 0.15 (power-exponent sd 0.3; one setting)
    "r1_pulxx_vp2_e180_vr_rc75_tl15": Recipe("r1_pulxx_vp2_e180_vr_rc75_tl15", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, tilt=0.15),
    "r1_pulx_vp2_e180_vr_tl15": Recipe("r1_pulx_vp2_e180_vr_tl15", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, tilt=0.15),
    # T2-D412: M6's tilt FBCNet line with CLASS-CONDITIONAL re-colouring (rc_cls 1; the GA panel's rcc); everything else as tl15
    "r1_pulxx_vp2_e180_vr_rc75cc_tl15": Recipe("r1_pulxx_vp2_e180_vr_rc75cc_tl15", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, rc_cls=1, tilt=0.15),
    # T2-D413: M6's tilt FBCNet line + MeanFlow synthetic new-people windows at 0.5 x the batch (SYN_BANK); everything else as tl15
    "r1_pulxx_vp2_e180_vr_rc75_tl15_mf50": Recipe("r1_pulxx_vp2_e180_vr_rc75_tl15_mf50", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, tilt=0.15, syn=0.5),
    # T2-D561 (the synthetic-dose question): the same line with synthetic windows = 20 % of each real batch (D413 / D413c ran only syn 0.5)
    "r1_pulxx_vp2_e180_vr_rc75_tl15_mf20": Recipe("r1_pulxx_vp2_e180_vr_rc75_tl15_mf20", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, tilt=0.15, syn=0.2),
    # T2-D561 (the owner: "why only fbcnet"): the ShallowNet / IFNet line at syn 0.2 / 0.5; run at align none (the syn path's requirement:
    # synthetic interpolated users have no EA whitening), all three arms of each family at align none
    "r1_pulx_vp2_e180_vr_mf20": Recipe("r1_pulx_vp2_e180_vr_mf20", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, syn=0.2),
    "r1_pulx_vp2_e180_vr_mf50": Recipe("r1_pulx_vp2_e180_vr_mf50", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, syn=0.5),
    "r1_pulxx_vp2_e180_vr_rc75_mf20": Recipe("r1_pulxx_vp2_e180_vr_rc75_mf20", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, syn=0.2),   # T2-D576: the stack's FBCNet line + syn 0.2
    "r1_pulx_vp2_e180_vr_smx4": Recipe("r1_pulx_vp2_e180_vr_smx4", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, side_mix=0.4),   # T2-D580: the side nets' line + joint mixup 0.4
    # T2-D382: M6's CNN lines at 360 epochs (T2-D204's e360 step re-asked on today's augmented recipes); copies unchanged
    "r1_pulxx_vp2_e360_vr_rc75_tl15": Recipe("r1_pulxx_vp2_e360_vr_rc75_tl15", epochs=360, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, tilt=0.15),
    "r1_pulx_vp2_e360_vr": Recipe("r1_pulx_vp2_e360_vr", epochs=360, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5),
    # T2-D388: M6's tilt FBCNet line with S1's pv10 copies (pu_vp 1.0: the COPIES train through the bank; the pooled net = the tl15 line's)
    "r1_pulxx_vp2_e180_vr_rc75_tl15_pv10": Recipe("r1_pulxx_vp2_e180_vr_rc75_tl15_pv10", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, tilt=0.15, pu_vp=1.0),
    # row 77 / T2-D305: MIRepNet's published fine-tune (AdamW lr 1e-3, wd 1e-6, batch 8, 10 epochs, cosine); no augmentation, no smoothing
    "mirep_ft": Recipe("mirep_ft", epochs=10, lr=1e-3, wd=1e-6, bs=8, label_smoothing=0.0),
    # T2-D308: the two lines + virtual sessions at p 0.5 (one setting)
    "r1_pulxx_vp2_e180_vr_rc75_vs5": Recipe("r1_pulxx_vp2_e180_vr_rc75_vs5", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, vs=0.5),
    "r1_pulx_vp2_e180_vr_vs5": Recipe("r1_pulx_vp2_e180_vr_vs5", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, vs=0.5),
    # backlog 69: FM -> CNN distillation on the L_V CNN recipes, alpha 0.5, T 2: kd = PROTEUS windows only, kdx = + unlabelled external windows
    "r1_pulxx_vp2_e180_vr_rc75_kd": Recipe("r1_pulxx_vp2_e180_vr_rc75_kd", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, kd=0.5),
    "r1_pulxx_vp2_e180_vr_rc75_kdx": Recipe("r1_pulxx_vp2_e180_vr_rc75_kdx", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, kd=0.5, kd_ext=1),
    "r1_pulx_vp2_e180_vr_kd": Recipe("r1_pulx_vp2_e180_vr_kd", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, kd=0.5),
    "r1_pulx_vp2_e180_vr_kdx": Recipe("r1_pulx_vp2_e180_vr_kdx", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, kd=0.5, kd_ext=1),
    # backlog 65 (a): FBCNet re-coloured at 0.75 toward the external covariance bank (rc75's targets -> ~116 real external users)
    "r1_pulxx_vp2_e180_vr_rcx75": Recipe("r1_pulxx_vp2_e180_vr_rcx75", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, rc_ext=1),
    # backlog 65 (b): + a rigid cap shift of 10 mm RMS per virtual participant, both CNN families (one bank per job)
    "r1_pulxx_vp2_e180_vr_rc75_cap10": Recipe("r1_pulxx_vp2_e180_vr_rc75_cap10", epochs=180, pu_lr=1e-3, pu_epochs=80, vp=2, vp_r=0.5, rc=0.75, vp_cap=0.010),
    "r1_pulx_vp2_e180_vr_cap10": Recipe("r1_pulx_vp2_e180_vr_cap10", epochs=180, pu_lr=1e-3, pu_epochs=40, vp=2, vp_r=0.5, vp_cap=0.010),
    "r1_pulxx_vp2_k256": Recipe("r1_pulxx_vp2_k256", pu_lr=1e-3, pu_epochs=80, vp=2, vp_k=256),    # T2-D202: a 256-draw bank
    "r1_pulx_vp2_k256": Recipe("r1_pulx_vp2_k256", pu_lr=1e-3, pu_epochs=40, vp=2, vp_k=256),
    "r1_lowlr": Recipe("r1_lowlr", lr=3e-4),
    "r1_gj": Recipe("r1_gj", gain_jit=0.2),                                 # T2-D112 G1
    "r1_ts": Recipe("r1_ts", tshift=30),                                    # G2 (+- 0.25 s at 120 Hz)
    "r1_cd": Recipe("r1_cd", chdrop=0.1),                                   # G3
    "r1_aug": Recipe("r1_aug", gain_jit=0.2, tshift=30, chdrop=0.1),        # G4
}
# T2-D111 per-user fine-tune from the pooled net (applied by t2p_bench --peruser)
PERUSER = Recipe("peruser", epochs=20, lr=3e-4, wd=1e-2, bs=32, label_smoothing=0.1, warmup_frac=0.0)


def peruser_for(recipe: Recipe) -> Recipe:
    """T2-D154: the per-user fine-tune recipe of a pooled recipe (PERUSER with its pu_lr / pu_epochs; r1 -> PERUSER); T2-D200: pu_vp > 0 ->
    the copies train through VP_BANK at strength pu_vp; T2-D226: the copies' label smoothing = pu_ls; T2-D227: their weight averaging = pu_swa;
    T2-D228: the part of the copy that trains = pu_train."""
    if recipe.pu_vp:
        return replace(PERUSER, lr=recipe.pu_lr, epochs=recipe.pu_epochs, vp=1, vp_a=float(recipe.pu_vp), label_smoothing=recipe.pu_ls, swa=recipe.pu_swa,
                       train_part=recipe.pu_train, kl_other=recipe.pu_kl, sbal=recipe.sbal, ft_mix=recipe.pu_mix)
    return replace(PERUSER, lr=recipe.pu_lr, epochs=recipe.pu_epochs, label_smoothing=recipe.pu_ls, swa=recipe.pu_swa,
                       train_part=recipe.pu_train, kl_other=recipe.pu_kl, sbal=recipe.sbal, ft_mix=recipe.pu_mix)

# model name -> (braindecode class name, extra kwargs). Temporal kernels of braindecode defaults assume
# 128-250 Hz; inputs here are 120 Hz (kit pipeline), so "_fs" variants rescale them where it matters.
MODEL_SPECS = {
    "eegnet": ("EEGNet", {}),
    "shallow": ("ShallowFBCSPNet", {}),
    "deep4": ("Deep4Net", {}),
    "conformer": ("EEGConformer", {}),
    "atcnet": ("ATCNet", {}),
    "eegnex": ("EEGNeX", {}),
    "ifnet": ("IFNet", {}),
    "fbcnet": ("FBCNet", {}),
    # T2-D115 one-change FBCNet variants
    "fbcnet_s2": ("FBCNet", {"stride_factor": 2}),
    "fbcnet_s8": ("FBCNet", {"stride_factor": 8}),
    "fbcnet_f16": ("FBCNet", {"n_filters_spat": 16}),
    "fbcnet_f64": ("FBCNet", {"n_filters_spat": 64}),
    "fbcnet_b830": ("FBCNet", {"n_bands": [(8.0, 10.0), (10.0, 12.0), (12.0, 14.0), (14.0, 16.0), (16.0, 18.0), (18.0, 22.0),
                                           (22.0, 26.0), (26.0, 30.0)]}),
    # T2-D122 bank widening (each replaces the default 9 x 4 Hz bank over 4-40 Hz)
    "fbcnet_b14": ("FBCNet", {"n_bands": [(1.0, 4.0)] + [(float(lo), float(lo + 4)) for lo in range(4, 37, 4)]}),
    "fbcnet_b4058": ("FBCNet", {"n_bands": [(float(lo), float(lo + 4)) for lo in range(4, 37, 4)] + [(40.0, 46.0), (46.0, 52.0), (52.0, 58.0)]}),
    "fbcnet_bboth": ("FBCNet", {"n_bands": [(1.0, 4.0)] + [(float(lo), float(lo + 4)) for lo in range(4, 37, 4)]
                                + [(40.0, 46.0), (46.0, 52.0), (52.0, 58.0)]}),
    # T2-D221 width x 2 under vp2-180 + vr (FBCNet 32 -> 64 spatial filters per band; ShallowNet 40 -> 80 temporal and spatial filters)
    "fbcnet_bboth_f64": ("FBCNet", {"n_bands": [(1.0, 4.0)] + [(float(lo), float(lo + 4)) for lo in range(4, 37, 4)]
                                    + [(40.0, 46.0), (46.0, 52.0), (52.0, 58.0)], "n_filters_spat": 64}),
    "shallow_w2": ("ShallowFBCSPNet", {"n_filters_time": 80, "n_filters_spat": 80}),
    # T2-D477 (seat G10 item 2): spectral coverage of the two EA families. IFNet has only run braindecode's bands [(4, 16), (16, 40)];
    # ShallowNet's 25-sample temporal filter is 208 ms at 120 Hz (61 samples = 508 ms resolves 1-4 Hz)
    # IFNet's spatial conv is grouped by band (n_filters_spat must divide by n_bands) and it takes ONE temporal kernel per band: keep the
    # default 32 filters PER BAND (64 / 2) and the defaults' kernels for the two old bands (63, 31), halving per band: 1-4 Hz 127, 40-58 Hz 15
    "ifnet_b3": ("IFNet", {"bands": [(1.0, 4.0), (4.0, 16.0), (16.0, 40.0)], "n_filters_spat": 96, "kernel_sizes": (127, 63, 31)}),
    "ifnet_b4": ("IFNet", {"bands": [(1.0, 4.0), (4.0, 16.0), (16.0, 40.0), (40.0, 58.0)], "n_filters_spat": 128, "kernel_sizes": (127, 63, 31, 15)}),
    "shallow_t61": ("ShallowFBCSPNet", {"filter_time_length": 61}),
    # T2-D518: config-diverse members around the shipped ShallowNet / IFNet (the hall-of-fame read: configs, not seeds, inside a family)
    "shallow_t13": ("ShallowFBCSPNet", {"filter_time_length": 13}),
    "shallow_t41": ("ShallowFBCSPNet", {"filter_time_length": 41}),
    "shallow_p50": ("ShallowFBCSPNet", {"pool_time_length": 50, "pool_time_stride": 10}),
    "shallow_f60": ("ShallowFBCSPNet", {"n_filters_time": 60, "n_filters_spat": 60}),
    "ifnet_k127": ("IFNet", {"kernel_sizes": (127, 63)}),
    "ifnet_k31": ("IFNet", {"kernel_sizes": (31, 15)}),
    "ifnet_s96": ("IFNet", {"n_filters_spat": 96}),
    "ifnet_st4": ("IFNet", {"stride_factor": 4}),
    "eegtcnet": ("EEGTCNet", {}),
}


def fbc_features(m, x):
    """braindecode FBCNet.forward up to (and including) flatten_layer: (B, C, T) -> (B, n_bands * n_filters * stride)."""
    x = m.spatial_conv(m.spectral_filtering(x))
    b, c, _, _ = x.shape
    x = m.padding_layer(x).view(b, c, m.stride_factor, m.n_times_padded // m.stride_factor)
    return m.flatten_layer(m.temporal_layer(x))


class UserHeadNet(torch.nn.Module):
    """T2-D111 FBCNet with user-aware heads (training users only; predict() never sees a user id).
    kind 'uid': + a linear user-id head on the features (auxiliary loss only; class logits = FBCNet's own head).
    kind 'moe': soft user-expert head, logits = W0 f + sum_k g_k(f) W_k f, g = softmax(G f) over the K training users,
    the gate trained with the auxiliary user-id CE; experts start at zero (= the shared head). forward(x, gate=None):
    gate = per-window posteriors (B, K) to use instead of softmax(G f) (the causal stream-smoothed twin)."""

    def __init__(self, base, kind, n_users):
        super().__init__()
        self.base, self.kind, self.n_users = base, kind, int(n_users)
        d = base.final_layer.in_features if hasattr(base.final_layer, "in_features") else base.final_layer.weight.shape[1]
        n_out = base.final_layer.weight.shape[0]
        self.uid = torch.nn.Linear(d, self.n_users)
        if kind == "moe":
            self.experts = torch.nn.Parameter(torch.zeros(self.n_users, n_out, d))

    def forward_all(self, x, gate=None):
        f = fbc_features(self.base, x)
        u = self.uid(f)
        logits = self.base.final_layer(f)
        if self.kind == "moe":
            g = torch.softmax(u.float(), 1).to(f.dtype) if gate is None else gate.to(f.dtype)
            logits = logits + torch.einsum("bk,kcd,bd->bc", g, self.experts, f)
        return logits, u

    def forward(self, x, gate=None):
        return self.forward_all(x, gate)[0]


MIREP_CH = []   # row 77 / T2-D305: the montage for the MIRepNet member (the caller sets it, e.g. t2p_bench from the export's meta)


def build_model(name, n_chans, n_classes, n_times, sfreq, n_users=0):
    if name in ("mirepnet", "mirepnet_bb"):             # row 77 / T2-D305: MIRepNet (8-30 Hz / broadband), pretrained, a new head
        from ni26.fm.mirepnet import MIRepMember
        assert len(MIREP_CH) == n_chans, "mirepnet needs ni26.train.MIREP_CH (the montage)"
        return MIRepMember(list(MIREP_CH), n_times, sfreq, n_classes, band=(8.0, 30.0) if name == "mirepnet" else None)
    if name.endswith("_lmlr"):                          # T2-D318 H-b: HEEGNet's Lorentz MLR head on an FBCNet's features
        from ni26.heegnet import LorentzHeadNet
        assert name[:-5].startswith("fbcnet") and name[:-5] in MODEL_SPECS, name
        base = build_model(name[:-5], n_chans, n_classes, n_times, sfreq)
        return LorentzHeadNet(base, fbc_features, base.final_layer.in_features, n_classes)
    if name in ("specnet", "specnet_sf"):               # T2-D554 / D554b: the 'speech recipe' family (+ a learned 41 -> 32 spatial projection)
        from ni26.specnet import SpecNet
        return SpecNet(n_chans, n_classes, n_times, sfreq, spatial=32 if name == "specnet_sf" else 0)
    if name == "rawnet":                                # T2-D521: the raw end-to-end net (learned compression + multi-scale temporal bank + level branch)
        from ni26.rawnet import RawNet
        return RawNet(n_chans, n_classes, n_times, sfreq)
    if name in ("fbcnet_uid", "fbcnet_moe"):
        assert n_users > 1, f"{name} needs the number of training users"
        return UserHeadNet(build_model("fbcnet", n_chans, n_classes, n_times, sfreq), name.split("_")[1], n_users)
    import braindecode.models as M
    cls_name, kw = MODEL_SPECS[name]
    cls = getattr(M, cls_name)
    params = inspect.signature(cls.__init__).parameters
    args = dict(n_chans=n_chans, n_outputs=n_classes, n_times=n_times, **kw)
    if "sfreq" in params:
        args["sfreq"] = sfreq
    return cls(**args)


def _cosine(step, total, warm):
    if step < warm:
        return (step + 1) / max(warm, 1)
    p = (step - warm) / max(total - warm, 1)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def predict_proba(net, X, bs=512):
    net.eval()
    out = [F.softmax(net(X[i:i + bs]).float(), dim=1) for i in range(0, len(X), bs)]
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def _pp_side(net, head, X, S, bs=512):
    """T2-D517: softmax(net(X) + head(S)), batched (the side head's inference)."""
    net.eval()
    return torch.cat([F.softmax((net(X[i:i + bs]) + head(S[i:i + bs])).float(), dim=1) for i in range(0, len(X), bs)]).cpu().numpy()


@torch.no_grad()
def predict_proba_vp(net, X, K, bs=512, shift_only=0):
    """backlog 71: test-time mean of the softmax over the identity + VP_BANK draws 0..K-2, each applied as augment applies a draw
    (gain x remap, then the circular time shift); the pooled nets' TTA arm, scored beside the plain predict_proba. T2-D302 (a):
    shift_only = S > 1 adds S - 1 views with ONLY the circular shift of draws K-1..K+S-3 (no remap, no gain): the mean over K + S - 1 views."""
    assert VP_BANK and 2 <= K and K - 1 + max(shift_only - 1, 0) <= len(VP_BANK["mats"]), "tta_vp needs VP_BANK and enough draws"
    M, G, S = VP_BANK["mats"], VP_BANK["gains"], VP_BANK["shifts"]
    acc = predict_proba(net, X, bs)
    for k in range(K - 1):
        Xk = G[k].to(X.device, X.dtype)[:, None] * torch.matmul(M[k].to(X.device, X.dtype), X)
        acc = acc + predict_proba(net, torch.roll(Xk, shifts=int(S[k]), dims=2), bs)
    for j in range(max(shift_only - 1, 0)):
        acc = acc + predict_proba(net, torch.roll(X, shifts=int(S[K - 1 + j]), dims=2), bs)
    return acc / (K + max(shift_only - 1, 0))


@torch.no_grad()
def predict_uid(net, X, bs=512):
    """UserHeadNet: per-window user posteriors (B, K)."""
    net.eval()
    return torch.cat([F.softmax(net.forward_all(X[i:i + bs])[1].float(), 1) for i in range(0, len(X), bs)]).cpu().numpy()


FW_K = 33      # T2-D229: warp factors per bank (odd: the middle one is the identity)
_FW_BANKS = {}


def fw_matrix(L, n_new):
    """T2-D229: the (L, L) matrix R with x @ R = T2-D191's iaf_warp of x (scipy FFT resample from L to n_new samples, then reflect pad /
    centre crop back to L; linear in x, so R's row i = the warp of the unit impulse e_i)."""
    from scipy.signal import resample
    if n_new == L:
        return np.eye(L)
    w = resample(np.eye(L), n_new, axis=-1)
    if n_new < L:
        p_ = L - n_new
        return np.pad(w, ((0, 0), (p_ // 2, p_ - p_ // 2)), mode="reflect")
    c_ = (n_new - L) // 2
    return w[:, c_:c_ + L]


TILT_FS = 120.0   # backlog 74: the CNN export's rate (fit_eval asserts it for tilt recipes)


def fw_bank(L, fw, device):
    """T2-D229: FW_K warp matrices for factors log-uniform over [1 / (1 + fw), 1 + fw] (n_new = round(L a)), (K, L, L) float32, cached."""
    key = (L, float(fw), str(device))
    if key not in _FW_BANKS:
        a = np.exp(np.linspace(-np.log1p(fw), np.log1p(fw), FW_K))
        _FW_BANKS[key] = torch.as_tensor(np.stack([fw_matrix(L, int(round(L * f))) for f in a]), dtype=torch.float32, device=device)
    return _FW_BANKS[key]


def rc_bank(X, u, n_users, rc, shrink=1e-3, mix=0, ext=None):
    """T2-D230: (n_users, n_users, C, C) float32 maps M[u, v] = I + rc (T_uv - I), T_uv = C_u^-1/2 (C_u^1/2 C_v C_u^1/2)^1/2 C_u^-1/2 (the
    Bures / optimal-transport map: T C_u T = C_v), C_u = user u's mean covariance of its demeaned windows (X: (N, C, L) tensor, u: (N,) user
    index) + shrink x mean eigenvalue x I; M[u, u] = I.
    T2-D237: mix 1 -> (n_users, K, C, C), K = (n-1)(n-2)/2: user u's maps toward the Bures midpoints G C_v G, G = (I + T_vw) / 2, of every pair
    v < w of the other users; mix 2 -> (n_users, n-1 + K, C, C): the real targets v != u (ascending), then the midpoints; mix 0 = the above.
    Backlog 65 (a): ext = (K, C, C) external covariances -> (n_users, K, C, C): user u's maps toward each external C_j, trace-matched to u
    (C_j x tr(C_u) / tr(C_j), + the same shrinkage), so T C_u T = that target."""
    def sq(A, pw):
        w, V = np.linalg.eigh(A)
        return (V * np.clip(w, 1e-12, None) ** pw) @ V.T
    Cs = []
    for k in range(n_users):
        x = X[u == k].double()
        x = x - x.mean(-1, keepdim=True)
        Ck = (torch.einsum("ncl,ndl->cd", x, x) / (x.shape[0] * x.shape[-1])).cpu().numpy()
        Cs.append(Ck + shrink * np.trace(Ck) / len(Ck) * np.eye(len(Ck)))
    C = len(Cs[0])
    M = np.tile(np.eye(C), (n_users, n_users, 1, 1))
    if ext is not None:                                 # backlog 65 (a): targets = the external users, trace-matched
        assert not mix, "backlog 65: rc_ext with rc_mix is not wired"
        ext = np.asarray(ext, np.float64)
        assert ext.ndim == 3 and ext.shape[1:] == (C, C), (ext.shape, C)
        out = np.empty((n_users, len(ext), C, C))
        for a in range(n_users):
            h, hi = sq(Cs[a], 0.5), sq(Cs[a], -0.5)
            ta = np.trace(Cs[a])
            for j, Cj in enumerate(ext):
                Ct = Cj * (ta / np.trace(Cj))
                Ct = Ct + shrink * np.trace(Ct) / C * np.eye(C)
                out[a, j] = np.eye(C) + rc * (hi @ sq(h @ Ct @ h, 0.5) @ hi - np.eye(C))
        return torch.as_tensor(out, dtype=torch.float32, device=X.device)
    for a in range(n_users):
        h, hi = sq(Cs[a], 0.5), sq(Cs[a], -0.5)
        for b in range(n_users):
            if a != b:
                Tm = hi @ sq(h @ Cs[b] @ h, 0.5) @ hi
                M[a, b] = np.eye(C) + rc * (Tm - np.eye(C))
    if mix:                                             # T2-D237: targets = the other users' pairwise Bures midpoints (+ the real ones at mix 2)
        assert mix in (1, 2) and n_users >= 3, (mix, n_users)
        mids = {}
        for b in range(n_users):
            hb, hib = sq(Cs[b], 0.5), sq(Cs[b], -0.5)
            for c in range(b + 1, n_users):
                G = (np.eye(C) + hib @ sq(hb @ Cs[c] @ hb, 0.5) @ hib) / 2
                mids[b, c] = G @ Cs[b] @ G
        out = []
        for a in range(n_users):
            h, hi = sq(Cs[a], 0.5), sq(Cs[a], -0.5)
            Ma = [M[a, b] for b in range(n_users) if b != a] if mix == 2 else []
            for (b, c), Ct in mids.items():
                if a not in (b, c):
                    Ma.append(np.eye(C) + rc * (hi @ sq(h @ Ct @ h, 0.5) @ hi - np.eye(C)))
            out.append(np.stack(Ma))
        M = np.stack(out)
    return torch.as_tensor(M, dtype=torch.float32, device=X.device)


def rc_bank_cls(X, u, y, n_users, n_classes, rc, shrink=1e-3):
    """T2-D412: (n_classes, n_users, n_users, C, C) maps, M[c] = rc_bank over class c's windows only (C_{u,c} = user u's mean covariance of
    its demeaned class-c windows, the same shrinkage), so T C_{u,c} T = C_{v,c} at rc 1; every user must have windows of every class."""
    out = []
    for c in range(n_classes):
        k = y == c
        assert all(bool((k & (u == j)).any()) for j in range(n_users)), f"T2-D412: a training user has no class-{c} window"
        out.append(rc_bank(X[k], u[k], n_users, rc, shrink=shrink))
    return torch.stack(out)


RC_EXT_BANK = {}   # backlog 65 (a): {"C": (K, C, C) numpy} external covariances (t2p_covbank), set by the caller for recipes with rc_ext
CNN_INIT = {}  # T2-D367: {"sd": a state_dict} -> fit_eval loads every tensor but final_layer.* into the pooled net before training (t2p_bench --cnn-init); {} = off
VS_BANK = {}   # T2-D308: {"mats": (K, C, C)} same-user drift maps on the training device, for recipes with vs (the caller sets it per task)
FIT_AUDIT = {}  # fit audit 2026-11-09 (lead/FIT_AUDIT_2026-11-09): {"clean": True} -> each epoch's curve row gains clean_train_ba / clean_train_ce (eval mode, the UNAUGMENTED training windows, no label smoothing); {"dropout0": True} -> every dropout p = 0; {} = off (bitwise)
SYN_BANK = {}  # T2-D413: {"X": (N, C, L) fp16, "y": (N,) long, "u": (N,) long = the dominant parent user} on the training device (t2p_bench --syn-bank)
LONG = {}      # T2-D486: {"lo": int, "hi": int}: the window-start range (samples) inside the long training windows (t2p_bench --long-jitter)
XDOM = {}      # T2-D507: {"head": bool, "w": float, "dsbn": bool} for the pooled net's EXTERNAL training rows (data["dtr"] > 0): their own
               # classifier head (a fresh copy of final_layer; discarded after training), their loss weight, their own BatchNorm statistics
               # (DomainBN; the net is unwrapped back to the PROTEUS BatchNorm after training). {} or no data["dtr"] = off (bitwise)


class DomainBN(torch.nn.Module):
    """T2-D507: one BatchNorm per domain; state["m"] = a (B,) bool mask of external rows for the current forward (None = all PROTEUS)."""

    def __init__(self, bn, state):
        super().__init__()
        self.bn0, self.bn1 = bn, copy.deepcopy(bn)
        self.bn1.reset_parameters()
        self.state = state

    def forward(self, x):
        m = self.state.get("m")
        if m is None or not bool(m.any()):
            return self.bn0(x)
        if bool(m.all()):
            return self.bn1(x)
        out = torch.empty_like(x)
        out[~m], out[m] = self.bn0(x[~m]), self.bn1(x[m])
        return out


def xdom_wrap(net, state):
    """T2-D507: replace every BatchNorm of `net` by a DomainBN (in place); returns the wrapped names."""
    names = [n for n, m in net.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for n in names:
        parent, _, leaf = n.rpartition(".")
        p = net.get_submodule(parent) if parent else net
        setattr(p, leaf, DomainBN(getattr(p, leaf), state))
    return names


def xdom_unwrap(net):
    """T2-D507: put each DomainBN's PROTEUS BatchNorm (bn0) back, so the state_dict keys are the plain net's."""
    names = [n for n, m in net.named_modules() if isinstance(m, DomainBN)]
    for n in names:
        parent, _, leaf = n.rpartition(".")
        p = net.get_submodule(parent) if parent else net
        setattr(p, leaf, getattr(p, leaf).bn0)
    return names


def long_crop(Lb, L, lo, hi, gen):
    """T2-D486: one L-sample window per row of the long ALIGNED training windows Lb (B, C, TL), start ~ U{lo..hi} drawn from `gen` (a CPU
    generator of its own, so every other draw of the fit stays the control's), re-demeaned per channel over the crop: a start at the contract
    offset reproduces the aligned contract window (the alignment is linear in the demeaned window)."""
    B, C_, TL = Lb.shape
    assert 0 <= lo <= hi <= TL - L, (lo, hi, TL, L)
    st = torch.randint(lo, hi + 1, (B,), generator=gen).to(Lb.device)
    ix = (st[:, None] + torch.arange(L, device=Lb.device)[None]).unsqueeze(1).expand(B, C_, L)
    x = torch.gather(Lb, 2, ix)
    return x - x.mean(-1, keepdim=True)


VP_BANK = {}   # T2-D193: {"mats": (K, C, C), "gains": (K, C), "shifts": (K,)} tensors on the training device, for recipes with vp


def augment(xb, recipe, ub=None, rcm=None, yc=None):
    """T2-D112: per-window session-shift augmentation of a training batch (B, C, T); identity when all knobs are off.
    T2-D230: ub = the windows' training-user indices and rcm = rc_bank(...) for recipes with rc.
    T2-D412: yc = the windows' class indices for rc_cls (rcm = rc_bank_cls(...): (n_classes, n_users, n_users, C, C))."""
    B, C, L = xb.shape
    if recipe.rc > 0:                                   # T2-D230: re-colour each window toward a random other training user
        assert ub is not None and rcm is not None, "recipe.rc needs the windows' users and rc_bank"
        if recipe.rc_cls:                               # T2-D412: toward v's covariance OF THE WINDOW'S CLASS
            assert yc is not None and rcm.dim() == 5, "T2-D412: rc_cls needs the windows' classes and rc_bank_cls"
            v = (ub + torch.randint(1, rcm.shape[1], (B,), device=xb.device)) % rcm.shape[1]
            Mv = rcm[yc, ub, v].to(xb.dtype)
        elif recipe.rc_mix or recipe.rc_ext:            # T2-D237 / backlog 65 (a): rcm = (n_users, K, C, C): K targets per user, one drawn per window
            assert recipe.rc_lo == 0, "T2-D237: rc_mix with a fixed strength only"
            Mv = rcm[ub, torch.randint(rcm.shape[1], (B,), device=xb.device)].to(xb.dtype)
        else:
            v = (ub + torch.randint(1, len(rcm), (B,), device=xb.device)) % len(rcm)
            Mv = rcm[ub, v].to(xb.dtype)
        if recipe.rc_lo > 0:                            # T2-D236: strength s ~ U(rc_lo, rc) per window; rcm was built at rc
            assert 0 < recipe.rc_lo < recipe.rc, (recipe.rc_lo, recipe.rc)
            s = (recipe.rc_lo + (recipe.rc - recipe.rc_lo) * torch.rand(B, 1, 1, device=xb.device, dtype=xb.dtype)) / recipe.rc
            eye = torch.eye(C, device=xb.device, dtype=xb.dtype)
            Mv = eye + s * (Mv - eye)
        xb = torch.bmm(Mv, xb)
    if recipe.vs > 0:                                   # T2-D308: one same-user drift map per window with probability vs (after rc)
        assert VS_BANK, "recipe.vs needs ni26.train.VS_BANK (t2p_bench.vs_bank)"
        Mb = VS_BANK["mats"]
        k = torch.randint(len(Mb), (B,), device=xb.device)
        on = (torch.rand(B, device=xb.device) < recipe.vs).view(B, 1, 1)
        xb = torch.where(on, torch.bmm(Mb[k].to(xb.dtype), xb), xb)
    if recipe.gain_jit > 0:
        xb = xb * torch.exp((torch.rand(B, C, 1, device=xb.device) * 2 - 1) * recipe.gain_jit)
    if recipe.tshift > 0:
        s = torch.randint(-recipe.tshift, recipe.tshift + 1, (B, 1, 1), device=xb.device)
        t = (torch.arange(L, device=xb.device).view(1, 1, L) - s) % L
        xb = torch.gather(xb, 2, t.expand(B, C, L))
    if recipe.chdrop > 0:
        xb = xb * (torch.rand(B, C, 1, device=xb.device) >= recipe.chdrop).to(xb.dtype)
    if recipe.vp:                                       # T2-D193: one virtual participant per window
        assert VP_BANK, "recipe.vp needs ni26.train.VP_BANK (t2p_bench.vp_bank)"
        M, G, S = VP_BANK["mats"], VP_BANK["gains"], VP_BANK["shifts"]
        assert M.shape[1] == C, (tuple(M.shape), C)
        k = torch.randint(len(M), (B,), device=xb.device)
        Mk, Gk = M[k], G[k]
        if recipe.vp_a != 1.0:                          # T2-D200: a weaker draw, interpolated toward the identity
            Mk = torch.eye(C, device=Mk.device, dtype=Mk.dtype) + recipe.vp_a * (Mk - torch.eye(C, device=Mk.device, dtype=Mk.dtype))
            Gk = 1.0 + recipe.vp_a * (Gk - 1.0)
        xb = Gk[:, :, None].to(xb.dtype) * torch.bmm(Mk.to(xb.dtype), xb)
        t = (torch.arange(L, device=xb.device).view(1, 1, L) - S[k].long().view(B, 1, 1)) % L
        xb = torch.gather(xb, 2, t.expand(B, C, L))
    if recipe.fw > 0:                                   # T2-D229: one random warp factor per window (after the vp draw; both are linear)
        R = fw_bank(L, recipe.fw, xb.device)
        xb = torch.bmm(xb, R[torch.randint(len(R), (B,), device=xb.device)].to(xb.dtype))
    if recipe.tilt > 0:                                 # backlog 74: one random spectral tilt per window (last; linear)
        f = torch.fft.rfftfreq(L, 1.0 / TILT_FS, device=xb.device)
        beta = torch.randn(B, 1, 1, device=xb.device) * recipe.tilt
        xb = torch.fft.irfft(torch.fft.rfft(xb.float(), dim=2) * ((f + 1.0) / 11.0) ** beta, n=L, dim=2).to(xb.dtype)
    return xb


def sam_grad(net, xb, yb, recipe: Recipe):
    """T2-D224: replace net's gradients (from the batch's first backward) by the SAM gradient: the CE gradient at w + rho g / ||g||,
    rho = recipe.sam, on the same batch; the weights are restored afterwards, and BatchNorm running statistics are frozen in the second pass."""
    ps = [p for p in net.parameters() if p.grad is not None]
    scale = recipe.sam / (torch.norm(torch.stack([p.grad.detach().norm(2) for p in ps]), 2) + 1e-12)
    with torch.no_grad():
        eps = [p.grad * scale for p in ps]
        for p, e in zip(ps, eps):
            p.add_(e)
    bns = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    mom = [(m.momentum, None if m.num_batches_tracked is None else m.num_batches_tracked.clone()) for m in bns]
    for m in bns:
        m.momentum = 0.0                                # running = (1 - 0) running + 0 batch: unchanged
    net.zero_grad(set_to_none=True)
    F.cross_entropy(net(xb), yb, label_smoothing=recipe.label_smoothing).backward()
    for m, (mo, nb) in zip(bns, mom):
        m.momentum = mo
        if nb is not None:
            m.num_batches_tracked.copy_(nb)             # the step counter too, so the state dict matches a plain step's
    with torch.no_grad():
        for p, e in zip(ps, eps):
            p.sub_(e)


def copy_data(X, y, u, k, bank, ratio, seed):
    """T2-D239: user k's copy data = its own windows (in order) + n = min(round(ratio x n_k), n_other) windows of the other users, drawn without
    replacement by a CPU generator seeded `seed`, each multiplied by bank[v, k] (v = its user; bank = rc_bank(X, u, n_users, 1.0))."""
    own = u == k
    oth = torch.nonzero(~own).flatten()
    n = min(int(round(ratio * int(own.sum()))), len(oth))
    pick = oth[torch.randperm(len(oth), generator=torch.Generator(device="cpu").manual_seed(seed))[:n].to(oth.device)]
    return torch.cat([X[own], torch.bmm(bank[u[pick], k].to(X.dtype), X[pick])]), torch.cat([y[own], y[pick]])


def stream_ids(utr, sessions):
    """T2-D478: training windows' stream ids (user x session), int64 numpy."""
    key = np.array([f"{int(u)}|{s_}" for u, s_ in zip(np.asarray(utr), np.asarray(sessions))])
    return np.unique(key, return_inverse=True)[1].astype(np.int64)


def sbal_batches(sid, bs, k, g):
    """T2-D478: one epoch of stream-balanced batches (CPU index tensors): each stream's windows in a random order cut into chunks of k, the chunks
    shuffled, bs // k chunks per batch (the last batch may hold fewer). Every window appears once per epoch. sid: (n,) int64 CPU tensor."""
    chunks = []
    for s_ in torch.unique(sid).tolist():
        rows = torch.nonzero(sid == s_).flatten()
        chunks += list(rows[torch.randperm(len(rows), generator=g)].split(int(k)))
    order = torch.randperm(len(chunks), generator=g).tolist()
    per = max(1, int(bs) // int(k))
    return [torch.cat([chunks[j] for j in order[i:i + per]]) for i in range(0, len(order), per)]


def sbal_steps(sid, bs, k):
    """T2-D478: the number of sbal batches per epoch (deterministic: it depends on the stream sizes only)."""
    n_chunks = sum(int(math.ceil(int((sid == s_).sum()) / int(k))) for s_ in torch.unique(sid).tolist())
    return int(math.ceil(n_chunks / max(1, int(bs) // int(k))))


def sbal_ce(logits, y, sid, label_smoothing):
    """T2-D478: CE on the logits minus the log of the batch's per-stream mean softmax (detached): a stream's class prior earns nothing."""
    z = logits.float()
    p = F.softmax(z, 1).detach()
    u, inv = torch.unique(sid, return_inverse=True)
    m = torch.zeros(len(u), z.shape[1], device=z.device, dtype=z.dtype).index_add_(0, inv, p)
    m = m / torch.bincount(inv, minlength=len(u)).clamp_min(1).to(z.dtype)[:, None]
    return F.cross_entropy(z - torch.log(m.clamp_min(1e-8))[inv], y, label_smoothing=label_smoothing)


def set_ce(lo, y, yset, ls):
    """T2-D594b: per row, -log sum_{k in set} p_k where the row's set holds >= 2 classes, else the plain CE with label smoothing ls; the mean."""
    ce = F.cross_entropy(lo, y, label_smoothing=ls, reduction="none")
    lp = F.log_softmax(lo.float(), 1)
    sv = -torch.logsumexp(lp.masked_fill(~yset, float("-inf")), 1)
    return torch.where(yset.sum(1) > 1, sv, ce.float()).mean()


def finetune(net, X, y, recipe: Recipe, seed: int, X_other=None, sid=None, Lx=None, yset=None):
    """T2-D111 (iii): continue training a copy of `net` on (X, y) tensors with `recipe` (no validation, last epoch). T2-D430: recipe.kl_other > 0
    needs X_other (the other training users' windows) and adds kl_other x KL(p_pooled || p_copy) on a batch of them per step."""
    import copy
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    p_other = None
    if Lx is not None:                                      # T2-D486: the copy's long windows (rows = X's), cropped per batch
        assert len(Lx) == len(X) and LONG, "T2-D486: Lx needs one long window per copy window and ni26.train.LONG"
        g_l = torch.Generator(device="cpu").manual_seed(seed + 9173)
    if recipe.kl_other > 0:                                 # T2-D430: the pooled net's probabilities on the strangers, once
        assert X_other is not None and len(X_other), "T2-D430: kl_other needs X_other"
        net.eval()
        with torch.no_grad():
            p_other = torch.cat([F.softmax(net(X_other[i:i + 512]).float(), 1) for i in range(0, len(X_other), 512)])
        g_o = torch.Generator(device="cpu").manual_seed(seed + 7919)
    net = copy.deepcopy(net)
    params = net.parameters()
    if recipe.train_part != "all":                          # T2-D228: freeze the other part (no grad, not in the optimiser, values = the pooled net's)
        assert recipe.train_part in ("head", "feat"), recipe.train_part
        params = []
        for k, p in net.named_parameters():
            if p.requires_grad and (k.startswith("final_layer.") == (recipe.train_part == "head")):
                params.append(p)
            else:
                p.requires_grad_(False)
        assert params, "T2-D228: nothing left to train"
    opt = torch.optim.AdamW(params, lr=recipe.lr, weight_decay=recipe.wd)
    if yset is not None:                                    # T2-D594b: set-valued rows (None = the code below, bitwise)
        assert not (recipe.sbal or recipe.kl_other or Lx is not None or y.dtype.is_floating_point or recipe.ft_mix > 0), "T2-D594b: yset on the plain CE branch only"
        assert yset.dim() == 2 and yset.shape[0] == len(X) and yset.dtype == torch.bool and bool(yset[torch.arange(len(y)), y].all()), \
            "T2-D594b: yset = a (N, K) bool mask holding each row's label"
    if recipe.ft_mix > 0:                                   # T2-D589: the copies' mixup (hard labels, plain CE only); its own generators
        assert not (recipe.sbal or recipe.kl_other or Lx is not None or y.dtype.is_floating_point), "T2-D589: ft_mix is wired for the plain hard-label CE branch only"
        g_m = torch.Generator(device="cpu").manual_seed(seed + 5890)
        rng_m = np.random.default_rng(seed + 5891)
    if recipe.sbal:                                         # T2-D478
        assert sid is not None and len(sid) == len(X) and recipe.kl_other == 0, "T2-D478: copies with sbal need sid (and no kl_other)"
        sid_c = torch.as_tensor(np.asarray(sid), dtype=torch.long)
    total = recipe.epochs * (sbal_steps(sid_c, recipe.bs, recipe.sbal) if recipe.sbal else int(math.ceil(len(X) / recipe.bs)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _cosine(s, total, int(recipe.warmup_frac * total)))
    start = recipe.epochs - int(round(recipe.swa * recipe.epochs)) if recipe.swa > 0 else recipe.epochs
    avg, n_avg = None, 0
    for ep in range(recipe.epochs):
        net.train()
        if recipe.sbal:                                     # T2-D478
            batches = [b_.to(X.device) for b_ in sbal_batches(sid_c, recipe.bs, recipe.sbal, g)]
        else:
            perm = torch.randperm(len(X), generator=g).to(X.device)
            batches = [perm[i:i + recipe.bs] for i in range(0, len(X), recipe.bs)]
        for b in batches:
            x0 = X[b] if Lx is None else long_crop(Lx[b], X.shape[2], LONG["lo"], LONG["hi"], g_l)   # T2-D486
            xb = augment(x0, recipe) if recipe.vp else x0           # T2-D200: copies through the bank (pu_vp); off = unchanged
            if recipe.sbal:                                 # T2-D478
                loss = sbal_ce(net(xb), y[b], sid_c.to(X.device)[b], recipe.label_smoothing)
            elif y.dtype.is_floating_point:                 # T2-D510: soft targets (N, K) = a teacher's posterior; cross-entropy to it
                loss = -(y[b] * F.log_softmax(net(xb).float(), 1)).sum(1).mean()
            elif yset is not None:                          # T2-D594b
                loss = set_ce(net(xb), y[b], yset[b], recipe.label_smoothing)
            elif recipe.ft_mix > 0:                         # T2-D589: lam ~ Beta(a, a), one pairing j per batch, the same lam for x and the labels
                lam = float(rng_m.beta(recipe.ft_mix, recipe.ft_mix))
                j = torch.randperm(len(b), generator=g_m).to(X.device)
                lo = net(lam * xb + (1 - lam) * xb[j])
                loss = lam * F.cross_entropy(lo, y[b], label_smoothing=recipe.label_smoothing) + \
                    (1 - lam) * F.cross_entropy(lo, y[b][j], label_smoothing=recipe.label_smoothing)
            else:
                loss = F.cross_entropy(net(xb), y[b], label_smoothing=recipe.label_smoothing)
            if p_other is not None:                         # T2-D430: stay pooled-like on strangers (BatchNorm / dropout in eval for this pass)
                o = torch.randint(0, len(X_other), (len(b),), generator=g_o).to(X.device)
                net.eval()
                lq = F.log_softmax(net(X_other[o]).float(), 1)
                net.train()
                loss = loss + recipe.kl_other * F.kl_div(lq, p_other[o], reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if ep >= start:                                     # T2-D227: running uniform mean of the end-of-epoch weights + float buffers
            n_avg += 1
            sd = net.state_dict()
            if avg is None:
                avg = {k: v.detach().clone() for k, v in sd.items() if v.is_floating_point()}
            else:
                with torch.no_grad():
                    for k in avg:
                        avg[k] += (sd[k].detach() - avg[k]) / n_avg
    if avg is not None:                                     # integer buffers (BatchNorm's step counter) stay the last epoch's
        net.load_state_dict({k: avg.get(k, v) for k, v in net.state_dict().items()})
    return net.eval()


MIXSTYLE_AT = {"fbcnet_bboth": "spatial_conv", "shallow": "bnorm"}   # backlog 70: the map whose per-channel time statistics are mixed


def mixstyle_hook(p, state, alpha=0.1):
    """backlog 70: a forward hook (training only): with probability p per call, out = norm(out) * mixed std + mixed mean, the statistics
    (per window and channel, over the map's last two dims, detached) mixed lam : 1 - lam with a random window of another training user and the
    same class in the batch (state["u"] / state["y"] = the batch's users / labels), lam ~ Beta(alpha, alpha); a window with no such partner in
    the batch is unchanged."""
    def hook(mod, inp, out):
        u, y = state.get("u"), state.get("y")
        if not mod.training or u is None or float(torch.rand(())) >= p:
            return None
        B = out.shape[0]
        mu = out.mean((2, 3), keepdim=True).detach()
        sig = (out.var((2, 3), keepdim=True) + 1e-6).sqrt().detach()
        sc = torch.rand(B, B, device=out.device)
        sc[(u[:, None] == u[None, :]) | (y[:, None] != y[None, :])] = -1.0
        j = sc.argmax(1)
        ok = (sc.max(1).values >= 0).view(B, 1, 1, 1).to(out.dtype)
        lam = torch.distributions.Beta(alpha, alpha).sample((B, 1, 1, 1)).to(out.device, out.dtype)
        lam = ok * lam + (1 - ok)
        return (out - mu) / sig * (lam * sig + (1 - lam) * sig[j]) + lam * mu + (1 - lam) * mu[j]
    return hook


def ustyle_hook(p, s, n_users, state, mom=0.1):
    """T2-D365: a forward PRE-hook on final_layer (training only). Per training user, running per-element mean / variance of the layer's input
    (EMA mom, detached, from the un-restyled batch; a user's first batch with >= 2 of its windows initialises it). With probability p per
    window whose user has statistics, f <- (f - mu_u) / sd_u * sd_t + mu_t, mu_t = (1 - s) mu_u + s mu_v, sd_t = (1 - s) sd_u + s sd_v, v != u
    a uniform other training user with statistics (state["u"] = the batch's users). The gradient flows through f; the statistics are constants."""
    st = {}

    def hook(mod, inp):
        u = state.get("u")
        if not mod.training or u is None:
            return None
        f = inp[0]
        B = f.shape[0]
        fd = f.detach().float()
        if not st:
            st["mu"] = torch.zeros((n_users,) + tuple(f.shape[1:]), device=f.device)
            st["var"] = torch.ones_like(st["mu"])
            st["seen"] = torch.zeros(n_users, dtype=torch.bool, device=f.device)
        for uu in torch.unique(u).tolist():
            m = u == uu
            if int(m.sum()) < 2:
                continue
            mu_b, var_b = fd[m].mean(0), fd[m].var(0, unbiased=False)
            if not bool(st["seen"][uu]):
                st["mu"][uu], st["var"][uu] = mu_b, var_b
                st["seen"][uu] = True
            else:
                st["mu"][uu].mul_(1 - mom).add_(mom * mu_b)
                st["var"][uu].mul_(1 - mom).add_(mom * var_b)
        ok = st["seen"].nonzero().flatten()
        if len(ok) < 2:
            return None
        sc = torch.rand(B, len(ok), device=f.device)
        sc[ok[None, :] == u[:, None]] = -1.0
        v = ok[sc.argmax(1)]
        on = (torch.rand(B, device=f.device) < p) & st["seen"][u]
        mu_u, sd_u = st["mu"][u], (st["var"][u] + 1e-5).sqrt()
        mu_v, sd_v = st["mu"][v], (st["var"][v] + 1e-5).sqrt()
        mu_t, sd_t = (1 - s) * mu_u + s * mu_v, (1 - s) * sd_u + s * sd_v
        f2 = (f - mu_u.to(f.dtype)) / sd_u.to(f.dtype) * sd_t.to(f.dtype) + mu_t.to(f.dtype)
        return (torch.where(on.view((B,) + (1,) * (f.dim() - 1)), f2, f),) + tuple(inp[1:])
    return hook


def xmix_batch(xb, yb, ub, p):
    """backlog 73: each window, with probability p, -> (lam x + (1 - lam) x_b) / sqrt(lam^2 + (1 - lam)^2), x_b a random window of the batch with
    the same label from another training user (ub), lam ~ U(0, 1); a window with no such partner is unchanged."""
    B = xb.shape[0]
    sc = torch.rand(B, B, device=xb.device)
    sc[(ub[:, None] == ub[None, :]) | (yb[:, None] != yb[None, :])] = -1.0
    j = sc.argmax(1)
    on = (sc.max(1).values >= 0) & (torch.rand(B, device=xb.device) < p)
    lam = torch.rand(B, 1, 1, device=xb.device, dtype=xb.dtype)
    mixed = (lam * xb + (1 - lam) * xb[j]) / torch.sqrt(lam ** 2 + (1 - lam) ** 2)
    return torch.where(on.view(B, 1, 1), mixed, xb)


class GradReverse(torch.autograd.Function):
    """T2-D293: identity forward, -lam x the gradient backward (DANN's gradient-reversal layer)."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def fit_eval(model_name, data, recipe: Recipe, seed: int, device="cuda", sfreq=120.0, log_every=1,
             return_state=False):
    """data: dict with Xtr,ytr, Xva,yva,cva (cells), Xte,yte,cte (numpy). Returns a result dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    g = torch.Generator(device="cpu").manual_seed(seed)

    def T(a, dt=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)

    Xtr, ytr = T(data["Xtr"]), T(data["ytr"], torch.long)
    Ltr = None
    if data.get("ltr") is not None:                     # T2-D486: the long aligned training windows, one per Xtr row, cropped per batch
        assert len(data["ltr"]) == len(data["Xtr"]) and LONG, "T2-D486: ltr needs one long window per training window and ni26.train.LONG"
        Ltr = T(data["ltr"])
        g_l = torch.Generator(device="cpu").manual_seed(seed + 9173)
    Xva = T(data["Xva"]) if len(data["yva"]) else None
    Xte = T(data["Xte"])
    K = int(max(data["ytr"].max(), data["yte"].max()) + 1)
    C, L = Xtr.shape[1], Xtr.shape[2]
    user_head = model_name in ("fbcnet_uid", "fbcnet_moe")
    assert not (recipe.sam and (recipe.mixup or user_head)), "T2-D224: SAM is wired for the plain CE branch only"
    assert not recipe.swa, "T2-D227: swa is a per-user copy field (pu_swa); the pooled net does not average"
    assert recipe.train_part == "all", "T2-D228: train_part is a per-user copy field (pu_train); the pooled net trains every part"
    utr = T(data["utr"], torch.long) if user_head else None
    if recipe.rc > 0:                                   # T2-D230: the users' re-colouring maps, from the training windows
        assert not user_head and int(data["utr"].max()) >= (2 if recipe.rc_mix else 1), "T2-D230: rc needs >= 2 training users (rc_mix 3) and the plain branch"
        urc = T(data["utr"], torch.long)
        if recipe.rc_ext:                               # backlog 65 (a)
            assert "C" in RC_EXT_BANK, "rc_ext needs RC_EXT_BANK (the caller loads t2p_covbank's npz)"
        if recipe.rc_cls:                               # T2-D412: class-conditional maps (fixed strength, real targets only)
            assert not (recipe.rc_mix or recipe.rc_ext or recipe.rc_lo), "T2-D412: rc_cls is wired with a fixed strength and real targets only"
            rcm = rc_bank_cls(Xtr, urc, ytr, int(data["utr"].max()) + 1, int(data["ytr"].max()) + 1, recipe.rc)
        else:
            rcm = rc_bank(Xtr, urc, int(data["utr"].max()) + 1, recipe.rc, mix=recipe.rc_mix, ext=RC_EXT_BANK["C"] if recipe.rc_ext else None)
    if recipe.syn > 0:                                  # T2-D413: the synthetic bank (plain CE branch only: every other branch indexes by b)
        assert SYN_BANK and tuple(SYN_BANK["X"].shape[1:]) == (C, L), "T2-D413: syn needs ni26.train.SYN_BANK of this task's window shape"
        assert not (user_head or recipe.mixup or recipe.sam or recipe.gdro or recipe.kd or recipe.user_adv or recipe.xmix or recipe.mixstyle
                    or recipe.ustyle or recipe.rc_cls or recipe.rc_mix or recipe.rc_ext), "T2-D413: syn is wired for the plain CE branch with class-pooled rc"
        assert int(SYN_BANK["y"].max()) < K and (recipe.rc == 0 or int(SYN_BANK["u"].max()) <= int(data["utr"].max())), "T2-D413: bank labels / users"
        gsyn = torch.Generator(device="cpu").manual_seed(seed + 4241)
    net = build_model(model_name, C, K, L, sfreq, n_users=int(data["utr"].max()) + 1 if user_head else 0).to(device)
    if CNN_INIT.get("sd") is not None:                  # T2-D367: a pretrained (SSL) init of the pooled net; final_layer stays this seed's fresh draw
        assert not user_head, "T2-D367: cnn init on the plain nets only"
        r_ = net.load_state_dict({k: v for k, v in CNN_INIT["sd"].items() if not k.startswith("final_layer.")}, strict=False)
        assert all(k.startswith("final_layer.") for k in r_.missing_keys) and not r_.unexpected_keys, (r_.missing_keys, r_.unexpected_keys)
    if FIT_AUDIT.get("dropout0"):                       # fit audit: the regularisers-off probe
        for mod in net.modules():
            if isinstance(mod, torch.nn.modules.dropout._DropoutNd):
                mod.p = 0.0
    n_params = sum(p.numel() for p in net.parameters())
    xd_on = data.get("dtr") is not None and bool(np.asarray(data["dtr"]).max() > 0) and bool(XDOM)   # T2-D507
    xd_state, xd_head, xd_hook = {}, None, None
    if xd_on:
        assert len(data["dtr"]) == len(data["Xtr"]), "T2-D507: one domain id per training window"
        assert not (user_head or recipe.mixup or recipe.sam or recipe.gdro or recipe.kd or recipe.user_adv or recipe.syn or recipe.sbal
                    or recipe.rc or recipe.xmix or recipe.mixstyle or recipe.ustyle), "T2-D507: XDOM is wired for the plain CE branch only"
        dtr_t = T(data["dtr"], torch.long)
        if XDOM.get("dsbn"):
            xdom_wrap(net, xd_state)
            net.to(device)
        if XDOM.get("head"):
            xd_head = copy.deepcopy(net.final_layer)
            for m_ in xd_head.modules():
                if m_ is not xd_head and hasattr(m_, "reset_parameters"):
                    m_.reset_parameters()
            xd_head.to(device)
            xd_hook = net.final_layer.register_forward_hook(
                lambda m_, inp, out: out if xd_state.get("m") is None or not bool(xd_state["m"].any())
                else torch.where(xd_state["m"].view(-1, *([1] * (out.dim() - 1))), xd_head(inp[0]), out))
        xd_w = float(XDOM.get("w", 1.0))
        print(f"[xdom] external rows {int((data['dtr'] > 0).sum())} / {len(data['dtr'])}: own head {bool(xd_head is not None)}, "
              f"loss weight {xd_w:g}, domain BN {bool(XDOM.get('dsbn'))}", flush=True)
    side_on = data.get("side_tr") is not None             # T2-D517: a per-window side vector added at the logits (a zero-init linear head)
    w_on = data.get("wtr") is not None                    # T2-D630: per-training-window loss weights (plain CE branch; absent = bitwise)
    if w_on:
        assert not (xd_on or side_on or user_head or recipe.mixup or recipe.sam or recipe.gdro or recipe.kd or recipe.user_adv or recipe.syn
                    or recipe.sbal), "T2-D630: the loss weights are wired for the plain CE branch only"
        assert len(data["wtr"]) == len(data["Xtr"]) and np.isfinite(data["wtr"]).all() and (np.asarray(data["wtr"]) > 0).all(), "T2-D630: one positive weight per row"
        wtr_t = torch.as_tensor(np.asarray(data["wtr"], np.float32), device=device)
    side_head = None
    assert not recipe.side_mix or side_on, "T2-D580: side_mix is the side-head branch's joint mixup (needs side vectors)"
    if side_on:
        assert not (xd_on or user_head or recipe.mixup or recipe.sam or recipe.gdro or recipe.kd or recipe.user_adv or recipe.syn or recipe.sbal
                    or recipe.rc or recipe.xmix or recipe.mixstyle or recipe.ustyle or recipe.ema), "T2-D517: the side head is wired for the plain CE branch only"
        assert len(data["side_tr"]) == len(data["Xtr"]) and len(data["side_te"]) == len(data["Xte"]), "T2-D517: one side row per window"
        S_tr, S_te = T(data["side_tr"]), T(data["side_te"])
        S_va = T(data["side_va"]) if len(data["yva"]) else None
        side_head = torch.nn.Linear(S_tr.shape[1], K).to(device)
        torch.nn.init.zeros_(side_head.weight)
        torch.nn.init.zeros_(side_head.bias)
        print(f"[side] side head: {S_tr.shape[1]} -> {K} (zero init), plain CE branch", flush=True)
    adv, hook, feat = None, None, {}
    if recipe.user_adv > 0:                             # T2-D293: the user adversary on final_layer's input (discarded after training)
        assert not (user_head or recipe.mixup or recipe.sam), "T2-D293: user_adv is wired for the plain CE branch only"
        uadv = T(data["utr"], torch.long)
        n_u = int(data["utr"].max()) + 1
        assert n_u >= 2, "T2-D293: user_adv needs >= 2 training users"
        hook = net.final_layer.register_forward_pre_hook(lambda m, inp: feat.__setitem__("f", inp[0]))
        net.eval()
        with torch.no_grad():                           # eval mode: no BatchNorm update, no dropout draw
            net(Xtr[:2])
        net.train()
        d_f = feat["f"].flatten(1).shape[1]
        mk = (lambda: torch.nn.Sequential(torch.nn.Linear(d_f, recipe.adv_hidden), torch.nn.ReLU(), torch.nn.Linear(recipe.adv_hidden, n_u))
              if recipe.adv_hidden > 0 else torch.nn.Linear(d_f, n_u))     # T2-D374: the MLP adversary (0 = T2-D293's linear one)
        adv = (torch.nn.ModuleList([mk() for _ in range(3)]) if recipe.adv_cond else mk()).to(device)
    assert not recipe.tilt or sfreq == TILT_FS, f"backlog 74: tilt assumes {TILT_FS} Hz windows, got {sfreq}"
    if recipe.xmix > 0:                                 # backlog 73
        assert not (user_head or recipe.mixup or recipe.sam), "backlog 73: xmix is wired for the plain CE branch"
        uxm = T(data["utr"], torch.long)
    ms_hook, ms_state = None, {}
    if recipe.mixstyle > 0:                             # backlog 70: MixStyle across training users (the hook is removed after training)
        assert model_name in MIXSTYLE_AT and not (user_head or recipe.mixup or recipe.sam), f"backlog 70: mixstyle is wired for {sorted(MIXSTYLE_AT)}, plain CE"
        ums = T(data["utr"], torch.long)
        ms_hook = getattr(net, MIXSTYLE_AT[model_name]).register_forward_hook(mixstyle_hook(recipe.mixstyle, ms_state))
    us_hook, us_state = None, {}
    if recipe.ustyle > 0:                               # T2-D365: user-level feature re-styling (the hook is removed after training)
        assert not (user_head or recipe.mixup or recipe.sam), "T2-D365: ustyle is wired for the plain CE branch only"
        assert int(data["utr"].max()) >= 1, "T2-D365: ustyle needs >= 2 training users"
        uus = T(data["utr"], torch.long)
        us_hook = net.final_layer.register_forward_pre_hook(ustyle_hook(recipe.ustyle, recipe.ustyle_s, int(data["utr"].max()) + 1, us_state))
    if recipe.kd > 0:                                   # backlog 69: the FM teacher's probabilities (and external windows)
        assert not (user_head or recipe.mixup or recipe.sam or recipe.user_adv), "backlog 69: kd is wired for the plain CE branch only"
        assert 0 < recipe.kd <= 1 and recipe.kd_t > 0, (recipe.kd, recipe.kd_t)
        ptr = T(data["ptr"])
        assert ptr.shape == (len(Xtr), K), (tuple(ptr.shape), len(Xtr), K)
        ptr_t = ptr.clamp_min(1e-8) ** (1.0 / recipe.kd_t)
        ptr_t = ptr_t / ptr_t.sum(1, keepdim=True)
        if recipe.kd_ext:
            Xkd, pkd = T(data["Xkd"]), T(data["pkd"])
            assert Xkd.shape[1:] == Xtr.shape[1:] and pkd.shape == (len(Xkd), K), (tuple(Xkd.shape), tuple(pkd.shape))
            pkd_t = pkd.clamp_min(1e-8) ** (1.0 / recipe.kd_t)
            pkd_t = pkd_t / pkd_t.sum(1, keepdim=True)
            gk = torch.Generator(device="cpu").manual_seed(seed + 7919)
    if adv is not None and (recipe.adv_lr_mult != 1.0 or recipe.adv_ramp or recipe.adv_hidden or recipe.adv_cond):   # T2-D374
        opt = torch.optim.AdamW([dict(params=list(net.parameters())), dict(params=list(adv.parameters()), lr=recipe.lr * recipe.adv_lr_mult)],
                                lr=recipe.lr, weight_decay=recipe.wd)
    else:
        opt = torch.optim.AdamW(list(net.parameters()) + (list(adv.parameters()) if adv is not None else [])
                                + (list(xd_head.parameters()) if xd_head is not None else [])
                                + (list(side_head.parameters()) if side_head is not None else []), lr=recipe.lr,   # T2-D507 / D517
                                weight_decay=recipe.wd)
    adv_log = dict(acc=[], hit=0, n=0)                  # T2-D374: the adversary's running user accuracy per epoch
    if recipe.sbal:                                     # T2-D478: stream-balanced batches + the prior-blind loss (plain CE branch only)
        assert not (user_head or recipe.mixup or recipe.sam or recipe.gdro or recipe.kd or recipe.user_adv or recipe.syn or recipe.xmix), \
            "T2-D478: sbal is wired for the plain CE branch only"
        assert len(data["str"]) == len(data["Xtr"]), "T2-D478: sbal needs data['str'] for every training window"
        sid_cpu = torch.as_tensor(stream_ids(data["utr"], data["str"]), dtype=torch.long)
        sid_dev = sid_cpu.to(device)
    steps_per_epoch = sbal_steps(sid_cpu, recipe.bs, recipe.sbal) if recipe.sbal else int(math.ceil(len(Xtr) / recipe.bs))
    total = recipe.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _cosine(s, total, int(recipe.warmup_frac * total)))
    sw = torch.optim.swa_utils
    ema = sw.AveragedModel(net, multi_avg_fn=sw.get_ema_multi_avg_fn(recipe.ema), use_buffers=True) if recipe.ema > 0 else None
    if recipe.gdro > 0:                                 # T2-D315: the users' DRO weights (uniform at start)
        assert not (user_head or recipe.mixup or recipe.kd or recipe.user_adv or recipe.sam), "T2-D315: gdro on the plain CE branch only"
        ugd = T(data["utr"], torch.long)
        n_ugd = int(data["utr"].max()) + 1
        q_gd = torch.full((n_ugd,), 1.0 / n_ugd, device=device)
    curves = []
    best = (-1.0, -1, None)
    t0 = time.time()
    for ep in range(recipe.epochs):
        net.train()
        if recipe.sbal:                                 # T2-D478
            batches = [b_.to(device) for b_ in sbal_batches(sid_cpu, recipe.bs, recipe.sbal, g)]
        else:
            perm = torch.randperm(len(Xtr), generator=g).to(device)
            batches = [perm[i:i + recipe.bs] for i in range(0, len(Xtr), recipe.bs)]
        tl, tc, tn = 0.0, 0, 0
        for b in batches:
            xr, yr, ur = Xtr[b], ytr[b], (urc[b] if recipe.rc > 0 else None)
            if Ltr is not None:                         # T2-D486: real neighbouring EEG of the same trial, a random start per window
                xr = long_crop(Ltr[b], L, LONG["lo"], LONG["hi"], g_l)
            if recipe.syn > 0:                          # T2-D413: round(syn x B) bank windows appended (syn 0: this block never runs)
                j = torch.randint(len(SYN_BANK["X"]), (int(round(recipe.syn * len(b))),), generator=gsyn).to(device)
                xr, yr = torch.cat([xr, SYN_BANK["X"][j].to(xr.dtype)]), torch.cat([yr, SYN_BANK["y"][j]])
                ur = torch.cat([ur, SYN_BANK["u"][j]]) if ur is not None else None
            xb, yb = augment(xr, recipe, *((ur, rcm) if recipe.rc > 0 else ()), yc=yr if recipe.rc_cls else None), yr
            if recipe.xmix > 0:                         # backlog 73
                xb = xmix_batch(xb, yb, uxm[b], recipe.xmix)
            if us_hook is not None:                     # T2-D365: the batch's users, for the hook
                us_state["u"] = uus[b]
            if ms_hook is not None:                     # backlog 70: the batch's users, for the hook
                ms_state["u"], ms_state["y"] = ums[b], yb
            if user_head:
                logits, ul = net.forward_all(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=recipe.label_smoothing) + \
                    recipe.uid_w * F.cross_entropy(ul.float(), utr[b])
            elif recipe.mixup > 0:
                lam = float(np.random.beta(recipe.mixup, recipe.mixup))
                j = torch.randperm(len(b), generator=g).to(device)
                logits = net(lam * xb + (1 - lam) * xb[j])
                loss = lam * F.cross_entropy(logits, yb, label_smoothing=recipe.label_smoothing) + \
                    (1 - lam) * F.cross_entropy(logits, yb[j], label_smoothing=recipe.label_smoothing)
            elif recipe.gdro > 0:                       # T2-D315
                logits = net(xb)
                ce = F.cross_entropy(logits.float(), yb, label_smoothing=recipe.label_smoothing, reduction="none")
                ub_ = ugd[b]
                cnt = torch.bincount(ub_, minlength=n_ugd).float()
                lu = torch.zeros(n_ugd, device=device).index_add_(0, ub_, ce) / cnt.clamp_min(1.0)
                on = cnt > 0
                with torch.no_grad():
                    q_gd[on] = q_gd[on] * torch.exp(recipe.gdro * lu[on].detach())
                    q_gd /= q_gd.sum()
                loss = (q_gd[on] * lu[on]).sum() / q_gd[on].sum()
            elif side_on:                               # T2-D517: the side vector's logits added to the net's
                if recipe.side_mix > 0:                 # T2-D580: ONE pairing j and ONE lambda for x, the side vector and the labels
                    lam = float(np.random.beta(recipe.side_mix, recipe.side_mix))
                    j = torch.randperm(len(b), generator=g).to(device)
                    Sb = S_tr[b]
                    logits = net(lam * xb + (1 - lam) * xb[j]) + side_head(lam * Sb + (1 - lam) * Sb[j])
                    loss = lam * F.cross_entropy(logits, yb, label_smoothing=recipe.label_smoothing) + \
                        (1 - lam) * F.cross_entropy(logits, yb[j], label_smoothing=recipe.label_smoothing)
                else:
                    logits = net(xb) + side_head(S_tr[b])
                    loss = F.cross_entropy(logits, yb, label_smoothing=recipe.label_smoothing)
            elif xd_on:                                 # T2-D507: external rows on their own head / BatchNorm, at their loss weight
                db_ = dtr_t[b] > 0
                xd_state["m"] = db_
                logits = net(xb)
                xd_state["m"] = None
                ce = F.cross_entropy(logits.float(), yb, label_smoothing=recipe.label_smoothing, reduction="none")
                wb_ = torch.where(db_, torch.full_like(ce, xd_w), torch.ones_like(ce))
                loss = (wb_ * ce).sum() / wb_.sum()
            else:
                logits = net(xb)
                if w_on:                                # T2-D630: sum w ce / sum w per batch (self-normalised; a global rescale of w cancels)
                    ce_w = F.cross_entropy(logits.float(), yb, label_smoothing=recipe.label_smoothing, reduction="none")
                    loss = (wtr_t[b] * ce_w).sum() / wtr_t[b].sum()
                else:
                    loss = (sbal_ce(logits, yb, sid_dev[b], recipe.label_smoothing) if recipe.sbal          # T2-D478
                            else F.cross_entropy(logits, yb, label_smoothing=recipe.label_smoothing))
                if adv is not None and not (recipe.adv_ramp or recipe.adv_hidden or recipe.adv_cond or recipe.adv_lr_mult != 1.0):   # T2-D293
                    loss = loss + F.cross_entropy(adv(GradReverse.apply(feat["f"].flatten(1), recipe.user_adv)).float(), uadv[b])
                elif adv is not None:                   # T2-D374: the tuned DANN
                    prog = min(1.0, sched.last_epoch / max(total, 1))
                    lam_ = recipe.user_adv * (2.0 / (1.0 + math.exp(-10.0 * prog)) - 1.0) if recipe.adv_ramp else recipe.user_adv
                    fr = GradReverse.apply(feat["f"].flatten(1), lam_)
                    if recipe.adv_cond:
                        la, cnt = 0.0, 0
                        for c in range(3):
                            k_ = yb == c
                            if k_.any():
                                lo = adv[c](fr[k_]).float()
                                la = la + F.cross_entropy(lo, uadv[b][k_], reduction="sum")
                                cnt += int(k_.sum())
                                adv_log["hit"] += int((lo.argmax(1) == uadv[b][k_]).sum())
                        loss = loss + la / max(cnt, 1)
                        adv_log["n"] += cnt
                    else:
                        lo = adv(fr).float()
                        loss = loss + F.cross_entropy(lo, uadv[b])
                        adv_log["hit"] += int((lo.argmax(1) == uadv[b]).sum()); adv_log["n"] += len(b)
                if recipe.kd > 0:                       # backlog 69
                    tk = recipe.kd_t
                    kl = F.kl_div(F.log_softmax(logits.float() / tk, 1), ptr_t[b], reduction="batchmean") * tk * tk
                    loss = (1 - recipe.kd) * loss + recipe.kd * kl
                    if recipe.kd_ext:
                        j = torch.randint(len(Xkd), (len(b),), generator=gk).to(device)
                        net.eval()
                        lo_k = net(Xkd[j])
                        net.train()
                        loss = loss + recipe.kd * F.kl_div(F.log_softmax(lo_k.float() / tk, 1), pkd_t[j], reduction="batchmean") * tk * tk
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if recipe.sam > 0:
                sam_grad(net, xb, yb, recipe)
            if recipe.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), recipe.max_norm)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(net)
            tl += float(loss.detach()) * len(yb)        # T2-D413: len(yb) = len(b) unless syn
            tc += int((logits.argmax(1) == yb).sum())
            tn += len(yb)
        row = dict(epoch=ep + 1, train_loss=tl / tn, train_acc=tc / tn)
        if FIT_AUDIT.get("clean"):                      # fit audit: clean train fit in eval mode (predict_proba sets eval; the next epoch sets train)
            ptr_c = predict_proba(net, Xtr)
            yt_c = data["ytr"]
            row["clean_train_ba"] = float(np.mean([np.mean(ptr_c[yt_c == k_].argmax(1) == k_) for k_ in np.unique(yt_c)]))
            row["clean_train_ce"] = float(-np.mean(np.log(np.clip(ptr_c[np.arange(len(yt_c)), yt_c], 1e-12, None))))
        if Xva is not None:
            pva = _pp_side(net, side_head, Xva, S_va) if side_on else predict_proba(net, Xva)
            row["val_cell_bacc"] = metrics.cell_balanced_accuracy(data["yva"], pva.argmax(1), data["cva"])[0]
            if row["val_cell_bacc"] > best[0]:
                best = (row["val_cell_bacc"], ep + 1, {k: v.detach().clone() for k, v in net.state_dict().items()})
        if (ep + 1) % log_every == 0 or ep + 1 == recipe.epochs:
            pte = _pp_side(net, side_head, Xte, S_te) if side_on else predict_proba(net, Xte)
            row["test_cell_bacc"] = metrics.cell_balanced_accuracy(data["yte"], pte.argmax(1), data["cte"])[0]  # analysis only
        if adv_log["n"]:                                # T2-D374: this epoch's adversary user accuracy
            adv_log["acc"].append(adv_log["hit"] / adv_log["n"]); adv_log["hit"] = adv_log["n"] = 0
        curves.append(row)
    if hook is not None and (recipe.adv_ramp or recipe.adv_hidden or recipe.adv_cond or recipe.adv_lr_mult != 1.0):   # T2-D374: engagement (T2-D371)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        net.eval()
        fs = []
        with torch.no_grad():
            for i_ in range(0, len(Xtr), 512):
                net(Xtr[i_:i_ + 512]); fs.append(feat["f"].flatten(1).float().cpu().numpy())
        Fp = np.concatenate(fs)
        sub = np.arange(0, len(Fp), max(1, len(Fp) // 2000))
        probe_acc = float(cross_val_score(LogisticRegression(max_iter=2000), Fp[sub], data["utr"][sub], cv=3).mean())
        print(f"[dann] adversary user acc by epoch {[round(a, 3) for a in adv_log['acc']]} | fresh linear user probe (last epoch, 3-fold) {probe_acc:.3f} "
              f"(chance {1 / (int(data['utr'].max()) + 1):.3f})", flush=True)
    if hook is not None:                                   # T2-D293: the adversary is discarded
        hook.remove()
    if ms_hook is not None:                                # backlog 70
        ms_hook.remove()
    if us_hook is not None:                                # T2-D365
        us_hook.remove()
    if ema is not None:                                    # T2-D153: from here on, the net = the EMA weights + float buffers
        raw = net.state_dict()                             # (integer buffers, e.g. BatchNorm's step counter, stay the raw net's)
        net.load_state_dict({k: v if v.is_floating_point() else raw[k] for k, v in ema.module.state_dict().items()})
    if xd_hook is not None:                                # T2-D507: the external head is discarded
        xd_hook.remove()
    if xd_on and XDOM.get("dsbn"):                         # T2-D507: back to the plain net (PROTEUS BatchNorm), plain state_dict keys
        xdom_unwrap(net)
        if best[2] is not None:
            best = (best[0], best[1], {k.replace(".bn0.", "."): v for k, v in best[2].items() if ".bn1." not in k})
    prob_last = _pp_side(net, side_head, Xte, S_te) if side_on else predict_proba(net, Xte)
    side_dump = None
    if side_on and os.environ.get("NI26_SIDE_DUMP") == "1":   # T2-D533: the last epoch's pre-head logits + the head (a read-time z clamp); unset = unchanged
        with torch.no_grad():
            net.eval()
            lg_ = torch.cat([net(Xte[i:i + 512]).float() for i in range(0, len(Xte), 512)]).cpu().numpy()
        side_dump = dict(logit_last=lg_, w=side_head.weight.detach().float().cpu().numpy(), b=side_head.bias.detach().float().cpu().numpy())
    prob_val_last = (_pp_side(net, side_head, Xva, S_va) if side_on else predict_proba(net, Xva)) if Xva is not None else None   # val windows at the last epoch
    uid_test = predict_uid(net, Xte) if user_head else None
    uid_val = predict_uid(net, Xva) if user_head and Xva is not None else None
    state_last = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()} if return_state else None
    prob_best = None
    if best[2] is not None:
        net.load_state_dict(best[2])
        prob_best = _pp_side(net, side_head, Xte, S_te) if side_on else predict_proba(net, Xte)   # (T2-D517: the head is the last epoch's)
    if recipe.gdro > 0:                                 # T2-D315: where the DRO weights ended (uniform = 1 / n_users each)
        print(f"[gdro] final user weights q = {[round(float(v), 3) for v in q_gd]}", flush=True)
    return dict(prob_val_last=prob_val_last, model=model_name, recipe=asdict(recipe), seed=seed, n_params=int(n_params),
                n_train=int(len(Xtr)), n_val=int(len(data["yva"])), n_test=int(len(Xte)),
                best_epoch=int(best[1]), best_val=float(best[0]), seconds=round(time.time() - t0, 1),
                curves=curves, prob_last=prob_last, prob_best=prob_best, state_last=state_last, uid_test=uid_test,
                uid_val=uid_val,
                **({"side_dump": side_dump} if side_dump is not None else {}),        # T2-D533 (NI26_SIDE_DUMP=1 only)
                side_state=({k: v.detach().cpu().clone() for k, v in side_head.state_dict().items()} if side_on and return_state else None))   # T2-D517 (W17)
