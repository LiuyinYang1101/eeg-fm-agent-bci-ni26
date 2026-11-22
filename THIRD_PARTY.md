# Third-party material

The MIT licence in [`LICENSE`](LICENSE) covers the code written for this project. The files below derive from other projects and keep their own terms.

| Path | Origin | Terms |
|---|---|---|
| `tracks/t2_bci/rebrain/` (except `__init__.py`) | ReBrain, `github.com/Xianglong-Bao/ReBrain` at commit `41881e56`; details in `tracks/t2_bci/rebrain/NOTICE.md` | Apache-2.0, text in `tracks/t2_bci/rebrain/LICENSE_REBRAIN` |
| `src/ni26/fm/neurorvq.py` | The encoder of NeuroRVQ, re-written in plain PyTorch from `github.com/KonstantinosBarmpas/NeuroRVQ` at commit `926e770d` | That repository's licence applies; the model is released under CC BY-NC 4.0 (non-commercial) |
| `src/ni26/fm/mirepnet.py`, `mirepnet_template.json` | A re-implementation of MIRepNet (`github.com/staraink/MIRepNet`) that loads its published weights | MIT, as the original |
| `src/ni26/heegnet.py` | Re-derived from the HEEGNet paper and the mathematics of its reference code; no code copied | The reference code is CC BY-NC 4.0 |
| `src/ni26/tsmnet.py` | A re-implementation of the TSMNet / SPDDSMBN idea (Kobler et al., NeurIPS 2022) | Written for this project |
| `src/ni26/fm/steeg_pt.py` | A plain-PyTorch forward pass for ST-EEGFormer, my own published model | Written for this project |

Pre-trained weights are not distributed here. Each published checkpoint (ST-EEGFormer, MIRepNet, NeuroRVQ, LaBraM and others named in the logs) is subject to its own licence.

Not exported, but imported by the FM scripts: the ST-EEGFormer model code (`steeg_mae.py`, `steeg_vit.py`) from its authors' repository, whose masked-autoencoder part states that it is adapted from Meta's MAE implementation. Check that repository's licence before reuse.

The competition's evaluation kit is not included; get it from the organisers' repository.
