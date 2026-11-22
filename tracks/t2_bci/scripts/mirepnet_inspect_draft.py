# DRAFT (2026-10-30, the MIRepNet feasibility check for T2; not used by any build): reads MIRepNet's channel template / 2D positions from its
# source file by ast.literal_eval only (no code from the download is executed), builds the 45 x 41 inverse-distance padding map onto our
# montage, and strict-loads MIRepNet.pth (weights_only) into a pure-torch copy of the encoder. Usage: python -I <this> <mirepnet src .py> <MIRepNet.pth>
import ast, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
src, wpath = sys.argv[1], sys.argv[2]
tree = ast.parse(open(src).read())
lit = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("use_channels_names", "channel_positions"):
        lit[node.targets[0].id] = ast.literal_eval(node.value)
tmpl, pos = lit["use_channels_names"], lit["channel_positions"]
ours = "C2 CP2 P2 Pz CPz Cz C1 CP1 P1 P3 PO7 P5 TP7 CP5 CP3 C3 PO8 P6 P4 TP8 CP6 CP4 C6 C4 FC6 FC4 FC2 F6 F4 F2 AF4 AF3 F5 F3 F1 Fz FC5 FC3 FC1 FCz C5".upper().split()
print("template", len(tmpl), tmpl)
have = [c for c in tmpl if c in ours]; miss = [c for c in tmpl if c not in ours]; extra = [c for c in ours if c not in tmpl]
print("in ours", len(have), "| missing", len(miss), miss, "| our extras", extra, "| ours w/o pos", [c for c in ours if c not in pos])
P = np.array([pos[c] for c in ours]); W = np.zeros((len(tmpl), len(ours)))
for i, c in enumerate(tmpl):
    if c in ours: W[i, ours.index(c)] = 1.0
    else:
        d = np.linalg.norm(P - np.array(pos[c]), axis=1); w = 1 / (d + 1e-6); W[i] = w / w.sum()
        top = np.argsort(-W[i])[:3]
        print(f"  {c:4s} <- " + ", ".join(f"{ours[j]} {W[i,j]:.2f}" for j in top) + f" (top3 sum {W[i,top].sum():.2f}; nearest d {d.min():.2f})")
class PE(nn.Module):
    def __init__(s, e=256, C=45):
        super().__init__(); s.conv1 = nn.Conv2d(1, 64, (1, 25)); s.conv2 = nn.Conv2d(64, 128, (C, 1)); s.bn = nn.BatchNorm2d(128)
        s.pool = nn.AvgPool2d((1, 75), (1, 15)); s.projection = nn.Sequential(nn.Conv2d(128, e, 1)); s.chan_embed = nn.Embedding(45, e)
    def forward(s, x):
        x = s.pool(F.elu(s.bn(s.conv2(s.conv1(x.unsqueeze(1))))))
        return s.projection(x).flatten(2).transpose(1, 2)          # b e 1 w -> b w e
class MHA(nn.Module):
    def __init__(s, e, h):
        super().__init__(); s.e, s.h = e, h
        s.keys, s.queries, s.values, s.projection = (nn.Linear(e, e) for _ in range(4))
    def forward(s, x):
        B, N, E = x.shape; sp = lambda t: t.view(B, N, s.h, E // s.h).transpose(1, 2)
        q, k, v = sp(s.queries(x)), sp(s.keys(x)), sp(s.values(x))
        a = torch.softmax(q @ k.transpose(-1, -2) / s.e ** 0.5, -1)     # NB: scale = sqrt(emb), not sqrt(head dim)
        return s.projection((a @ v).transpose(1, 2).reshape(B, N, E))
class Res(nn.Module):
    def __init__(s, fn): super().__init__(); s.fn = fn
    def forward(s, x): return x + s.fn(x)
def block(e):
    return nn.Sequential(Res(nn.Sequential(nn.LayerNorm(e), MHA(e, 8), nn.Dropout(0.5))),
                         Res(nn.Sequential(nn.LayerNorm(e), nn.Sequential(nn.Linear(e, 4 * e), nn.GELU(), nn.Dropout(0.5), nn.Linear(4 * e, e)), nn.Dropout(0.5))))
class MIRep(nn.Module):
    def __init__(s, e=256, depth=6, n=3):
        super().__init__(); s.embedding = PE(e); s.transformer = nn.Sequential(*[block(e) for _ in range(depth)]); s.clshead = nn.Linear(e, n)
    def forward(s, x): return s.clshead(s.transformer(s.embedding(x)).mean(1))
sd = torch.load(wpath, map_location="cpu", weights_only=True)
enc = {k: v for k, v in sd.items() if not (k.startswith("decoder") or k == "mask_token")}
m = MIRep().eval(); r = m.load_state_dict(enc, strict=True); print("strict load:", r)
n_all = sum(p.numel() for p in m.parameters()); n_ce = m.embedding.chan_embed.weight.numel()
print(f"params encoder+head {n_all:,} (chan_embed unused {n_ce:,}) -> used {n_all-n_ce:,}; ckpt-only decoder+mask {sum(v.numel() for k,v in sd.items() if k.startswith('decoder') or k=='mask_token'):,}")
torch.set_num_threads(1); x = torch.randn(1, 45, 1000)
with torch.no_grad():
    y = m(x); print("out", tuple(y.shape), "tokens", m.embedding(x).shape[1])
    for _ in range(3): m(x)
    t = time.perf_counter(); [m(x) for _ in range(20)]; print(f"CPU 1-thread fwd per window {(time.perf_counter()-t)/20*1e3:.1f} ms")
