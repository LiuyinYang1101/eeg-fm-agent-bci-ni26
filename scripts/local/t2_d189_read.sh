#!/usr/bin/env bash
# T2-D189 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the four
# ni26-t2p-d189-<kit|seen1>-<a|b>-<jobid> dirs:
# 1 link both seed halves into t2_d189/stl_runs; 2 route the per-user delta predictions (`t2p_fpcalib blend --r 1`: the r 1/2 is already
#   inside the weights) -> t2_d189/route_stl; 3 t2p_pkgshape on spec.json (c_stlu vs cS2 decides, bar d141; lora_alone / single_alone =
#   the member-alone views) and on spec_3seed.json (the confirm: c3_stlu vs c3_stl, mix >= +0.0025 with >= 8 / 12 up); 4 the kill check:
#   member-alone seen delta on seeds 0-5 (lora_alone - single_alone, seen1 rows i < 6) <= 0 -> KILL
#   mem-guard bash scripts/local/t2_d189_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
mkdir -p "$O/t2_d189/stl_runs"
for p in kit seen1; do for h in a b; do
  d=( "$S"/ni26-t2p-d189-$p-$h-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need exactly one fetched dir for $p $h: ${d[*]}"; exit 2; }
  for f in "${d[0]}"runs/*.npz; do ln -sfn "$f" "$O/t2_d189/stl_runs/$(basename "$f")"; done
done; done
echo "$(ls "$O/t2_d189/stl_runs" | wc -l) linked files (expect 48)"
for p in kit seen1; do
  rec=$([ $p = kit ] && echo fmE30b64le0.0001lh0.001wd0.05ll1ema0dp0mx0.4cgcptstlarge-d89b || echo fmE30b64le0.0001lh0.001wd0.05ll1ema0dp0mx0.4cgcptpt-d111-with)
  $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" --runs "$O/t2_d189/stl_runs" --protocols $p \
    --out "$O/t2_d189/route_stl" --model fm-st_large-mean --align sz --recipe "$rec" --r 1.0 --tag 1 --seeds 0,1,2,3,4,5,6,7,8,9,10,11
done
for x in "spec read" "spec_3seed read_3seed"; do set -- $x
  $PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d189/$1.json" --out "$O/t2_d189/$2.json" > "$O/t2_d189/$2.txt" 2>&1; echo "$2 rc $?"
done
$PY -I - "$O/t2_d189" <<'PYEOF' | tee "$O/t2_d189/verdict.txt"
import json, sys
D = sys.argv[1]
r = json.load(open(f"{D}/read.json"))
rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
ks = [rows[("seen1", i, "lora_alone")]["seen"] - rows[("seen1", i, "single_alone")]["seen"] for i in range(6)]
kill = sum(ks) / 6 <= 0
print(f"kill check (member alone, seen1 seen, seeds 0-5): mean delta {sum(ks) / 6:+.4f} ({sum(k > 0 for k in ks)} / 6 up) -> {'KILL' if kill else 'continue'}")
a = r["arms"]["c_stlu"]; bar = a["mix"]["d"] >= 0.005 and a["mix"]["up"] >= 9 and a["kit_test"]["d"] >= -0.005
print(f"deciding c_stlu vs cS2: mix {a['mix']['d']:+.4f} ({a['mix']['up']} / 12), kit test {a['kit_test']['d']:+.4f}, seen {a['seen']['d']:+.4f}, new {a['new']['d']:+.4f} -> bar d141 {'MET' if bar else 'not met'}")
c = json.load(open(f"{D}/read_3seed.json"))["arms"]["c3_stlu"]; conf = c["mix"]["d"] >= 0.0025 and c["mix"]["up"] >= 8
print(f"3-seed confirm c3_stlu vs c3_stl: mix {c['mix']['d']:+.4f} ({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f} -> {'MET' if conf else 'not met'}")
print("VERDICT:", "KILLED" if kill else ("ADOPT" if bar and conf else "NOT ADOPTED"))
PYEOF
