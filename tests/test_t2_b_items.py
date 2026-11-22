#!/usr/bin/env python
"""T2 night plan (the D582 B crash, 18:29): the set-B pair reads configure the reader with the LAB's 'on' (W19C + D553 + the B item; no D557,
which the lab admits on set C only). Every B form must configure and give the intended member swaps:
  REF  NI26_LAB_D582B=ref               -> the five headed slots = D582's bref (r forms);
  D582 NI26_LAB_D582B=arm               -> s forms;      D583 NI26_LAB_D583B=arm -> w forms;      D588 NI26_LAB_D588B=arm -> f forms;
  D586 NI26_LAB_D582B=ref + D586B=arm   -> the r forms with ONLY slot 827 -> ysd@827.
+ set C (ON with D557) and set D (ON without D557) for D582 / D583 / D586 / D588. Each case runs the reader head (as t2_lab.py does) in a subprocess with only its own NI26_* env. Needs the set-B dumps (outputs/).
  python tests/test_t2_b_items.py  ->  "OK"
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = r'''
import json, os, sys
items, on = sys.argv[1], json.loads(sys.argv[2])
sys.argv = [sys.argv[0], "--draws", "828", "--items", items]
src = open("scripts/local/t2_d540a_stack_read.py").read()
g = {"__name__": "chk", "__file__": os.path.abspath("scripts/local/t2_d540a_stack_read.py")}
exec(compile(src.split("\nBASE = read([])\n")[0], "head", "exec"), g)
g["configure"](on)
print("SWAP " + json.dumps({str(k): v for k, v in sorted(g["FM_SWAP"].items())}))
'''
W19C = ["D539b", "D543", "D546b", "D553"]
CASES = {
    "REF": ({"NI26_LAB_D582B": "ref"}, "D582B", {"827": "rsd", "828": "rcb@835", "829": "rsd", "830": "rcb@836", "831": "rsd"}),
    "D582": ({"NI26_LAB_D582B": "arm"}, "D582B", {"827": "ssd", "828": "scb@835", "829": "ssd", "830": "scb@836", "831": "ssd"}),
    "D583": ({"NI26_LAB_D583B": "arm"}, "D583B", {"827": "wsd", "828": "wcb@835", "829": "wsd", "830": "wcb@836", "831": "wsd"}),
    "D588": ({"NI26_LAB_D588B": "arm"}, "D588B", {"827": "fsd", "828": "fcb@835", "829": "fsd", "830": "fcb@836", "831": "fsd"}),
    "D586": ({"NI26_LAB_D582B": "ref", "NI26_LAB_D586B": "arm"}, "D582B,D586B",
             {"827": "ysd@827", "828": "rcb@835", "829": "rsd", "830": "rcb@836", "831": "rsd"}),
    "D596": ({"NI26_LAB_D596B": "arm"}, "D596B", {"827": "Vsd", "828": "Vcb@835", "829": "Vsd", "830": "Vcb@836", "831": "Vsd"}),
    "D611": ({"NI26_LAB_D582B": "ref", "NI26_LAB_D611B": "arm"}, "D582B,D611B",   # T2-D611: the r forms with ONLY slot 831 -> its band twin
             {"827": "rsd", "828": "rcb@835", "829": "rsd", "830": "rcb@836", "831": "bpbsd@831"}),
    "D618": ({"NI26_LAB_D582B": "ref", "NI26_LAB_D618B": "arm"}, "D582B,D618B",   # T2-D618: the r forms with ONLY slot 831 -> its mirror twin
             {"827": "rsd", "828": "rcb@835", "829": "rsd", "830": "rcb@836", "831": "mrbsd@831"}),
}


# set C (the lab's ON carries D557: NI26_LAB_D557=1) and set D (no D557 in ON: set D's base IS W20+D557's members)
CD = {
    "C D583": (831, {"NI26_LAB_D557": "1", "NI26_LAB_D583": "1"}, "D557,D583", {"827": "usd@837", "828": "ucb@840", "831": "usd@839"}),
    "C D586": (831, {"NI26_LAB_D557": "1", "NI26_LAB_D586": "1"}, "D557,D586", {"827": "vsd@837", "828": "kcb@840", "831": "ksd@839"}),
    "D D582": (834, {"NI26_LAB_D582": "1"}, "D582", {"827": "tsd@843", "828": "tcb@846", "831": "tsd@845"}),
    "D D583": (834, {"NI26_LAB_D583": "1"}, "D583", {"827": "xsd@843", "828": "xcb@846", "831": "xsd@845"}),
    "D D586": (834, {"NI26_LAB_D586": "1"}, "D586", {"827": "zsd@843", "828": "pcb@846", "831": "psd@845"}),
    "D D588": (834, {"NI26_LAB_D588": "1"}, "D588", {"827": "isd@843", "828": "icb@846", "831": "isd@845"}),
    "C D596": (831, {"NI26_LAB_D557": "1", "NI26_LAB_D596": "1"}, "D557,D596", {"827": "Usd@837", "828": "Ucb@840", "831": "Usd@839"}),
    "D D596": (834, {"NI26_LAB_D596": "1"}, "D596", {"827": "Wsd@843", "828": "Wcb@846", "831": "Wsd@845"}),
    "C D611": (831, {"NI26_LAB_D557": "1", "NI26_LAB_D611": "1"}, "D557,D611", {"827": "ksd@837", "828": "kcb@840", "831": "bpcsd@839"}),
    "D D611": (834, {"NI26_LAB_D611": "1"}, "D611", {"827": "psd@843", "828": "pcb@846", "831": "bpdsd@845"}),
    "C D618": (831, {"NI26_LAB_D557": "1", "NI26_LAB_D618": "1"}, "D557,D618", {"827": "ksd@837", "828": "kcb@840", "831": "mrcsd@839"}),
    "D D618": (834, {"NI26_LAB_D618": "1"}, "D618", {"827": "psd@843", "828": "pcb@846", "831": "mrdsd@845"}),
}


def run(d0, env_, items, on):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NI26_")}
    env.update(env_, OMP_NUM_THREADS="2")
    case = CASE.replace('"--draws", "828"', f'"--draws", "{d0}"')
    return subprocess.run([sys.executable, "-c", case, items, json.dumps(on)], cwd=ROOT, env=env, capture_output=True, text=True)


def main():
    for name, (d0, env_, add, want) in CD.items():
        items = ",".join(W19C + (["D555"] if d0 == 831 and "D557" not in add else []) + [a for a in add.split(",") if a != "D557"] + (["D557"] if d0 == 834 or "D557" in add else []))
        on = W19C + add.split(",")                                      # C: D557 in ON (LAB_D557); D: not
        r = run(d0, env_, items, on)
        line = [l_ for l_ in r.stdout.splitlines() if l_.startswith("SWAP ")]
        assert r.returncode == 0 and line, (name, r.stderr[-600:])
        got = json.loads(line[0][5:])
        assert all(got.get(k) == v for k, v in want.items()), (name, got)
        print(f"{name}: configure(lab on) OK, slots {[got[k] for k in sorted(want)]}", flush=True)
    for name, (env_, items_b, want) in CASES.items():
        items = ",".join(W19C + ["D557"] + items_b.split(","))         # the lab's ITEMS string on 828 (D557 included)
        on = W19C + items_b.split(",")                                  # the lab's ON on 828 (no D557: LAB_D557 is set C only)
        r = run(828, env_, items, on)
        line = [l_ for l_ in r.stdout.splitlines() if l_.startswith("SWAP ")]
        assert r.returncode == 0 and line, (name, r.stderr[-600:])
        got = json.loads(line[0][5:])
        assert all(got.get(k) == v for k, v in want.items()), (name, got)
        print(f"{name}: configure(lab on) OK, slots {[got[k] for k in sorted(want)]}", flush=True)
    print("OK")


if __name__ == "__main__":
    main()
