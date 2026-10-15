#!/usr/bin/env python3
"""Aggregate deviation tables from all *-schemes directories.
Reports failures/completeness, maskgripper recoveries, and full-track deviation.
Usage: aggregate_schemes.py [data_root]
"""
import argparse, csv, glob, os
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("data_root", nargs="?", type=Path,
                    default=Path(os.environ.get("MASK_DATA", "/home/ss/data/1000_onesb_labpicking")))
args = parser.parse_args()
base = args.data_root.resolve()
csvs = sorted(glob.glob(str(base / "*-schemes/_compare/deviation.csv")))
if not csvs:
    raise SystemExit(f"no deviation.csv files found under {base}")
f = lambda x: float(x) if x not in (None, "NA", "") else None

print(f"sessions: {len(csvs)}\n")
print(f"{'session':<16}{'eps':>4}{'mh fail':>9}{'mg fail':>9}{'mg saves':>10}{'mg-mh dev':>11}")
tot_eps = tot_mh = tot_mg = tot_saved = 0
both_dev = []
for c in csvs:
    sess = Path(c).parents[1].name.replace("-schemes", "")
    rows = list(csv.DictReader(open(c)))
    mh_fail = mg_fail = saved = 0
    sdev = []
    for r in rows:
        nin = f(r["nin"]); mh = f(r["maskhalf_n"]); mg = f(r["maskgripper_n"])
        if nin is not None and mh is not None and mh < nin: mh_fail += 1
        if nin is not None and mg is not None and mg < nin: mg_fail += 1
        if (nin is not None and mh is not None and mg is not None
                and mh < nin and mg >= nin): saved += 1
        if nin is not None and mh == nin and mg == nin:  # Comparable quality when both track fully
            d = f(r["rmse_mg_vs_mh_mm"])
            if d is not None: sdev.append(d)
    dev_med = np.median(sdev) if sdev else float("nan")
    print(f"{sess:<16}{len(rows):>4}{mh_fail:>7}{mg_fail:>7}{saved:>9}{dev_med:>9.1f}mm")
    tot_eps += len(rows); tot_mh += mh_fail; tot_mg += mg_fail; tot_saved += saved; both_dev += sdev

overall_med = np.median(both_dev) if both_dev else float("nan")
print(f"{'TOTAL':<16}{tot_eps:>4}{tot_mh:>7}{tot_mg:>7}{tot_saved:>9}{overall_med:>9.1f}mm")
print(f"\nSummary: {tot_eps} episodes | maskhalf failures {tot_mh} | maskgripper failures {tot_mg} | maskgripper recoveries {tot_saved}")
print(f"Quality (both tracked fully, n={len(both_dev)}): median maskgripper-vs-maskhalf deviation {overall_med:.1f}mm")
