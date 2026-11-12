_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import spearmanr

BASE = _REPO + "/data/KGSS_결과파일"
SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"

with open(f"{BASE}/(2)kgss_variable_dictionary.json", encoding="utf-8") as f:
    VARDICT = json.load(f)
with open(f"{BASE}/(7)kgss_variable_metadata.json", encoding="utf-8") as f:
    META = json.load(f)

TARGETS = ["CANTRUST", "BEFAIR", "HELPFUL", "PARTYLR", "CURGOV", "SATPOL", "SATFIN"]
YEAR = 2023
RNG = np.random.default_rng(42)

target_parts = {t: META.get(t, {}).get("part") for t in TARGETS}
print("target parts:", target_parts)

same_part_pool = set()
for t, part in target_parts.items():
    for v, info in META.items():
        if info.get("part") == part and v != t and v in VARDICT:
            same_part_pool.add(v)

other_part_vars = [v for v, info in META.items()
                    if info.get("part") not in set(target_parts.values())
                    and v in VARDICT]
background_pool = set(RNG.choice(other_part_vars, size=min(300, len(other_part_vars)), replace=False))

candidate_pool = (same_part_pool | background_pool) - set(TARGETS)
print("same_part_pool:", len(same_part_pool), "background_pool:", len(background_pool),
      "total candidates:", len(candidate_pool))

usecols = ["YEAR"] + TARGETS + sorted(candidate_pool)
print("loading", len(usecols), "columns from .sav ...")
df, meta = pyreadstat.read_sav(SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()
print("rows in year", YEAR, ":", len(df))


def to_nan(series, var):
    missing = set()
    for code in VARDICT.get(var, {}).get("missing_codes", []):
        try:
            missing.add(float(code))
        except ValueError:
            pass
    return series.where(~series.isin(missing))


for t in TARGETS:
    df[t] = to_nan(df[t], t)
for c in candidate_pool:
    if c in df.columns:
        df[c] = to_nan(df[c], c)

rows = []
for t in TARGETS:
    for c in candidate_pool:
        if c not in df.columns:
            continue
        pair = df[[t, c]].dropna()
        if len(pair) < 30:
            continue
        try:
            r, p = spearmanr(pair[t], pair[c])
        except Exception:
            continue
        if np.isnan(r):
            continue
        same_part = META.get(c, {}).get("part") == target_parts[t]
        rows.append({
            "target": t, "candidate": c, "r": r, "abs_r": abs(r),
            "n": len(pair), "same_part": same_part,
            "c_type": META.get(c, {}).get("type"),
            "t_type": META.get(t, {}).get("type"),
        })

res = pd.DataFrame(rows)
res.to_csv(__import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/leakage_r_results.csv", index=False)
print("\ntotal pairs computed:", len(res))

print("\n=== |r| percentiles, same-part vs other-part ===")
for grp, sub in res.groupby("same_part"):
    print(f"same_part={grp}, n={len(sub)}")
    print(sub["abs_r"].describe(percentiles=[.5, .75, .9, .95, .99]))
    print()

print("=== |r| percentiles by target-type pairing ===")
res["pair_type"] = res["t_type"].fillna("?") + "->" + res["c_type"].fillna("?")
for grp, sub in res.groupby("pair_type"):
    if len(sub) < 5:
        continue
    print(f"{grp}: n={len(sub)}, median={sub['abs_r'].median():.3f}, p90={sub['abs_r'].quantile(.9):.3f}, p95={sub['abs_r'].quantile(.95):.3f}, max={sub['abs_r'].max():.3f}")

print("\n=== Top 8 highest |r| per target (sanity check for structural duplicates) ===")
for t in TARGETS:
    sub = res[res.target == t].sort_values("abs_r", ascending=False).head(8)
    print(f"\n-- {t} ({VARDICT[t]['label']}) --")
    for _, row in sub.iterrows():
        label = VARDICT.get(row["candidate"], {}).get("label", "?")
        print(f"  r={row['r']:+.3f} n={row['n']:.0f} same_part={row['same_part']}  {row['candidate']} | {label}")
