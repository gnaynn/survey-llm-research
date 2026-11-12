_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

os.environ.setdefault(
    "HF_HOME",
    _REPO + "/.cache/hf",
)

BASE = _REPO + "/data/KGSS_결과파일"
EXP001 = _REPO + "/01_exploration/experiments/exp001_leakage_calibration/output_r_pairs.csv"
OUT = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/leakage_rs_results.csv"

with open(f"{BASE}/(2)kgss_variable_dictionary.json", encoding="utf-8") as f:
    VARDICT = json.load(f)

res = pd.read_csv(EXP001)
print("loaded pairs:", len(res))

all_vars = sorted(set(res["target"]) | set(res["candidate"]))
labels = {v: VARDICT.get(v, {}).get("label", v) for v in all_vars}

model = SentenceTransformer("nlpai-lab/KURE-v1")
texts = [labels[v] for v in all_vars]
embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
emb_map = dict(zip(all_vars, embs))


def cos(a, b):
    return float(np.dot(a, b))


res["s"] = [cos(emb_map[t], emb_map[c]) for t, c in zip(res["target"], res["candidate"])]
res["abs_s"] = res["s"].abs()
res.to_csv(OUT, index=False)
print("saved:", OUT)

print("\n=== s distribution ===")
print(res["s"].describe(percentiles=[.5, .75, .9, .95, .99]))

print("\n=== known cases ===")
for t, c in [("PARTYLR", "LEFTRIGT"), ("CURGOV", "VOTE221"), ("CURGOV", "CONBLUHO")]:
    row = res[(res.target == t) & (res.candidate == c)]
    if len(row):
        r = row.iloc[0]
        print(f"{t}({labels[t]}) <-> {c}({labels[c]}): r={r.r:+.3f}  s={r.s:+.3f}")
    else:
        print(f"{t}<->{c}: not in pair set")

print("\n=== r vs s correlation (does high r predict high s in general?) ===")
print("pearson corr(|r|, s):", res["abs_r"].corr(res["s"]))

print("\n=== top 10 by r, showing s alongside (to see r/s agreement or divergence) ===")
top = res.sort_values("abs_r", ascending=False).head(15)
for _, row in top.iterrows():
    print(f"  r={row.r:+.3f} s={row.s:+.3f}  {row.target} <-> {row.candidate}  "
          f"| {labels[row.target]} <-> {labels[row.candidate]}")
