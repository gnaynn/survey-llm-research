"""
Phase 1용 arm 구성 로직. exp008 build_arms.py와 동일 로직이나,
r을 retrieval_pool 데이터로만 재계산(held-out 누출 방지).
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = _REPO + "/data/KGSS_결과파일"
EXP002_CSV = _REPO + "/01_exploration/experiments/exp002_embedding_calibration/output_rs_pairs.csv"

with open(f"{BASE}/(2)kgss_variable_dictionary.json", encoding="utf-8") as f:
    VARDICT = json.load(f)
with open(f"{BASE}/(7)kgss_variable_metadata.json", encoding="utf-8") as f:
    META = json.load(f)

N_EXT = 10


def missing_set(var):
    out = set()
    for c in VARDICT.get(var, {}).get("missing_codes", []):
        try:
            out.add(float(c))
        except ValueError:
            pass
    return out


def decode(var, val):
    if val is None or pd.isna(val):
        return None
    if val in missing_set(var):
        return None
    return VARDICT.get(var, {}).get("values", {}).get(str(int(val)), str(val))


def recompute_eligible_pool(target, retrieval_df):
    """s는 exp002(텍스트 임베딩, 응답자 데이터 무관)에서 재사용, r은 retrieval_pool로만 재계산."""
    s_lookup = pd.read_csv(EXP002_CSV)
    s_lookup = s_lookup[s_lookup.target == target][["candidate", "s"]].set_index("candidate")["s"].to_dict()

    rows = []
    for cand in s_lookup:
        if cand not in retrieval_df.columns or target not in retrieval_df.columns:
            continue
        t_dec = retrieval_df[target].apply(lambda v: decode(target, v))
        c_dec_raw = retrieval_df[cand]
        c_missing = missing_set(cand)
        c_dec = c_dec_raw.where(~c_dec_raw.isin(c_missing))
        pair = pd.DataFrame({"t": t_dec, "c": c_dec}).dropna()
        # t는 라벨 문자열이라 순서 코드로 다시 변환해야 상관계수 계산 가능
        t_vals = retrieval_df.loc[pair.index, target]
        c_vals = pair["c"]
        if len(pair) < 30:
            continue
        try:
            r, _ = spearmanr(t_vals, c_vals)
        except Exception:
            continue
        if np.isnan(r):
            continue
        rows.append({"candidate": cand, "r": r, "s": s_lookup[cand]})

    df = pd.DataFrame(rows)
    df["exclude"] = (df["r"].abs() >= 0.30) | (df["s"] >= 0.50)
    return df[~df["exclude"]].copy()


def build_arms(target, target_part, eligible_df, respondent_row, rng):
    answered = []
    for _, r in eligible_df.iterrows():
        c = r["candidate"]
        if c not in respondent_row.index:
            continue
        dec = decode(c, respondent_row[c])
        if dec is not None:
            answered.append({"var": c, "label": VARDICT[c]["label"], "value": dec,
                              "r": r["r"], "s": r["s"], "part": META.get(c, {}).get("part")})
    ans_df = pd.DataFrame(answered)

    arms = {}
    if len(ans_df) == 0:
        return {"arm1_corr": [], "arm2_embed": [], "arm3_theory": [], "arm4_random": []}

    top_r = ans_df.reindex(ans_df["r"].abs().sort_values(ascending=False).index).head(N_EXT)
    arms["arm1_corr"] = top_r.to_dict("records")

    top_s = ans_df.sort_values("s", ascending=False).head(N_EXT)
    arms["arm2_embed"] = top_s.to_dict("records")

    same_part = ans_df[ans_df["part"] == target_part].copy()
    if len(same_part):
        same_part["n_years"] = same_part["var"].map(lambda v: META.get(v, {}).get("n_years", 0))
        same_part["missing_rate_pct"] = same_part["var"].map(lambda v: META.get(v, {}).get("missing_rate_pct", 100))
        same_part = same_part.sort_values(["n_years", "missing_rate_pct"], ascending=[False, True])
    arms["arm3_theory"] = same_part.head(N_EXT).to_dict("records")

    n_pick = min(N_EXT, len(ans_df))
    idx = rng.sample(list(ans_df.index), n_pick)
    arms["arm4_random"] = ans_df.loc[idx].to_dict("records")

    return arms
