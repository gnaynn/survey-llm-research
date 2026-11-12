"""
axis1 arm0~4 프로필 구성 로직.
- arm0: 9개 핵심만
- arm1: eligible pool 중 이 응답자가 실제로 답한 문항 중 |r| 상위 10개
- arm2: eligible pool 중 이 응답자가 실제로 답한 문항 중 s 상위 10개
- arm3: 타겟과 같은 18파트, eligible, 이 응답자가 답한 것. N초과시 n_years desc -> missing_rate_pct asc 캡
- arm4: eligible pool 중 이 응답자가 답한 문항에서 무작위 10개
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import random

import pandas as pd

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


def get_eligible_pool(target):
    """exp001+exp002 결과 재사용, r>=0.30 or s>=0.50 -> 제외."""
    res = pd.read_csv(EXP002_CSV)
    sub = res[res.target == target].copy()
    sub["exclude"] = (sub["r"].abs() >= 0.30) | (sub["s"] >= 0.50)
    eligible = sub[~sub["exclude"]].copy()
    return eligible  # columns: candidate, r, s, same_part, ...


def build_arms(target, target_part, eligible_df, respondent_row, rng):
    """respondent_row: pandas Series (usecols에 eligible 후보 컬럼 다 포함되어 있어야 함)"""
    # 이 응답자가 실제로 답한 eligible 후보만 추림
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
        arms["arm1_corr"] = []
        arms["arm2_embed"] = []
        arms["arm3_theory"] = []
        arms["arm4_random"] = []
        return arms

    # arm1: |r| 상위 10
    top_r = ans_df.reindex(ans_df["r"].abs().sort_values(ascending=False).index).head(N_EXT)
    arms["arm1_corr"] = top_r.to_dict("records")

    # arm2: s 상위 10
    top_s = ans_df.sort_values("s", ascending=False).head(N_EXT)
    arms["arm2_embed"] = top_s.to_dict("records")

    # arm3: 같은 파트, n_years desc -> missing_rate asc 캡
    same_part = ans_df[ans_df["part"] == target_part].copy()
    if len(same_part):
        same_part["n_years"] = same_part["var"].map(lambda v: META.get(v, {}).get("n_years", 0))
        same_part["missing_rate_pct"] = same_part["var"].map(lambda v: META.get(v, {}).get("missing_rate_pct", 100))
        same_part = same_part.sort_values(["n_years", "missing_rate_pct"], ascending=[False, True])
    arms["arm3_theory"] = same_part.head(N_EXT).to_dict("records")

    # arm4: 무작위 10
    n_pick = min(N_EXT, len(ans_df))
    idx = rng.sample(list(ans_df.index), n_pick)
    arms["arm4_random"] = ans_df.loc[idx].to_dict("records")

    return arms
