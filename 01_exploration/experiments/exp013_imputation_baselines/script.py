_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import random

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

BASE = _REPO + "/data/KGSS_결과파일"
SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"

with open(f"{BASE}/(2)kgss_variable_dictionary.json", encoding="utf-8") as f:
    VARDICT = json.load(f)

CORE9 = ["SEX", "AGE", "MARITAL", "EDUC", "INCOME", "EMPLY", "REGION", "URBAN", "OCC"]
YEAR = 2023
N_HELDOUT = 100
SPLIT_SEED = 42
SAMPLE_SEED = 7

TARGETS = {
    "SATFIN": ['매우 만족', '약간 만족', '만족도 불만족도 아니다', '약간 불만족', '매우 불만족'],
    "HELPFUL": ["남을 도우려고 한다", "경우에 따라 다르다", "자신만을 위한다"],
    "PARTYLR": ['매우 진보적', '다소 진보적', '중도', '다소 보수적', '매우 보수적'],
}


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


results = []
for target, options in TARGETS.items():
    usecols = ["YEAR"] + CORE9 + [target]
    df, meta = pyreadstat.read_sav(SAV, usecols=usecols)
    df = df[df["YEAR"] == YEAR].copy()
    df["target_decoded"] = df[target].apply(lambda v: decode(target, v))
    pool = df[df["target_decoded"].notna()].copy()

    # exp009~011과 동일한 분할 재현
    rng_split = random.Random(SPLIT_SEED)
    all_idx = list(pool.index)
    rng_split.shuffle(all_idx)
    n_retrieval = int(len(all_idx) * 0.7)
    retrieval_pool = pool.loc[all_idx[:n_retrieval]]
    heldout_pool = pool.loc[all_idx[n_retrieval:]]

    random.seed(SAMPLE_SEED)
    sample_idx = random.sample(list(heldout_pool.index), min(N_HELDOUT, len(heldout_pool)))
    sample = heldout_pool.loc[sample_idx]

    idx_map = {o: i for i, o in enumerate(options)}

    # ---- arm5: 최빈값(mode) imputation (retrieval_pool 기준) ----
    mode_val = retrieval_pool["target_decoded"].mode()[0]
    pred5 = [mode_val] * len(sample)

    # ---- arm6: kNN (9개 핵심 프로필 원핫 인코딩, k=15) ----
    for v in CORE9:
        retrieval_pool[v + "_dec"] = retrieval_pool[v].apply(lambda x: decode(v, x) or "정보없음")
        sample[v + "_dec"] = sample[v].apply(lambda x: decode(v, x) or "정보없음")
    feat_cols = [v + "_dec" for v in CORE9]

    enc = OneHotEncoder(handle_unknown="ignore")
    X_train = enc.fit_transform(retrieval_pool[feat_cols])
    X_test = enc.transform(sample[feat_cols])
    y_train = retrieval_pool["target_decoded"]

    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)
    pred6 = knn.predict(X_test)

    actual = sample["target_decoded"].tolist()

    for arm_name, preds in [("arm5_median_mode", pred5), ("arm6_knn_regression", pred6)]:
        for a, p in zip(actual, preds):
            dist = abs(idx_map[p] - idx_map[a])
            results.append({
                "target": target, "arm": arm_name, "actual": a, "predicted": p,
                "correct": p == a, "distance": dist,
            })

res_df = pd.DataFrame(results)
res_df.to_csv(__import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/imputation_results.csv", index=False)

print("=== 타겟별 imputation baseline 결과 ===")
for target in TARGETS:
    for arm in ["arm5_median_mode", "arm6_knn_regression"]:
        sub = res_df[(res_df.target == target) & (res_df.arm == arm)]
        print(f"{target} / {arm}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}")
    print()

print("=== Pooled(3개 타겟, n=300) ===")
for arm in ["arm5_median_mode", "arm6_knn_regression"]:
    sub = res_df[res_df.arm == arm]
    print(f"{arm}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}")
