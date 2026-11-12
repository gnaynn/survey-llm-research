"""
교란 요인 점검: script_A_llm_vs_ml.py에서 Ridge/Logistic이 LLM이 고른
K=8(stage1_relevance) 문항을 썼기 때문에, "LLM reranking vs 통계모델"이 아니라
"LLM이 고른 feature 위의 LLM reranking vs LLM이 고른 feature 위의 통계모델"이었을
가능성이 있음 — LLM의 기여가 reranking이 아니라 feature selection에 있었을 수 있음.

이 스크립트는 같은 비교를 Random-8 / Statistical-8 feature로 재실행(오프라인, 완전
재사용): output_4condition.csv에 이미 있는 LLM-reranked-on-random8/stat8 예측
(n=30/target)과, retrieval_pool로 Random-8/Stat-8 feature 위에서 학습한 Ridge/Logistic
직접예측을 비교. LLM의 우위가 feature selection이 아니라 reranking 자체에서 나오는지
분리 검증.
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import random
import sys

import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import wilcoxon, spearmanr
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L  # noqa: E402

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp020_llm_proxy_selection_validation"
EXP018_DIR = _REPO + "/01_exploration/experiments/exp018_axis2_target_aware_reranking"
FOURCOND_CSV = f"{EXP018_DIR}/output_4condition.csv"

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
YEAR = 2023
SPLIT_SEED = 42
RANDOM8_SEED = 123  # script_4condition_retrieval.py와 동일 — 같은 random8 재현

fourcond = pd.read_csv(FOURCOND_CSV)
all_results = []

for TARGET in TARGETS:
    print(f"===== {TARGET} =====", flush=True)
    ordinal_vars, _ = L.get_ordinal_candidates(TARGET)

    usecols_all = list(dict.fromkeys(["YEAR", "RESPID"] + CORE9 + [TARGET] + ordinal_vars))
    df, meta = pyreadstat.read_sav(SAV, usecols=usecols_all)
    df = df[df["YEAR"] == YEAR].copy()
    df["target_decoded"] = df[TARGET].apply(lambda v: L.decode(TARGET, v))
    pool = df[df["target_decoded"].notna()].copy()

    rng_split = random.Random(SPLIT_SEED)
    all_idx = list(pool.index)
    rng_split.shuffle(all_idx)
    n_retrieval = int(len(all_idx) * 0.7)
    retrieval_pool = pool.loc[all_idx[:n_retrieval]].copy()
    heldout_pool = pool.loc[all_idx[n_retrieval:]]

    # script_4condition_retrieval.py와 동일하게 eligible/stat8/random8 재현
    eligible = L.recompute_eligible_pool(TARGET, retrieval_pool, candidate_vars=ordinal_vars)
    eligible_vars = list(eligible["candidate"])
    stat8_vars = list(eligible.assign(absr=eligible["r"].abs()).sort_values("absr", ascending=False)["candidate"])[:8]
    rng_r8 = random.Random(RANDOM8_SEED)
    random8_vars = rng_r8.sample(eligible_vars, 8)
    print(f"stat8={stat8_vars}", flush=True)
    print(f"random8={random8_vars}", flush=True)

    options = [v for k, v in L.VARDICT[TARGET]["values"].items() if float(k) not in L.missing_set(TARGET)]
    idx_map = {o: i for i, o in enumerate(options)}
    code_to_idx = {}
    for k, v in L.VARDICT[TARGET]["values"].items():
        if v in idx_map:
            code_to_idx[float(k)] = idx_map[v]

    for cond_name, k_vars in [("random8", random8_vars), ("stat8", stat8_vars)]:
        ms_map = {v: L.missing_set(v) for v in k_vars}
        train_X = pd.DataFrame({v: retrieval_pool[v].where(~retrieval_pool[v].isin(ms_map[v])).astype(float) for v in k_vars})
        train_y_code = retrieval_pool[TARGET].astype(float)
        med = train_X.median()
        train_X_imp = train_X.fillna(med)
        train_y_idx = train_y_code.map(lambda c: code_to_idx.get(c, np.nan))
        valid_mask = train_y_idx.notna()
        Xtr, ytr_idx = train_X_imp[valid_mask], train_y_idx[valid_mask].astype(int)

        logit = LogisticRegression(max_iter=2000).fit(Xtr, ytr_idx)
        ridge = Ridge(alpha=1.0).fit(Xtr, ytr_idx)

        llm_sub = fourcond[(fourcond.target == TARGET) & (fourcond.arm == f"arm_{cond_name}_retrieval")].copy()

        for _, row in llm_sub.iterrows():
            respid = row["respid"]
            hrow_df = heldout_pool[heldout_pool["RESPID"] == respid]
            if len(hrow_df) == 0:
                continue
            hrow = hrow_df.iloc[0]
            actual = row["actual"]

            x = pd.DataFrame([{v: hrow.get(v) for v in k_vars}]).astype(float)
            for v in k_vars:
                if pd.notna(x.at[0, v]) and x.at[0, v] in ms_map[v]:
                    x.at[0, v] = np.nan
            x_imp = x.fillna(med)

            pred_logit = options[int(np.clip(round(logit.predict(x_imp)[0]), 0, len(options) - 1))]
            pred_ridge = options[int(np.clip(round(ridge.predict(x_imp)[0]), 0, len(options) - 1))]

            for method, pred in [("Logistic", pred_logit), ("Ridge", pred_ridge), ("LLM", row["predicted"])]:
                all_results.append({
                    "target": TARGET, "respid": respid, "feature_set": cond_name, "method": method,
                    "actual": actual, "predicted": pred, "correct": pred == actual,
                    "distance": abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1,
                })

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_confound_check.csv", index=False)

print("\n\n=== Confound check 결과 (feature_set별) ===")
for fs in ["random8", "stat8"]:
    print(f"\n--- feature_set={fs} ---")
    for target in TARGETS:
        for m in ["Logistic", "Ridge", "LLM"]:
            sub = res_df[(res_df.feature_set == fs) & (res_df.target == target) & (res_df.method == m)]
            if len(sub) == 0:
                continue
            print(f"{target:8s} {m:10s}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")
    print("  pooled:")
    for m in ["Logistic", "Ridge", "LLM"]:
        sub = res_df[(res_df.feature_set == fs) & (res_df.method == m)]
        print(f"  {m:10s}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

print("\n=== 유의성 (LLM vs Ridge/Logistic, feature_set별, pooled) ===")
res_df["uid"] = res_df["target"] + "_" + res_df["respid"].astype(str)
for fs in ["random8", "stat8"]:
    sub = res_df[res_df.feature_set == fs]
    pv_c = sub.pivot(index="uid", columns="method", values="correct")
    pv_d = sub.pivot(index="uid", columns="method", values="distance")
    for m in ["Logistic", "Ridge"]:
        t = pd.crosstab(pv_c["LLM"], pv_c[m])
        try:
            pm = mcnemar(t, exact=True).pvalue
        except Exception:
            pm = None
        stat, pw = wilcoxon(pv_d["LLM"], pv_d[m])
        print(f"[{fs}] LLM vs {m}: McNemar p={pm}  Wilcoxon MAE p={pw:.4f}  diff={(pv_d[m]-pv_d['LLM']).mean():+.3f}")
