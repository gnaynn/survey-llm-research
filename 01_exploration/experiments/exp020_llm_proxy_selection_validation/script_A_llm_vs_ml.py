"""
exp020 Plan A/B — LLM vs 통계/ML 베이스라인. 완전 오프라인(신규 LLM 호출 없음).

두 계열의 비교를 한 번에 수행:
  (1) 이웃-기반(neighbor-based), 같은 demo_knn 30명 후보 안에서 Top-8만 다르게 선택:
      Random / Z-Euclidean(동일가중치) / Weighted-KNN(|r|가중) / LLM(기존 데이터 재사용)
  (2) 직접예측(direct-model) 계열, 후보 검색 자체를 생략 — retrieval_pool로 K=8→TARGET
      모델을 학습해 타겟 응답자 본인의 K=8 값에서 바로 예측:
      Logistic / Ridge(ordinal 근사) / RandomForest / GradientBoosting

목적: "LLM의 proxy selection이 단순 거리계산뿐 아니라 학습 기반 모델과 비교해도
경쟁력 있는가?"를 오프라인으로, API 실패 위험 없이 검증.
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import ast
import json
import random
import sys

import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import wilcoxon, spearmanr
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L  # noqa: E402

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp020_llm_proxy_selection_validation"
EXP018_DIR = _REPO + "/01_exploration/experiments/exp018_axis2_target_aware_reranking"
STAGE1_DIR = _REPO + "/01_exploration/experiments/exp019_question_aware_temporal_retrieval"
EXISTING_CSV = f"{EXP018_DIR}/output_pilot_v2_enriched.csv"

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
YEAR = 2023
SPLIT_SEED = 42
RANDOM_SEED = 55
N_CANDIDATES = 30
N_FINAL = 8
RNG_MODEL_SEED = 42

existing_df = pd.read_csv(EXISTING_CSV)
llm_rows_all = existing_df[existing_df.arm == "arm_neighbor_target_enriched"].copy()
mode_rows_all = existing_df[existing_df.arm == "arm_mode"].copy()

all_results = []

for TARGET in TARGETS:
    print(f"===== {TARGET} =====", flush=True)
    with open(f"{STAGE1_DIR}/stage1_relevance_{TARGET}.json", encoding="utf-8") as f:
        stage1 = json.load(f)
    K_VARS = [s["var"] for s in stage1]

    options = [v for k, v in L.VARDICT[TARGET]["values"].items() if float(k) not in L.missing_set(TARGET)]
    idx_map = {o: i for i, o in enumerate(options)}
    code_to_idx = {}  # 원 코드값(float) -> ordinal index, 학습용
    for k, v in L.VARDICT[TARGET]["values"].items():
        if v in idx_map:
            code_to_idx[float(k)] = idx_map[v]

    usecols = list(dict.fromkeys(["YEAR", "RESPID"] + CORE9 + [TARGET] + K_VARS))
    df, meta = pyreadstat.read_sav(SAV, usecols=usecols)
    df = df[df["YEAR"] == YEAR].copy()
    df["target_decoded"] = df[TARGET].apply(lambda v: L.decode(TARGET, v))
    pool = df[df["target_decoded"].notna()].copy()

    rng_split = random.Random(SPLIT_SEED)
    all_idx = list(pool.index)
    rng_split.shuffle(all_idx)
    n_retrieval = int(len(all_idx) * 0.7)
    retrieval_pool = pool.loc[all_idx[:n_retrieval]].copy()
    heldout_pool = pool.loc[all_idx[n_retrieval:]]

    # ---- demo_knn 후보 30명 (기존과 동일 로직, 결정론적 재현) ----
    for v in CORE9:
        retrieval_pool[v + "_dec"] = retrieval_pool[v].apply(lambda x: L.decode(v, x) or "정보없음")
    feat_cols = [v + "_dec" for v in CORE9]
    enc = OneHotEncoder(handle_unknown="ignore")
    X_retrieval = enc.fit_transform(retrieval_pool[feat_cols]).toarray()
    knn = NearestNeighbors(n_neighbors=N_CANDIDATES).fit(X_retrieval)

    # ---- K=8 z-score 통계 (retrieval_pool 기준) + |r| 가중치 ----
    ms_map = {v: L.missing_set(v) for v in K_VARS}
    stats = {}
    weights_r = {}
    for v in K_VARS:
        col = retrieval_pool[v].where(~retrieval_pool[v].isin(ms_map[v])).astype(float)
        mean, std = col.mean(), col.std()
        stats[v] = (mean, std)
        pair = pd.DataFrame({"t": retrieval_pool["target_decoded"], "c": col}).dropna()
        try:
            r, _ = spearmanr(pair["t"].map(lambda x: idx_map.get(x, np.nan)), pair["c"])
        except Exception:
            r = 0.0
        weights_r[v] = abs(r) if not np.isnan(r) else 0.0

    # ---- ML 모델 학습 (retrieval_pool, K=8 -> TARGET 원 코드) ----
    train_X = pd.DataFrame({v: retrieval_pool[v].where(~retrieval_pool[v].isin(ms_map[v])).astype(float) for v in K_VARS})
    train_y_code = retrieval_pool[TARGET].astype(float)
    med = train_X.median()
    train_X_imp = train_X.fillna(med)
    train_y_idx = train_y_code.map(lambda c: code_to_idx.get(c, np.nan))
    valid_mask = train_y_idx.notna()
    Xtr, ytr_idx, ytr_code = train_X_imp[valid_mask], train_y_idx[valid_mask].astype(int), train_y_code[valid_mask]

    logit = LogisticRegression(max_iter=2000).fit(Xtr, ytr_idx)
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr_idx)
    rf = RandomForestClassifier(n_estimators=300, random_state=RNG_MODEL_SEED, min_samples_leaf=3).fit(Xtr, ytr_idx)
    gbm = GradientBoostingClassifier(n_estimators=200, random_state=RNG_MODEL_SEED, max_depth=2).fit(Xtr, ytr_idx)

    def direct_predict(target_feat_row):
        x = pd.DataFrame([{v: target_feat_row.get(v) for v in K_VARS}])
        x = x.where(~pd.isna(x), np.nan)
        for v in K_VARS:
            val = target_feat_row.get(v)
            if val is not None and not pd.isna(val) and val in ms_map[v]:
                x[v] = np.nan
        x_imp = x.fillna(med)
        preds = {}
        preds["Logistic"] = options[int(np.clip(round(logit.predict(x_imp)[0]), 0, len(options) - 1))]
        preds["Ridge"] = options[int(np.clip(round(ridge.predict(x_imp)[0]), 0, len(options) - 1))]
        preds["RandomForest"] = options[int(np.clip(round(rf.predict(x_imp)[0]), 0, len(options) - 1))]
        preds["GradientBoosting"] = options[int(np.clip(round(gbm.predict(x_imp)[0]), 0, len(options) - 1))]
        return preds

    def aggregate_median(answers):
        idxs = sorted(idx_map[a] for a in answers if a in idx_map)
        if not idxs:
            return None
        mid = idxs[len(idxs) // 2] if len(idxs) % 2 == 1 else idxs[len(idxs) // 2 - 1]
        return options[mid]

    used_respids = llm_rows_all[llm_rows_all.target == TARGET]["respid"].tolist()
    llm_sub = llm_rows_all[llm_rows_all.target == TARGET].set_index("respid")

    rng_rand = random.Random(RANDOM_SEED)

    for i, respid in enumerate(used_respids):
        hrow_df = heldout_pool[heldout_pool["RESPID"] == respid]
        if len(hrow_df) == 0:
            continue
        hrow = hrow_df.iloc[0]
        actual = hrow["target_decoded"]

        target_feat = enc.transform([[L.decode(v, hrow[v]) or "정보없음" for v in CORE9]]).toarray()
        _, nn_idx = knn.kneighbors(target_feat)
        cand_rows = [retrieval_pool.iloc[j] for j in nn_idx[0]]

        # --- 이웃 기반: Random / Z-Euclidean / Weighted-KNN (같은 30명) ---
        rand8 = rng_rand.sample(range(N_CANDIDATES), N_FINAL)
        rand_answers = [cand_rows[r]["target_decoded"] for r in rand8]

        dists_eq, dists_w = [], []
        for c in cand_rows:
            sq_eq, sq_w, wsum = [], 0.0, 0.0
            for v in K_VARS:
                mean, std = stats[v]
                if pd.isna(mean) or pd.isna(std) or std == 0:
                    continue
                tv, cv = hrow.get(v), c.get(v)
                if tv is None or pd.isna(tv) or tv in ms_map[v]:
                    continue
                if cv is None or pd.isna(cv) or cv in ms_map[v]:
                    continue
                tz, cz = (float(tv) - mean) / std, (float(cv) - mean) / std
                d2 = (tz - cz) ** 2
                sq_eq.append(d2)
                w = weights_r[v]
                sq_w += w * d2
                wsum += w
            dists_eq.append(np.sqrt(np.mean(sq_eq)) if sq_eq else np.inf)
            dists_w.append(np.sqrt(sq_w / wsum) if wsum > 0 else np.inf)

        eq_top8 = np.argsort(dists_eq)[:N_FINAL]
        w_top8 = np.argsort(dists_w)[:N_FINAL]
        eq_answers = [cand_rows[j]["target_decoded"] for j in eq_top8]
        w_answers = [cand_rows[j]["target_decoded"] for j in w_top8]

        neighbor_preds = {
            "Random8": aggregate_median(rand_answers),
            "Z-Euclidean8": aggregate_median(eq_answers),
            "Weighted-KNN8": aggregate_median(w_answers),
        }

        # --- 직접예측 모델 (후보 검색 없음, target 본인 K=8 값으로 바로 예측) ---
        direct_preds = direct_predict(hrow)

        # --- LLM (기존 데이터 재사용) ---
        llm_pred = llm_sub.loc[respid, "predicted"] if respid in llm_sub.index else None

        all_preds = {**neighbor_preds, **direct_preds, "LLM": llm_pred}
        for method, pred in all_preds.items():
            all_results.append({
                "target": TARGET, "respid": respid, "method": method,
                "actual": actual, "predicted": pred, "correct": pred == actual,
                "distance": abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1,
            })

    print(f"{TARGET}: {len(used_respids)}명 평가 완료", flush=True)

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_A_comparison.csv", index=False)

METHODS = ["Random8", "Z-Euclidean8", "Weighted-KNN8", "Logistic", "Ridge", "RandomForest", "GradientBoosting", "LLM"]

print("\n\n=== Plan A/B 결과 ===")
for target in TARGETS:
    print(f"\n--- {target} ---")
    for m in METHODS:
        sub = res_df[(res_df.target == target) & (res_df.method == m)]
        if len(sub) == 0:
            continue
        print(f"{m:18s}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

print("\n--- pooled ---")
for m in METHODS:
    sub = res_df[res_df.method == m]
    print(f"{m:18s}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

# arm_mode 포함 pooled 유의성
mode_rows_all2 = mode_rows_all[mode_rows_all.apply(lambda r: r["respid"] in llm_rows_all[llm_rows_all.target == r["target"]]["respid"].values, axis=1)].copy()
mode_rows_all2["uid"] = mode_rows_all2["target"] + "_" + mode_rows_all2["respid"].astype(str)
res_df["uid"] = res_df["target"] + "_" + res_df["respid"].astype(str)

print("\n=== LLM vs 각 방법 (pooled, paired) ===")
pv_c = res_df.pivot(index="uid", columns="method", values="correct")
pv_d = res_df.pivot(index="uid", columns="method", values="distance")
mode_c = mode_rows_all2.set_index("uid")["correct"].reindex(pv_c.index)
mode_d = mode_rows_all2.set_index("uid")["distance"].reindex(pv_d.index)

for m in METHODS:
    if m == "LLM":
        continue
    t = pd.crosstab(pv_c["LLM"], pv_c[m])
    try:
        pm = mcnemar(t, exact=True).pvalue
    except Exception:
        pm = None
    stat, pw = wilcoxon(pv_d["LLM"], pv_d[m])
    print(f"LLM vs {m:18s}: McNemar p={pm}  Wilcoxon MAE p={pw:.4f}  diff(LLM유리:+)={(pv_d[m]-pv_d['LLM']).mean():+.3f}")

print("\n=== 참고: arm_mode 대비 각 방법 ===")
for m in METHODS:
    t = pd.crosstab(mode_c, pv_c[m])
    try:
        pm = mcnemar(t, exact=True).pvalue
    except Exception:
        pm = None
    stat, pw = wilcoxon(mode_d, pv_d[m])
    print(f"mode vs {m:18s}: McNemar p={pm}  Wilcoxon MAE p={pw:.4f}")
