"""
exp019 파일럿 — Stage1(관련문항 선정)에서 나온 K개 문항으로 Question-specific Response Profile을
구성하고, 3가지 가중치(동일/LLM점수/실제|r|)로 거리를 계산해 이웃을 검색, 실제 답변으로 집계.
LLM은 1단계(문항당 1회)에서만 쓰이고, 그 뒤로는 전부 계산(LLM 미사용).
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import random
import sys

import numpy as np
import pandas as pd
import pyreadstat
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

load_dotenv(_REPO + "/.env")
sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L  # noqa: E402

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp019_question_aware_temporal_retrieval"

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
YEAR = 2023
SPLIT_SEED = 42
SAMPLE_SEED = 7
N_PILOT = 15
N_DEMO_CAND = 8  # demo_knn 기준선용
N_FINAL = 8       # qaware 최종 이웃 수
MIN_JOINT = 4     # 거리 계산에 필요한 최소 공통 응답 문항 수


def aggregate_median(answers, options):
    idx_map = {o: i for i, o in enumerate(options)}
    idxs = sorted(idx_map[a] for a in answers if a in idx_map)
    if not idxs:
        return None
    mid = idxs[len(idxs) // 2] if len(idxs) % 2 == 1 else idxs[len(idxs) // 2 - 1]
    return options[mid]


all_results = []
relevance_log = []

for TARGET in TARGETS:
    print(f"\n===== {TARGET} =====", flush=True)
    with open(f"{OUTDIR}/stage1_relevance_{TARGET}.json", encoding="utf-8") as f:
        stage1 = json.load(f)
    K_VARS = [s["var"] for s in stage1]
    LLM_SCORE = {s["var"]: s["score"] for s in stage1}
    print(f"K={len(K_VARS)}개 문항: {K_VARS}", flush=True)

    options = [v for k, v in L.VARDICT[TARGET]["values"].items() if float(k) not in L.missing_set(TARGET)]
    idx_map = {o: i for i, o in enumerate(options)}

    usecols = ["YEAR", "RESPID"] + CORE9 + [TARGET] + K_VARS
    usecols = list(dict.fromkeys(usecols))
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
    print(f"retrieval_pool={len(retrieval_pool)}, heldout_pool={len(heldout_pool)}", flush=True)

    random.seed(SAMPLE_SEED)
    sample_idx = random.sample(list(heldout_pool.index), min(N_PILOT, len(heldout_pool)))
    sample = heldout_pool.loc[sample_idx]
    print(f"evaluating n={len(sample)}", flush=True)

    mode_val = retrieval_pool["target_decoded"].mode()[0]

    # ---- K개 문항: retrieval_pool 기준 평균/표준편차 (population z-score) ----
    pop_stats = {}
    for v in K_VARS:
        ms = L.missing_set(v)
        col = retrieval_pool[v].where(~retrieval_pool[v].isin(ms))
        pop_stats[v] = (col.mean(), col.std())
        # 실제 |r| (statweight용)
        t_dec = retrieval_pool["target_decoded"]
        pair = pd.DataFrame({"t": retrieval_pool[TARGET], "c": col}).dropna()
        if len(pair) >= 30:
            r, _ = spearmanr(pair["t"], pair["c"])
        else:
            r = 0.0
        LLM_SCORE.setdefault(v, 0)
        relevance_log.append({"target": TARGET, "var": v, "llm_score": LLM_SCORE[v], "actual_r": r})

    STAT_WEIGHT = {row["var"]: abs(row["actual_r"]) for row in relevance_log if row["target"] == TARGET}

    def response_profile(row):
        prof = {}
        for v in K_VARS:
            ms = L.missing_set(v)
            val = row[v]
            if val is None or pd.isna(val) or val in ms:
                continue
            mean, std = pop_stats[v]
            if pd.isna(std) or std == 0:
                continue
            prof[v] = (float(val) - mean) / std
        return prof

    # 후보(retrieval_pool) 전원의 response profile 미리 계산
    cand_profiles = {j: response_profile(retrieval_pool.loc[j]) for j in retrieval_pool.index}

    def weighted_neighbors(target_prof, weights, n_final):
        scored = []
        for j, cprof in cand_profiles.items():
            common = set(target_prof) & set(cprof)
            if len(common) < MIN_JOINT:
                continue
            wsum = sum(weights.get(v, 0) for v in common) or 1e-9
            dist = sum(weights.get(v, 0) * (target_prof[v] - cprof[v]) ** 2 for v in common) / wsum
            scored.append((dist, j))
        scored.sort(key=lambda x: x[0])
        return [j for _, j in scored[:n_final]]

    # ---- demo_knn 기준선 준비 ----
    for v in CORE9:
        retrieval_pool[v + "_dec"] = retrieval_pool[v].apply(lambda x: L.decode(v, x) or "정보없음")
    feat_cols = [v + "_dec" for v in CORE9]
    enc = OneHotEncoder(handle_unknown="ignore")
    X_retrieval = enc.fit_transform(retrieval_pool[feat_cols]).toarray()
    knn = NearestNeighbors(n_neighbors=N_DEMO_CAND).fit(X_retrieval)

    equal_w = {v: 1.0 for v in K_VARS}
    llm_w = {v: LLM_SCORE.get(v, 1) for v in K_VARS}
    stat_w = {v: max(STAT_WEIGHT.get(v, 0.05), 0.05) for v in K_VARS}  # 0 나눗셈 방지 최소값

    for i, (idx, row) in enumerate(sample.iterrows()):
        actual = row["target_decoded"]

        # arm_mode
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_mode",
                             "actual": actual, "predicted": mode_val, "correct": mode_val == actual,
                             "distance": abs(idx_map.get(mode_val, len(options) - 1) - idx_map[actual])})

        # arm_demo_knn
        target_feat = enc.transform([[L.decode(v, row[v]) or "정보없음" for v in CORE9]]).toarray()
        _, nn_idx = knn.kneighbors(target_feat)
        demo_answers = [retrieval_pool.iloc[j]["target_decoded"] for j in nn_idx[0]]
        pred_demo = aggregate_median(demo_answers, options)
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_demo_knn",
                             "actual": actual, "predicted": pred_demo, "correct": pred_demo == actual,
                             "distance": abs(idx_map[pred_demo] - idx_map[actual]) if pred_demo in idx_map else len(options) - 1})

        target_prof = response_profile(row)

        for arm_name, weights in [("arm_qaware_equal", equal_w), ("arm_qaware_llmweight", llm_w), ("arm_qaware_statweight", stat_w)]:
            neighbors_idx = weighted_neighbors(target_prof, weights, N_FINAL)
            neighbor_answers = [retrieval_pool.loc[j]["target_decoded"] for j in neighbors_idx]
            pred = aggregate_median(neighbor_answers, options) if neighbor_answers else pred_demo
            all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": arm_name,
                                 "actual": actual, "predicted": pred, "correct": pred == actual,
                                 "distance": abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1,
                                 "n_neighbors_found": len(neighbors_idx), "n_target_items_observed": len(target_prof)})

        if (i + 1) % 5 == 0 or (i + 1) == len(sample):
            print(f"[{TARGET}][{i+1}/{len(sample)}] respid={row.get('RESPID', idx)} "
                  f"n_target_items={len(target_prof)}", flush=True)
            pd.DataFrame(all_results).to_csv(f"{OUTDIR}/output_pilot.csv", index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_pilot.csv", index=False)
pd.DataFrame(relevance_log).to_csv(f"{OUTDIR}/output_relevance_log.csv", index=False)

print(f"\n\n=== exp019 파일럿 결과 (n={N_PILOT} x {len(TARGETS)} targets) ===", flush=True)
ARMS = ["arm_mode", "arm_demo_knn", "arm_qaware_equal", "arm_qaware_llmweight", "arm_qaware_statweight"]
for target in TARGETS:
    print(f"\n--- {target} ---")
    for arm in ARMS:
        sub = res_df[(res_df.target == target) & (res_df.arm == arm)]
        if len(sub) == 0:
            continue
        print(f"{arm}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

print("\n--- pooled ---")
for arm in ARMS:
    sub = res_df[res_df.arm == arm]
    print(f"{arm}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

print("\n--- LLM 관련성 점수 vs 실제 |r| ---")
rel = pd.DataFrame(relevance_log)
print(rel.to_string(index=False))
print("\nPearson corr(llm_score, |actual_r|):", rel[["llm_score"]].assign(abs_r=rel["actual_r"].abs()).corr().iloc[0, 1])
