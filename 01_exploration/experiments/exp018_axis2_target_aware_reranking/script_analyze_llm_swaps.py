"""
exp018 진단 — LLM이 demo_knn top-8 대비 "누구를 빼고 누구를 넣었는지" 전수 분석.
새 LLM 호출 없음: output_pilot_v2_enriched.csv의 arm_neighbor_target_enriched가 이미
로깅한 picks(30명 후보 중 순번)를 가지고, 동일 demo_knn 30명 후보를 결정론적으로
재구성해 "demo_top8(랭크 1~8)"과 "LLM top8"을 비교.

핵심 질문: LLM이 새로 넣은 사람(added)의 실제 target 답변이, LLM이 뺀 사람
(dropped)의 실제 target 답변보다 진짜 타겟의 실제 답변에 더 가까운가? 그렇다면 LLM이
demographic similarity가 아니라 특정 survey response pattern으로 더 나은 참고인을 찾는다는
설명이 성립. 아니라면 그것도 중요한 결과(패턴 없음).
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import ast
import json
import random

import pandas as pd
import pyreadstat
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import wilcoxon
import sys

sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L  # noqa: E402

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp018_axis2_target_aware_reranking"
STAGE1_DIR = _REPO + "/01_exploration/experiments/exp019_question_aware_temporal_retrieval"
EXISTING_CSV = f"{OUTDIR}/output_pilot_v2_enriched.csv"

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
YEAR = 2023
SPLIT_SEED = 42
N_CANDIDATES = 30
N_FINAL = 8

existing_df = pd.read_csv(EXISTING_CSV)
target_rows_all = existing_df[existing_df.arm == "arm_neighbor_target_enriched"].copy()

case_records = []      # per-case summary
swap_records = []      # per-swapped-person detail

for TARGET in TARGETS:
    print(f"===== {TARGET} =====", flush=True)
    with open(f"{STAGE1_DIR}/stage1_relevance_{TARGET}.json", encoding="utf-8") as f:
        stage1 = json.load(f)
    K_VARS = [s["var"] for s in stage1]

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

    for v in CORE9:
        retrieval_pool[v + "_dec"] = retrieval_pool[v].apply(lambda x: L.decode(v, x) or "정보없음")
    feat_cols = [v + "_dec" for v in CORE9]
    enc = OneHotEncoder(handle_unknown="ignore")
    X_retrieval = enc.fit_transform(retrieval_pool[feat_cols]).toarray()
    knn = NearestNeighbors(n_neighbors=N_CANDIDATES).fit(X_retrieval)

    sub = target_rows_all[target_rows_all.target == TARGET]
    for _, row in sub.iterrows():
        respid = row["respid"]
        hrow = heldout_pool[heldout_pool["RESPID"] == respid]
        if len(hrow) == 0:
            continue
        hrow = hrow.iloc[0]
        actual = row["actual"]
        actual_idx = idx_map.get(actual)

        target_feat = enc.transform([[L.decode(v, hrow[v]) or "정보없음" for v in CORE9]]).toarray()
        _, nn_idx = knn.kneighbors(target_feat)
        cand_rows = [retrieval_pool.iloc[j] for j in nn_idx[0]]  # rank order 1..30

        picks = ast.literal_eval(row["picks"]) if isinstance(row["picks"], str) else row["picks"]
        llm_ranks = set(picks)
        demo_ranks = set(range(1, N_FINAL + 1))

        dropped_ranks = sorted(demo_ranks - llm_ranks)
        added_ranks = sorted(llm_ranks - demo_ranks)
        overlap_ranks = sorted(demo_ranks & llm_ranks)

        def cand_info(rank):
            c = cand_rows[rank - 1]
            ans = c["target_decoded"]
            dist_to_actual = abs(idx_map[ans] - actual_idx) if (ans in idx_map and actual_idx is not None) else None
            k_vals = {v: L.decode(v, c.get(v)) for v in K_VARS}
            return {"rank": rank, "answer": ans, "dist_to_actual": dist_to_actual, **{f"k_{v}": val for v, val in k_vals.items()}}

        dropped_info = [cand_info(r) for r in dropped_ranks]
        added_info = [cand_info(r) for r in added_ranks]

        dropped_mean_dist = pd.Series([d["dist_to_actual"] for d in dropped_info if d["dist_to_actual"] is not None]).mean() if dropped_info else None
        added_mean_dist = pd.Series([d["dist_to_actual"] for d in added_info if d["dist_to_actual"] is not None]).mean() if added_info else None
        dropped_mean_rank = pd.Series([d["rank"] for d in dropped_info]).mean() if dropped_info else None
        added_mean_rank = pd.Series([d["rank"] for d in added_info]).mean() if added_info else None

        target_k_vals = {v: L.decode(v, hrow.get(v)) for v in K_VARS}

        case_records.append({
            "target": TARGET, "respid": respid, "actual": actual,
            "llm_correct": row["correct"], "llm_distance": row["distance"],
            "n_overlap": len(overlap_ranks), "n_dropped": len(dropped_ranks), "n_added": len(added_ranks),
            "dropped_mean_dist_to_actual": dropped_mean_dist, "added_mean_dist_to_actual": added_mean_dist,
            "dropped_mean_rank": dropped_mean_rank, "added_mean_rank": added_mean_rank,
        })

        for d in dropped_info:
            swap_records.append({"target": TARGET, "respid": respid, "group": "dropped", "actual": actual, **d})
        for a in added_info:
            swap_records.append({"target": TARGET, "respid": respid, "group": "added", "actual": actual, **a})

case_df = pd.DataFrame(case_records)
swap_df = pd.DataFrame(swap_records)
case_df.to_csv(f"{OUTDIR}/output_swap_case_summary.csv", index=False)
swap_df.to_csv(f"{OUTDIR}/output_swap_detail.csv", index=False)

print(f"\n총 {len(case_df)}건, 평균 overlap={case_df['n_overlap'].mean():.2f}/8, "
      f"평균 dropped={case_df['n_dropped'].mean():.2f}, 평균 added={case_df['n_added'].mean():.2f}")

# 핵심 검정: dropped 그룹과 added 그룹 중 어느 쪽이 실제 target에 더 가까운가 (paired, case 단위)
paired = case_df.dropna(subset=["dropped_mean_dist_to_actual", "added_mean_dist_to_actual"])
print(f"\n=== 핵심 비교 (both dropped/added 존재하는 케이스, n={len(paired)}) ===")
print(f"dropped(KNN이 고르고 LLM이 버린 사람)의 실제답-타겟실제답 평균거리: {paired['dropped_mean_dist_to_actual'].mean():.3f}")
print(f"added(LLM이 새로 고른 사람)의 실제답-타겟실제답 평균거리:          {paired['added_mean_dist_to_actual'].mean():.3f}")
stat, p = wilcoxon(paired["dropped_mean_dist_to_actual"], paired["added_mean_dist_to_actual"])
print(f"Wilcoxon paired p={p:.4f} (dropped-added diff mean={(paired['dropped_mean_dist_to_actual']-paired['added_mean_dist_to_actual']).mean():+.3f}, 양수면 added가 더 가까움)")

print(f"\n평균 demo_knn 랭크: dropped={paired['dropped_mean_rank'].mean():.1f}/30(당연히 1~8), "
      f"added={paired['added_mean_rank'].mean():.1f}/30 (LLM이 얼마나 먼 순위까지 가서 데려왔는지)")

# 타겟별로도 분리
print("\n=== 타겟별 ===")
for t in TARGETS:
    sub = paired[paired.target == t]
    print(f"{t} (n={len(sub)}): dropped_dist={sub['dropped_mean_dist_to_actual'].mean():.3f}  "
          f"added_dist={sub['added_mean_dist_to_actual'].mean():.3f}")

# swap_df 기준: added/dropped 각 개인 단위로 다시 (n이 더 많음, unpaired지만 참고용)
print(f"\n=== 개인 단위(unpaired) 비교 ===")
sd = swap_df.dropna(subset=["dist_to_actual"])
print(sd.groupby("group")["dist_to_actual"].agg(["mean", "count"]))
