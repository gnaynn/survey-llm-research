"""
exp018 v2d — 프롬프트로 typical-slip을 고치는 건 실패(v2c)했으니, LLM 판단이 아니라
LLM이 이미 뽑은 8명 이웃의 "답 합의도"를 외부 신호로 써서 hybrid 규칙을 테스트.
새 LLM 호출 없음 — output_pilot_v2_enriched.csv의 arm_neighbor_target_enriched가 이미
로깅한 picks(선택된 후보 순번)를 가지고, 동일한 demo_knn 후보풀을 결정론적으로 재구성해
그 8명의 실제 target 응답을 복원한 뒤, 합의도(최다득표 비율)로 LLM median을 쓸지 mode로
되돌릴지 결정.

주의: threshold를 같은 데이터에서 스윕해서 고르므로 이 스크립트의 "최적 threshold" 결과는
과적합 가능성이 있음 — 방향성 확인용이며, 확정하려면 새 held-out에서 재검증 필요.
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import ast
import random
import sys

import pandas as pd
import pyreadstat
from collections import Counter
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L  # noqa: E402

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp018_axis2_target_aware_reranking"
EXISTING_CSV = f"{OUTDIR}/output_pilot_v2_enriched.csv"

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
YEAR = 2023
SPLIT_SEED = 42
N_CANDIDATES = 30

existing_df = pd.read_csv(EXISTING_CSV)
target_rows_all = existing_df[existing_df.arm == "arm_neighbor_target_enriched"].copy()
mode_rows_all = existing_df[existing_df.arm == "arm_mode"].copy()

records = []

for TARGET in TARGETS:
    print(f"===== {TARGET} =====", flush=True)
    usecols = ["YEAR", "RESPID"] + CORE9 + [TARGET]
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

        target_feat = enc.transform([[L.decode(v, hrow[v]) or "정보없음" for v in CORE9]]).toarray()
        _, nn_idx = knn.kneighbors(target_feat)
        cand_rows = [retrieval_pool.iloc[j] for j in nn_idx[0]]

        picks = ast.literal_eval(row["picks"]) if isinstance(row["picks"], str) else row["picks"]
        neighbor_answers = [cand_rows[p - 1]["target_decoded"] for p in picks]
        cnt = Counter(neighbor_answers)
        top_answer, top_count = cnt.most_common(1)[0]
        agreement = top_count / len(neighbor_answers) if neighbor_answers else 0.0

        records.append({
            "target": TARGET, "respid": respid, "actual": row["actual"],
            "llm_pred": row["predicted"], "llm_correct": row["correct"], "llm_distance": row["distance"],
            "agreement": agreement, "n_neighbors": len(neighbor_answers),
        })

rec_df = pd.DataFrame(records)
rec_df["uid"] = rec_df["target"] + "_" + rec_df["respid"].astype(str)
mode_rows_all["uid"] = mode_rows_all["target"] + "_" + mode_rows_all["respid"].astype(str)
rec_df = rec_df.merge(mode_rows_all[["uid", "predicted", "correct", "distance"]].rename(
    columns={"predicted": "mode_pred", "correct": "mode_correct", "distance": "mode_distance"}), on="uid")

rec_df.to_csv(f"{OUTDIR}/output_hybrid_agreement.csv", index=False)

print(f"\n합의도(agreement) 분포:\n{rec_df['agreement'].describe()}")
print(f"\n합의도별 LLM 정확도 (전체 n={len(rec_df)}):")
for lo, hi in [(0, 0.375), (0.375, 0.5), (0.5, 0.625), (0.625, 0.75), (0.75, 1.01)]:
    m = (rec_df.agreement >= lo) & (rec_df.agreement < hi)
    if m.sum() == 0:
        continue
    print(f"  [{lo:.3f},{hi:.3f}) n={m.sum():3d}  LLM acc={rec_df[m]['llm_correct'].mean():.3f}  "
          f"mode acc={rec_df[m]['mode_correct'].mean():.3f}")

print("\n=== threshold 스윕: agreement >= T면 LLM 예측, 아니면 mode로 대체 ===")
best = None
for T in [0.375, 0.5, 0.625, 0.75, 0.875, 1.01]:
    use_llm = rec_df["agreement"] >= T
    hybrid_correct = rec_df["llm_correct"].where(use_llm, rec_df["mode_correct"])
    hybrid_distance = rec_df["llm_distance"].where(use_llm, rec_df["mode_distance"])
    acc = hybrid_correct.mean()
    mae = hybrid_distance.mean()
    print(f"T={T:.3f}: LLM 사용 {use_llm.mean():.1%}  acc={acc:.3f}  MAE={mae:.3f}")
    if best is None or acc > best[1]:
        best = (T, acc, mae)

print(f"\n최고 threshold(같은 데이터 기준, 과적합 주의): T={best[0]} acc={best[1]:.3f} MAE={best[2]:.3f}")

T = best[0]
use_llm = rec_df["agreement"] >= T
hybrid_correct = rec_df["llm_correct"].where(use_llm, rec_df["mode_correct"])
hybrid_distance = rec_df["llm_distance"].where(use_llm, rec_df["mode_distance"])

print(f"\n=== 최고 threshold(T={T})로 mode 대비 유의성 검정 ===")
t = pd.crosstab(rec_df["mode_correct"], hybrid_correct)
print(t)
print(f"McNemar p={mcnemar(t, exact=True).pvalue:.4f}")
stat, p = wilcoxon(rec_df["mode_distance"], hybrid_distance)
print(f"Wilcoxon MAE p={p:.4f} diff={(rec_df['mode_distance']-hybrid_distance).mean():+.3f}")

is_typical = rec_df["actual"] == rec_df["mode_pred"]
print(f"\ntypical(n={is_typical.sum()}): hybrid acc={hybrid_correct[is_typical].mean():.3f} "
      f"(LLM only={rec_df[is_typical]['llm_correct'].mean():.3f}, mode={rec_df[is_typical]['mode_correct'].mean():.3f})")
print(f"atypical(n={(~is_typical).sum()}): hybrid acc={hybrid_correct[~is_typical].mean():.3f} "
      f"(LLM only={rec_df[~is_typical]['llm_correct'].mean():.3f}, mode={rec_df[~is_typical]['mode_correct'].mean():.3f})")
