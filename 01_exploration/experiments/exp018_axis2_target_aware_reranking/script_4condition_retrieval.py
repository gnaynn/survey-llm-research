"""
exp018 v4 — "타겟 문항과 관련된 문항으로 검색하는 것이 인구통계 검색보다 유용한가?"
4조건 비교, axis2 LLM 재랭킹(target-aware, N_FINAL=8)은 전 조건 동일 고정:

  1. arm_demo_retrieval     : CORE9 인구통계 원핫 거리로 Top-30 (K-var 없음, 프로필도 인구통계만)
  2. arm_random8_retrieval  : leakage-eligible 척도형 풀에서 무작위 8개 z-score 거리로 Top-30
  3. arm_stat8_retrieval    : 같은 풀에서 |r| 최상위 8개 z-score 거리로 Top-30
  4. arm_llm8_retrieval     : LLM이 고른 8개(stage1_relevance_*.json) z-score 거리로 Top-30
                              (기존 output_pilot_v2_enriched.csv의 arm_neighbor_target_enriched 재사용, 재호출 없음)

respid는 기존 v2-extended와 동일 100개/타겟 재사용(직접 비교 가능하게). 조건1/2/3만 신규 LLM 호출.
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import ast
import json
import random
import re
import sys

import numpy as np
import pandas as pd
import pyreadstat
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

load_dotenv(_REPO + "/.env")

sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L  # noqa: E402

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp018_axis2_target_aware_reranking"
STAGE1_DIR = _REPO + "/01_exploration/experiments/exp019_question_aware_temporal_retrieval"
EXISTING_CSV = f"{OUTDIR}/output_pilot_v2_enriched.csv"
OUT_CSV = f"{OUTDIR}/output_4condition.csv"

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
YEAR = 2023
SPLIT_SEED = 42
RANDOM8_SEED = 123
N_CANDIDATES = 30
N_FINAL = 8
TEMPERATURE = 0.3
MIN_COMMON = 4
N_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # smoke default; pass 100 for full run
CONDITIONS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["demo", "random8", "stat8"]


def profile9(row):
    return {v: (L.decode(v, row[v]) or "정보 없음") for v in CORE9}


def profile9_text(p9):
    return f"성별:{p9['SEX']}, 나이:{p9['AGE']}, 혼인:{p9['MARITAL']}, 학력:{p9['EDUC']}, 소득:{p9['INCOME']}, 취업:{p9['EMPLY']}, 지역:{p9['REGION']}, 도농:{p9['URBAN']}, 직업:{p9['OCC']}"


def enriched_text(row, k_vars):
    p9 = profile9(row)
    base = profile9_text(p9)
    extra = []
    for v in k_vars:
        dec = L.decode(v, row.get(v))
        if dec is not None:
            extra.append(f"{L.VARDICT[v]['label']}:{dec}")
    if extra:
        return base + " | " + ", ".join(extra)
    return base


RERANK_SYSTEM = """당신은 사회조사 응답자들 중 "누구를 참고하면 좋을지" 판단하는 보조자입니다.
오직 후보 순위와 이유만 반환하며, 어떤 설문 문항에 대해서도 직접 답을 생성하지 않습니다."""


def build_target_aware_prompt(target_text, candidates_text_list, target_label, question, options):
    cand_lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(candidates_text_list))
    options_str = "\n".join(f"  - {o}" for o in options)
    return f"""[타겟 인물]
{target_text}

[후보 {len(candidates_text_list)}명]
{cand_lines}

[예측하려는 문항 — {target_label}]
{question}

[보기]
{options_str}

타겟 인물이 이 문항에 어떻게 답할지 **예측하는 데 가장 informative한** 사람 {N_FINAL}명을
순서대로 골라주세요. 단순히 전반적으로 비슷한 사람이 아니라, 이 특정 문항에 대한 답을
추론하는 데 실제로 도움이 될 사람을 고르세요. 각 선택 옆에 이유를 한 줄로 적으세요.

**중요: 이 문항에 대한 답을 절대 생성하지 마세요. 오직 후보 번호 순위와 이유만 반환하세요.**

형식(반드시 이 형식으로):
1위. 번호X — 이유
2위. 번호Y — 이유
...
{N_FINAL}위. 번호Z — 이유"""


def parse_rerank_numbers(response, n_candidates, n_final):
    if not response:
        return []
    nums = re.findall(r"번호\s*(\d+)", response)
    if len(nums) < n_final:
        nums = re.findall(r"(?:^|\n)\s*\d+위\D*?(\d+)", response)
    seen, out = [], []
    for n in nums:
        v = int(n)
        if 1 <= v <= n_candidates and v not in seen:
            seen.append(v)
            out.append(v)
        if len(out) == n_final:
            break
    return out


def aggregate_median(answers, options):
    idx_map = {o: i for i, o in enumerate(options)}
    idxs = sorted(idx_map[a] for a in answers if a in idx_map)
    if not idxs:
        return None
    mid = idxs[len(idxs) // 2] if len(idxs) % 2 == 1 else idxs[len(idxs) // 2 - 1]
    return options[mid]


def build_pool_zscores(retrieval_pool, k_vars):
    Z = pd.DataFrame(index=retrieval_pool.index)
    stats = {}
    for v in k_vars:
        ms = L.missing_set(v)
        col = retrieval_pool[v].where(~retrieval_pool[v].isin(ms)).astype(float)
        mean, std = col.mean(), col.std()
        if pd.isna(mean) or pd.isna(std) or std == 0:
            continue
        Z[v] = (col - mean) / std
        stats[v] = (mean, std)
    return Z, stats


def zscore_top_n(target_row, Z, stats, k_vars, n=N_CANDIDATES, min_common=MIN_COMMON):
    valid_vars, target_z = [], {}
    for v in k_vars:
        if v not in stats:
            continue
        mean, std = stats[v]
        ms = L.missing_set(v)
        tv = target_row.get(v)
        if tv is None or pd.isna(tv) or tv in ms:
            continue
        target_z[v] = (float(tv) - mean) / std
        valid_vars.append(v)
    if not valid_vars:
        return []
    sub = Z[valid_vars]
    diffs_sq = (sub - pd.Series(target_z)) ** 2
    common_count = sub.notna().sum(axis=1)
    mean_sq = diffs_sq.mean(axis=1, skipna=True)
    dist = np.sqrt(mean_sq)
    dist = dist[common_count >= min_common].sort_values()
    return list(dist.index[:n])


existing_df = pd.read_csv(EXISTING_CSV)
all_results = []

for TARGET in TARGETS:
    print(f"\n===== {TARGET} =====", flush=True)
    with open(f"{STAGE1_DIR}/stage1_relevance_{TARGET}.json", encoding="utf-8") as f:
        stage1 = json.load(f)
    LLM8_VARS = [s["var"] for s in stage1]

    TARGET_LABEL = L.VARDICT[TARGET]["label"]
    QUESTION = L.VARDICT[TARGET]["question"]
    options = [v for k, v in L.VARDICT[TARGET]["values"].items() if float(k) not in L.missing_set(TARGET)]
    idx_map = {o: i for i, o in enumerate(options)}

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

    eligible = L.recompute_eligible_pool(TARGET, retrieval_pool, candidate_vars=ordinal_vars)
    eligible_vars = list(eligible["candidate"])

    stat8_vars = list(eligible.assign(absr=eligible["r"].abs()).sort_values("absr", ascending=False)["candidate"])[:8]
    rng_r8 = random.Random(RANDOM8_SEED)
    random8_vars = rng_r8.sample(eligible_vars, 8)
    print(f"eligible={len(eligible_vars)}  stat8={stat8_vars}  random8={random8_vars}", flush=True)

    # demo_knn (조건1)
    for v in CORE9:
        retrieval_pool[v + "_dec"] = retrieval_pool[v].apply(lambda x: L.decode(v, x) or "정보없음")
    feat_cols = [v + "_dec" for v in CORE9]
    enc = OneHotEncoder(handle_unknown="ignore")
    X_retrieval = enc.fit_transform(retrieval_pool[feat_cols]).toarray()
    knn = NearestNeighbors(n_neighbors=N_CANDIDATES).fit(X_retrieval)

    zscore_pools = {}
    if "random8" in CONDITIONS:
        zscore_pools["random8"] = (build_pool_zscores(retrieval_pool, random8_vars), random8_vars)
    if "stat8" in CONDITIONS:
        zscore_pools["stat8"] = (build_pool_zscores(retrieval_pool, stat8_vars), stat8_vars)

    used_respids = existing_df[(existing_df.target == TARGET) & (existing_df.arm == "arm_neighbor_target_enriched")]["respid"].tolist()
    used_respids = used_respids[:N_LIMIT]
    sample = heldout_pool[heldout_pool["RESPID"].isin(used_respids)]
    print(f"재사용 respid {len(used_respids)}개 중 매칭 {len(sample)}개 평가", flush=True)

    for i, (idx, hrow) in enumerate(sample.iterrows()):
        actual = hrow["target_decoded"]

        for cond in CONDITIONS:
            if cond == "demo":
                k_vars = []
                target_feat = enc.transform([[L.decode(v, hrow[v]) or "정보없음" for v in CORE9]]).toarray()
                _, nn_idx = knn.kneighbors(target_feat)
                cand_idx_list = [retrieval_pool.index[j] for j in nn_idx[0]]
            else:
                (Z, stats), k_vars = zscore_pools[cond]
                cand_idx_list = zscore_top_n(hrow, Z, stats, k_vars)
                if len(cand_idx_list) < N_FINAL:
                    continue  # 후보 부족 시 스킵 (드묾)

            cand_rows = [retrieval_pool.loc[j] for j in cand_idx_list]
            n_cands = len(cand_rows)

            target_text = enriched_text(hrow, k_vars)
            cand_texts = [enriched_text(r, k_vars) for r in cand_rows]

            top8_fallback = [r["target_decoded"] for r in cand_rows[:N_FINAL]]
            pred_fallback = aggregate_median(top8_fallback, options)

            prompt = build_target_aware_prompt(target_text, cand_texts, TARGET_LABEL, QUESTION, options)
            resp = L.call_luxia([{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": prompt}], temperature=TEMPERATURE)
            picks = parse_rerank_numbers(resp, n_cands, N_FINAL)
            neighbors = [cand_rows[p - 1]["target_decoded"] for p in picks]
            pred = aggregate_median(neighbors, options) if neighbors else pred_fallback
            leak_hit = any(opt in (resp or "") for opt in options)

            all_results.append({
                "target": TARGET, "respid": hrow.get("RESPID", idx), "arm": f"arm_{cond}_retrieval",
                "actual": actual, "predicted": pred, "correct": pred == actual,
                "distance": abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1,
                "n_picked": len(picks), "picks": picks, "raw_response": resp, "leak_hit": leak_hit,
                "n_candidates": n_cands,
            })

        if (i + 1) % 5 == 0 or (i + 1) == len(sample):
            print(f"[{TARGET}][{i+1}/{len(sample)}] respid={hrow.get('RESPID', idx)}", flush=True)
            pd.DataFrame(all_results).to_csv(OUT_CSV, index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(OUT_CSV, index=False)

print(f"\n\n=== 4조건(신규 {len(CONDITIONS)}개) 결과 ===", flush=True)
for target in TARGETS:
    print(f"\n--- {target} ---")
    for cond in CONDITIONS:
        sub = res_df[(res_df.target == target) & (res_df.arm == f"arm_{cond}_retrieval")]
        if len(sub) == 0:
            continue
        print(f"arm_{cond}_retrieval: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

print("\n--- pooled ---")
for cond in CONDITIONS:
    sub = res_df[res_df.arm == f"arm_{cond}_retrieval"]
    if len(sub) == 0:
        continue
    print(f"arm_{cond}_retrieval: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}  leak={sub['leak_hit'].mean():.3f}")
