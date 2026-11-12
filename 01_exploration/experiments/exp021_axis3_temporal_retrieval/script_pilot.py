"""
exp021 — axis3: 검색 풀(candidate pool)의 연도 범위만 바꿔가며 비교.
파이프라인은 exp018 v2-extended와 완전히 동일(demo_knn 30 -> K=8 LLM-enrichment ->
target-aware 재랭킹 8명 -> median). 예측 대상(held-out)은 항상 2023년 응답자로 고정,
기존 output_pilot_v2_enriched.csv의 동일 100명/타겟을 그대로 재사용해 "동일연도" 조건은
신규 호출 없이 재사용하고, 확장 연도 조건만 신규 LLM 호출.

주의: KGSS는 반복횡단(repeated cross-section)이라 RESPID가 연도마다 재사용됨(2021년
RESPID=101과 2023년 RESPID=101은 다른 사람) — 후보 식별은 반드시 pandas index(연도
필터링 전 원본 행 인덱스, 전체 파일에서 유일)로 하고 RESPID로 하지 않는다.
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import ast
import json
import random
import re
import sys

import pandas as pd
import pyreadstat
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

load_dotenv(_REPO + "/.env")

sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L  # noqa: E402

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp021_axis3_temporal_retrieval"
EXP018_DIR = _REPO + "/01_exploration/experiments/exp018_axis2_target_aware_reranking"
STAGE1_DIR = _REPO + "/01_exploration/experiments/exp019_question_aware_temporal_retrieval"
EXISTING_CSV = f"{EXP018_DIR}/output_pilot_v2_enriched.csv"

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
TARGET_YEAR = 2023
SPLIT_SEED = 42
N_CANDIDATES = 30
N_FINAL = 8
TEMPERATURE = 0.3
N_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CONDITIONS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["wave1", "wave2", "all"]

YEAR_RANGES = {
    "same": [2023],
    "wave1": [2021, 2023, 2025],
    "wave2": [2018, 2021, 2023, 2025],
    "all": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2016, 2018, 2021, 2023, 2025],
}

existing_df = pd.read_csv(EXISTING_CSV)


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


all_results = []

for TARGET in TARGETS:
    print(f"\n===== {TARGET} =====", flush=True)
    with open(f"{STAGE1_DIR}/stage1_relevance_{TARGET}.json", encoding="utf-8") as f:
        stage1 = json.load(f)
    K_VARS = [s["var"] for s in stage1]

    TARGET_LABEL = L.VARDICT[TARGET]["label"]
    QUESTION = L.VARDICT[TARGET]["question"]
    options = [v for k, v in L.VARDICT[TARGET]["values"].items() if float(k) not in L.missing_set(TARGET)]
    idx_map = {o: i for i, o in enumerate(options)}

    usecols = list(dict.fromkeys(["YEAR", "RESPID"] + CORE9 + [TARGET] + K_VARS))
    df_all, meta = pyreadstat.read_sav(SAV, usecols=usecols)
    df_all["target_decoded"] = df_all[TARGET].apply(lambda v: L.decode(TARGET, v))

    # 2023년 타겟 풀 + 기존과 동일한 70/30 분할 (SPLIT_SEED=42) -- held-out 재현
    df_2023 = df_all[df_all["YEAR"] == TARGET_YEAR].copy()
    pool_2023 = df_2023[df_2023["target_decoded"].notna()].copy()
    rng_split = random.Random(SPLIT_SEED)
    all_idx_2023 = list(pool_2023.index)
    rng_split.shuffle(all_idx_2023)
    n_retrieval = int(len(all_idx_2023) * 0.7)
    retrieval_pool_2023 = pool_2023.loc[all_idx_2023[:n_retrieval]].copy()
    heldout_pool = pool_2023.loc[all_idx_2023[n_retrieval:]]

    used_respids = existing_df[(existing_df.target == TARGET) & (existing_df.arm == "arm_neighbor_target_enriched")]["respid"].tolist()
    used_respids = used_respids[:N_LIMIT]
    sample = heldout_pool[heldout_pool["RESPID"].isin(used_respids)]
    print(f"재사용 respid {len(used_respids)}개 중 매칭 {len(sample)}개 평가", flush=True)

    for cond in CONDITIONS:
        years = YEAR_RANGES[cond]
        print(f"\n--- 조건={cond}, years={years} ---", flush=True)

        parts = [retrieval_pool_2023]  # 2023은 기존과 동일하게 70% split만
        for y in years:
            if y == TARGET_YEAR:
                continue
            df_y = df_all[df_all["YEAR"] == y].copy()
            pool_y = df_y[df_y["target_decoded"].notna()].copy()
            parts.append(pool_y)
        candidate_pool = pd.concat(parts, axis=0)
        candidate_pool = candidate_pool[~candidate_pool.index.duplicated(keep="first")]
        print(f"candidate_pool 크기: {len(candidate_pool)}", flush=True)

        for v in CORE9:
            candidate_pool[v + "_dec"] = candidate_pool[v].apply(lambda x: L.decode(v, x) or "정보없음")
        feat_cols = [v + "_dec" for v in CORE9]
        enc = OneHotEncoder(handle_unknown="ignore")
        X_cand = enc.fit_transform(candidate_pool[feat_cols]).toarray()
        knn = NearestNeighbors(n_neighbors=min(N_CANDIDATES, len(candidate_pool))).fit(X_cand)

        for i, (idx, hrow) in enumerate(sample.iterrows()):
            actual = hrow["target_decoded"]

            target_feat = enc.transform([[L.decode(v, hrow[v]) or "정보없음" for v in CORE9]]).toarray()
            _, nn_idx = knn.kneighbors(target_feat)
            cand_rows = [candidate_pool.iloc[j] for j in nn_idx[0]]
            n_cands = len(cand_rows)

            target_text = enriched_text(hrow, K_VARS)
            cand_texts = [enriched_text(r, K_VARS) for r in cand_rows]

            top8_fallback = [r["target_decoded"] for r in cand_rows[:N_FINAL]]
            pred_fallback = aggregate_median(top8_fallback, options)

            prompt = build_target_aware_prompt(target_text, cand_texts, TARGET_LABEL, QUESTION, options)
            resp = L.call_luxia([{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": prompt}], temperature=TEMPERATURE)
            picks = parse_rerank_numbers(resp, n_cands, N_FINAL)
            neighbors = [cand_rows[p - 1]["target_decoded"] for p in picks]
            pred = aggregate_median(neighbors, options) if neighbors else pred_fallback
            neighbor_years = [int(cand_rows[p - 1]["YEAR"]) for p in picks] if picks else []
            leak_hit = any(opt in (resp or "") for opt in options)

            all_results.append({
                "target": TARGET, "respid": hrow.get("RESPID", idx), "condition": cond,
                "actual": actual, "predicted": pred, "correct": pred == actual,
                "distance": abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1,
                "n_picked": len(picks), "picks": picks, "raw_response": resp, "leak_hit": leak_hit,
                "n_candidates": n_cands, "neighbor_years": neighbor_years,
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(sample):
                print(f"[{TARGET}][{cond}][{i+1}/{len(sample)}] respid={hrow.get('RESPID', idx)}", flush=True)
                pd.DataFrame(all_results).to_csv(f"{OUTDIR}/output_pilot.csv", index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_pilot.csv", index=False)

print(f"\n\n=== axis3 조건별 결과 (신규 {len(CONDITIONS)}개 조건) ===")
for target in TARGETS:
    print(f"\n--- {target} ---")
    for cond in CONDITIONS:
        sub = res_df[(res_df.target == target) & (res_df.condition == cond)]
        if len(sub) == 0:
            continue
        print(f"{cond:8s}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}  leak={sub['leak_hit'].mean():.3f}")

print("\n--- pooled ---")
for cond in CONDITIONS:
    sub = res_df[res_df.condition == cond]
    if len(sub) == 0:
        continue
    print(f"{cond:8s}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")
