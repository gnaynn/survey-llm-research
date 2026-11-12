"""
exp021 v2 — v1(인구통계 KNN 기반 30명 검색 + 연도 확장)을 폐기하고, exp018 4조건 실험이
이미 유의하게 낫다고 확인한 K=8(LLM-8) 관련문항 z-score 거리 기반 30명 검색으로 재설계.
연도 범위만 바꿔가며 비교(동일연도/±1회차/±2회차/전체), 재랭킹은 동일(target-aware LLM).

동일연도 조건은 exp018의 output_4condition.csv(arm_llm8_retrieval, n=30/타겟)를 그대로
재사용 — 신규 호출 없음. wave1/wave2/all만 신규 호출, 같은 30명/타겟 respid로 맞춰서
직접 비교 가능하게 함.

z-score 표준화는 그 조건의 실제 candidate_pool(연도 범위만큼)에서 계산 — 연도가 늘어날
때 그 풀 자체의 분포 기준으로 정규화해야 일관됨(고정된 2023 기준으로 다른 연도 사람을
재는 것보다 원칙적으로 맞음).
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

load_dotenv(_REPO + "/.env")

sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L  # noqa: E402

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp021_axis3_temporal_retrieval"
EXP018_DIR = _REPO + "/01_exploration/experiments/exp018_axis2_target_aware_reranking"
STAGE1_DIR = _REPO + "/01_exploration/experiments/exp019_question_aware_temporal_retrieval"
FOURCOND_CSV = f"{EXP018_DIR}/output_4condition.csv"

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
TARGET_YEAR = 2023
SPLIT_SEED = 42
N_CANDIDATES = 30
N_FINAL = 8
TEMPERATURE = 0.3
MIN_COMMON = 4
N_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CONDITIONS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["same", "wave1", "wave2", "all"]

YEAR_RANGES = {
    "same": [2023],
    "wave1": [2021, 2023, 2025],
    "wave2": [2018, 2021, 2023, 2025],
    "all": [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2016, 2018, 2021, 2023, 2025],
}

# 참고용: exp018 4조건 실험의 respid 목록(재사용해 동일 인물로 평가) — 단, 그 실험의
# "arm_llm8_retrieval"은 실제로는 demo_knn 기반 데이터를 재라벨링한 것이었으므로
# (2026-08-10 확인) 예측값은 재사용하지 않고 respid 목록만 재사용함.
fourcond = pd.read_csv(FOURCOND_CSV)


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


def build_pool_zscores(pool, k_vars):
    Z = pd.DataFrame(index=pool.index)
    stats = {}
    for v in k_vars:
        ms = L.missing_set(v)
        col = pool[v].where(~pool[v].isin(ms)).astype(float)
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

    df_2023 = df_all[df_all["YEAR"] == TARGET_YEAR].copy()
    pool_2023 = df_2023[df_2023["target_decoded"].notna()].copy()
    rng_split = random.Random(SPLIT_SEED)
    all_idx_2023 = list(pool_2023.index)
    rng_split.shuffle(all_idx_2023)
    n_retrieval = int(len(all_idx_2023) * 0.7)
    retrieval_pool_2023 = pool_2023.loc[all_idx_2023[:n_retrieval]].copy()
    heldout_pool = pool_2023.loc[all_idx_2023[n_retrieval:]]

    # 4조건 실험과 동일 인물로 평가(재사용은 respid 목록만 — 예측값은 재사용 안 함, 위 주석 참고)
    ref_respids = fourcond[(fourcond.target == TARGET) & (fourcond.arm == "arm_demo_retrieval")]["respid"].tolist()
    used_respids = ref_respids[:N_LIMIT]
    sample = heldout_pool[heldout_pool["RESPID"].isin(used_respids)]
    print(f"참조 respid {len(used_respids)}개 중 매칭 {len(sample)}개 — 모든 조건 동일 인물로 평가", flush=True)

    for cond in CONDITIONS:
        years = YEAR_RANGES[cond]
        print(f"\n--- 조건={cond}, years={years} ---", flush=True)

        parts = [retrieval_pool_2023]
        for y in years:
            if y == TARGET_YEAR:
                continue
            df_y = df_all[df_all["YEAR"] == y].copy()
            pool_y = df_y[df_y["target_decoded"].notna()].copy()
            parts.append(pool_y)
        candidate_pool = pd.concat(parts, axis=0)
        candidate_pool = candidate_pool[~candidate_pool.index.duplicated(keep="first")]
        print(f"candidate_pool 크기: {len(candidate_pool)}", flush=True)

        Z, stats = build_pool_zscores(candidate_pool, K_VARS)

        for i, (idx, hrow) in enumerate(sample.iterrows()):
            actual = hrow["target_decoded"]

            cand_idx_list = zscore_top_n(hrow, Z, stats, K_VARS)
            if len(cand_idx_list) < N_FINAL:
                print(f"  [스킵] respid={hrow.get('RESPID')} 후보 부족({len(cand_idx_list)})", flush=True)
                continue
            cand_rows = [candidate_pool.loc[j] for j in cand_idx_list]
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
                pd.DataFrame(all_results).to_csv(f"{OUTDIR}/output_pilot_v2.csv", index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_pilot_v2.csv", index=False)

print(f"\n\n=== axis3 v2(K=8 기반 검색) 결과 ===")
for target in TARGETS:
    print(f"\n--- {target} ---")
    for cond in CONDITIONS:
        sub = res_df[(res_df.target == target) & (res_df.condition == cond)]
        if len(sub) == 0:
            continue
        print(f"{cond:8s}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

print("\n--- pooled ---")
for cond in CONDITIONS:
    sub = res_df[res_df.condition == cond]
    if len(sub) == 0:
        continue
    print(f"{cond:8s}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")
