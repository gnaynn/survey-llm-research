"""
exp018 v3 — 대조실험: 후보군(demo_knn Top-30, exp018과 동일)과 프로필 내용(CORE9+관련문항8,
exp019 Stage1 재사용)을 완전히 고정한 채, **랭킹 방식만** 바꿔서 비교.
① arm_llm_rank_30: LLM이 30명 프로필을 보고 target-aware로 8명 선택 (exp018 v2와 동일 프롬프트)
② arm_distance_rank_30: 같은 K=8개 문항의 z-score 가중(동일가중치) 거리로 8명 선택 (exp019 로직,
   단 검색 범위를 800명대 전체가 아니라 이 30명 후보로 제한)

결과가 ①>②면 "후보군 제한 + LLM 종합판단"이 핵심, ①≈② 또는 ②가 낫다면 "후보군 제한(demo_knn)"
자체가 exp018 성공의 핵심이라는 뜻.

동일 SPLIT_SEED/SAMPLE_SEED/N_PILOT=50이라 exp018 v2와 같은 held-out 응답자로 respid 단위 비교 가능.
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
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

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
YEAR = 2023
SPLIT_SEED = 42
SAMPLE_SEED = 7
N_PILOT = 50
N_CANDIDATES = 30
N_FINAL = 8
MIN_JOINT = 4
TEMPERATURE = 0.3


def profile9(row):
    return {v: (L.decode(v, row[v]) or "정보 없음") for v in CORE9}


def profile9_text(p9):
    return f"성별:{p9['SEX']}, 나이:{p9['AGE']}, 혼인:{p9['MARITAL']}, 학력:{p9['EDUC']}, 소득:{p9['INCOME']}, 취업:{p9['EMPLY']}, 지역:{p9['REGION']}, 도농:{p9['URBAN']}, 직업:{p9['OCC']}"


def enriched_text(row, k_vars):
    base = profile9_text(profile9(row))
    extra = [f"{L.VARDICT[v]['label']}:{dec}" for v in k_vars if (dec := L.decode(v, row.get(v))) is not None]
    return base + (" | " + ", ".join(extra) if extra else "")


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
    print(f"K={len(K_VARS)}개: {K_VARS}", flush=True)

    TARGET_LABEL = L.VARDICT[TARGET]["label"]
    QUESTION = L.VARDICT[TARGET]["question"]
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

    # ---- population z-score 기준 (exp019와 동일: retrieval_pool 전체로 평균/표준편차) ----
    pop_stats = {}
    for v in K_VARS:
        ms = L.missing_set(v)
        col = retrieval_pool[v].where(~retrieval_pool[v].isin(ms))
        pop_stats[v] = (col.mean(), col.std())

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

    # ---- demo_knn Top-30 후보군 (exp018과 동일) ----
    for v in CORE9:
        retrieval_pool[v + "_dec"] = retrieval_pool[v].apply(lambda x: L.decode(v, x) or "정보없음")
    feat_cols = [v + "_dec" for v in CORE9]
    enc = OneHotEncoder(handle_unknown="ignore")
    X_retrieval = enc.fit_transform(retrieval_pool[feat_cols]).toarray()
    knn = NearestNeighbors(n_neighbors=N_CANDIDATES).fit(X_retrieval)

    for i, (idx, row) in enumerate(sample.iterrows()):
        actual = row["target_decoded"]

        target_feat = enc.transform([[L.decode(v, row[v]) or "정보없음" for v in CORE9]]).toarray()
        _, nn_idx = knn.kneighbors(target_feat)
        cand_rows = [(retrieval_pool.index[j], retrieval_pool.iloc[j]) for j in nn_idx[0]]

        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_mode",
                             "actual": actual, "predicted": mode_val, "correct": mode_val == actual,
                             "distance": abs(idx_map.get(mode_val, len(options) - 1) - idx_map[actual])})

        top8_demo = [r["target_decoded"] for _, r in cand_rows[:N_FINAL]]
        pred_demo = aggregate_median(top8_demo, options)
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_demo_top8",
                             "actual": actual, "predicted": pred_demo, "correct": pred_demo == actual,
                             "distance": abs(idx_map[pred_demo] - idx_map[actual]) if pred_demo in idx_map else len(options) - 1})

        target_text = enriched_text(row, K_VARS)
        cand_texts = [enriched_text(r, K_VARS) for _, r in cand_rows]

        # ---- ① arm_llm_rank_30 ----
        prompt = build_target_aware_prompt(target_text, cand_texts, TARGET_LABEL, QUESTION, options)
        resp = L.call_luxia([{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": prompt}], temperature=TEMPERATURE)
        picks = parse_rerank_numbers(resp, N_CANDIDATES, N_FINAL)
        neighbors = [cand_rows[p - 1][1]["target_decoded"] for p in picks]
        pred_llm = aggregate_median(neighbors, options) if neighbors else pred_demo
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_llm_rank_30",
                             "actual": actual, "predicted": pred_llm, "correct": pred_llm == actual,
                             "distance": abs(idx_map[pred_llm] - idx_map[actual]) if pred_llm in idx_map else len(options) - 1,
                             "n_picked": len(picks)})

        # ---- ② arm_distance_rank_30 (같은 30명 안에서 z-score 거리, 동일가중치) ----
        target_prof = response_profile(row)
        scored = []
        for j, r in cand_rows:
            cprof = response_profile(r)
            common = set(target_prof) & set(cprof)
            if len(common) < MIN_JOINT:
                continue
            dist = sum((target_prof[v] - cprof[v]) ** 2 for v in common) / len(common)
            scored.append((dist, j, r))
        scored.sort(key=lambda x: x[0])
        top_dist = scored[:N_FINAL]
        dist_answers = [r["target_decoded"] for _, _, r in top_dist]
        pred_dist = aggregate_median(dist_answers, options) if dist_answers else pred_demo
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_distance_rank_30",
                             "actual": actual, "predicted": pred_dist, "correct": pred_dist == actual,
                             "distance": abs(idx_map[pred_dist] - idx_map[actual]) if pred_dist in idx_map else len(options) - 1,
                             "n_neighbors_found": len(top_dist)})

        if (i + 1) % 5 == 0 or (i + 1) == len(sample):
            print(f"[{TARGET}][{i+1}/{len(sample)}] respid={row.get('RESPID', idx)} llm_pick={len(picks)}", flush=True)
            pd.DataFrame(all_results).to_csv(f"{OUTDIR}/output_pilot_v3_ablation.csv", index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_pilot_v3_ablation.csv", index=False)

ARMS = ["arm_mode", "arm_demo_top8", "arm_llm_rank_30", "arm_distance_rank_30"]
print(f"\n\n=== exp018 v3(대조실험) 결과 (n={N_PILOT} x {len(TARGETS)} targets) ===", flush=True)
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
