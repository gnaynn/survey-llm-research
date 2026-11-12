"""
exp017 파일럿 — 축2 LLM 재랭킹 이웃검색. LLM은 "누가 이웃인지"만 판단하고,
최종 예측은 항상 실제 이웃 8명의 진짜 답변에서 median으로 집계(LLM이 답 직접 생성 안 함).

핵심 비교: 2단계 입력 프로필이 raw(9개 인구통계)냐 axis1(또래이탈 z-score narration)이냐에 따라
성능 차이가 나는지 — 이게 이번 파일럿에서 가장 중요한 질문.
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
OUTDIR = _REPO + "/01_exploration/experiments/exp017_axis2_llm_reranking"

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
YEAR = 2023
SPLIT_SEED = 42
SAMPLE_SEED = 7
N_PILOT = 15
N_CANDIDATES = 20  # 1단계 후보
N_FINAL = 8         # 2단계 최종 이웃
PEER_DEF = "A1"
N_DEV_FEATS = 3
TEMPERATURE = 0.3


def profile9_text(p9):
    return f"성별:{p9['SEX']}, 나이:{p9['AGE']}, 혼인:{p9['MARITAL']}, 학력:{p9['EDUC']}, 소득:{p9['INCOME']}, 취업:{p9['EMPLY']}, 지역:{p9['REGION']}, 도농:{p9['URBAN']}, 직업:{p9['OCC']}"


def profile9(row):
    p = {}
    for v in CORE9:
        dec = L.decode(v, row[v])
        p[v] = dec if dec else "정보 없음"
    return p


# ---------- axis1 narration 캐시 (retrieval_pool 멤버당 1회만 계산) ----------
_axis1_cache = {}


def get_axis1_narrative(idx, row, retrieval_pool, ordinal_vars, eligible_set):
    if idx in _axis1_cache:
        return _axis1_cache[idx]
    zscores, _ = L.compute_person_zscore(row, retrieval_pool, ordinal_vars, peer_def=PEER_DEF)
    # 2026-08-08 정책 변경: leakage 필터(|r|>=0.30 또는 s>=0.50) 다시 적용
    feats = L.extract_deviation_features(row, zscores, eligible_set, n_top=N_DEV_FEATS)
    narrative = L.narrate(feats, temperature=TEMPERATURE) if feats else "(또래 대비 뚜렷한 이탈 문항 없음)"
    _axis1_cache[idx] = narrative
    return narrative


RERANK_SYSTEM = """당신은 사회조사 응답자들 중 "누가 서로 비슷하게 답할지" 판단하는 보조자입니다.
절대로 타겟 문항 자체에 대한 답을 생성하지 마세요 — 오직 후보 순위와 이유만 반환합니다."""


def build_rerank_prompt(topic_label, target_text, candidates_text_list):
    cand_lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(candidates_text_list))
    return f"""[타겟 인물]
{target_text}

[후보 {len(candidates_text_list)}명]
{cand_lines}

타겟 인물이 "{topic_label}" 관련 질문에 어떻게 답할지 예측하려고 합니다.
후보 {len(candidates_text_list)}명 중, 타겟 인물과 가치관·경험·상황이 실제로 겹쳐서 이 주제에
비슷하게 답할 가능성이 높은 사람 {N_FINAL}명을 순서대로 골라주세요. 각 선택 옆에 이유를 한 줄로 적으세요.
단순히 인구통계(나이/성별/직업 등)만 같다고 고르지 말고, 실제 가치관·경험이 겹칠지 판단하세요.

**중요: 타겟 문항 자체에 대한 답을 절대 생성하지 마세요. 오직 후보 번호 순위와 이유만 반환하세요.**

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
leakage_mentions = []

for TARGET in TARGETS:
    print(f"\n===== {TARGET} =====", flush=True)
    TOPIC_LABEL = L.VARDICT[TARGET]["label"]
    options = [v for k, v in L.VARDICT[TARGET]["values"].items() if float(k) not in L.missing_set(TARGET)]
    idx_map = {o: i for i, o in enumerate(options)}

    ordinal_vars, _ = L.get_ordinal_candidates(TARGET)
    cands_pool_cols = L.get_candidate_pool(TARGET)

    usecols = ["YEAR", "RESPID"] + CORE9 + [TARGET] + cands_pool_cols
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

    # leakage 필터 적용(2026-08-08 정책 변경): |r|>=0.30 또는 s>=0.50인 후보는 axis1 narration에서 제외
    eligible_df = L.recompute_eligible_pool(TARGET, retrieval_pool, candidate_vars=ordinal_vars)
    eligible_set = set(eligible_df["candidate"])
    print(f"eligible(leakage-filtered) 후보: {len(eligible_set)} / {len(ordinal_vars)}", flush=True)

    # demo_knn 준비 (exp014와 동일 로직)
    for v in CORE9:
        retrieval_pool[v + "_dec"] = retrieval_pool[v].apply(lambda x: L.decode(v, x) or "정보없음")
    feat_cols = [v + "_dec" for v in CORE9]
    enc = OneHotEncoder(handle_unknown="ignore")
    X_retrieval = enc.fit_transform(retrieval_pool[feat_cols]).toarray()
    knn = NearestNeighbors(n_neighbors=N_CANDIDATES).fit(X_retrieval)

    for i, (idx, row) in enumerate(sample.iterrows()):
        p9 = profile9(row)
        actual = row["target_decoded"]

        target_feat = enc.transform([[L.decode(v, row[v]) or "정보없음" for v in CORE9]]).toarray()
        _, nn_idx = knn.kneighbors(target_feat)
        cand_rows = [(retrieval_pool.index[j], retrieval_pool.iloc[j]) for j in nn_idx[0]]

        # arm_mode
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_mode",
                             "actual": actual, "predicted": mode_val, "correct": mode_val == actual,
                             "distance": abs(idx_map.get(mode_val, len(options) - 1) - idx_map[actual])})

        # arm_demo_top8: LLM 없이 demo_knn 상위 8명 그대로 median
        top8_demo = [r["target_decoded"] for _, r in cand_rows[:N_FINAL]]
        pred_demo = aggregate_median(top8_demo, options)
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_demo_top8",
                             "actual": actual, "predicted": pred_demo, "correct": pred_demo == actual,
                             "distance": abs(idx_map[pred_demo] - idx_map[actual]) if pred_demo in idx_map else len(options) - 1})

        # ---- arm_neighbor_raw ----
        target_text_raw = profile9_text(p9)
        cand_texts_raw = [profile9_text(profile9(r)) for _, r in cand_rows]
        raw_prompt = build_rerank_prompt(TOPIC_LABEL, target_text_raw, cand_texts_raw)
        raw_resp = L.call_luxia([{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": raw_prompt}], temperature=TEMPERATURE)
        picks_raw = parse_rerank_numbers(raw_resp, N_CANDIDATES, N_FINAL)
        neighbors_raw = [cand_rows[p - 1][1]["target_decoded"] for p in picks_raw]
        pred_raw = aggregate_median(neighbors_raw, options) if neighbors_raw else pred_demo
        leak_hit_raw = any(opt in (raw_resp or "") for opt in options)
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_neighbor_raw",
                             "actual": actual, "predicted": pred_raw, "correct": pred_raw == actual,
                             "distance": abs(idx_map[pred_raw] - idx_map[actual]) if pred_raw in idx_map else len(options) - 1,
                             "n_picked": len(picks_raw), "raw_response": raw_resp})
        leakage_mentions.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_neighbor_raw", "leak_hit": leak_hit_raw})

        # ---- arm_neighbor_axis1 ----
        target_narr = get_axis1_narrative(idx, row, retrieval_pool, ordinal_vars, eligible_set)
        target_text_axis1 = f"{profile9_text(p9)}. {target_narr}"
        cand_texts_axis1 = []
        for cidx, r in cand_rows:
            narr = get_axis1_narrative(cidx, r, retrieval_pool, ordinal_vars, eligible_set)
            cand_texts_axis1.append(f"{profile9_text(profile9(r))}. {narr}")
        axis1_prompt = build_rerank_prompt(TOPIC_LABEL, target_text_axis1, cand_texts_axis1)
        axis1_resp = L.call_luxia([{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": axis1_prompt}], temperature=TEMPERATURE)
        picks_axis1 = parse_rerank_numbers(axis1_resp, N_CANDIDATES, N_FINAL)
        neighbors_axis1 = [cand_rows[p - 1][1]["target_decoded"] for p in picks_axis1]
        pred_axis1 = aggregate_median(neighbors_axis1, options) if neighbors_axis1 else pred_demo
        leak_hit_axis1 = any(opt in (axis1_resp or "") for opt in options)
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_neighbor_axis1",
                             "actual": actual, "predicted": pred_axis1, "correct": pred_axis1 == actual,
                             "distance": abs(idx_map[pred_axis1] - idx_map[actual]) if pred_axis1 in idx_map else len(options) - 1,
                             "n_picked": len(picks_axis1), "raw_response": axis1_resp})
        leakage_mentions.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_neighbor_axis1", "leak_hit": leak_hit_axis1})

        print(f"[{TARGET}][{i+1}/{len(sample)}] respid={row.get('RESPID', idx)} "
              f"raw_pick={len(picks_raw)} axis1_pick={len(picks_axis1)} axis1_cache={len(_axis1_cache)}", flush=True)
        pd.DataFrame(all_results).to_csv(f"{OUTDIR}/output_pilot_filtered.csv", index=False)
        pd.DataFrame(leakage_mentions).to_csv(f"{OUTDIR}/output_leakage_log_filtered.csv", index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_pilot_filtered.csv", index=False)
pd.DataFrame(leakage_mentions).to_csv(f"{OUTDIR}/output_leakage_log_filtered.csv", index=False)

print(f"\n\n=== exp017 파일럿 결과 (n={N_PILOT} x {len(TARGETS)} targets) ===", flush=True)
for target in TARGETS:
    print(f"\n--- {target} ---")
    for arm in ["arm_mode", "arm_demo_top8", "arm_neighbor_raw", "arm_neighbor_axis1"]:
        sub = res_df[(res_df.target == target) & (res_df.arm == arm)]
        if len(sub) == 0:
            continue
        print(f"{arm}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

print("\n--- pooled ---")
for arm in ["arm_mode", "arm_demo_top8", "arm_neighbor_raw", "arm_neighbor_axis1"]:
    sub = res_df[res_df.arm == arm]
    print(f"{arm}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

leak_df = pd.DataFrame(leakage_mentions)
print(f"\n--- leakage 로그 (2단계 근거에 타겟 옵션 문구 등장 비율) ---")
print(leak_df.groupby("arm")["leak_hit"].mean())
