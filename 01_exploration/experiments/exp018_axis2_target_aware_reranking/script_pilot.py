"""
exp018 파일럿 — axis2: "일반적으로 비슷한 사람"(raw) vs "이 문항 예측에 informative한 사람"(target-aware)
재랭킹 기준 비교. exp017의 candidate filtering(demo_knn)·집계(median) 로직 재사용.

leakage 금지: target-aware 방식도 질문+보기만 보여주고, 타겟/후보의 실제 target 문항 답변은 절대
보여주지 않음. LLM은 target 답을 생성하지 않고 순위+근거만 반환.
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
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
import lib_peer_deviation as L  # noqa: E402  (VARDICT/decode/missing_set/call_luxia 재사용)

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp018_axis2_target_aware_reranking"

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
YEAR = 2023
SPLIT_SEED = 42
SAMPLE_SEED = 7
N_PILOT = 15
N_CANDIDATES = 30  # 1단계 후보 (exp017의 20에서 스펙대로 30으로 확대)
N_FINAL = 8
TEMPERATURE = 0.3


def profile9_text(p9):
    return f"성별:{p9['SEX']}, 나이:{p9['AGE']}, 혼인:{p9['MARITAL']}, 학력:{p9['EDUC']}, 소득:{p9['INCOME']}, 취업:{p9['EMPLY']}, 지역:{p9['REGION']}, 도농:{p9['URBAN']}, 직업:{p9['OCC']}"


def profile9(row):
    p = {}
    for v in CORE9:
        dec = L.decode(v, row[v])
        p[v] = dec if dec else "정보 없음"
    return p


RERANK_SYSTEM = """당신은 사회조사 응답자들 중 "누구를 참고하면 좋을지" 판단하는 보조자입니다.
오직 후보 순위와 이유만 반환하며, 어떤 설문 문항에 대해서도 직접 답을 생성하지 않습니다."""


def build_raw_prompt(target_text, candidates_text_list):
    cand_lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(candidates_text_list))
    return f"""[타겟 인물]
{target_text}

[후보 {len(candidates_text_list)}명]
{cand_lines}

타겟 인물과 **전반적으로** 가장 비슷한 사람 {N_FINAL}명을 순서대로 골라주세요.
각 선택 옆에 이유를 한 줄로 적으세요.

형식(반드시 이 형식으로):
1위. 번호X — 이유
2위. 번호Y — 이유
...
{N_FINAL}위. 번호Z — 이유"""


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
overlap_log = []

for TARGET in TARGETS:
    print(f"\n===== {TARGET} =====", flush=True)
    TARGET_LABEL = L.VARDICT[TARGET]["label"]
    QUESTION = L.VARDICT[TARGET]["question"]
    options = [v for k, v in L.VARDICT[TARGET]["values"].items() if float(k) not in L.missing_set(TARGET)]
    idx_map = {o: i for i, o in enumerate(options)}

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
    print(f"retrieval_pool={len(retrieval_pool)}, heldout_pool={len(heldout_pool)}", flush=True)

    random.seed(SAMPLE_SEED)
    sample_idx = random.sample(list(heldout_pool.index), min(N_PILOT, len(heldout_pool)))
    sample = heldout_pool.loc[sample_idx]
    print(f"evaluating n={len(sample)}", flush=True)

    mode_val = retrieval_pool["target_decoded"].mode()[0]

    # demo_knn 준비 (exp017과 동일 로직, 재사용)
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

        target_text = profile9_text(p9)
        cand_texts = [profile9_text(profile9(r)) for _, r in cand_rows]

        # ---- arm_neighbor_raw ----
        raw_prompt = build_raw_prompt(target_text, cand_texts)
        raw_resp = L.call_luxia([{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": raw_prompt}], temperature=TEMPERATURE)
        picks_raw = parse_rerank_numbers(raw_resp, N_CANDIDATES, N_FINAL)
        neighbors_raw = [cand_rows[p - 1][1]["target_decoded"] for p in picks_raw]
        pred_raw = aggregate_median(neighbors_raw, options) if neighbors_raw else pred_demo
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_neighbor_raw",
                             "actual": actual, "predicted": pred_raw, "correct": pred_raw == actual,
                             "distance": abs(idx_map[pred_raw] - idx_map[actual]) if pred_raw in idx_map else len(options) - 1,
                             "n_picked": len(picks_raw), "picks": picks_raw, "raw_response": raw_resp})

        # ---- arm_neighbor_target ----
        target_prompt = build_target_aware_prompt(target_text, cand_texts, TARGET_LABEL, QUESTION, options)
        target_resp = L.call_luxia([{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": target_prompt}], temperature=TEMPERATURE)
        picks_target = parse_rerank_numbers(target_resp, N_CANDIDATES, N_FINAL)
        neighbors_target = [cand_rows[p - 1][1]["target_decoded"] for p in picks_target]
        pred_target = aggregate_median(neighbors_target, options) if neighbors_target else pred_demo
        # leakage 점검: 근거 텍스트에 실제 옵션 문구가 등장하는지(답을 흘렸는지)
        leak_hit = any(opt in (target_resp or "") for opt in options)
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_neighbor_target",
                             "actual": actual, "predicted": pred_target, "correct": pred_target == actual,
                             "distance": abs(idx_map[pred_target] - idx_map[actual]) if pred_target in idx_map else len(options) - 1,
                             "n_picked": len(picks_target), "picks": picks_target, "raw_response": target_resp,
                             "leak_hit": leak_hit})

        overlap = len(set(picks_raw) & set(picks_target))
        overlap_log.append({"target": TARGET, "respid": row.get("RESPID", idx),
                             "overlap": overlap, "picks_raw": picks_raw, "picks_target": picks_target})

        print(f"[{TARGET}][{i+1}/{len(sample)}] respid={row.get('RESPID', idx)} "
              f"raw_pick={len(picks_raw)} target_pick={len(picks_target)} overlap={overlap}/8", flush=True)
        pd.DataFrame(all_results).to_csv(f"{OUTDIR}/output_pilot.csv", index=False)
        pd.DataFrame(overlap_log).to_csv(f"{OUTDIR}/output_overlap_log.csv", index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_pilot.csv", index=False)
pd.DataFrame(overlap_log).to_csv(f"{OUTDIR}/output_overlap_log.csv", index=False)

print(f"\n\n=== exp018 파일럿 결과 (n={N_PILOT} x {len(TARGETS)} targets) ===", flush=True)
for target in TARGETS:
    print(f"\n--- {target} ---")
    for arm in ["arm_mode", "arm_demo_top8", "arm_neighbor_raw", "arm_neighbor_target"]:
        sub = res_df[(res_df.target == target) & (res_df.arm == arm)]
        if len(sub) == 0:
            continue
        print(f"{arm}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

print("\n--- pooled ---")
for arm in ["arm_mode", "arm_demo_top8", "arm_neighbor_raw", "arm_neighbor_target"]:
    sub = res_df[res_df.arm == arm]
    print(f"{arm}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

ov = pd.DataFrame(overlap_log)
print(f"\n--- raw vs target-aware top8 overlap (0~8) ---")
print(f"평균 overlap: {ov['overlap'].mean():.2f} / 8")
print(ov.groupby("target")["overlap"].mean())

leak_df = res_df[res_df.arm == "arm_neighbor_target"]
print(f"\n--- leakage 로그 (target-aware 근거에 실제 옵션 문구 등장 비율) ---")
print(f"{leak_df['leak_hit'].mean():.3f}")
