"""
exp018 v2c — v2-extended 진단 결과: LLM이 "informative한 이웃"을 고르라는 지시 때문에
뚜렷한 입장을 가진 이웃 쪽으로 쏠려서, 정작 타겟이 중도/무난(=pool 최빈값)인 경우
17~21%를 놓침(typical-slip). 반면 atypical(비주류) 응답자는 mode가 못 맞히는 걸
15~24% 복구함 — 이 이득이 typical 손실을 못 넘는 게 SATFIN에서 mode에 지는 이유.

이 실험은 프롬프트에 "타겟의 관련문항 응답이 뚜렷한 방향성 없이 중간적/혼재돼 있으면
이웃도 한쪽으로 쏠리지 않게 다양하게 고르라"는 지시를 추가해, 코드로 hybrid 규칙을 만드는
대신 LLM 스스로 typical-slip을 줄일 수 있는지 테스트. 후보 풀(demo_knn 30명)과 대상
respid는 v2-extended(output_pilot_v2_enriched.csv)와 완전히 동일하게 재구성해 직접 비교
가능하게 함 — arm_mode/arm_demo_top8/arm_neighbor_raw_enriched는 재사용(재호출 없음),
target-aware 프롬프트만 balance 지시를 추가한 새 arm(arm_neighbor_target_v2c_balance)
하나만 새로 호출.
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
import lib_peer_deviation as L  # noqa: E402

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp018_axis2_target_aware_reranking"
STAGE1_DIR = _REPO + "/01_exploration/experiments/exp019_question_aware_temporal_retrieval"
EXISTING_CSV = f"{OUTDIR}/output_pilot_v2_enriched.csv"
OUT_CSV = f"{OUTDIR}/output_pilot_v2c_balance.csv"

CORE9 = L.CORE9
TARGETS = ["SATFIN", "PARTYLR"]
YEAR = 2023
SPLIT_SEED = 42
N_CANDIDATES = 30
N_FINAL = 8
TEMPERATURE = 0.3
N_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # smoke default; pass 100 for full run


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


def build_target_aware_prompt_v2c(target_text, candidates_text_list, target_label, question, options):
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
추론하는 데 실제로 도움이 될 사람을 고르세요.

**중요 — 쏠림 방지**: 타겟의 프로필에 나타난 관련 문항 응답들이 어느 한쪽으로 뚜렷하지 않고
중간적이거나 혼재돼 있다면(예: 일부는 진보적 응답, 일부는 보수적 응답처럼 방향이 섞여 있거나,
애초에 강한 태도가 드러나지 않는다면), 이웃 후보도 한쪽 방향으로만 쏠리게 고르지 말고 다양한/
중간적인 성향의 사람을 섞어서 고르세요. 타겟의 응답이 실제로 뚜렷한 한쪽 방향을 보일 때만
그 방향에 맞는 이웃 위주로 고르세요. 근거 없이 극단적이거나 한쪽으로 쏠린 이웃만 고르는 것을
피하세요.

각 선택 옆에 이유를 한 줄로 적으세요.

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


existing_df = pd.read_csv(EXISTING_CSV)

all_results = []

for TARGET in TARGETS:
    print(f"\n===== {TARGET} (v2c balance-prompt) =====", flush=True)
    with open(f"{STAGE1_DIR}/stage1_relevance_{TARGET}.json", encoding="utf-8") as f:
        stage1 = json.load(f)
    K_VARS = [s["var"] for s in stage1]

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

    # v2-extended와 정확히 같은 respid 집합 재사용 (직접 비교 가능하게)
    used_respids = existing_df[(existing_df.target == TARGET) & (existing_df.arm == "arm_neighbor_target_enriched")]["respid"].tolist()
    used_respids = used_respids[:N_LIMIT]
    sample = heldout_pool[heldout_pool["RESPID"].isin(used_respids)]
    print(f"재사용 respid {len(used_respids)}개 중 매칭 {len(sample)}개 평가", flush=True)

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

        target_text = enriched_text(row, K_VARS)
        cand_texts = [enriched_text(r, K_VARS) for _, r in cand_rows]

        top8_demo = [r["target_decoded"] for _, r in cand_rows[:N_FINAL]]
        pred_demo = aggregate_median(top8_demo, options)

        prompt = build_target_aware_prompt_v2c(target_text, cand_texts, TARGET_LABEL, QUESTION, options)
        resp = L.call_luxia([{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": prompt}], temperature=TEMPERATURE)
        picks = parse_rerank_numbers(resp, N_CANDIDATES, N_FINAL)
        neighbors = [cand_rows[p - 1][1]["target_decoded"] for p in picks]
        pred = aggregate_median(neighbors, options) if neighbors else pred_demo
        leak_hit = any(opt in (resp or "") for opt in options)
        all_results.append({"target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_neighbor_target_v2c_balance",
                             "actual": actual, "predicted": pred, "correct": pred == actual,
                             "distance": abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1,
                             "n_picked": len(picks), "picks": picks, "raw_response": resp, "leak_hit": leak_hit})

        if (i + 1) % 5 == 0 or (i + 1) == len(sample):
            print(f"[{TARGET}][{i+1}/{len(sample)}] respid={row.get('RESPID', idx)} n_picked={len(picks)}", flush=True)
            pd.DataFrame(all_results).to_csv(OUT_CSV, index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(OUT_CSV, index=False)

print(f"\n\n=== v2c(balance-prompt) 결과 ===", flush=True)
for target in TARGETS:
    sub = res_df[res_df.target == target]
    if len(sub) == 0:
        continue
    print(f"{target}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")
print(f"\npooled: acc={res_df['correct'].mean():.3f}  MAE={res_df['distance'].mean():.3f}  n={len(res_df)}")
print(f"leakage 로그: {res_df['leak_hit'].mean():.3f}")

# v2-extended의 arm_neighbor_target_enriched와 uid 기준 직접 비교
res_df["uid"] = res_df["target"] + "_" + res_df["respid"].astype(str)
old = existing_df[existing_df.arm == "arm_neighbor_target_enriched"].copy()
old["uid"] = old["target"] + "_" + old["respid"].astype(str)
mode_df = existing_df[existing_df.arm == "arm_mode"].copy()
mode_df["uid"] = mode_df["target"] + "_" + mode_df["respid"].astype(str)

merged = res_df.merge(old[["uid", "correct", "distance"]], on="uid", suffixes=("_v2c", "_v2old"))
merged = merged.merge(mode_df[["uid", "predicted"]].rename(columns={"predicted": "mode_pred"}), on="uid")
is_typical_slip = (merged["actual"] == merged["mode_pred"]) & (~merged["correct_v2old"])

print(f"\n=== 기존 typical-slip 케이스 중 v2c가 회복한 비율 ===")
n_slip = is_typical_slip.sum()
n_fixed = (is_typical_slip & merged["correct_v2c"]).sum()
print(f"기존 typical-slip {n_slip}건 중 v2c로 맞춘 것: {n_fixed}건")

print(f"\n=== 전체 acc/MAE 비교 (같은 respid 집합) ===")
print(f"v2(old, target_enriched): acc={merged['correct_v2old'].mean():.3f} MAE={merged['distance_v2old'].mean():.3f}")
print(f"v2c(balance):            acc={merged['correct_v2c'].mean():.3f} MAE={merged['distance_v2c'].mean():.3f}")
