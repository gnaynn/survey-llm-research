"""
Latent Respondent State 파이프라인 실제 예시 1건 (PARTYLR, 타겟 1명 + 후보 5명).
RAG(과거 관련 문항) 단계는 exp015의 또래이탈 z-score 상위3을 재사용(가장 구체적으로 이미 구현된
"이 사람을 이해하는 데 관련된 문항 찾기" 방식).
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import random
import sys

import pandas as pd
import pyreadstat
from dotenv import load_dotenv

load_dotenv(_REPO + "/.env")
sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
CORE9 = L.CORE9
TARGET = "PARTYLR"
YEAR = 2023
SPLIT_SEED, SAMPLE_SEED = 42, 7
N_CAND = 5

ordinal_vars, _ = L.get_ordinal_candidates(TARGET)
eligible_df = L.recompute_eligible_pool(TARGET, None, candidate_vars=ordinal_vars) if False else None
cands_pool_cols = L.get_candidate_pool(TARGET)
options = [v for k, v in L.VARDICT[TARGET]["values"].items() if float(k) not in L.missing_set(TARGET)]

usecols = ["YEAR", "RESPID"] + CORE9 + [TARGET] + cands_pool_cols
df, meta = pyreadstat.read_sav(SAV, usecols=list(dict.fromkeys(usecols)))
df = df[df["YEAR"] == YEAR].copy()
df["target_decoded"] = df[TARGET].apply(lambda v: L.decode(TARGET, v))
pool = df[df["target_decoded"].notna()].copy()

rng_split = random.Random(SPLIT_SEED)
all_idx = list(pool.index)
rng_split.shuffle(all_idx)
n_retrieval = int(len(all_idx) * 0.7)
retrieval_pool = pool.loc[all_idx[:n_retrieval]].copy()
heldout_pool = pool.loc[all_idx[n_retrieval:]]

eligible_df = L.recompute_eligible_pool(TARGET, retrieval_pool, candidate_vars=ordinal_vars)
eligible_set = set(eligible_df["candidate"])

random.seed(SAMPLE_SEED)
target_idx = random.sample(list(heldout_pool.index), 1)[0]
target_row = heldout_pool.loc[target_idx]

# 후보 5명: 무작위(데모 예시라 demo_knn 대신 간단히)
cand_idx = random.Random(99).sample(list(retrieval_pool.index), N_CAND)
cand_rows = [(j, retrieval_pool.loc[j]) for j in cand_idx]


def profile9(row):
    return {v: (L.decode(v, row[v]) or "정보 없음") for v in CORE9}


SYNTH_SYSTEM = """당신은 사회조사 응답자의 답변 패턴을 보고 그 사람의 성향을 요약하는 분석가입니다.
주어진 사실(인구통계 + 여러 문항 응답)을 바탕으로, 이 사람의 응답 성향을 1~2문장으로 요약하세요.
사실에 없는 내용을 지어내지 말고, 주어진 답변들의 공통된 방향성만 요약하세요."""


def synthesize_state(row):
    p9 = profile9(row)
    zscores, _ = L.compute_person_zscore(row, retrieval_pool, ordinal_vars, peer_def="A1")
    feats = L.extract_deviation_features(row, zscores, eligible_set, n_top=3)
    lines = [f"- {k}: {v}" for k, v in p9.items()]
    for f in feats:
        lines.append(f"- {f['label']}: {f['respondent_value']}")
    facts = "\n".join(lines)
    resp = L.call_luxia([
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user", "content": f"다음 사실들로 이 사람의 응답 성향을 요약하세요:\n{facts}"},
    ])
    return resp, feats


print("=" * 70)
print("[1] 타겟 응답자 — Latent Respondent State 합성")
print("=" * 70)
target_state, target_feats = synthesize_state(target_row)
print("사용된 RAG 관련 문항(또래이탈 top3):", [f["label"] for f in target_feats])
print("\n>> Latent State:", target_state)
print("\n(실제 PARTYLR 정답 — LLM에게는 안 보여줌):", target_row["target_decoded"])

print("\n" + "=" * 70)
print("[2] 후보 5명 — 각자의 Latent Respondent State 합성")
print("=" * 70)
cand_states = []
for j, row in cand_rows:
    state, feats = synthesize_state(row)
    cand_states.append((j, row, state))
    print(f"\n후보 (respid={row.get('RESPID')}): {state}")
    print(f"   (실제 PARTYLR 답변 — 최종 집계에만 씀): {row['target_decoded']}")

print("\n" + "=" * 70)
print("[3] LLM 재랭킹 — 성향 서술만 비교해서 상위 3명 선택 (답 생성 안 함)")
print("=" * 70)
cand_lines = "\n".join(f"{i+1}. {s}" for i, (j, row, s) in enumerate(cand_states))
rerank_prompt = f"""[타겟 인물의 응답 성향]
{target_state}

[후보 5명의 응답 성향]
{cand_lines}

타겟 인물과 응답 성향이 가장 비슷한 순서로 3명을 고르세요. 각 선택에 이유를 한 줄로 적으세요.
정치 성향(진보/보수)을 직접 언급하거나 추측하지 말고, 성향 서술 자체의 유사성으로만 판단하세요.

형식:
1위. 번호X — 이유
2위. 번호Y — 이유
3위. 번호Z — 이유"""
rerank_resp = L.call_luxia([
    {"role": "system", "content": "당신은 응답 성향 서술을 비교해 비슷한 사람을 고르는 보조자입니다. 문항 답을 직접 생성하지 않습니다."},
    {"role": "user", "content": rerank_prompt},
])
print(rerank_resp)

import re
picks = []
for n in re.findall(r"번호\s*(\d+)", rerank_resp or ""):
    v = int(n)
    if 1 <= v <= N_CAND and v not in picks:
        picks.append(v)
picks = picks[:3]

print("\n" + "=" * 70)
print("[4] 최종 집계 — 선택된 후보들의 실제 PARTYLR 답변으로 median")
print("=" * 70)
selected_answers = [cand_states[p - 1][1]["target_decoded"] for p in picks]
print("선택된 후보 번호:", picks)
print("그들의 실제 답변:", selected_answers)

idx_map = {o: i for i, o in enumerate(options)}
idxs = sorted(idx_map[a] for a in selected_answers)
mid = idxs[len(idxs) // 2]
pred = options[mid]
print(f"\n>> 최종 예측: {pred}")
print(f">> 실제 정답: {target_row['target_decoded']}")
print(f">> 정답 여부: {pred == target_row['target_decoded']}")
