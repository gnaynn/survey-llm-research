"""
exp001 — 188문항 k-sweep에 "stat"(순수 통계, Pearson |r|) 방법을 추가.
기존 run_k_sweep.py와 동일한 188개 타겟·동일한 응답자 샘플(seed 42)을
재현해서 method="stat"만 새로 예측·기록한다 (ours/rag/random/demo는 이미 있으므로 재실행 안 함).

stat 랭킹: 타겟 문항의 원자료 응답값(SPSS 코드, 결측 제외)과 각 후보 문항의 원자료 값 사이
Pearson 상관계수(|r|)를 전체 응답자 풀에서 계산 → |r| 내림차순 top-K_MAX.
문항 텍스트/의미는 전혀 사용하지 않음 (순수 데이터 패턴).

CLI: K_VALUES(콤마구분, 기본 3,5,8,12,15)
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()  # repo root
import json
import random
import sys

import numpy as np
import pandas as pd
import pyreadstat

sys.path.insert(0, _REPO + "/02_llm_question_selection")
import utils as LV

OUTDIR = _REPO + "/02_llm_question_selection"
QUESTIONS_JSON = _REPO + "/data/questions/2025_questions.json"

YEAR = 2025
TEMPERATURE = 0.3
N_PILOT_TARGETS = 188
N_RESP = 5
EXCLUDED_PARTS = {
    "15_법_질서_범죄", "8_여가_문화", "10_인권_소수자", "14_환경_재난",
    "18_기타_분류_애매", "7_건강_질병",
}
K_VALUES = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [3, 5, 8, 12, 15]
K_MAX = max(K_VALUES)

RUN_TAG = f"ksweep_stat_{'-'.join(map(str, K_VALUES))}_{N_PILOT_TARGETS}q{N_RESP}r"
PILOT_CSV = f"{OUTDIR}/output_{RUN_TAG}.csv"
TOKENS_CSV = f"{OUTDIR}/output_tokens_{RUN_TAG}.csv"

with open(QUESTIONS_JSON, encoding="utf-8") as f:
    qdata = json.load(f)
pool_all = [q for q in qdata["questions"] if q["topic_part"] not in EXCLUDED_PARTS]
print(f"후보 풀: {len(pool_all)}개", flush=True)

random.seed(42)
pilot_targets = random.sample(pool_all, min(N_PILOT_TARGETS, len(pool_all)))
print(f"파일럿 타겟 {len(pilot_targets)}개 (기존 188q5r ours/rag/random 실험과 동일한 seed로 재현)", flush=True)

all_cand_vars = [q["var"] for q in pool_all]
usecols = list(dict.fromkeys(["YEAR", "RESPID"] + LV.CORE9 + all_cand_vars))
df, meta = pyreadstat.read_sav(LV.SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()

# ---- 상관계수 계산용: 모든 후보 변수의 결측 코드 처리된 numeric 프레임 (한 번만) ----
print("numeric 프레임 구성 중 (결측 코드 NaN 처리)...", flush=True)
num_cols = {}
for v in all_cand_vars:
    ms = LV.missing_set(v)
    num_cols[v] = df[v].where(~df[v].isin(ms)).astype(float)
num_df = pd.concat(num_cols, axis=1)
print("numeric 프레임 완료:", num_df.shape, flush=True)

SYS_PREDICT = """당신은 사회조사 응답자의 프로필을 보고 그 사람이 특정 문항에 어떻게 답했을지 예측하는
보조자입니다. 주어진 보기 중 정확히 하나를 골라 그 문구 그대로 답하세요."""


def profile_line(q, row):
    ans = LV.decode(q["var"], row.get(q["var"]))
    if ans is None:
        return None
    return f"Q: {q['question']} A: {ans}"


def build_predict_prompt(p9_text, context_lines, target_q, options):
    ctx = " | ".join(context_lines) if context_lines else "(관련 문항 정보 없음)"
    target_disp = f"Q: {target_q['question']}"
    options_str = "\n".join(f"  - {o}" for o in options)
    return f"""[타겟 인물]
{p9_text}
| {ctx}

[예측하려는 문항]
{target_disp}

[보기]
{options_str}

타겟 인물이 이 문항에 어떻게 답했을지, 위 보기 중 정확히 하나를 그대로 골라 답하세요.
다른 설명 없이 보기 문구만 출력하세요."""


import os
all_results = pd.read_csv(PILOT_CSV).to_dict("records") if os.path.exists(PILOT_CSV) else []
token_log = pd.read_csv(TOKENS_CSV).to_dict("records") if os.path.exists(TOKENS_CSV) else []
done_targets = {r["target"] for r in all_results}
print(f"기존 결과 이어붙임: pilot {len(all_results)}행, 완료된 타겟 {len(done_targets)}개", flush=True)

for ti, target_q in enumerate(pilot_targets):
    tvar = target_q["var"]
    if tvar in done_targets:
        continue
    print(f"\n===== [{ti+1}/{len(pilot_targets)}] 타겟: {tvar} ({target_q['label']}) =====", flush=True)
    options, idx_map = LV.target_options(tvar)
    candidates = [q for q in pool_all if q["var"] != tvar]

    dft = df.copy()
    dft["target_decoded"] = dft[tvar].apply(lambda v: LV.decode_stripped(tvar, v))
    dft = dft[dft["target_decoded"].notna()]
    for v in LV.CORE9:
        dft = dft[dft[v].notna()]
    sample = dft.sample(n=min(N_RESP, len(dft)), random_state=42)
    print(f"  응답자 {len(sample)}명", flush=True)

    # ---- stat: Pearson |r| 랭킹 ----
    cand_vars = [q["var"] for q in candidates]
    corrs = num_df[cand_vars].corrwith(num_df[tvar]).abs()
    corrs = corrs.fillna(0)
    order_vars = corrs.sort_values(ascending=False).index.tolist()[:K_MAX]
    var_to_q = {q["var"]: q for q in candidates}
    stat_ranked = [var_to_q[v] for v in order_vars]
    print(f"  stat 랭킹 top-{K_MAX}: {[(v, round(corrs[v], 3)) for v in order_vars[:5]]}...", flush=True)

    for _, row in sample.iterrows():
        actual = row["target_decoded"]
        p9_text = LV.profile9_text(LV.profile9(row))

        for k in K_VALUES:
            ctx_qs = stat_ranked[:k]
            ctx_lines = [ln for q in ctx_qs if (ln := profile_line(q, row)) is not None]
            prompt = build_predict_prompt(p9_text, ctx_lines, target_q, options)
            content, usage = LV.call_luxia_tracked(
                [{"role": "system", "content": SYS_PREDICT}, {"role": "user", "content": prompt}],
                temperature=TEMPERATURE)
            pred = LV.match_answer(content, options)
            correct = (pred == actual)
            distance = abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1
            all_results.append({
                "target": tvar, "method": "stat", "k": k, "respid": row.get("RESPID"),
                "actual": actual, "predicted": pred, "raw_response": content,
                "correct": correct, "distance": distance,
            })
            token_log.append({"target": tvar, "stage": "predict", "method": "stat", "k": k,
                               "respid": row.get("RESPID"), **(usage or {})})

    pd.DataFrame(all_results).to_csv(PILOT_CSV, index=False)
    pd.DataFrame(token_log).to_csv(TOKENS_CSV, index=False)
    print(f"  [{tvar}] 저장됨 (누적 {len(all_results)}행)", flush=True)

res_df = pd.DataFrame(all_results)
res_df.to_csv(PILOT_CSV, index=False)
pd.DataFrame(token_log).to_csv(TOKENS_CSV, index=False)

print("\n\n=== 결과 (stat method x k) ===")
for k in K_VALUES:
    sub = res_df[res_df.k == k]
    print(f"k={k:2d}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

tok_df = pd.DataFrame(token_log)
tok_valid = tok_df.dropna(subset=["prompt_tokens"]) if "prompt_tokens" in tok_df.columns else pd.DataFrame()
if len(tok_valid) > 0:
    print(f"\n총 호출 {len(tok_valid)}회, 총 tokens={tok_valid['total_tokens'].sum():.0f}")
