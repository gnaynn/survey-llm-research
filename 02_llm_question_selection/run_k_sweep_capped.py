"""
exp001 — 188문항 k-sweep에 "배터리(연속 시리즈 문항) 최대 2개 캡"을 rag/stat에도 적용해서
method="rag_capped", "stat_capped"로 재실행. ours는 이미 프롬프트에 이 규칙이 있었고
(원래 rag/stat엔 없어서 twin/배터리 문항으로 점수를 부풀릴 수 있었던 불공정 비교를 바로잡음).

배터리 판정: label의 콜론 앞부분에서 끝자리 숫자를 뗀 문자열(예: "정부지출의 증감7: 실업수당"
-> "정부지출의 증감") 을 1차 키로, 콜론이 없는 소수 항목은 변수명 4글자 이상 공통 접두사 +
동일 topic_part를 2차 키로 사용. 랭킹 만들 때 같은 배터리 키에서 이미 2개 뽑혔으면 스킵.

동일한 188개 타겟·동일한 응답자 샘플(seed 42)을 재현.

CLI: K_VALUES(콤마구분, 기본 3,5,8,12,15)
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()  # repo root
import json
import random
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import pyreadstat
from sentence_transformers import SentenceTransformer

sys.path.insert(0, _REPO + "/02_llm_question_selection")
import utils as LV

OUTDIR = _REPO + "/02_llm_question_selection"
QUESTIONS_JSON = _REPO + "/data/questions/2025_questions.json"

YEAR = 2025
TEMPERATURE = 0.3
N_PILOT_TARGETS = 188
N_RESP = 5
BATTERY_CAP = 2
EXCLUDED_PARTS = {
    "15_법_질서_범죄", "8_여가_문화", "10_인권_소수자", "14_환경_재난",
    "18_기타_분류_애매", "7_건강_질병",
}
K_VALUES = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [3, 5, 8, 12, 15]
K_MAX = max(K_VALUES)

RUN_TAG = f"ksweep_capped_{'-'.join(map(str, K_VALUES))}_{N_PILOT_TARGETS}q{N_RESP}r"
PILOT_CSV = f"{OUTDIR}/output_{RUN_TAG}.csv"
TOKENS_CSV = f"{OUTDIR}/output_tokens_{RUN_TAG}.csv"

with open(QUESTIONS_JSON, encoding="utf-8") as f:
    qdata = json.load(f)
pool_all = [q for q in qdata["questions"] if q["topic_part"] not in EXCLUDED_PARTS]
print(f"후보 풀: {len(pool_all)}개", flush=True)

random.seed(42)
pilot_targets = random.sample(pool_all, min(N_PILOT_TARGETS, len(pool_all)))
print(f"파일럿 타겟 {len(pilot_targets)}개 (기존 188q5r 실험과 동일 seed로 재현)", flush=True)

# ---------------- 배터리 키 계산 (전체 후보 풀 기준, 한 번만) ----------------
def strip_trailing_num(s):
    return re.sub(r"\d+\s*$", "", s).strip()

battery_key = {}
for q in pool_all:
    label = q["label"]
    if ":" in label:
        prefix = strip_trailing_num(label.split(":")[0].strip())
        if prefix:
            battery_key[q["var"]] = "label:" + prefix

remaining = [q for q in pool_all if q["var"] not in battery_key]
groups = defaultdict(list)
for q in remaining:
    m = re.match(r"^([A-Za-z]{4,})", q["var"])
    if m:
        groups[(m.group(1), q["topic_part"])].append(q["var"])
for (prefix, tp), vars_ in groups.items():
    if len(vars_) >= 2:
        for v in vars_:
            battery_key[v] = f"varpfx:{prefix}"

n_grouped = len(battery_key)
print(f"배터리로 묶인 문항: {n_grouped}/{len(pool_all)} (나머지는 단독 취급, 캡 영향 없음)", flush=True)


def cap_by_battery(ranked_vars, k_max, cap=BATTERY_CAP):
    """유사도/상관 내림차순 정렬된 var 리스트에서, 배터리별 최대 cap개까지만 순서 유지하며 채택."""
    out = []
    counts = defaultdict(int)
    for v in ranked_vars:
        bk = battery_key.get(v)  # None이면 단독 문항, 캡 없음
        if bk is not None and counts[bk] >= cap:
            continue
        out.append(v)
        if bk is not None:
            counts[bk] += 1
        if len(out) >= k_max:
            break
    return out


all_cand_vars = [q["var"] for q in pool_all]
usecols = list(dict.fromkeys(["YEAR", "RESPID"] + LV.CORE9 + all_cand_vars))
df, meta = pyreadstat.read_sav(LV.SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()

print("numeric 프레임 구성 중 (Pearson용, 결측 NaN 처리)...", flush=True)
num_cols = {}
for v in all_cand_vars:
    ms = LV.missing_set(v)
    num_cols[v] = df[v].where(~df[v].isin(ms)).astype(float)
num_df = pd.concat(num_cols, axis=1)
print("numeric 프레임 완료:", num_df.shape, flush=True)

print("임베딩 모델 로드 중(KURE-v1)...", flush=True)
EMB_MODEL = SentenceTransformer("nlpai-lab/KURE-v1")
ALL_QUESTIONS = [q["question"] for q in pool_all]
ALL_EMB = EMB_MODEL.encode(ALL_QUESTIONS, normalize_embeddings=True, show_progress_bar=False)
VAR_TO_EMB_IDX = {q["var"]: i for i, q in enumerate(pool_all)}
print("임베딩 완료", flush=True)

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
    var_to_q = {q["var"]: q for q in candidates}

    dft = df.copy()
    dft["target_decoded"] = dft[tvar].apply(lambda v: LV.decode_stripped(tvar, v))
    dft = dft[dft["target_decoded"].notna()]
    for v in LV.CORE9:
        dft = dft[dft[v].notna()]
    sample = dft.sample(n=min(N_RESP, len(dft)), random_state=42)
    print(f"  응답자 {len(sample)}명", flush=True)

    # ---- rag_capped: 임베딩 유사도 정렬 후 배터리 캡 적용 ----
    target_emb = EMB_MODEL.encode([target_q["question"]], normalize_embeddings=True, show_progress_bar=False)[0]
    cand_vars_all = [q["var"] for q in candidates]
    cand_idx = [VAR_TO_EMB_IDX[v] for v in cand_vars_all]
    cand_emb = ALL_EMB[cand_idx]
    sims = cand_emb @ target_emb
    order = np.argsort(-sims)
    rag_order_vars = [cand_vars_all[i] for i in order]
    rag_capped_vars = cap_by_battery(rag_order_vars, K_MAX)
    rag_ranked = [var_to_q[v] for v in rag_capped_vars]

    # ---- stat_capped: Pearson |r| 정렬 후 배터리 캡 적용 ----
    corrs = num_df[cand_vars_all].corrwith(num_df[tvar]).abs().fillna(0)
    stat_order_vars = corrs.sort_values(ascending=False).index.tolist()
    stat_capped_vars = cap_by_battery(stat_order_vars, K_MAX)
    stat_ranked = [var_to_q[v] for v in stat_capped_vars]

    print(f"  rag_capped top-{K_MAX}: {rag_capped_vars}", flush=True)
    print(f"  stat_capped top-{K_MAX}: {stat_capped_vars}", flush=True)

    for _, row in sample.iterrows():
        actual = row["target_decoded"]
        p9_text = LV.profile9_text(LV.profile9(row))

        for method, ranked in [("rag_capped", rag_ranked), ("stat_capped", stat_ranked)]:
            for k in K_VALUES:
                ctx_qs = ranked[:k]
                ctx_lines = [ln for q in ctx_qs if (ln := profile_line(q, row)) is not None]
                prompt = build_predict_prompt(p9_text, ctx_lines, target_q, options)
                content, usage = LV.call_luxia_tracked(
                    [{"role": "system", "content": SYS_PREDICT}, {"role": "user", "content": prompt}],
                    temperature=TEMPERATURE)
                pred = LV.match_answer(content, options)
                correct = (pred == actual)
                distance = abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1
                all_results.append({
                    "target": tvar, "method": method, "k": k, "respid": row.get("RESPID"),
                    "actual": actual, "predicted": pred, "raw_response": content,
                    "correct": correct, "distance": distance,
                })
                token_log.append({"target": tvar, "stage": "predict", "method": method, "k": k,
                                   "respid": row.get("RESPID"), **(usage or {})})

    pd.DataFrame(all_results).to_csv(PILOT_CSV, index=False)
    pd.DataFrame(token_log).to_csv(TOKENS_CSV, index=False)
    print(f"  [{tvar}] 저장됨 (누적 {len(all_results)}행)", flush=True)

res_df = pd.DataFrame(all_results)
res_df.to_csv(PILOT_CSV, index=False)
pd.DataFrame(token_log).to_csv(TOKENS_CSV, index=False)

print("\n\n=== 결과 (method x k) ===")
for method in ["rag_capped", "stat_capped"]:
    for k in K_VALUES:
        sub = res_df[(res_df.method == method) & (res_df.k == k)]
        if len(sub) == 0:
            continue
        print(f"{method:12s} k={k:2d}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

tok_df = pd.DataFrame(token_log)
tok_valid = tok_df.dropna(subset=["prompt_tokens"]) if "prompt_tokens" in tok_df.columns else pd.DataFrame()
if len(tok_valid) > 0:
    print(f"\n총 호출 {len(tok_valid)}회, 총 tokens={tok_valid['total_tokens'].sum():.0f}")
