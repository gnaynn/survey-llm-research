"""
exp001 — Rank-curve 실험. ours의 top-15 랭킹(기존 rankings.json 재사용, select 재호출 없음)을
k=1개씩 늘려가며 누적 예측, k=0(demo)~15 전체 곡선을 만들기 위해 기존에 없는 k만 신규 실행.
30문항(기존 40문항 중 앞 30개, 무작위 샘플의 부분집합이라 편향 없음) × 5명.

CLI: MISSING_K(콤마구분, 기본 1,2,4,6,7,9,10,11,13,14)
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()  # repo root
import json
import sys

import pandas as pd
import pyreadstat

sys.path.insert(0, _REPO + "/02_llm_question_selection")
import utils as LV

OUTDIR = _REPO + "/02_llm_question_selection"
QUESTIONS_JSON = _REPO + "/data/questions/2025_questions.json"
RANKINGS_JSON = OUTDIR + "/rankings_40.json"  # 이전 실행에서 저장한 LLM 랭킹 (저장소 미포함, 재생성 필요)

YEAR = 2025
TEMPERATURE = 0.3
N_RESP = 5
MISSING_K = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 4, 6, 7, 9, 10, 11, 13, 14]

TARGETS_30 = ['CESD2', 'CESD3', 'CESD6', 'CNSPIMPT', 'CONBUS', 'CONMEDIC', 'CONPRESS', 'DEMTODAY', 'DRVLSCAR',
              'FREESPCH3A', 'GOVSPD1', 'GOVSPD2', 'GOVSPD3', 'GOVSPD5', 'GOVSPD8', 'HATOPSEX', 'HLPDMONF', 'HUMNAT',
              'INTDFPOL', 'INTTROTH', 'KIDSOL06', 'KRPROUD', 'OTHERS1', 'OTHERS2', 'OTHNEEDS', 'PAIDLESS',
              'PATRACH223', 'PROUDDEM', 'PROUDECO', 'PROUDSCI']

PILOT_CSV = f"{OUTDIR}/output_rankcurve_30q5r.csv"
TOKENS_CSV = f"{OUTDIR}/output_tokens_rankcurve_30q5r.csv"

with open(QUESTIONS_JSON, encoding="utf-8") as f:
    qdata = json.load(f)
EXCLUDED_PARTS = {
    "15_법_질서_범죄", "8_여가_문화", "10_인권_소수자", "14_환경_재난",
    "18_기타_분류_애매", "7_건강_질병",
}
pool_all = [q for q in qdata["questions"] if q["topic_part"] not in EXCLUDED_PARTS]
var_to_q = {q["var"]: q for q in pool_all}

with open(RANKINGS_JSON, encoding="utf-8") as f:
    rankings_40 = json.load(f)
rankings = {t: rankings_40[t] for t in TARGETS_30}
print(f"타겟 {len(rankings)}개, 신규 k: {MISSING_K}", flush=True)

all_cand_vars = [q["var"] for q in pool_all]
usecols = list(dict.fromkeys(["YEAR", "RESPID"] + LV.CORE9 + all_cand_vars))
df, meta = pyreadstat.read_sav(LV.SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()

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


all_results = []
token_log = []

for tvar, (ours_list, rag_list) in rankings.items():
    target_q = var_to_q[tvar]
    print(f"\n===== 타겟: {tvar} ({target_q['label']}) =====", flush=True)
    options, idx_map = LV.target_options(tvar)
    ours_ranked = [var_to_q[v] for v in ours_list]

    dft = df.copy()
    dft["target_decoded"] = dft[tvar].apply(lambda v: LV.decode_stripped(tvar, v))
    dft = dft[dft["target_decoded"].notna()]
    for v in LV.CORE9:
        dft = dft[dft[v].notna()]
    sample = dft.sample(n=min(N_RESP, len(dft)), random_state=42)
    print(f"응답자 {len(sample)}명, 기존 랭킹 재사용: {[q['var'] for q in ours_ranked[:5]]}...", flush=True)

    for _, row in sample.iterrows():
        actual = row["target_decoded"]
        p9_text = LV.profile9_text(LV.profile9(row))

        for k in MISSING_K:
            ctx_qs = ours_ranked[:k]
            ctx_lines = [ln for q in ctx_qs if (ln := profile_line(q, row)) is not None]
            prompt = build_predict_prompt(p9_text, ctx_lines, target_q, options)
            content, usage = LV.call_luxia_tracked(
                [{"role": "system", "content": SYS_PREDICT}, {"role": "user", "content": prompt}],
                temperature=TEMPERATURE)
            pred = LV.match_answer(content, options)
            correct = (pred == actual)
            distance = abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1
            all_results.append({
                "target": tvar, "method": "ours", "k": k, "respid": row.get("RESPID"),
                "actual": actual, "predicted": pred, "raw_response": content,
                "correct": correct, "distance": distance,
            })
            token_log.append({"target": tvar, "stage": "predict", "method": "ours", "k": k,
                               "respid": row.get("RESPID"), **(usage or {})})

    pd.DataFrame(all_results).to_csv(PILOT_CSV, index=False)
    pd.DataFrame(token_log).to_csv(TOKENS_CSV, index=False)
    print(f"  [{tvar}] 저장됨", flush=True)

res_df = pd.DataFrame(all_results)
res_df.to_csv(PILOT_CSV, index=False)
pd.DataFrame(token_log).to_csv(TOKENS_CSV, index=False)

print("\n\n=== 결과 (신규 k만) ===")
for k in MISSING_K:
    sub = res_df[res_df.k == k]
    print(f"k={k:2d}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

tok_df = pd.DataFrame(token_log)
tok_valid = tok_df.dropna(subset=["prompt_tokens"]) if "prompt_tokens" in tok_df.columns else pd.DataFrame()
if len(tok_valid) > 0:
    print(f"\n총 호출 {len(tok_valid)}회, 총 tokens={tok_valid['total_tokens'].sum():.0f}")
