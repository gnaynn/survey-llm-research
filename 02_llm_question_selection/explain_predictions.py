"""
exp001 — 5개 타겟(k=0,3,5,8,15) 예측에 대한 "이유" 생성(2차 콜).
1차 예측은 이미 존재하는 결과(output_rankcurve_full_0to15.csv)를 그대로 재사용 —
예측 자체를 다시 안 하고, 같은 컨텍스트 + 이미 나온 답을 다시 보여주며 "왜 그렇게 답했는지"만 물음.
(정확도에 영향 없음 — 순수 설명용 콜)
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
RANKCURVE_CSV = f"{OUTDIR}/output_rankcurve_full_0to15.csv"

YEAR = 2025
TEMPERATURE = 0.3
TARGETS_5 = ["OTHERS1", "OTHNEEDS", "INTTROTH", "CONMEDIC", "OTHERS2", "CONBUS"]
K_POINTS = [0, 3, 5, 8, 15]

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

all_cand_vars = [q["var"] for q in pool_all]
usecols = list(dict.fromkeys(["YEAR", "RESPID"] + LV.CORE9 + all_cand_vars))
df, meta = pyreadstat.read_sav(LV.SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()

rc = pd.read_csv(RANKCURVE_CSV)
rc = rc[(rc.target.isin(TARGETS_5)) & (rc.k.isin(K_POINTS))]
print(f"이유 생성 대상: {len(rc)}건 (5타겟 x 5명 x 5k = {5*5*5} 예상)", flush=True)

SYS_REASON = """당신은 방금 사회조사 응답자의 프로필을 보고 특정 문항에 대한 응답을 예측했습니다.
그 예측을 내린 이유를, 주어진 정보 중 무엇을 근거로 판단했는지 포함해서 간결하게 설명하세요."""


def profile_line(q, row):
    ans = LV.decode(q["var"], row.get(q["var"]))
    if ans is None:
        return None
    return f"Q: {q['question']} A: {ans}"


def build_reason_prompt(p9_text, context_lines, target_q, options, answer):
    ctx = " | ".join(context_lines) if context_lines else "(관련 문항 정보 없음)"
    target_disp = f"Q: {target_q['question']}"
    options_str = "\n".join(f"  - {o}" for o in options)
    return f"""[타겟 인물]
{p9_text}
| {ctx}

[예측했던 문항]
{target_disp}

[보기]
{options_str}

[방금 당신이 예측한 답]
{answer}

이 답을 예측한 이유를 설명하세요. 위에 제공된 정보(인구통계, 관련 문항 답변) 중 구체적으로 어떤
것을 근거로 이 답을 골랐는지 짚어서, 2~3문장으로 간결하게 설명하세요."""


results = []
for tvar in TARGETS_5:
    target_q = var_to_q[tvar]
    options, idx_map = LV.target_options(tvar)
    ours_ranked_vars = rankings_40[tvar][0]
    ours_ranked = [var_to_q[v] for v in ours_ranked_vars]
    print(f"\n===== 타겟: {tvar} ({target_q['label']}) =====", flush=True)

    dft = df.copy()
    dft["target_decoded"] = dft[tvar].apply(lambda v: LV.decode_stripped(tvar, v))
    dft = dft[dft["target_decoded"].notna()]
    for v in LV.CORE9:
        dft = dft[dft[v].notna()]
    sample = dft.sample(n=5, random_state=42)
    sample_by_respid = {row["RESPID"]: row for _, row in sample.iterrows()}

    sub = rc[rc.target == tvar]
    for _, r in sub.iterrows():
        k = int(r["k"])
        respid = r["respid"]
        answer = r["predicted"]
        if pd.isna(answer):
            continue
        row = sample_by_respid.get(respid)
        if row is None:
            continue
        p9_text = LV.profile9_text(LV.profile9(row))
        ctx_qs = ours_ranked[:k]
        ctx_lines = [ln for q in ctx_qs if (ln := profile_line(q, row)) is not None]
        prompt = build_reason_prompt(p9_text, ctx_lines, target_q, options, answer)
        content, usage = LV.call_luxia_tracked(
            [{"role": "system", "content": SYS_REASON}, {"role": "user", "content": prompt}],
            temperature=TEMPERATURE)
        results.append({
            "target": tvar, "k": k, "respid": respid, "actual": r["actual"],
            "predicted": answer, "correct": r["correct"], "reasoning": content,
        })
        pd.DataFrame(results).to_csv(f"{OUTDIR}/output_reasoning_5q.csv", index=False)
    print(f"  [{tvar}] {len(sub)}건 처리 완료", flush=True)

print(f"\n전체 완료: {len(results)}건 저장됨 -> output_reasoning_5q.csv", flush=True)
