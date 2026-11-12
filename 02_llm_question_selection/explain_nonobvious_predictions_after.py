"""
exp001 — "키워드/임베딩 유사도로는 안 설명되지만 실제로 성적을 올린" 31개 사례에 대해,
원래 explain_predictions.py와 동일한 방식으로 이유를 물음: 실제 그 응답자의 프로필+컨텍스트+
이미 나온(맞은) 예측 답을 그대로 보여주고, "왜 이 답을 예측했는지"를 2~3문장으로 설명 요청.
(1차 예측은 이미 존재하는 결과를 그대로 재사용 — 다시 예측 안 함, 순수 설명용 2차 콜 31회)
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

YEAR = 2025
TEMPERATURE = 0.3

with open(QUESTIONS_JSON, encoding="utf-8") as f:
    qdata = json.load(f)
EXCLUDED_PARTS = {
    "15_법_질서_범죄", "8_여가_문화", "10_인권_소수자", "14_환경_재난",
    "18_기타_분류_애매", "7_건강_질병",
}
pool_all = [q for q in qdata["questions"] if q["topic_part"] not in EXCLUDED_PARTS]
var_to_q = {q["var"]: q for q in pool_all}
all_cand_vars = [q["var"] for q in pool_all]

rankings = json.load(open(f"{OUTDIR}/rankcurve_40q5point_5r_ours_rankings.json", encoding="utf-8"))
rc = pd.read_csv(f"{OUTDIR}/output_rankcurve_40q5point_5r.csv")
rep = pd.read_csv(f"{OUTDIR}/decisive_pairs_representative_cases.csv")
print(f"이유 생성 대상: {len(rep)}건", flush=True)

usecols = list(dict.fromkeys(["YEAR", "RESPID"] + LV.CORE9 + all_cand_vars))
df, meta = pyreadstat.read_sav(LV.SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()
df_by_respid = {row["RESPID"]: row for _, row in df.iterrows()}

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
for _, r in rep.iterrows():
    tvar, item, k, respid = r["target"], r["added_item"], int(r["k"]), r["respid"]
    target_q = var_to_q[tvar]
    options, idx_map = LV.target_options(tvar)
    ranked_vars = rankings.get(tvar, [])
    ranked = [var_to_q[v] for v in ranked_vars[:k]]

    row = df_by_respid.get(respid)
    if row is None:
        print(f"  [{tvar}<-{item}] 응답자 {respid} 못 찾음, 스킵", flush=True)
        continue

    pred_row = rc[(rc.target == tvar) & (rc.k == k) & (rc.respid == respid)]
    if len(pred_row) == 0:
        print(f"  [{tvar}<-{item}] 예측 기록 없음, 스킵", flush=True)
        continue
    answer = pred_row.iloc[0]["predicted"]

    p9_text = LV.profile9_text(LV.profile9(row))
    ctx_lines = [ln for q in ranked if (ln := profile_line(q, row)) is not None]
    prompt = build_reason_prompt(p9_text, ctx_lines, target_q, options, answer)
    content, usage = LV.call_luxia_tracked(
        [{"role": "system", "content": SYS_REASON}, {"role": "user", "content": prompt}],
        temperature=TEMPERATURE)
    results.append({
        "target": tvar, "added_item": item, "k": k, "respid": respid,
        "predicted": answer, "reasoning": content,
    })
    print(f"[{len(results)}/{len(rep)}] {tvar} <- {item} (respid={respid}, k={k}) 완료", flush=True)
    pd.DataFrame(results).to_csv(f"{OUTDIR}/output_nonobvious_reasoning_v2.csv", index=False)

print(f"\n전체 완료: {len(results)}건 -> output_nonobvious_reasoning_v2.csv", flush=True)
