"""
exp001 — "키워드/임베딩 유사도로는 설명 안 되지만 실제로 성적을 올린" 31개 (타겟, 추가문항)
쌍에 대해, LLM에게 "왜 이 문항이 타겟 예측에 도움이 될 수 있는지" 직접 설명을 요청.
(예측 자체는 안 함 — 순수 설명용 2차 콜, 31회)
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()  # repo root
import json
import sys

import pandas as pd

sys.path.insert(0, _REPO + "/02_llm_question_selection")
import utils as LV

OUTDIR = _REPO + "/02_llm_question_selection"
QUESTIONS_JSON = _REPO + "/data/questions/2025_questions.json"

TEMPERATURE = 0.3

with open(QUESTIONS_JSON, encoding="utf-8") as f:
    qdata = json.load(f)
EXCLUDED_PARTS = {
    "15_법_질서_범죄", "8_여가_문화", "10_인권_소수자", "14_환경_재난",
    "18_기타_분류_애매", "7_건강_질병",
}
pool_all = [q for q in qdata["questions"] if q["topic_part"] not in EXCLUDED_PARTS]
var_to_q = {q["var"]: q for q in pool_all}

pairs = pd.read_csv(f"{OUTDIR}/decisive_nonobvious_pairs.csv")
print(f"설명 요청 대상: {len(pairs)}쌍", flush=True)

SYS = """당신은 사회조사 문항 간의 숨은 관련성을 분석하는 조사방법론 전문가입니다."""


def build_prompt(target_q, item_q):
    return f"""[타겟 문항]
{target_q}

[관련 있다고 판단된 문항]
{item_q}

이 두 문항은 표면적인 단어나 주제가 겹치지 않습니다. 그럼에도 "관련 있는 문항" 문항에 대한
응답이 "타겟 문항"에 대한 응답을 예측하는 데 실제로 도움이 되었습니다.

이 두 문항이 어떤 심리적 성향, 가치관, 혹은 잠재적 특성을 통해 연결될 수 있는지 2~3문장으로
설명하세요. 만약 특별한 연결고리가 떠오르지 않는다면 "뚜렷한 연결고리를 찾기 어려움"이라고
답하세요."""


results = []
for _, r in pairs.iterrows():
    tvar, ivar = r["target"], r["added_item"]
    tq = var_to_q[tvar]["question"]
    iq = var_to_q[ivar]["question"]
    prompt = build_prompt(tq, iq)
    content, usage = LV.call_luxia_tracked(
        [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
        temperature=TEMPERATURE)
    results.append({
        "target": tvar, "added_item": ivar, "cos_sim": r["cos_sim"],
        "target_q": tq, "item_q": iq, "llm_explanation": content,
    })
    print(f"[{len(results)}/{len(pairs)}] {tvar} <- {ivar} 완료", flush=True)
    pd.DataFrame(results).to_csv(f"{OUTDIR}/output_nonobvious_reasoning.csv", index=False)

print(f"\n전체 완료: {len(results)}건 -> output_nonobvious_reasoning.csv", flush=True)
