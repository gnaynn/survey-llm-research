_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import os
import random
import re

import pandas as pd
import pyreadstat
import requests
from dotenv import load_dotenv

load_dotenv(_REPO + "/.env")

BASE = _REPO + "/data/KGSS_결과파일"
SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUT = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/exp004_results.csv"

LUXIA_API_URL = os.getenv("LUXIA_API_URL", "https://bridge.luxiacloud.com/luxia/v1/chat")
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
LUXIA_MODEL = os.getenv("LUXIA_MODEL", "luxia3-llm-32b-0731")

CORE9 = ["SEX", "AGE", "MARITAL", "EDUC", "INCOME", "EMPLY", "REGION", "URBAN", "OCC"]
TARGET = "SATFIN"
YEAR = 2023
N_SAMPLE = 20
SEED = 7

with open(f"{BASE}/(2)kgss_variable_dictionary.json", encoding="utf-8") as f:
    VARDICT = json.load(f)


def missing_set(var):
    out = set()
    for c in VARDICT.get(var, {}).get("missing_codes", []):
        try:
            out.add(float(c))
        except ValueError:
            pass
    return out


def decode(var, val):
    if val is None or pd.isna(val):
        return None
    if val in missing_set(var):
        return None
    return VARDICT.get(var, {}).get("values", {}).get(str(int(val)), str(val))


# ---------------------------------------------------------------------------
# 1. 데이터 로드 + 샘플링
# ---------------------------------------------------------------------------

usecols = ["YEAR"] + CORE9 + [TARGET]
df, meta = pyreadstat.read_sav(SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()

df["target_decoded"] = df[TARGET].apply(lambda v: decode(TARGET, v))
pool = df[df["target_decoded"].notna()].copy()
print(f"연도 {YEAR}, {TARGET} 유효 응답자: {len(pool)}명 (전체 {len(df)}명 중)")

random.seed(SEED)
sample_idx = random.sample(list(pool.index), min(N_SAMPLE, len(pool)))
sample = pool.loc[sample_idx]

TARGET_OPTIONS = [v for k, v in VARDICT[TARGET]["values"].items()
                  if float(k) not in missing_set(TARGET)]
# values dict 순서가 곧 척도 순서 (1=매우 만족 ... 5=매우 불만족)
print("타겟 문항:", VARDICT[TARGET]["label"], "-", VARDICT[TARGET]["question"])
print("보기:", TARGET_OPTIONS)

# ---------------------------------------------------------------------------
# 2. 프롬프트 빌더 (persona-dashboard backend/prompts.py 패턴 적용)
# ---------------------------------------------------------------------------

ANTI_DRIFT_RULES = """## 답변 규칙 (반드시 지킬 것)
1. 위 인물 본인의 시점에서 1인칭으로 답하라.
2. 본인의 실제 인구통계적 배경(성별·연령·소득·직업 등)에 근거해 자연스럽게 답하라.
   일반적/교과서적 지식을 끌어와 답하지 말 것.
3. 자기 의견을 단언적으로 말하라. 조건·회피·분석체("~에 따라 다를 수 있다") 금지.
4. 당신은 AI가 아니라 위 인물 본인이다. AI라는 사실을 언급하지 말 것.
"""

_CIRCLE = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧']


def build_system_prompt(profile: dict) -> str:
    basic_info = json.dumps(profile, ensure_ascii=False, indent=2)
    return f"""당신은 아래에 묘사된 인물 본인입니다. 이 인물로서 1인칭으로 답하세요.
2023년 한국종합사회조사(KGSS)에 실제로 응답하는 상황입니다.

## 기본 정보
{basic_info}
{ANTI_DRIFT_RULES}"""


def build_user_prompt(question_label: str, question_text: str, options: list) -> str:
    options_str = "\n".join(
        f"  {_CIRCLE[i] if i < len(_CIRCLE) else str(i+1)} {opt}"
        for i, opt in enumerate(options)
    )
    return f"""다음 질문에 답해주세요.

[질문 - {question_label}]
{question_text}

[보기]
{options_str}

[답변 방식]
- 가장 자신에게 맞는 보기 하나를 고르세요.
- ①②③ 기호를 먼저 말한 뒤, 그 내용을 자연스럽게 이어 말하세요.
  (예: "② 약간 만족합니다. 그럭저럭 먹고 살 만해서요.")
- 전체 1~2 문장의 자연스러운 구어체로.
"""


# ---------------------------------------------------------------------------
# 3. 객관식 응답 파싱 (persona-dashboard backend/postprocess.py에서 가져옴)
# ---------------------------------------------------------------------------

CIRCLE_NUMS = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5, '⑥': 6, '⑦': 7, '⑧': 8}


def parse_objective_response_robust(response: str, options: list) -> dict:
    if not response or not options:
        return {"selected_option": None, "match_method": "none", "raw_response": response}
    resp = response.strip()
    for sym, idx_1based in CIRCLE_NUMS.items():
        if sym in resp:
            idx = idx_1based - 1
            if 0 <= idx < len(options):
                return {"selected_option": options[idx], "match_method": "circle_symbol", "raw_response": response}
    num_match = re.search(r'(?<!\d)([1-9])\s*(?:번|\.|\))', resp)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if 0 <= idx < len(options):
            return {"selected_option": options[idx], "match_method": "number_suffix", "raw_response": response}
    standalone = re.match(r'^([1-9])[\s\.,]', resp)
    if standalone:
        idx = int(standalone.group(1)) - 1
        if 0 <= idx < len(options):
            return {"selected_option": options[idx], "match_method": "number_start", "raw_response": response}
    for opt in options:
        if opt in resp:
            return {"selected_option": opt, "match_method": "exact", "raw_response": response}
    return {"selected_option": None, "match_method": "none", "raw_response": response}


# ---------------------------------------------------------------------------
# 4. Luxia 호출
# ---------------------------------------------------------------------------

def call_luxia(messages):
    headers = {"apikey": LUXIA_API_KEY, "Content-Type": "application/json"}
    payload = {"model": LUXIA_MODEL, "messages": messages}
    resp = requests.post(LUXIA_API_URL, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        return None
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 5. 실행
# ---------------------------------------------------------------------------

results = []
for i, (idx, row) in enumerate(sample.iterrows()):
    profile = {}
    for v in CORE9:
        dec = decode(v, row[v])
        profile[v] = dec if dec else "정보 없음"

    messages = [
        {"role": "system", "content": build_system_prompt(profile)},
        {"role": "user", "content": build_user_prompt(
            VARDICT[TARGET]["label"], VARDICT[TARGET]["question"], TARGET_OPTIONS)},
    ]
    raw = call_luxia(messages)
    parsed = parse_objective_response_robust(raw, TARGET_OPTIONS) if raw else {"selected_option": None, "match_method": "api_fail", "raw_response": None}
    correct = parsed["selected_option"] == row["target_decoded"]
    results.append({
        "respid": row.get("RESPID", idx),
        "profile": profile,
        "actual": row["target_decoded"],
        "predicted": parsed["selected_option"],
        "match_method": parsed["match_method"],
        "correct": correct,
        "raw_response": raw,
    })
    print(f"[{i+1}/{len(sample)}] actual={row['target_decoded']!r:20} predicted={parsed['selected_option']!r:20} {'OK' if correct else 'X'}")

res_df = pd.DataFrame(results)
res_df.to_csv(OUT, index=False)

n_valid = res_df["predicted"].notna().sum()
acc = res_df["correct"].mean()
acc_valid = res_df[res_df["predicted"].notna()]["correct"].mean() if n_valid else float("nan")

print("\n=== 요약 ===")
print(f"전체 {len(res_df)}명 중 파싱 성공 {n_valid}명")
print(f"전체 정확도(파싱실패=오답 처리): {acc:.3f}")
print(f"파싱 성공분만 정확도: {acc_valid:.3f}")
print(f"무작위 추측 기대 정확도(1/{len(TARGET_OPTIONS)}): {1/len(TARGET_OPTIONS):.3f}")
print("\n실제 분포:", res_df["actual"].value_counts().to_dict())
print("예측 분포:", res_df["predicted"].value_counts().to_dict())
