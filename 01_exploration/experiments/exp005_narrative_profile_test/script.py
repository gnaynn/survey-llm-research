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
OUT = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/exp005_results.csv"

LUXIA_API_URL = os.getenv("LUXIA_API_URL", "https://bridge.luxiacloud.com/luxia/v1/chat")
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
LUXIA_MODEL = os.getenv("LUXIA_MODEL", "luxia3-llm-32b-0731")

CORE9 = ["SEX", "AGE", "MARITAL", "EDUC", "INCOME", "EMPLY", "REGION", "URBAN", "OCC"]
TARGET = "SATFIN"
YEAR = 2023
N_SAMPLE = 20
SEED = 7  # exp004와 동일 seed -> 같은 20명

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


usecols = ["YEAR"] + CORE9 + [TARGET]
df, meta = pyreadstat.read_sav(SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()
df["target_decoded"] = df[TARGET].apply(lambda v: decode(TARGET, v))
pool = df[df["target_decoded"].notna()].copy()

random.seed(SEED)
sample_idx = random.sample(list(pool.index), min(N_SAMPLE, len(pool)))
sample = pool.loc[sample_idx]  # exp004와 동일 20명 (동일 seed, 동일 pool 순서)

TARGET_OPTIONS = [v for k, v in VARDICT[TARGET]["values"].items() if float(k) not in missing_set(TARGET)]

# ---------------------------------------------------------------------------
# 줄글(narrative) 프로필 빌더 -- exp004와 정보량은 동일, 형태만 JSON -> 문장
# ---------------------------------------------------------------------------


def build_narrative(profile: dict) -> str:
    sex = profile["SEX"]
    age = profile["AGE"]
    marital = profile["MARITAL"]
    educ = profile["EDUC"]
    emply = profile["EMPLY"]
    occ = profile["OCC"]
    income = profile["INCOME"]
    region = profile["REGION"]
    urban = profile["URBAN"]

    sentences = []
    sentences.append(f"저는 {region}({urban})에 사는 {age} {marital} {sex}입니다.")
    if educ != "정보 없음":
        sentences.append(f"최종 학력은 {educ}입니다.")
    if emply != "정보 없음":
        if emply == "취업" and occ != "정보 없음":
            sentences.append(f"현재 {occ}로 일하고 있습니다.")
        else:
            sentences.append(f"현재 {emply} 상태입니다.")
    if income != "정보 없음":
        sentences.append(f"저희 가구의 월평균 소득은 {income} 정도입니다.")
    return " ".join(sentences)


ANTI_DRIFT_RULES = """## 답변 규칙 (반드시 지킬 것)
1. 위 인물 본인의 시점에서 1인칭으로 답하라.
2. 본인의 실제 배경(성별·연령·소득·직업 등)에 근거해 자연스럽게 답하라.
   일반적/교과서적 지식을 끌어와 답하지 말 것.
3. 자기 의견을 단언적으로 말하라. 조건·회피·분석체("~에 따라 다를 수 있다") 금지.
4. 당신은 AI가 아니라 위 인물 본인이다. AI라는 사실을 언급하지 말 것.
"""

_CIRCLE = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧']


def build_system_prompt_narrative(profile: dict) -> str:
    bio = build_narrative(profile)
    return f"""당신은 아래에 묘사된 인물 본인입니다. 이 인물로서 1인칭으로 답하세요.
2023년 한국종합사회조사(KGSS)에 실제로 응답하는 상황입니다.

## 나에 대한 소개
{bio}
{ANTI_DRIFT_RULES}"""


def build_user_prompt(question_label, question_text, options):
    options_str = "\n".join(f"  {_CIRCLE[i]} {opt}" for i, opt in enumerate(options))
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


CIRCLE_NUMS = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5, '⑥': 6, '⑦': 7, '⑧': 8}


def parse_objective_response_robust(response, options):
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


def call_luxia(messages):
    headers = {"apikey": LUXIA_API_KEY, "Content-Type": "application/json"}
    payload = {"model": LUXIA_MODEL, "messages": messages}
    resp = requests.post(LUXIA_API_URL, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        return None
    return resp.json()["choices"][0]["message"]["content"]


results = []
for i, (idx, row) in enumerate(sample.iterrows()):
    profile = {}
    for v in CORE9:
        dec = decode(v, row[v])
        profile[v] = dec if dec else "정보 없음"

    messages = [
        {"role": "system", "content": build_system_prompt_narrative(profile)},
        {"role": "user", "content": build_user_prompt(VARDICT[TARGET]["label"], VARDICT[TARGET]["question"], TARGET_OPTIONS)},
    ]
    raw = call_luxia(messages)
    parsed = parse_objective_response_robust(raw, TARGET_OPTIONS) if raw else {"selected_option": None, "match_method": "api_fail", "raw_response": None}
    correct = parsed["selected_option"] == row["target_decoded"]
    results.append({
        "respid": row.get("RESPID", idx),
        "narrative": build_narrative(profile),
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
print("\n=== 요약 (narrative 프로필) ===")
print(f"전체 {len(res_df)}명 중 파싱 성공 {n_valid}명")
print(f"정확도: {acc:.3f}")
print("실제 분포:", res_df["actual"].value_counts().to_dict())
print("예측 분포:", res_df["predicted"].value_counts().to_dict())
