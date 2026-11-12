"""
exp016 파일럿 — Step A/B/C로 LLM이 스스로 찾아낸 문항(9개 core + 21개, RELIABLE 등 leakage twin 제외)을
"스토리라인 있는 줄글" 프로필로 엮어 CANTRUST를 mimicking. 소규모(n=30) 파일럿으로 파이프라인 확인.

비교: arm0(core9 JSON baseline) / arm_mode(classical 최빈값) / arm_narrative(신규, 31개 줄글)
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import random
import re

import pandas as pd
import pyreadstat
from dotenv import load_dotenv

load_dotenv(_REPO + "/.env")

import os
import requests

BASE = _REPO + "/data/KGSS_결과파일"
SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp016_llm_factor_matching"

with open(f"{BASE}/(2)kgss_variable_dictionary.json", encoding="utf-8") as f:
    VARDICT = json.load(f)

LUXIA_API_URL = os.getenv("LUXIA_API_URL", "https://bridge.luxiacloud.com/luxia/v1/chat")
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
LUXIA_MODEL = os.getenv("LUXIA_MODEL", "luxia3-llm-32b-0731")
TEMPERATURE = 0.3

CORE9 = ["SEX", "AGE", "MARITAL", "EDUC", "INCOME", "EMPLY", "REGION", "URBAN", "OCC"]
TARGET = "CANTRUST"
YEAR = 2023
N_PILOT = 30
SPLIT_SEED = 42
SAMPLE_SEED = 7
N_EXAMPLES = 3

with open(f"{OUTDIR}/final_selected_no_leak_{TARGET}.json", encoding="utf-8") as f:
    SELECTED = json.load(f)
print(f"선택된 추가 문항 {len(SELECTED)}개: {SELECTED}", flush=True)


def missing_set(var):
    s = set()
    for c in VARDICT.get(var, {}).get("missing_codes", []):
        try:
            s.add(float(c))
        except ValueError:
            pass
    return s


def decode(var, val):
    if val is None or pd.isna(val):
        return None
    if val in missing_set(var):
        return None
    return VARDICT.get(var, {}).get("values", {}).get(str(int(val)), str(val))


def call_luxia(messages, temperature=TEMPERATURE, retries=2, timeout=120):
    headers = {"apikey": LUXIA_API_KEY, "Content-Type": "application/json"}
    payload = {"model": LUXIA_MODEL, "messages": messages, "temperature": temperature}
    for attempt in range(retries + 1):
        try:
            resp = requests.post(LUXIA_API_URL, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            print(f"  status={resp.status_code} body={resp.text[:200]}", flush=True)
        except Exception as e:
            print("  exception:", e, flush=True)
    return None


def profile9(row):
    p = {}
    for v in CORE9:
        dec = decode(v, row[v])
        p[v] = dec if dec else "정보 없음"
    return p


def short_desc(p9):
    return f"{p9['AGE']} {p9['SEX']}, {p9['OCC']}, 월 가구소득 {p9['INCOME']}"


NARRATIVE_SYSTEM = """당신은 사회조사 응답자 프로필을 작성하는 작가입니다. 아래 나열된 사실들(인구통계 +
그 사람의 여러 문항 응답)을 바탕으로, 그 사람을 소개하는 **스토리라인 있는 짧은 인물 소개글**을 쓰세요.

## 규칙 (반드시 지킬 것)
1. 나열된 사실 외의 새로운 사실(숫자, 사건 등)을 지어내지 마라.
2. 사실들을 그냥 나열하지 말고, 이 사람의 일상/가치관/관계가 자연스럽게 드러나는 하나의 짧은 이야기처럼 엮어라
   (예: "그는 OO에서 태어나 자라 지금은 ~하며 살고 있다. 평소 ~하고, ~에는 …" 식의 흐름).
3. 사실들 사이의 개연성 있는 연결(예: 종교 활동이 활발하면 공동체 관계가 두터울 것 등)은 자연스러운
   추론으로 표현해도 되지만, 과도하게 단정하지 마라.
4. 5~7문장, 3인칭으로 서술하라."""


def build_narrative_facts(row, p9):
    lines = [f"- {k}: {v}" for k, v in p9.items()]
    for v in SELECTED:
        dec = decode(v, row.get(v))
        if dec is not None:
            lines.append(f"- {VARDICT[v]['label']}: {dec}")
    return "\n".join(lines)


def narrate_profile(row, p9):
    facts = build_narrative_facts(row, p9)
    messages = [
        {"role": "system", "content": NARRATIVE_SYSTEM},
        {"role": "user", "content": f"다음 사실들로 인물 소개글을 써주세요:\n{facts}"},
    ]
    return call_luxia(messages)


ANTI_DRIFT_RULES = """## 답변 규칙 (반드시 지킬 것)
1. 위 인물 본인의 시점에서 1인칭으로 답하라.
2. 본인의 실제 배경에 근거해 자연스럽게 답하라. 일반적/교과서적 지식을 끌어와 답하지 말 것.
3. 자기 의견을 단언적으로 말하라. 조건·회피·분석체 금지.
4. 당신은 AI가 아니라 위 인물 본인이다.
5. "비슷한 사람들"은 참고 예시일 뿐, 당신은 어디까지나 "기본 정보"에 묘사된 인물 본인이다.
"""
_CIRCLE = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧']
CIRCLE_NUMS = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5, '⑥': 6, '⑦': 7, '⑧': 8}


def build_system_prompt_json(p9, examples):
    ex_lines = [f"- {short_desc(e9)}: \"{a}\"라고 답함" for e9, a in examples]
    basic_info = json.dumps(p9, ensure_ascii=False, indent=2)
    return f"""당신은 아래 "기본 정보"에 묘사된 인물 본인입니다. 이 인물로서 1인칭으로 답하세요.
2023년 한국종합사회조사(KGSS)에 실제로 응답하는 상황입니다.

## 비슷한 사람들 (참고 — 같은 질문에 실제로 답한 다른 사람들)
{chr(10).join(ex_lines)}

## 기본 정보 (당신)
{basic_info}
{ANTI_DRIFT_RULES}"""


def build_system_prompt_narrative(narrative_text, examples):
    ex_lines = [f"- {short_desc(e9)}: \"{a}\"라고 답함" for e9, a in examples]
    return f"""당신은 아래 "인물 소개"에 묘사된 인물 본인입니다. 이 인물로서 1인칭으로 답하세요.
2023년 한국종합사회조사(KGSS)에 실제로 응답하는 상황입니다.

## 비슷한 사람들 (참고 — 같은 질문에 실제로 답한 다른 사람들)
{chr(10).join(ex_lines)}

## 인물 소개 (당신)
{narrative_text}
{ANTI_DRIFT_RULES}"""


def build_user_prompt(options):
    options_str = "\n".join(f"  {_CIRCLE[i]} {opt}" for i, opt in enumerate(options))
    return f"""다음 질문에 답해주세요.

[질문 - {VARDICT[TARGET]['label']}]
{VARDICT[TARGET]['question']}

[보기]
{options_str}

[답변 방식]
- 가장 자신에게 맞는 보기 하나를 고르세요.
- ①②③ 기호를 먼저 말한 뒤, 그 내용을 자연스럽게 이어 말하세요.
- 전체 1~2 문장의 자연스러운 구어체로.
"""


def parse_objective_response_robust(response, options):
    if not response or not options:
        return {"selected_option": None, "match_method": "none"}
    resp = response.strip()
    for sym, idx_1based in CIRCLE_NUMS.items():
        if sym in resp:
            idx = idx_1based - 1
            if 0 <= idx < len(options):
                return {"selected_option": options[idx], "match_method": "circle_symbol"}
    m = re.search(r'(?<!\d)([1-9])\s*(?:번|\.|\))', resp)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(options):
            return {"selected_option": options[idx], "match_method": "number_suffix"}
    m2 = re.match(r'^([1-9])[\s\.,]', resp)
    if m2:
        idx = int(m2.group(1)) - 1
        if 0 <= idx < len(options):
            return {"selected_option": options[idx], "match_method": "number_start"}
    for opt in options:
        if opt in resp:
            return {"selected_option": opt, "match_method": "exact"}
    return {"selected_option": None, "match_method": "none"}


options = [v for k, v in VARDICT[TARGET]["values"].items() if float(k) not in missing_set(TARGET)]
idx_map = {o: i for i, o in enumerate(options)}

usecols = ["YEAR", "RESPID"] + CORE9 + [TARGET] + SELECTED
usecols = list(dict.fromkeys(usecols))
print(f"loading {len(usecols)} columns...", flush=True)
df, meta = pyreadstat.read_sav(SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()
df["target_decoded"] = df[TARGET].apply(lambda v: decode(TARGET, v))
pool = df[df["target_decoded"].notna()].copy()
print(f"pool size: {len(pool)}", flush=True)

rng_split = random.Random(SPLIT_SEED)
all_idx = list(pool.index)
rng_split.shuffle(all_idx)
n_retrieval = int(len(all_idx) * 0.7)
retrieval_pool = pool.loc[all_idx[:n_retrieval]]
heldout_pool = pool.loc[all_idx[n_retrieval:]]
print(f"retrieval_pool={len(retrieval_pool)}, heldout_pool={len(heldout_pool)}", flush=True)

random.seed(SAMPLE_SEED)
sample_idx = random.sample(list(heldout_pool.index), min(N_PILOT, len(heldout_pool)))
sample = heldout_pool.loc[sample_idx]
print(f"evaluating n={len(sample)}", flush=True)

mode_val = retrieval_pool["target_decoded"].mode()[0]

all_results = []
for i, (idx, row) in enumerate(sample.iterrows()):
    p9 = profile9(row)
    ex_idx = random.Random(1000 + i).sample(list(retrieval_pool.index), N_EXAMPLES)
    examples = [(profile9(retrieval_pool.loc[j]), retrieval_pool.loc[j, "target_decoded"]) for j in ex_idx]
    actual = row["target_decoded"]

    # arm_mode
    all_results.append({
        "respid": row.get("RESPID", idx), "arm": "arm_mode", "actual": actual, "predicted": mode_val,
        "correct": mode_val == actual, "distance": abs(idx_map.get(mode_val, len(options) - 1) - idx_map[actual]),
    })

    # arm0 baseline (core9 JSON)
    msgs0 = [
        {"role": "system", "content": build_system_prompt_json(p9, examples)},
        {"role": "user", "content": build_user_prompt(options)},
    ]
    raw0 = call_luxia(msgs0)
    parsed0 = parse_objective_response_robust(raw0, options) if raw0 else {"selected_option": None, "match_method": "api_fail"}
    pred0 = parsed0["selected_option"]
    all_results.append({
        "respid": row.get("RESPID", idx), "arm": "arm0_baseline", "actual": actual, "predicted": pred0,
        "correct": pred0 == actual,
        "distance": abs(idx_map[pred0] - idx_map[actual]) if pred0 in idx_map else len(options) - 1,
        "raw_response": raw0,
    })

    # arm_narrative (31개, 스토리라인 줄글)
    narrative = narrate_profile(row, p9)
    msgsN = [
        {"role": "system", "content": build_system_prompt_narrative(narrative, examples)},
        {"role": "user", "content": build_user_prompt(options)},
    ]
    rawN = call_luxia(msgsN)
    parsedN = parse_objective_response_robust(rawN, options) if rawN else {"selected_option": None, "match_method": "api_fail"}
    predN = parsedN["selected_option"]
    all_results.append({
        "respid": row.get("RESPID", idx), "arm": "arm_narrative_31", "actual": actual, "predicted": predN,
        "correct": predN == actual,
        "distance": abs(idx_map[predN] - idx_map[actual]) if predN in idx_map else len(options) - 1,
        "raw_response": rawN, "narrative": narrative,
    })

    if (i + 1) % 5 == 0 or (i + 1) == len(sample):
        print(f"[{i+1}/{len(sample)}] respid={row.get('RESPID', idx)} done", flush=True)
        pd.DataFrame(all_results).to_csv(f"{OUTDIR}/output_narrative_pilot.csv", index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_narrative_pilot.csv", index=False)

print(f"\n=== exp016 파일럿 결과 (n={N_PILOT}, {TARGET}) ===", flush=True)
for arm in ["arm0_baseline", "arm_mode", "arm_narrative_31"]:
    sub = res_df[res_df.arm == arm]
    acc = sub["correct"].mean()
    mae = sub["distance"].mean()
    print(f"{arm}: acc={acc:.3f}  MAE={mae:.3f}  n={len(sub)}", flush=True)
