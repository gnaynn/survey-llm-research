_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pyreadstat
import requests
from dotenv import load_dotenv

sys.path.insert(0, _REPO + "/01_exploration/experiments/exp009_phase1_satfin_n100")
from build_arms_v2 import VARDICT, META, recompute_eligible_pool, build_arms, decode, missing_set

load_dotenv(_REPO + "/.env")

BASE = _REPO + "/data/KGSS_결과파일"
SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUT = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/phase1_helpful_results.csv"
PROGRESS = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/phase1_helpful_progress.txt"

LUXIA_API_URL = os.getenv("LUXIA_API_URL", "https://bridge.luxiacloud.com/luxia/v1/chat")
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
LUXIA_MODEL = os.getenv("LUXIA_MODEL", "luxia3-llm-32b-0731")
TEMPERATURE = 0.3

CORE9 = ["SEX", "AGE", "MARITAL", "EDUC", "INCOME", "EMPLY", "REGION", "URBAN", "OCC"]
TARGET = "HELPFUL"
TARGET_PART = META[TARGET]["part"]
YEAR = 2023
N_HELDOUT = 100
SPLIT_SEED = 42
SAMPLE_SEED = 7
N_EXAMPLES = 3

TARGET_OPTIONS = ["남을 도우려고 한다", "경우에 따라 다르다", "자신만을 위한다"]  # 이타주의<->이기주의 스펙트럼 순서로 수동 정렬 (MAE 의미있게)

# 후보 컬럼: exp002의 SATFIN 후보 전체 로드
s_df = pd.read_csv(_REPO + "/01_exploration/experiments/exp002_embedding_calibration/output_rs_pairs.csv")
candidate_vars = list(s_df[s_df.target == TARGET]["candidate"])

usecols = ["YEAR"] + CORE9 + [TARGET] + candidate_vars
usecols = list(dict.fromkeys(usecols))
print(f"loading {len(usecols)} columns...", flush=True)
df, meta = pyreadstat.read_sav(SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()
df["target_decoded"] = df[TARGET].apply(lambda v: decode(TARGET, v))
pool = df[df["target_decoded"].notna()].copy()
print("pool size:", len(pool), flush=True)

# 1단계: retrieval pool 70% / held-out 30% 분할
rng_split = random.Random(SPLIT_SEED)
all_idx = list(pool.index)
rng_split.shuffle(all_idx)
n_retrieval = int(len(all_idx) * 0.7)
retrieval_idx = all_idx[:n_retrieval]
heldout_idx = all_idx[n_retrieval:]
retrieval_pool = pool.loc[retrieval_idx]
heldout_pool = pool.loc[heldout_idx]
print(f"retrieval_pool={len(retrieval_pool)}, heldout_pool={len(heldout_pool)}", flush=True)

# eligible pool을 retrieval_pool로만 재계산 (leakage 방지)
print("recomputing eligible pool on retrieval_pool only...", flush=True)
eligible = recompute_eligible_pool(TARGET, retrieval_pool)
print(f"eligible candidates: {len(eligible)}", flush=True)

# held-out에서 n=100 샘플
random.seed(SAMPLE_SEED)
sample_idx = random.sample(list(heldout_pool.index), min(N_HELDOUT, len(heldout_pool)))
sample = heldout_pool.loc[sample_idx]
print(f"evaluating n={len(sample)}", flush=True)


def profile9(row):
    p = {}
    for v in CORE9:
        dec = decode(v, row[v])
        p[v] = dec if dec else "정보 없음"
    return p


def short_desc(p9):
    return f"{p9['AGE']} {p9['SEX']}, {p9['OCC']}, 월 가구소득 {p9['INCOME']}"


ANTI_DRIFT_RULES = """## 답변 규칙 (반드시 지킬 것)
1. 위 인물 본인의 시점에서 1인칭으로 답하라.
2. 본인의 실제 배경에 근거해 자연스럽게 답하라. 일반적/교과서적 지식을 끌어와 답하지 말 것.
3. 자기 의견을 단언적으로 말하라. 조건·회피·분석체 금지.
4. 당신은 AI가 아니라 위 인물 본인이다.
5. "비슷한 사람들"은 참고 예시일 뿐, 당신은 어디까지나 "기본 정보"에 묘사된 인물 본인이다.
"""
_CIRCLE = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧']


def build_system_prompt(profile_dict, examples):
    ex_lines = [f"- {short_desc(p9)}: \"{TARGET_OPTIONS.index(a)+1}번 {a}\"라고 답함" for p9, a in examples]
    basic_info = json.dumps(profile_dict, ensure_ascii=False, indent=2)
    return f"""당신은 아래 "기본 정보"에 묘사된 인물 본인입니다. 이 인물로서 1인칭으로 답하세요.
2023년 한국종합사회조사(KGSS)에 실제로 응답하는 상황입니다.

## 비슷한 사람들 (참고 — 같은 질문에 실제로 답한 다른 사람들)
{chr(10).join(ex_lines)}

## 기본 정보 (당신)
{basic_info}
{ANTI_DRIFT_RULES}"""


def build_user_prompt():
    options_str = "\n".join(f"  {_CIRCLE[i]} {opt}" for i, opt in enumerate(TARGET_OPTIONS))
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


CIRCLE_NUMS = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5, '⑥': 6, '⑦': 7, '⑧': 8}


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


def call_luxia(messages, retries=2):
    headers = {"apikey": LUXIA_API_KEY, "Content-Type": "application/json"}
    payload = {"model": LUXIA_MODEL, "messages": messages, "temperature": TEMPERATURE}
    for attempt in range(retries + 1):
        try:
            resp = requests.post(LUXIA_API_URL, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print("  exception:", e, flush=True)
    return None


ARM_NAMES = ["arm0_baseline", "arm1_corr", "arm2_embed", "arm3_theory", "arm4_random"]

all_results = []
for i, (idx, row) in enumerate(sample.iterrows()):
    p9 = profile9(row)
    ex_idx = random.Random(1000 + i).sample(list(retrieval_pool.index), N_EXAMPLES)
    examples = [(profile9(retrieval_pool.loc[j]), retrieval_pool.loc[j, "target_decoded"]) for j in ex_idx]

    arms_ext = build_arms(TARGET, TARGET_PART, eligible, row, random.Random(2000 + i))

    profiles = {
        "arm0_baseline": dict(p9),
        "arm1_corr": {**p9, **{r["var"]: r["value"] for r in arms_ext["arm1_corr"]}},
        "arm2_embed": {**p9, **{r["var"]: r["value"] for r in arms_ext["arm2_embed"]}},
        "arm3_theory": {**p9, **{r["var"]: r["value"] for r in arms_ext["arm3_theory"]}},
        "arm4_random": {**p9, **{r["var"]: r["value"] for r in arms_ext["arm4_random"]}},
    }

    for arm_name in ARM_NAMES:
        prof = profiles[arm_name]
        messages = [
            {"role": "system", "content": build_system_prompt(prof, examples)},
            {"role": "user", "content": build_user_prompt()},
        ]
        raw = call_luxia(messages)
        parsed = parse_objective_response_robust(raw, TARGET_OPTIONS) if raw else {"selected_option": None, "match_method": "api_fail"}
        actual = row["target_decoded"]
        pred = parsed["selected_option"]
        correct = pred == actual
        dist = abs(TARGET_OPTIONS.index(pred) - TARGET_OPTIONS.index(actual)) if pred in TARGET_OPTIONS else len(TARGET_OPTIONS) - 1
        all_results.append({
            "respid": row.get("RESPID", idx), "arm": arm_name, "n_extra": len(prof) - 9,
            "actual": actual, "predicted": pred, "match_method": parsed["match_method"],
            "correct": correct, "distance": dist, "raw_response": raw,
        })

    if (i + 1) % 5 == 0 or (i + 1) == len(sample):
        msg = f"[{i+1}/{len(sample)}] respid={row.get('RESPID', idx)} done"
        print(msg, flush=True)
        with open(PROGRESS, "w") as f:
            f.write(msg + "\n")
        pd.DataFrame(all_results).to_csv(OUT, index=False)  # 중간 저장

res_df = pd.DataFrame(all_results)
res_df.to_csv(OUT, index=False)

print(f"\n\n=== Phase 1 최종 요약 (n=100, {TARGET}) ===", flush=True)
for arm in ARM_NAMES:
    sub = res_df[res_df.arm == arm]
    acc = sub["correct"].mean()
    mae = sub["distance"].mean()
    dist = Counter(sub["predicted"].dropna())
    n_valid = sub["predicted"].notna().sum()
    max_share = max(dist.values()) / n_valid if n_valid else float("nan")
    print(f"{arm}: acc={acc:.3f}  MAE={mae:.3f}  max_share={max_share:.3f}  n={len(sub)}", flush=True)
