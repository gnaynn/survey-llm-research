"""
exp016 — axis1 신규 방향: "LLM이 프로필을 직접 추천"하는 방식의 1단계 파일럿.
규칙 기반 선별(상관계수/임베딩/이론적개념군) 대신, LLM의 추론으로 프로필을 구성할 수 있는지 확인.

Step A — 문항 목록 없이, 타겟 질문 텍스트만 주고 "사람들이 이 질문에 답할 때 근거로 삼을 요인"을 LLM이 스스로 나열 (1회 호출).
Step B — Step A의 요인들을, 실제 KGSS 전체 문항 목록(타겟 7개 제외 3,346개)과 매칭 (1회 호출).

첫 타겟: CANTRUST. 파일럿이라 아직 예측(mimicking)까지는 안 가고, "LLM이 관련 요인과 그에 대응하는
실제 문항을 스스로 찾아낼 수 있는가"만 확인.
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv(_REPO + "/.env")

BASE = _REPO + "/data/KGSS_결과파일"
OUTDIR = _REPO + "/01_exploration/experiments/exp016_llm_factor_matching"

with open(f"{BASE}/(2)kgss_variable_dictionary.json", encoding="utf-8") as f:
    VARDICT = json.load(f)
with open(f"{BASE}/(7)kgss_variable_metadata.json", encoding="utf-8") as f:
    META = json.load(f)

LUXIA_API_URL = os.getenv("LUXIA_API_URL", "https://bridge.luxiacloud.com/luxia/v1/chat")
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
LUXIA_MODEL = os.getenv("LUXIA_MODEL", "luxia3-llm-32b-0731")
TEMPERATURE = 0.3

TARGETS = ["CANTRUST", "BEFAIR", "HELPFUL", "PARTYLR", "CURGOV", "SATPOL", "SATFIN"]
TARGET = "CANTRUST"


def call_luxia(messages, temperature=TEMPERATURE, retries=2, timeout=180):
    headers = {"apikey": LUXIA_API_KEY, "Content-Type": "application/json"}
    payload = {"model": LUXIA_MODEL, "messages": messages, "temperature": temperature}
    for attempt in range(retries + 1):
        try:
            resp = requests.post(LUXIA_API_URL, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            print(f"  status={resp.status_code} body={resp.text[:300]}", flush=True)
        except Exception as e:
            print("  exception:", e, flush=True)
    return None


YEAR_FILTER = "2023"

# ---------- Step A (이미 저장된 결과 재사용) ----------

question_text = VARDICT[TARGET]["question"]
label = VARDICT[TARGET]["label"]

step_a_path = f"{OUTDIR}/step_a_{TARGET}.txt"
with open(step_a_path, encoding="utf-8") as f:
    saved = f.read()
step_a_result = saved.split("[응답]\n", 1)[1].strip()
print("=== Step A (저장된 결과 재사용) ===", flush=True)
print(step_a_result, flush=True)

# ---------- Step B (후보를 YEAR_FILTER년에 실제 조사된 문항만으로 제한) ----------

candidate_vars = [v for v in VARDICT.keys()
                   if META.get(v, {}).get("response_format") != "no_valid_values"
                   and v not in TARGETS
                   and YEAR_FILTER in VARDICT[v].get("years", [])]
item_list_lines = [f"{v}: {VARDICT[v]['label']}" for v in candidate_vars]
item_list_text = "\n".join(item_list_lines)
print(f"\n{YEAR_FILTER}년 조사 문항만 필터링 -> 후보 문항 수: {len(candidate_vars)}, 목록 글자수: {len(item_list_text)}", flush=True)

step_b_prompt = f"""아래는 위에서 나온 요인들입니다:
{step_a_result}

아래는 전체 설문 문항 목록입니다 (문항명: 문항 라벨):
{item_list_text}

각 요인에 대해, 그 요인을 실제로 측정할 수 있는 문항이 있으면 문항명으로 연결해주세요.
대응하는 문항이 없으면 "해당 없음"으로 남기세요.
형식: 요인 → 문항명(문항 라벨) 또는 "해당 없음" """

print("\n=== Step B 호출 (전체 문항 목록 포함) ===", flush=True)
messages_b = [{"role": "user", "content": step_b_prompt}]
step_b_result = call_luxia(messages_b, timeout=300)
print(step_b_result, flush=True)

with open(f"{OUTDIR}/step_b_{TARGET}.txt", "w", encoding="utf-8") as f:
    f.write(f"[질문 (요인 부분만 발췌, 문항목록은 생략)]\n요인:\n{step_a_result}\n\n[응답]\n{step_b_result}\n")

print("\n완료.", flush=True)
