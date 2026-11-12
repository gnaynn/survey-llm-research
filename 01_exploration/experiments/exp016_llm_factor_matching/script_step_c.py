"""
exp016 Step C — Step B에서 매칭된(검증 완료) 70개 후보 중, LLM 스스로 "진짜 예측에 중요한" 문항만
추리게 한다. 통계(r/s)로 다시 거르면 원래 비판받은 방식으로 회귀하는 것이므로, 순수 LLM 판단으로만 축소.
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
with open(f"{OUTDIR}/validated_candidates_CANTRUST.json", encoding="utf-8") as f:
    CANDIDATES = json.load(f)

LUXIA_API_URL = os.getenv("LUXIA_API_URL", "https://bridge.luxiacloud.com/luxia/v1/chat")
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
LUXIA_MODEL = os.getenv("LUXIA_MODEL", "luxia3-llm-32b-0731")
TARGET = "CANTRUST"
N_FINAL = 12


def call_luxia(messages, temperature=0.3, retries=2, timeout=120):
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


cand_lines = "\n".join(f"{v}: {label}" for v, label in CANDIDATES.items())
question_text = VARDICT[TARGET]["question"]

step_c_prompt = f"""타겟 질문: "{question_text}" ({TARGET})

아래는 이 질문에 대한 응답을 예측하는 데 도움이 될 것으로 앞서 매칭된 후보 문항 {len(CANDIDATES)}개입니다:
{cand_lines}

이 중 실제로 개인별 예측에 가장 중요하고 서로 중복되지 않는 문항 {N_FINAL}개만 골라주세요.
같은 배터리(예: 여러 대상에 대한 신뢰도, 여러 조직 가입 여부)에서는 대표성 있는 1~2개만 남기고,
나머지 비슷한 항목은 과감히 빼세요.

출력 형식: 문항명만 한 줄에 하나씩, 총 {N_FINAL}개. 다른 설명 없이 문항명 목록만."""

print("=== Step C 호출 ===", flush=True)
result = call_luxia([{"role": "user", "content": step_c_prompt}])
print(result, flush=True)

with open(f"{OUTDIR}/step_c_{TARGET}.txt", "w", encoding="utf-8") as f:
    f.write(f"[질문]\n{step_c_prompt}\n\n[응답]\n{result}\n")

# 응답에서 실제 VARDICT에 존재하는 문항명만 추출(검증)
import re
tokens = re.findall(r"[A-Z][A-Z0-9_]{2,}", result or "")
selected = [t for t in dict.fromkeys(tokens) if t in CANDIDATES]
print("\n검증된 최종 선택:", len(selected), selected, flush=True)
with open(f"{OUTDIR}/final_selected_{TARGET}.json", "w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)
