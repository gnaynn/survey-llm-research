"""
Survey-LLM-Lab 2 공용 유틸. 데이터/디코딩 로직은 기존 Survey-LLM-Lab의
lib_peer_deviation.py를 그대로 재사용(데이터는 동일한 것 사용 방침), LLM 호출만
토큰 사용량을 캡처하도록 새로 작성.
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()  # repo root
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(_REPO + "/.env")

sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L  # noqa: E402

VARDICT = L.VARDICT
CORE9 = L.CORE9
decode = L.decode
missing_set = L.missing_set

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"

LUXIA_API_URL = os.getenv("LUXIA_API_URL", "https://bridge.luxiacloud.com/luxia/v1/chat")
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
LUXIA_MODEL = os.getenv("LUXIA_MODEL", "luxia3-llm-32b-0731")


def call_luxia_tracked(messages, temperature=0.3, retries=4):
    """content, usage(dict: prompt_tokens/completion_tokens/total_tokens) 튜플 반환. 실패 시 (None, None).
    non-200(레이트리밋 등)도 로그 남기고 지수 백오프 후 재시도."""
    import time
    headers = {"apikey": LUXIA_API_KEY, "Content-Type": "application/json"}
    payload = {"model": LUXIA_MODEL, "messages": messages, "temperature": temperature}
    for attempt in range(retries + 1):
        try:
            resp = requests.post(LUXIA_API_URL, json=payload, headers=headers, timeout=90)
            if resp.status_code == 200:
                d = resp.json()
                content = d["choices"][0]["message"]["content"]
                usage = d.get("usage", {}) or {}
                return content, usage
            else:
                print(f"  http {resp.status_code}: {resp.text[:200]}", flush=True)
        except Exception as e:
            print("  exception:", e, flush=True)
        if attempt < retries:
            time.sleep(min(2 ** attempt, 20))
    return None, None


def profile9(row):
    return {v: (decode(v, row[v]) or "정보 없음") for v in CORE9}


def profile9_text(p9):
    return (f"성별:{p9['SEX']}, 나이:{p9['AGE']}, 혼인:{p9['MARITAL']}, 학력:{p9['EDUC']}, "
            f"소득:{p9['INCOME']}, 취업:{p9['EMPLY']}, 지역:{p9['REGION']}, 도농:{p9['URBAN']}, 직업:{p9['OCC']}")


def target_options(target_var):
    options = [v.strip() for k, v in VARDICT[target_var]["values"].items() if float(k) not in missing_set(target_var)]
    idx_map = {o: i for i, o in enumerate(options)}
    return options, idx_map


def decode_stripped(var, val):
    """decode()는 VARDICT 라벨을 앞뒤 공백 그대로 반환(일부 변수는 라벨에 leading space가 있음).
    target_options()가 공백을 strip하므로, 타겟 실제값 비교에는 이 함수를 써서 일치시킨다."""
    d = decode(var, val)
    return d.strip() if d is not None else None


def match_answer(response_text, options):
    """LLM 응답 텍스트를 옵션 문자열 중 하나로 매칭. 정확히 포함되는 옵션 중 가장 긴 것을 채택."""
    if not response_text:
        return None
    candidates = [o for o in options if o in response_text]
    if not candidates:
        return None
    return max(candidates, key=len)


def build_pool_zscores(pool_df, k_vars):
    """pool_df 기준 각 k_var의 (z-score 컬럼 DataFrame, {var: (mean, std)}) 반환."""
    import pandas as pd
    Z = pd.DataFrame(index=pool_df.index)
    stats = {}
    for v in k_vars:
        ms = missing_set(v)
        col = pool_df[v].where(~pool_df[v].isin(ms)).astype(float)
        mean, std = col.mean(), col.std()
        if pd.isna(mean) or pd.isna(std) or std == 0:
            continue
        Z[v] = (col - mean) / std
        stats[v] = (mean, std)
    return Z, stats


def zscore_nearest(target_row, Z, stats, k_vars, n=8, min_common=2):
    """target_row와 Z(풀의 z-score) 기준 가장 가까운 n명의 인덱스 리스트(공통응답 min_common개 미만 제외)."""
    import numpy as np
    import pandas as pd
    valid_vars, target_z = [], {}
    for v in k_vars:
        if v not in stats:
            continue
        mean, std = stats[v]
        ms = missing_set(v)
        tv = target_row.get(v)
        if tv is None or pd.isna(tv) or tv in ms:
            continue
        target_z[v] = (float(tv) - mean) / std
        valid_vars.append(v)
    if not valid_vars:
        return []
    sub = Z[valid_vars]
    diffs_sq = (sub - pd.Series(target_z)) ** 2
    common_count = sub.notna().sum(axis=1)
    mean_sq = diffs_sq.mean(axis=1, skipna=True)
    dist = np.sqrt(mean_sq)
    dist = dist[common_count >= min_common].sort_values()
    return list(dist.index[:n])


def aggregate_median(answers, options):
    idx_map = {o: i for i, o in enumerate(options)}
    idxs = sorted(idx_map[a] for a in answers if a in idx_map)
    if not idxs:
        return None
    mid = idxs[len(idxs) // 2] if len(idxs) % 2 == 1 else idxs[len(idxs) // 2 - 1]
    return options[mid]
