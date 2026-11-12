"""
axis1 신규 방식 공용 라이브러리: 또래집단(peer group) 대비 이탈(z-score) 기반 프로필 구성.

4단계:
  1. classify_ordinal()      - 척도형(Likert) 문항 식별 (LLM 미사용)
  2. compute_person_zscore() - 또래집단 정의 + z-score 계산 (LLM 미사용, 순수 통계)
  3. extract_deviation_features() - leakage 필터 적용 후 |z| 상위 N개 추출 (LLM 미사용)
  4. build_narration_prompt()/narrate() - LLM 서술화 (응답자당 1회 호출)
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import os
import re

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

BASE = _REPO + "/data/KGSS_결과파일"
EXP002_CSV = _REPO + "/01_exploration/experiments/exp002_embedding_calibration/output_rs_pairs.csv"

with open(f"{BASE}/(2)kgss_variable_dictionary.json", encoding="utf-8") as f:
    VARDICT = json.load(f)
with open(f"{BASE}/(7)kgss_variable_metadata.json", encoding="utf-8") as f:
    META = json.load(f)

LUXIA_API_URL = os.getenv("LUXIA_API_URL", "https://bridge.luxiacloud.com/luxia/v1/chat")
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
LUXIA_MODEL = os.getenv("LUXIA_MODEL", "luxia3-llm-32b-0731")

CORE9 = ["SEX", "AGE", "MARITAL", "EDUC", "INCOME", "EMPLY", "REGION", "URBAN", "OCC"]

# ---------- 공통 유틸 (기존 build_arms_v2.py와 동일 로직) ----------

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


# ---------- Stage 1: 척도형(ordinal) 문항 식별 ----------

# 값 라벨에 이 키워드가 있으면 순서형(Likert/빈도척도)일 가능성이 높다고 판단.
# 완벽한 의미론적 분류가 아니라 휴리스틱임 — audit CSV로 사람이 검토 가능하게 남겨둠.
ORDINAL_KEYWORDS = [
    "매우", "다소", "약간", "전혀", "거의", "보통", "자주", "가끔", "항상", "한번도",
    "좀처럼", "대체로", "대체적으로", "드물게", "매일", "1주일", "1개월", "1달", "1년",
    "높다", "낮다", "많다", "적다", "찬성", "반대", "만족", "긍정", "부정", "좋다", "나쁘다", "신뢰",
]

# missing_codes에 안 잡혔지만 사실상 결측인 라벨 (예: MYTH218의 "Not Applicable")
SOFT_MISSING_PATTERNS = [
    "not applicable", "응답거부", "모름", "모르겠", "무응답", "해당없음", "해당 없음",
    "dk", "iap", "비해당", "n/a",
]


def _is_soft_missing(label):
    lab = str(label).strip().lower()
    return any(p in lab for p in SOFT_MISSING_PATTERNS)


def nonmissing_sorted_values(var):
    """(코드, 라벨) 리스트. missing_codes + soft-missing 라벨 모두 제외, 코드 오름차순."""
    vals = VARDICT.get(var, {}).get("values", {})
    ms = missing_set(var)
    items = []
    for k, v in vals.items():
        try:
            fk = float(k)
        except ValueError:
            continue
        if fk in ms:
            continue
        if _is_soft_missing(v):
            continue
        items.append((fk, str(v).strip()))
    items.sort()
    return items


def classify_ordinal(var):
    """(is_ordinal: bool, n_levels: int, labels: list[str]) 반환.
    규칙: 유효 수준 3~11개 + (중간 라벨이 전부 숫자만이거나 라벨에 척도 키워드 포함) -> 순서형으로 판단.
    """
    nm = nonmissing_sorted_values(var)
    labels = [lab for _, lab in nm]
    n = len(labels)
    if n < 3 or n > 11:
        return False, n, labels
    inner = labels[1:-1]
    inner_numeric = len(inner) > 0 and all(s.replace(".", "", 1).isdigit() for s in inner)
    has_kw = any(any(kw in lab for kw in ORDINAL_KEYWORDS) for lab in labels)
    return bool(inner_numeric or has_kw), n, labels


def get_candidate_pool(target):
    """exp002에서 미리 계산된 target별 후보 문항 목록(same-part + 배경 랜덤 300, 총 153개/타겟)."""
    s_df = pd.read_csv(EXP002_CSV)
    return list(s_df[s_df.target == target]["candidate"])


def get_ordinal_candidates(target, audit_path=None):
    """target의 후보 풀 중 척도형만 필터링. audit_path 지정 시 전체 분류 결과를 CSV로 저장(검토용)."""
    cands = get_candidate_pool(target)
    rows = []
    for c in cands:
        is_ord, n_levels, labels = classify_ordinal(c)
        rows.append({
            "candidate": c, "label": VARDICT.get(c, {}).get("label", ""),
            "is_ordinal": is_ord, "n_levels": n_levels, "value_labels": " / ".join(labels),
        })
    audit_df = pd.DataFrame(rows)
    if audit_path:
        audit_df.to_csv(audit_path, index=False)
    ordinal_vars = list(audit_df[audit_df.is_ordinal]["candidate"])
    return ordinal_vars, audit_df


# ---------- leakage 배제 (기존 axis1 방식 그대로: |r|>=0.30 또는 s>=0.50) ----------

def recompute_eligible_pool(target, retrieval_df, candidate_vars=None):
    """r은 retrieval_pool로 재계산(held-out 누출 방지), s는 exp002 사전계산값 재사용.
    candidate_vars 지정 시 그 목록으로 한정(예: ordinal 후보만).
    반환: DataFrame[candidate, r, s, exclude] 중 exclude=False(=eligible)만.
    """
    s_df = pd.read_csv(EXP002_CSV)
    s_lookup = s_df[s_df.target == target][["candidate", "s"]].set_index("candidate")["s"].to_dict()
    if candidate_vars is None:
        candidate_vars = list(s_lookup.keys())

    rows = []
    for cand in candidate_vars:
        if cand not in s_lookup or cand not in retrieval_df.columns or target not in retrieval_df.columns:
            continue
        t_dec = retrieval_df[target].apply(lambda v: decode(target, v))
        c_dec_raw = retrieval_df[cand]
        c_missing = missing_set(cand)
        c_dec = c_dec_raw.where(~c_dec_raw.isin(c_missing))
        pair = pd.DataFrame({"t": t_dec, "c": c_dec}).dropna()
        if len(pair) < 30:
            continue
        t_vals = retrieval_df.loc[pair.index, target]
        c_vals = pair["c"]
        try:
            r, _ = spearmanr(t_vals, c_vals)
        except Exception:
            continue
        if np.isnan(r):
            continue
        rows.append({"candidate": cand, "r": r, "s": s_lookup[cand]})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["exclude"] = (df["r"].abs() >= 0.30) | (df["s"] >= 0.50)
    return df[~df["exclude"]].copy()


# ---------- Stage 2: 또래집단 정의 + z-score ----------

PEER_DEFS = {
    "A1": ["SEX", "EDUC"],          # 연령대(±5) + 성별 + 학력
    "A2": [],                        # 연령대(±5) + 성별만
    "A3": ["SEX", "EDUC", "REGION"],  # 연령대(±5) + 성별 + 학력 + 지역
}
AGE_WINDOW = 5
MIN_PEER_N = 10


def _peer_mask(retrieval_pool, age, sex, educ, region, extra_fields, age_window=AGE_WINDOW):
    m = (retrieval_pool["AGE"] - age).abs() <= age_window
    if "SEX" in extra_fields:
        m &= retrieval_pool["SEX"] == sex
    else:
        m &= retrieval_pool["SEX"] == sex  # 성별은 A2에도 포함되는 고정 조건 (peer_def 정의상)
    if "EDUC" in extra_fields:
        m &= retrieval_pool["EDUC"] == educ
    if "REGION" in extra_fields:
        m &= retrieval_pool["REGION"] == region
    return m


def compute_person_zscore(row, retrieval_pool, ordinal_vars, peer_def="A1"):
    """한 응답자에 대해 ordinal_vars 전체의 z-score를 벡터화(컬럼 방향)로 계산.
    또래집단 fallback: A1/A3 조건 -> EDUC(및 REGION) 완화 -> age window 확대(±10) -> 성별만.
    반환: (zscores: dict{var: z}, meta: dict{peer_n, fallback_level})
    """
    age, sex, educ, region = row.get("AGE"), row.get("SEX"), row.get("EDUC"), row.get("REGION")
    extra_fields = PEER_DEFS[peer_def]

    fallback_chain = []
    fallback_chain.append(("full", extra_fields, AGE_WINDOW))
    if "EDUC" in extra_fields or "REGION" in extra_fields:
        fallback_chain.append(("no_educ_region", [], AGE_WINDOW))
    fallback_chain.append(("wide_age", [], AGE_WINDOW * 2))
    fallback_chain.append(("sex_only_wide", [], AGE_WINDOW * 4))

    mask = None
    used_level, peer_n = "none", 0
    for level, fields, win in fallback_chain:
        mask = _peer_mask(retrieval_pool, age, sex, educ, region, fields, age_window=win)
        n = int(mask.sum())
        if n >= MIN_PEER_N:
            used_level, peer_n = level, n
            break
    else:
        used_level, peer_n = "insufficient", int(mask.sum()) if mask is not None else 0

    peers = retrieval_pool[mask] if mask is not None else retrieval_pool.iloc[0:0]

    zscores = {}
    if peer_n < 2:
        return zscores, {"peer_n": peer_n, "fallback_level": used_level}

    cols = [v for v in ordinal_vars if v in peers.columns]
    for v in cols:
        ms = missing_set(v)
        col = peers[v].where(~peers[v].isin(ms))
        mean, std = col.mean(), col.std()
        if pd.isna(mean) or pd.isna(std) or std == 0:
            continue
        own = row.get(v)
        if own is None or pd.isna(own) or own in ms:
            continue
        z = (float(own) - float(mean)) / float(std)
        zscores[v] = {"z": z, "own_code": float(own), "peer_mean_code": float(mean), "peer_std": float(std)}

    return zscores, {"peer_n": peer_n, "fallback_level": used_level}


# ---------- Stage 3: leakage 필터 적용 후 |z| 상위 N개 이탈 문항 추출 ----------

def _nearest_label(var, code):
    """peer 평균 코드값에 가장 가까운 유효 코드의 라벨(라벨이 없는 순수 숫자 척도는 근사값 표기)."""
    nm = nonmissing_sorted_values(var)
    if not nm:
        return None
    nearest = min(nm, key=lambda kv: abs(kv[0] - code))
    return nearest[1]


def extract_deviation_features(row, zscores, eligible_set, n_top):
    """zscores 중 eligible_set(leakage 필터 통과)에 속한 것만 |z| 기준 상위 n_top 추출.
    본인 응답 라벨과 또래평균에 해당하는(근사) 라벨을 함께 담아 narration 단계에서
    z의 부호를 자의적으로 해석(방향 오인)하지 않고 라벨 대 라벨로 비교할 수 있게 한다.
    """
    items = [(v, d) for v, d in zscores.items() if v in eligible_set]
    items.sort(key=lambda x: -abs(x[1]["z"]))
    top = items[:n_top]
    out = []
    for v, d in top:
        out.append({
            "var": v,
            "label": VARDICT.get(v, {}).get("label", v),
            "question": VARDICT.get(v, {}).get("question", ""),
            "z": round(d["z"], 3),
            "respondent_value": decode(v, row.get(v)),
            "peer_mean_label": _nearest_label(v, d["peer_mean_code"]),
            "peer_mean_code": round(d["peer_mean_code"], 2),
            "own_code": d["own_code"],
        })
    return out


# ---------- Stage 4: LLM 서술화 ----------

NARRATION_SYSTEM = """당신은 사회조사 데이터 분석가입니다. 한 응답자가 또래집단(비슷한 나이/성별/학력대) 대비 \
어느 문항에서 얼마나 벗어난 답을 했는지 문항별로 "본인 응답 라벨"과 "또래집단 평균에 해당하는 라벨"이 함께 주어집니다. \
이를 바탕으로 "이 사람만의 특징"을 2~3문장으로 서술하세요.

## 규칙 (반드시 지킬 것)
1. 주어진 문항명·응답 라벨·또래평균 라벨 외의 사실을 새로 만들어내지 마라. 원 데이터에 없는 숫자를 임의로 덧붙이지 마라.
2. "왜 그럴 것 같다"는 인과적 추측은 허용하되, 과도하게 확신하는 표현("반드시", "확실히")은 쓰지 마라.
3. **방향은 반드시 "본인 응답 라벨" vs "또래 평균 라벨"의 실제 의미를 비교해서 서술하라.** \
z 부호(+/-)는 내부 코드값의 위치 차이일 뿐이며, 문항마다 코드 순서(1이 긍정 쪽인지 부정 쪽인지)가 다르므로 \
z가 양수/음수라는 이유만으로 "더 높다/낮다", "더 동의한다/반대한다" 같은 의미를 추론하지 마라. \
오직 두 라벨의 실제 문구를 비교해서 방향을 판단하라. |z| 크기는 "또래 대비 얼마나 이례적인지"를 나타내는 용도로만 써라.
4. 3인칭으로, 분석 노트처럼 간결하게 써라."""


def build_narration_prompt(deviation_features):
    lines = []
    for f in deviation_features:
        lines.append(
            f"- [{f['label']}] 질문: \"{f['question']}\" / 본인 응답 라벨: \"{f['respondent_value']}\" "
            f"/ 또래집단 평균에 해당하는 라벨(근사): \"{f['peer_mean_label']}\" / 이례성 크기 |z|={abs(f['z']):.2f}"
        )
    user = "다음은 한 응답자의 또래집단 대비 이탈 문항 목록입니다. 각 항목은 [본인 응답 라벨]과 [또래 평균 라벨]을 " \
        "직접 비교할 수 있게 제공됩니다.\n\n" + "\n".join(lines) + \
        "\n\n이 사람만의 특징을 2~3문장으로 서술하세요. 방향(더 동의/덜 동의, 더 만족/덜 만족 등)은 반드시 " \
        "두 라벨의 실제 문구 비교로 판단하고, z 부호로 임의 추론하지 마세요."
    return [
        {"role": "system", "content": NARRATION_SYSTEM},
        {"role": "user", "content": user},
    ]


def call_luxia(messages, temperature=0.3, retries=2):
    headers = {"apikey": LUXIA_API_KEY, "Content-Type": "application/json"}
    payload = {"model": LUXIA_MODEL, "messages": messages, "temperature": temperature}
    for attempt in range(retries + 1):
        try:
            resp = requests.post(LUXIA_API_URL, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print("  exception:", e, flush=True)
    return None


def narrate(deviation_features, temperature=0.3):
    if not deviation_features:
        return None
    messages = build_narration_prompt(deviation_features)
    return call_luxia(messages, temperature=temperature)
