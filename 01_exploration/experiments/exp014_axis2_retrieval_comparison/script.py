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
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

load_dotenv(_REPO + "/.env")
os.environ.setdefault(
    "HF_HOME",
    _REPO + "/.cache/hf",
)

BASE = _REPO + "/data/KGSS_결과파일"
SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUT = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/axis2_results.csv"
PROGRESS = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/axis2_progress.txt"

LUXIA_API_URL = os.getenv("LUXIA_API_URL", "https://bridge.luxiacloud.com/luxia/v1/chat")
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")
LUXIA_MODEL = os.getenv("LUXIA_MODEL", "luxia3-llm-32b-0731")
TEMPERATURE = 0.3

CORE9 = ["SEX", "AGE", "MARITAL", "EDUC", "INCOME", "EMPLY", "REGION", "URBAN", "OCC"]
YEAR = 2023
N_HELDOUT = 100
SPLIT_SEED = 42
SAMPLE_SEED = 7
N_EXAMPLES = 3

TARGETS = {
    "SATFIN": ['매우 만족', '약간 만족', '만족도 불만족도 아니다', '약간 불만족', '매우 불만족'],
    "HELPFUL": ["남을 도우려고 한다", "경우에 따라 다르다", "자신만을 위한다"],
    "PARTYLR": ['매우 진보적', '다소 진보적', '중도', '다소 보수적', '매우 보수적'],
}

with open(f"{BASE}/(2)kgss_variable_dictionary.json", encoding="utf-8") as f:
    VARDICT = json.load(f)

print("loading embedding model...", flush=True)
EMB_MODEL = SentenceTransformer("nlpai-lab/KURE-v1")


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


def profile9(row):
    p = {}
    for v in CORE9:
        dec = decode(v, row[v])
        p[v] = dec if dec else "정보 없음"
    return p


def short_desc(p9):
    return f"{p9['AGE']} {p9['SEX']}, {p9['OCC']}, 월 가구소득 {p9['INCOME']}"


def narrative(p9):
    return (f"{p9['REGION']}({p9['URBAN']})에 사는 {p9['AGE']} {p9['MARITAL']} {p9['SEX']}. "
            f"최종학력 {p9['EDUC']}, {p9['EMPLY']} 상태로 {p9['OCC']} 종사, "
            f"가구 월소득 {p9['INCOME']}.")


ANTI_DRIFT_RULES = """## 답변 규칙 (반드시 지킬 것)
1. 위 인물 본인의 시점에서 1인칭으로 답하라.
2. 본인의 실제 배경에 근거해 자연스럽게 답하라. 일반적/교과서적 지식을 끌어와 답하지 말 것.
3. 자기 의견을 단언적으로 말하라. 조건·회피·분석체 금지.
4. 당신은 AI가 아니라 위 인물 본인이다.
5. "비슷한 사람들"은 참고 예시일 뿐, 당신은 어디까지나 "기본 정보"에 묘사된 인물 본인이다.
"""
_CIRCLE = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧']


def build_system_prompt(target_profile, examples):
    ex_lines = [f"- {short_desc(p)}: \"{options.index(a)+1}번 {a}\"라고 답함" for p, a, options in examples]
    basic_info = json.dumps(target_profile, ensure_ascii=False, indent=2)
    return f"""당신은 아래 "기본 정보"에 묘사된 인물 본인입니다. 이 인물로서 1인칭으로 답하세요.
2023년 한국종합사회조사(KGSS)에 실제로 응답하는 상황입니다.

## 비슷한 사람들 (참고 — 같은 질문에 실제로 답한 다른 사람들)
{chr(10).join(ex_lines)}

## 기본 정보 (당신)
{basic_info}
{ANTI_DRIFT_RULES}"""


def build_user_prompt(target, options):
    options_str = "\n".join(f"  {_CIRCLE[i]} {opt}" for i, opt in enumerate(options))
    return f"""다음 질문에 답해주세요.

[질문 - {VARDICT[target]['label']}]
{VARDICT[target]['question']}

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


def call_luxia(messages):
    headers = {"apikey": LUXIA_API_KEY, "Content-Type": "application/json"}
    payload = {"model": LUXIA_MODEL, "messages": messages, "temperature": TEMPERATURE}
    try:
        resp = requests.post(LUXIA_API_URL, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("  exception:", e, flush=True)
    return None


all_results = []
for target, options in TARGETS.items():
    print(f"\n===== TARGET: {target} =====", flush=True)
    usecols = ["YEAR"] + CORE9 + [target]
    df, meta = pyreadstat.read_sav(SAV, usecols=usecols)
    df = df[df["YEAR"] == YEAR].copy()
    df["target_decoded"] = df[target].apply(lambda v: decode(target, v))
    pool = df[df["target_decoded"].notna()].copy()

    rng_split = random.Random(SPLIT_SEED)
    all_idx = list(pool.index)
    rng_split.shuffle(all_idx)
    n_retrieval = int(len(all_idx) * 0.7)
    retrieval_pool = pool.loc[all_idx[:n_retrieval]].copy()
    heldout_pool = pool.loc[all_idx[n_retrieval:]]

    random.seed(SAMPLE_SEED)
    sample_idx = random.sample(list(heldout_pool.index), min(N_HELDOUT, len(heldout_pool)))
    sample = heldout_pool.loc[sample_idx]

    # --- 인구통계 kNN 준비 ---
    for v in CORE9:
        retrieval_pool[v + "_dec"] = retrieval_pool[v].apply(lambda x: decode(v, x) or "정보없음")
    feat_cols = [v + "_dec" for v in CORE9]
    enc = OneHotEncoder(handle_unknown="ignore")
    X_retrieval = enc.fit_transform(retrieval_pool[feat_cols]).toarray()
    knn_demo = NearestNeighbors(n_neighbors=N_EXAMPLES).fit(X_retrieval)

    # --- 임베딩 준비: 각자의 narrative를 벡터화 ---
    retrieval_profiles = [profile9(row) for _, row in retrieval_pool.iterrows()]
    retrieval_narratives = [narrative(p) for p in retrieval_profiles]
    retrieval_emb = EMB_MODEL.encode(retrieval_narratives, normalize_embeddings=True, show_progress_bar=False)

    for i, (idx, row) in enumerate(sample.iterrows()):
        target_profile = profile9(row)

        # 방식1: 인구통계 kNN
        target_feat = enc.transform([[decode(v, row[v]) or "정보없음" for v in CORE9]]).toarray()
        _, nn_idx = knn_demo.kneighbors(target_feat)
        demo_examples = []
        for j in nn_idx[0]:
            r = retrieval_pool.iloc[j]
            demo_examples.append((profile9(r), r["target_decoded"], options))

        # 방식2: 임베딩 유사도
        target_narr = narrative(target_profile)
        target_emb = EMB_MODEL.encode([target_narr], normalize_embeddings=True)[0]
        sims = retrieval_emb @ target_emb
        top_idx = np.argsort(-sims)[:N_EXAMPLES]
        embed_examples = []
        for j in top_idx:
            r = retrieval_pool.iloc[j]
            embed_examples.append((profile9(r), r["target_decoded"], options))

        actual = row["target_decoded"]

        for arm_name, examples in [("arm_demo_knn", demo_examples), ("arm_embed_sim", embed_examples)]:
            messages = [
                {"role": "system", "content": build_system_prompt(target_profile, examples)},
                {"role": "user", "content": build_user_prompt(target, options)},
            ]
            raw = call_luxia(messages)
            parsed = parse_objective_response_robust(raw, options) if raw else {"selected_option": None, "match_method": "api_fail"}
            pred = parsed["selected_option"]
            correct = pred == actual
            dist = abs(options.index(pred) - options.index(actual)) if pred in options else len(options) - 1
            all_results.append({
                "target": target, "respid": row.get("RESPID", idx), "arm": arm_name,
                "actual": actual, "predicted": pred, "match_method": parsed["match_method"],
                "correct": correct, "distance": dist, "raw_response": raw,
            })

        if (i + 1) % 10 == 0 or (i + 1) == len(sample):
            msg = f"[{target}] [{i+1}/{len(sample)}] done"
            print(msg, flush=True)
            with open(PROGRESS, "w") as f:
                f.write(msg + "\n")
            pd.DataFrame(all_results).to_csv(OUT, index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(OUT, index=False)

print("\n\n=== axis2 결과 요약 ===", flush=True)
for target in TARGETS:
    for arm in ["arm_demo_knn", "arm_embed_sim"]:
        sub = res_df[(res_df.target == target) & (res_df.arm == arm)]
        print(f"{target} / {arm}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}")
    print()

print("=== Pooled (3 targets, n=300) ===")
for arm in ["arm_demo_knn", "arm_embed_sim"]:
    sub = res_df[res_df.arm == arm]
    print(f"{arm}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}")
