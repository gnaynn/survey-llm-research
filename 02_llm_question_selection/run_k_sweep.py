"""
exp001 — k-sweep 실험. full 조건은 제외하고, 컨텍스트 개수 k를 3,5,8,12,15로 늘려가며
ours(LLM 랭킹 선택) / rag(임베딩 유사도) / random(고정 랜덤 순열) 세 방법을 비교.
각 방법은 타겟당 "최대 k" 랭킹을 한 번만 만들고, k별로 그 랭킹의 앞부분을 잘라 씀
(select/임베딩/랜덤 순열 계산은 타겟당 1회로 재사용 — k마다 다시 안 함).
demo는 k와 무관한 고정 참조 조건으로 함께 기록.

CLI: N_PILOT_TARGETS, N_RESP, K_VALUES(콤마구분, 기본 3,5,8,12,15)
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()  # repo root
import json
import random
import re
import sys

import numpy as np
import pandas as pd
import pyreadstat
from sentence_transformers import SentenceTransformer

sys.path.insert(0, _REPO + "/02_llm_question_selection")
import utils as LV

OUTDIR = _REPO + "/02_llm_question_selection"
QUESTIONS_JSON = _REPO + "/data/questions/2025_questions.json"

YEAR = 2025
TEMPERATURE = 0.3
EXCLUDED_PARTS = {
    "15_법_질서_범죄", "8_여가_문화", "10_인권_소수자", "14_환경_재난",
    "18_기타_분류_애매", "7_건강_질병",
}
N_PILOT_TARGETS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
N_RESP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
K_VALUES = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [3, 5, 8, 12, 15]
K_MAX = max(K_VALUES)

RUN_TAG = f"ksweep_{'-'.join(map(str, K_VALUES))}_{N_PILOT_TARGETS}q{N_RESP}r"
PILOT_CSV = f"{OUTDIR}/output_{RUN_TAG}.csv"
TOKENS_CSV = f"{OUTDIR}/output_tokens_{RUN_TAG}.csv"

with open(QUESTIONS_JSON, encoding="utf-8") as f:
    qdata = json.load(f)
pool_all = [q for q in qdata["questions"] if q["topic_part"] not in EXCLUDED_PARTS]
print(f"후보 풀(188 기대): {len(pool_all)}개", flush=True)

# 이전 topic-cycling(seed 7)으로 뽑은 15개 샘플이 demo에 유리하게 편향된 게 확인되어,
# 순수 무작위 샘플링으로 교체(대표성 확보 목적, seed 다르게 사용해 이전 샘플과 구분).
random.seed(42)
pilot_targets = random.sample(pool_all, min(N_PILOT_TARGETS, len(pool_all)))
print(f"파일럿 타겟 {len(pilot_targets)}개(무작위 샘플): {[q['var'] for q in pilot_targets]}", flush=True)

all_cand_vars = [q["var"] for q in pool_all]
usecols = list(dict.fromkeys(["YEAR", "RESPID"] + LV.CORE9 + all_cand_vars))
df, meta = pyreadstat.read_sav(LV.SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()

print("임베딩 모델 로드 중(KURE-v1)...", flush=True)
EMB_MODEL = SentenceTransformer("nlpai-lab/KURE-v1")
# 188개 후보 질문 임베딩은 한 번만 계산해서 재사용
ALL_QUESTIONS = [q["question"] for q in pool_all]
ALL_EMB = EMB_MODEL.encode(ALL_QUESTIONS, normalize_embeddings=True, show_progress_bar=False)
VAR_TO_EMB_IDX = {q["var"]: i for i, q in enumerate(pool_all)}
print("임베딩 완료", flush=True)

SYS_SELECT = """당신은 사회조사 문항 중 어떤 것이 특정 문항 예측에 도움이 되는지 판단하는 보조자입니다.
오직 문항 번호와 이유만 반환합니다."""

SYS_PREDICT = """당신은 사회조사 응답자의 프로필을 보고 그 사람이 특정 문항에 어떻게 답했을지 예측하는
보조자입니다. 주어진 보기 중 정확히 하나를 골라 그 문구 그대로 답하세요."""


def cand_display(q):
    return f"Q: {q['question']}"


def build_rank_select_prompt(target_q, candidates, k_max):
    target_disp = cand_display(target_q)
    blocks = []
    cur_part = None
    for i, c in enumerate(candidates):
        if c["topic_part"] != cur_part:
            cur_part = c["topic_part"]
            blocks.append(f"\n[{cur_part}]")
        blocks.append(f"{i+1}. {cand_display(c)}")
    lines = "\n".join(blocks)
    return f"""[예측하려는 문항]
{target_disp}

[후보 {len(candidates)}개]
{lines}

이 중 위 문항을 예측하는 데 **도움이 되는 순서대로** 최대 {k_max}개까지 번호를 고르고,
각각 이유를 한 줄씩 적으세요. 가장 도움이 되는 것을 1번으로, 그다음을 2번으로... 순서대로
나열하세요. 도움이 되는 게 {k_max}개보다 적다고 판단되면 그만큼만 적어도 됩니다.

**중요 — 같은 계열(배터리) 문항 쏠림 방지 (엄격 규칙)**: 후보 중에는 같은 주제를 여러 항목으로
나눠 물은 연속 문항들이 섞여 있습니다(예: 정부지출 증감을 분야별로 여러 번 묻거나, 여러 기관에
대한 신뢰를 각각 따로 묻는 경우, 지난 한 주간의 감정을 여러 감정별로 묻는 경우). **같은 계열(문항
설계상 하나의 시리즈로 묶이는 배터리) 문항은 순위 안에 최대 2개까지만 포함할 수 있습니다 — 3개
이상 포함하는 것은 금지입니다.** 같은 배터리에서 2개를 넘겨 고르고 싶어지면, 그 대신 다른 관점에서
도움이 될 만한 문항을 찾아 다양성을 확보하세요.

형식(반드시 이 형식으로, 순위 순서대로):
1위. 번호X — 이유
2위. 번호Y — 이유
..."""


def parse_rank_numbers(response, n_candidates):
    if not response:
        return []
    nums = re.findall(r"번호\s*(\d+)", response)
    seen, out = [], []
    for n in nums:
        v = int(n)
        if 1 <= v <= n_candidates and v not in seen:
            seen.append(v)
            out.append(v)
    return out


def profile_line(q, row):
    ans = LV.decode(q["var"], row.get(q["var"]))
    if ans is None:
        return None
    return f"Q: {q['question']} A: {ans}"


def build_predict_prompt(p9_text, context_lines, target_q, options):
    ctx = " | ".join(context_lines) if context_lines else "(관련 문항 정보 없음)"
    target_disp = f"Q: {target_q['question']}"
    options_str = "\n".join(f"  - {o}" for o in options)
    return f"""[타겟 인물]
{p9_text}
| {ctx}

[예측하려는 문항]
{target_disp}

[보기]
{options_str}

타겟 인물이 이 문항에 어떻게 답했을지, 위 보기 중 정확히 하나를 그대로 골라 답하세요.
다른 설명 없이 보기 문구만 출력하세요."""


all_results = []
token_log = []

for target_q in pilot_targets:
    tvar = target_q["var"]
    print(f"\n===== 타겟: {tvar} ({target_q['label']}) =====", flush=True)
    options, idx_map = LV.target_options(tvar)
    candidates = [q for q in pool_all if q["var"] != tvar]
    candidates.sort(key=lambda q: q["topic_part"])

    dft = df.copy()
    dft["target_decoded"] = dft[tvar].apply(lambda v: LV.decode_stripped(tvar, v))
    dft = dft[dft["target_decoded"].notna()]
    for v in LV.CORE9:
        dft = dft[dft[v].notna()]
    sample = dft.sample(n=min(N_RESP, len(dft)), random_state=42)
    print(f"응답자 {len(sample)}명 (전체 유효 {len(dft)}명 중)", flush=True)

    # ---- ours: 랭킹 select 1회 ----
    sel_prompt = build_rank_select_prompt(target_q, candidates, K_MAX)
    content, usage = LV.call_luxia_tracked(
        [{"role": "system", "content": SYS_SELECT}, {"role": "user", "content": sel_prompt}],
        temperature=TEMPERATURE)
    token_log.append({"target": tvar, "stage": "select", "method": "ours", "respid": None, **(usage or {})})
    picks = parse_rank_numbers(content, len(candidates))
    ours_ranked = [candidates[p - 1] for p in picks]
    if len(ours_ranked) < K_MAX:
        # 부족하면 나머지는 남은 후보 순서대로 채움(k별 슬라이스가 항상 K_MAX개 확보되도록)
        picked_vars = {q["var"] for q in ours_ranked}
        ours_ranked += [q for q in candidates if q["var"] not in picked_vars][: K_MAX - len(ours_ranked)]
    print(f"  ours 랭킹 top-{K_MAX}: {[q['var'] for q in ours_ranked[:K_MAX]]}", flush=True)

    # ---- rag: 임베딩 코사인 유사도 랭킹 ----
    target_emb = EMB_MODEL.encode([target_q["question"]], normalize_embeddings=True, show_progress_bar=False)[0]
    cand_idx = [VAR_TO_EMB_IDX[q["var"]] for q in candidates]
    cand_emb = ALL_EMB[cand_idx]
    sims = cand_emb @ target_emb
    order = np.argsort(-sims)
    rag_ranked = [candidates[i] for i in order[:K_MAX]]
    print(f"  rag 랭킹 top-{K_MAX}: {[q['var'] for q in rag_ranked]}", flush=True)

    # ---- random: 고정 랜덤 순열 ----
    rng_r = random.Random(123)
    random_ranked = rng_r.sample(candidates, K_MAX)

    for _, row in sample.iterrows():
        actual = row["target_decoded"]
        p9_text = LV.profile9_text(LV.profile9(row))

        # demo: k 무관 고정 조건
        prompt = build_predict_prompt(p9_text, [], target_q, options)
        content, usage = LV.call_luxia_tracked(
            [{"role": "system", "content": SYS_PREDICT}, {"role": "user", "content": prompt}],
            temperature=TEMPERATURE)
        pred = LV.match_answer(content, options)
        all_results.append({
            "target": tvar, "method": "demo", "k": 0, "respid": row.get("RESPID"),
            "actual": actual, "predicted": pred, "raw_response": content,
            "correct": pred == actual,
            "distance": abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1,
        })
        token_log.append({"target": tvar, "stage": "predict", "method": "demo", "k": 0,
                           "respid": row.get("RESPID"), **(usage or {})})

        for method, ranked in [("ours", ours_ranked), ("rag", rag_ranked), ("random", random_ranked)]:
            for k in K_VALUES:
                ctx_qs = ranked[:k]
                ctx_lines = [ln for q in ctx_qs if (ln := profile_line(q, row)) is not None]
                prompt = build_predict_prompt(p9_text, ctx_lines, target_q, options)
                content, usage = LV.call_luxia_tracked(
                    [{"role": "system", "content": SYS_PREDICT}, {"role": "user", "content": prompt}],
                    temperature=TEMPERATURE)
                pred = LV.match_answer(content, options)
                correct = (pred == actual)
                distance = abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1
                all_results.append({
                    "target": tvar, "method": method, "k": k, "respid": row.get("RESPID"),
                    "actual": actual, "predicted": pred, "raw_response": content,
                    "correct": correct, "distance": distance,
                })
                token_log.append({"target": tvar, "stage": "predict", "method": method, "k": k,
                                   "respid": row.get("RESPID"), **(usage or {})})

    pd.DataFrame(all_results).to_csv(PILOT_CSV, index=False)
    pd.DataFrame(token_log).to_csv(TOKENS_CSV, index=False)
    print(f"  [{tvar}] 저장됨", flush=True)

res_df = pd.DataFrame(all_results)
res_df.to_csv(PILOT_CSV, index=False)
pd.DataFrame(token_log).to_csv(TOKENS_CSV, index=False)

print("\n\n=== 결과 (method x k) ===")
print(f"{'method':8s} {'k':>4s} {'acc':>8s} {'MAE':>8s} {'n':>6s}")
sub = res_df[res_df.method == "demo"]
print(f"{'demo':8s} {'-':>4s} {sub['correct'].mean():8.3f} {sub['distance'].mean():8.3f} {len(sub):6d}")
for method in ["ours", "rag", "random"]:
    for k in K_VALUES:
        sub = res_df[(res_df.method == method) & (res_df.k == k)]
        if len(sub) == 0:
            continue
        print(f"{method:8s} {k:4d} {sub['correct'].mean():8.3f} {sub['distance'].mean():8.3f} {len(sub):6d}")

print("\n=== 토큰 사용량 ===")
tok_df = pd.DataFrame(token_log)
tok_df_valid = tok_df.dropna(subset=["prompt_tokens"]) if "prompt_tokens" in tok_df.columns else pd.DataFrame()
if len(tok_df_valid) > 0:
    print(f"총 호출 {len(tok_df_valid)}회, 총 tokens={tok_df_valid['total_tokens'].sum():.0f}")
