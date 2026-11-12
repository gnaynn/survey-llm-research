"""
exp001 — 5점 척도 문항 40개(기존 30문항 실험의 5점 척도 11개 + 188문항 풀의 5점 척도 중
새로 뽑은 29개, seed 42) × 5명, k=1~15 전체 rankcurve. method="ours"만 (LLM 랭킹 선택 1회 +
k별 예측). 30개 실험(run_rank_curve_30q.py)과 동일한 파이프라인/응답자 샘플링(seed 42) 구조.

모든 타겟이 5점 척도로 통일되어 있어 WD 정규화 없이도 문항 간 척도 편향 없이 비교 가능.

CLI: K_VALUES(콤마구분, 기본 1~15 전체)
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()  # repo root
import json
import random
import re
import sys

import pandas as pd
import pyreadstat

sys.path.insert(0, _REPO + "/02_llm_question_selection")
import utils as LV

OUTDIR = _REPO + "/02_llm_question_selection"
QUESTIONS_JSON = _REPO + "/data/questions/2025_questions.json"

YEAR = 2025
TEMPERATURE = 0.3
N_RESP = 5
EXCLUDED_PARTS = {
    "15_법_질서_범죄", "8_여가_문화", "10_인권_소수자", "14_환경_재난",
    "18_기타_분류_애매", "7_건강_질병",
}
K_VALUES = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, 16))
K_MAX = max(K_VALUES)

TARGETS_40 = ['ABLITAPP', 'ABLITINT', 'ABLITSER', 'BETSENSE', 'CNSPIMPT', 'CNSPRESP',
              'GOVSPD1', 'GOVSPD2', 'GOVSPD3', 'GOVSPD5', 'GOVSPD8', 'HATOPSEX',
              'INSULTFR', 'INTCONFM', 'INTDFPOL', 'ISOL4WKS', 'KIDSOL06', 'LUCKEFFO',
              'OLCOMFOT', 'OTHERS1', 'OTHERS2', 'PARTYLR', 'REGGRP1', 'REGGRP2',
              'REGGRP3', 'REGGRP9', 'RELSCI2', 'SATFIN', 'SATPOL', 'SOCNET10',
              'SUCDAMBT', 'SUCDKNOW', 'SUCDPAED', 'SUCDPOL', 'SUCDREL', 'SUCDWLTH',
              'SURREQST', 'TECHEASY', 'TECHOPPO', 'USECOMMU']

RUN_TAG = f"rankcurve_40q5point_{N_RESP}r"
PILOT_CSV = f"{OUTDIR}/output_{RUN_TAG}.csv"
TOKENS_CSV = f"{OUTDIR}/output_tokens_{RUN_TAG}.csv"
RANKING_CACHE_JSON = f"{OUTDIR}/{RUN_TAG}_ours_rankings.json"

with open(QUESTIONS_JSON, encoding="utf-8") as f:
    qdata = json.load(f)
pool_all = [q for q in qdata["questions"] if q["topic_part"] not in EXCLUDED_PARTS]
var_to_q = {q["var"]: q for q in pool_all}
print(f"후보 풀: {len(pool_all)}개, 타겟 40개: {TARGETS_40}", flush=True)

all_cand_vars = [q["var"] for q in pool_all]
usecols = list(dict.fromkeys(["YEAR", "RESPID"] + LV.CORE9 + all_cand_vars))
df, meta = pyreadstat.read_sav(LV.SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()

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


import os
all_results = pd.read_csv(PILOT_CSV).to_dict("records") if os.path.exists(PILOT_CSV) else []
token_log = pd.read_csv(TOKENS_CSV).to_dict("records") if os.path.exists(TOKENS_CSV) else []
done_targets = {r["target"] for r in all_results}
rankings_cache = json.load(open(RANKING_CACHE_JSON, encoding="utf-8")) if os.path.exists(RANKING_CACHE_JSON) else {}
print(f"기존 결과 이어붙임: pilot {len(all_results)}행, 완료된 타겟 {len(done_targets)}개, "
      f"캐시된 랭킹 {len(rankings_cache)}개", flush=True)

for ti, tvar in enumerate(TARGETS_40):
    if tvar in done_targets:
        continue
    target_q = var_to_q[tvar]
    print(f"\n===== [{ti+1}/{len(TARGETS_40)}] 타겟: {tvar} ({target_q['label']}) =====", flush=True)
    options, idx_map = LV.target_options(tvar)
    candidates = [q for q in pool_all if q["var"] != tvar]
    candidates.sort(key=lambda q: q["topic_part"])

    dft = df.copy()
    dft["target_decoded"] = dft[tvar].apply(lambda v: LV.decode_stripped(tvar, v))
    dft = dft[dft["target_decoded"].notna()]
    for v in LV.CORE9:
        dft = dft[dft[v].notna()]
    sample = dft.sample(n=min(N_RESP, len(dft)), random_state=42)
    print(f"  응답자 {len(sample)}명", flush=True)

    if tvar in rankings_cache:
        ours_vars = rankings_cache[tvar]
        ours_ranked = [var_to_q[v] for v in ours_vars if v in var_to_q]
        print(f"  ours 랭킹(캐시 재사용) top-{K_MAX}: {ours_vars[:K_MAX]}", flush=True)
    else:
        sel_prompt = build_rank_select_prompt(target_q, candidates, K_MAX)
        content, usage = LV.call_luxia_tracked(
            [{"role": "system", "content": SYS_SELECT}, {"role": "user", "content": sel_prompt}],
            temperature=TEMPERATURE)
        token_log.append({"target": tvar, "stage": "select", "method": "ours", "respid": None, **(usage or {})})
        picks = parse_rank_numbers(content, len(candidates))
        ours_ranked = [candidates[p - 1] for p in picks]
        if len(ours_ranked) < K_MAX:
            picked_vars = {q["var"] for q in ours_ranked}
            ours_ranked += [q for q in candidates if q["var"] not in picked_vars][: K_MAX - len(ours_ranked)]
        rankings_cache[tvar] = [q["var"] for q in ours_ranked]
        json.dump(rankings_cache, open(RANKING_CACHE_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  ours 랭킹(신규) top-{K_MAX}: {rankings_cache[tvar][:K_MAX]}", flush=True)

    for _, row in sample.iterrows():
        actual = row["target_decoded"]
        p9_text = LV.profile9_text(LV.profile9(row))

        for k in K_VALUES:
            ctx_qs = ours_ranked[:k]
            ctx_lines = [ln for q in ctx_qs if (ln := profile_line(q, row)) is not None]
            prompt = build_predict_prompt(p9_text, ctx_lines, target_q, options)
            content, usage = LV.call_luxia_tracked(
                [{"role": "system", "content": SYS_PREDICT}, {"role": "user", "content": prompt}],
                temperature=TEMPERATURE)
            pred = LV.match_answer(content, options)
            correct = (pred == actual)
            distance = abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1
            all_results.append({
                "target": tvar, "method": "ours", "k": k, "respid": row.get("RESPID"),
                "actual": actual, "predicted": pred, "raw_response": content,
                "correct": correct, "distance": distance,
            })
            token_log.append({"target": tvar, "stage": "predict", "method": "ours", "k": k,
                               "respid": row.get("RESPID"), **(usage or {})})

    pd.DataFrame(all_results).to_csv(PILOT_CSV, index=False)
    pd.DataFrame(token_log).to_csv(TOKENS_CSV, index=False)
    print(f"  [{tvar}] 저장됨 (누적 {len(all_results)}행)", flush=True)

res_df = pd.DataFrame(all_results)
res_df.to_csv(PILOT_CSV, index=False)
pd.DataFrame(token_log).to_csv(TOKENS_CSV, index=False)

print("\n\n=== 결과 (k별) ===")
for k in K_VALUES:
    sub = res_df[res_df.k == k]
    print(f"k={k:2d}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")

tok_df = pd.DataFrame(token_log)
tok_valid = tok_df.dropna(subset=["prompt_tokens"]) if "prompt_tokens" in tok_df.columns else pd.DataFrame()
if len(tok_valid) > 0:
    print(f"\n총 호출 {len(tok_valid)}회, 총 tokens={tok_valid['total_tokens'].sum():.0f}")
