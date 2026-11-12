"""
Survey-LLM-Lab 2 exp001 — 프로필 구성(1단계) 파일럿.
축2/축3 없이, 문항 하나(Q_t)를 예측할 때 LLM이 관련문항(Q_c) K=8개를 고르고,
그 문항들에 대한 타겟 본인의 실제 답을 프로필에 넣어 LLM이 Q_t 답을 "직접 생성".
2025년 데이터, 188개 후보 풀(203개에서 민감/소표본 주제 제외), n_pilot 문항 × n_resp 응답자.

조건 4개: ours(LLM-K8) / full(전체 187) / demo(인구통계만) / random(무작위 K8)
포맷 2종: A=Q&A구조(질문 원문) / B=라벨구조
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
K = 8
TEMPERATURE = 0.3
EXCLUDED_PARTS = {
    "15_법_질서_범죄", "8_여가_문화", "10_인권_소수자", "14_환경_재난",
    "18_기타_분류_애매", "7_건강_질병",
}
N_PILOT_TARGETS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N_RESP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
VARIANTS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["A", "B"]
RUN_CONDITIONS = sys.argv[4].split(",") if len(sys.argv) > 4 else [
    "mode", "demo", "peer_stat_demo", "random", "peer_stat_random", "ours", "peer_stat", "full"]
RUN_TAG = "-".join(VARIANTS) + "_" + "-".join(RUN_CONDITIONS)
PILOT_CSV = f"{OUTDIR}/output_pilot_{RUN_TAG}.csv"
TOKENS_CSV = f"{OUTDIR}/output_tokens_{RUN_TAG}.csv"

with open(QUESTIONS_JSON, encoding="utf-8") as f:
    qdata = json.load(f)
pool_all = [q for q in qdata["questions"] if q["topic_part"] not in EXCLUDED_PARTS]
print(f"후보 풀(188 기대): {len(pool_all)}개", flush=True)

# 파일럿 타겟 5개: 주제 다양하게 고정 선택 (재현성 위해 seed 고정 셔플 후 앞에서 N개, 서로 다른 주제 우선)
random.seed(7)
by_part = {}
for q in pool_all:
    by_part.setdefault(q["topic_part"], []).append(q)
parts_order = sorted(by_part.keys())
pilot_targets = []
pi = 0
while len(pilot_targets) < N_PILOT_TARGETS and pi < 5000:
    part = parts_order[pi % len(parts_order)]
    cands = [q for q in by_part[part] if q not in pilot_targets]
    if cands:
        pilot_targets.append(random.choice(cands))
    pi += 1
pilot_targets = pilot_targets[:N_PILOT_TARGETS]
print(f"파일럿 타겟 {len(pilot_targets)}개: {[q['var'] for q in pilot_targets]}", flush=True)

# ---------------- 데이터 로드 ----------------
all_cand_vars = [q["var"] for q in pool_all]
usecols = list(dict.fromkeys(["YEAR", "RESPID"] + LV.CORE9 + all_cand_vars))
df, meta = pyreadstat.read_sav(LV.SAV, usecols=usecols)
df = df[df["YEAR"] == YEAR].copy()

# ---------------- 프롬프트 빌더 ----------------
SYS_SELECT = """당신은 사회조사 문항 중 어떤 것이 특정 문항 예측에 도움이 되는지 판단하는 보조자입니다.
오직 문항 번호와 이유만 반환합니다."""

SYS_PREDICT = """당신은 사회조사 응답자의 프로필을 보고 그 사람이 특정 문항에 어떻게 답했을지 예측하는
보조자입니다. 주어진 보기 중 정확히 하나를 골라 그 문구 그대로 답하세요."""


def cand_display(q, variant):
    return f"Q: {q['question']}" if variant == "A" else q["label"]


def build_select_prompt(target_q, candidates, variant):
    """candidates는 topic_part 기준으로 정렬되어 있다고 가정(호출 전 정렬 필요).
    같은 주제 블록으로 묶어 제시해 flat list에서 형제 문항을 놓치는 lost-in-the-middle을 완화."""
    target_disp = cand_display(target_q, variant)
    blocks = []
    cur_part = None
    for i, c in enumerate(candidates):
        if c["topic_part"] != cur_part:
            cur_part = c["topic_part"]
            blocks.append(f"\n[{cur_part}]")
        blocks.append(f"{i+1}. {cand_display(c, variant)}")
    lines = "\n".join(blocks)
    return f"""[예측하려는 문항]
{target_disp}

[후보 {len(candidates)}개]
{lines}

이 중 위 문항을 예측하는 데 **실제로 도움이 된다고 판단하는 문항만** 번호로 고르고, 각각
이유를 한 줄씩 적으세요. 개수 제한은 없습니다 — 진짜 도움이 되는 게 2개면 2개만, 10개면
10개를 고르세요. 확신 없는 문항을 억지로 채우지 마세요.

**중요 — 같은 계열(배터리) 문항 쏠림 방지 (엄격 규칙)**: 후보 중에는 같은 주제를 여러 항목으로
나눠 물은 연속 문항들이 섞여 있습니다(예: 정부지출 증감을 분야별로 여러 번 묻거나, 여러 기관에
대한 신뢰를 각각 따로 묻는 경우). **같은 계열(문항 설계상 하나의 시리즈로 묶이는 배터리) 문항은
최대 2개까지만 고를 수 있습니다 — 3개 이상 고르는 것은 금지입니다.**

형식(반드시 이 형식으로, 개수는 자유):
1. 번호X — 이유
2. 번호Y — 이유
..."""


def parse_select_numbers(response, n_candidates, k=None):
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


def profile_line(q, row, variant):
    ans = LV.decode(q["var"], row.get(q["var"]))
    if ans is None:
        return None
    if variant == "A":
        return f"Q: {q['question']} A: {ans}"
    return f"{q['label']}:{ans}"


def build_predict_prompt(p9_text, context_lines, target_q, options, variant):
    ctx = " | ".join(context_lines) if context_lines else "(관련 문항 정보 없음)"
    target_disp = f"Q: {target_q['question']}" if variant == "A" else target_q["label"]
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


# ---------------- 실행 ----------------
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

    mode_val = dft["target_decoded"].mode()[0]

    eligible = [c for c in candidates]
    rng_r = random.Random(123)
    random8_qs = rng_r.sample(eligible, K)
    random8_vars = [q["var"] for q in random8_qs]

    Z_demo, stats_demo = (None, None)
    if "peer_stat_demo" in RUN_CONDITIONS:
        Z_demo, stats_demo = LV.build_pool_zscores(dft, LV.CORE9)
    Z_rand, stats_rand = (None, None)
    if "peer_stat_random" in RUN_CONDITIONS:
        Z_rand, stats_rand = LV.build_pool_zscores(dft, random8_vars)

    for variant in VARIANTS:
        print(f"  --- variant {variant} ---", flush=True)
        llm8_qs = []
        need_select = ("ours" in RUN_CONDITIONS) or ("peer_stat" in RUN_CONDITIONS)
        if need_select:
            # step1: ours/peer_stat 공용 LLM 선정 (문항당 1회, 재사용)
            sel_prompt = build_select_prompt(target_q, candidates, variant)
            content, usage = LV.call_luxia_tracked(
                [{"role": "system", "content": SYS_SELECT}, {"role": "user", "content": sel_prompt}],
                temperature=TEMPERATURE)
            token_log.append({"target": tvar, "variant": variant, "stage": "select", "respid": None, "condition": "ours",
                               **(usage or {})})
            picks = parse_select_numbers(content, len(candidates))
            llm8_qs = [candidates[p - 1] for p in picks] if picks else candidates[:K]
            print(f"  선정된 {len(llm8_qs)}개: {[q['var'] for q in llm8_qs]}", flush=True)

        peer_Z, peer_stats = (None, None)
        if "peer_stat" in RUN_CONDITIONS and llm8_qs:
            peer_vars = [q["var"] for q in llm8_qs]
            peer_Z, peer_stats = LV.build_pool_zscores(dft, peer_vars)

        for _, row in sample.iterrows():
            actual = row["target_decoded"]
            p9_text = LV.profile9_text(LV.profile9(row))

            # peer_stat: LLM 호출 없이 통계로만 계산
            if "peer_stat" in RUN_CONDITIONS and peer_Z is not None:
                peer_vars = [q["var"] for q in llm8_qs]
                dist_series_idx = LV.zscore_nearest(row, peer_Z.drop(index=row.name, errors="ignore"),
                                                     peer_stats, peer_vars, n=K, min_common=2)
                peer_answers = [dft.loc[i, "target_decoded"] for i in dist_series_idx]
                pred = LV.aggregate_median(peer_answers, options) if peer_answers else None
                correct = (pred == actual)
                distance = abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1
                all_results.append({
                    "target": tvar, "variant": variant, "condition": "peer_stat", "respid": row.get("RESPID"),
                    "actual": actual, "predicted": pred, "raw_response": f"n_peers={len(peer_answers)}",
                    "correct": correct, "distance": distance,
                })

            # mode: LLM/통계 계산 없이 그냥 최빈값
            if "mode" in RUN_CONDITIONS:
                pred = mode_val
                correct = (pred == actual)
                distance = abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1
                all_results.append({
                    "target": tvar, "variant": variant, "condition": "mode", "respid": row.get("RESPID"),
                    "actual": actual, "predicted": pred, "raw_response": "mode_baseline",
                    "correct": correct, "distance": distance,
                })

            # peer_stat_demo: 인구통계 9개 기준 통계적 최근접 이웃
            if "peer_stat_demo" in RUN_CONDITIONS and Z_demo is not None:
                idx_demo = LV.zscore_nearest(row, Z_demo.drop(index=row.name, errors="ignore"),
                                              stats_demo, LV.CORE9, n=K, min_common=2)
                ans_demo = [dft.loc[i, "target_decoded"] for i in idx_demo]
                pred = LV.aggregate_median(ans_demo, options) if ans_demo else None
                correct = (pred == actual)
                distance = abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1
                all_results.append({
                    "target": tvar, "variant": variant, "condition": "peer_stat_demo", "respid": row.get("RESPID"),
                    "actual": actual, "predicted": pred, "raw_response": f"n_peers={len(ans_demo)}",
                    "correct": correct, "distance": distance,
                })

            # peer_stat_random: 무작위 K개 문항 기준 통계적 최근접 이웃
            if "peer_stat_random" in RUN_CONDITIONS and Z_rand is not None:
                idx_rand = LV.zscore_nearest(row, Z_rand.drop(index=row.name, errors="ignore"),
                                              stats_rand, random8_vars, n=K, min_common=2)
                ans_rand = [dft.loc[i, "target_decoded"] for i in idx_rand]
                pred = LV.aggregate_median(ans_rand, options) if ans_rand else None
                correct = (pred == actual)
                distance = abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1
                all_results.append({
                    "target": tvar, "variant": variant, "condition": "peer_stat_random", "respid": row.get("RESPID"),
                    "actual": actual, "predicted": pred, "raw_response": f"n_peers={len(ans_rand)}",
                    "correct": correct, "distance": distance,
                })

            conditions = {
                "ours": llm8_qs,
                "random": random8_qs,
                "demo": [],
                "full": candidates,
            }
            conditions = {c: v for c, v in conditions.items() if c in RUN_CONDITIONS}
            for cond, ctx_qs in conditions.items():
                ctx_lines = [ln for q in ctx_qs if (ln := profile_line(q, row, variant)) is not None]
                prompt = build_predict_prompt(p9_text, ctx_lines, target_q, options, variant)
                content, usage = LV.call_luxia_tracked(
                    [{"role": "system", "content": SYS_PREDICT}, {"role": "user", "content": prompt}],
                    temperature=TEMPERATURE)
                pred = LV.match_answer(content, options)
                correct = (pred == actual)
                distance = abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1

                all_results.append({
                    "target": tvar, "variant": variant, "condition": cond, "respid": row.get("RESPID"),
                    "actual": actual, "predicted": pred, "raw_response": content,
                    "correct": correct, "distance": distance,
                })
                token_log.append({"target": tvar, "variant": variant, "stage": "predict", "respid": row.get("RESPID"),
                                   "condition": cond, **(usage or {})})

        pd.DataFrame(all_results).to_csv(PILOT_CSV, index=False)
        pd.DataFrame(token_log).to_csv(TOKENS_CSV, index=False)
        print(f"  [{tvar}][{variant}] 진행상황 저장됨", flush=True)

res_df = pd.DataFrame(all_results)
tok_df = pd.DataFrame(token_log)
res_df.to_csv(PILOT_CSV, index=False)
tok_df.to_csv(TOKENS_CSV, index=False)

print("\n\n=== 결과 ===")
for variant in VARIANTS:
    print(f"\n--- variant {variant} ---")
    for cond in ["mode", "demo", "peer_stat_demo", "random", "peer_stat_random", "ours", "peer_stat", "full"]:
        sub = res_df[(res_df.variant == variant) & (res_df.condition == cond)]
        if len(sub) == 0:
            continue
        n_parsed = sub["predicted"].notna().sum()
        print(f"{cond:8s}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  "
              f"n={len(sub)}  파싱성공={n_parsed}/{len(sub)}")

print("\n=== 토큰 사용량 ===")
tok_df_valid = tok_df.dropna(subset=["prompt_tokens"]) if "prompt_tokens" in tok_df.columns else pd.DataFrame()
if len(tok_df_valid) > 0:
    for variant in VARIANTS:
        sub = tok_df_valid[tok_df_valid.variant == variant]
        print(f"\n--- variant {variant} ---")
        print(f"총 호출 {len(sub)}회, 총 prompt_tokens={sub['prompt_tokens'].sum():.0f}, "
              f"총 completion_tokens={sub['completion_tokens'].sum():.0f}, "
              f"총 tokens={sub['total_tokens'].sum():.0f}")
        for cond in ["select"] + ["mode", "demo", "peer_stat_demo", "random", "peer_stat_random", "ours", "peer_stat", "full"]:
            s2 = sub[(sub.condition == cond) | (sub.stage == cond)]
            if len(s2) == 0:
                continue
            print(f"  {cond:8s}: 평균 prompt_tokens={s2['prompt_tokens'].mean():.0f}  n={len(s2)}")
else:
    print("토큰 정보 없음(usage 필드 누락)")
