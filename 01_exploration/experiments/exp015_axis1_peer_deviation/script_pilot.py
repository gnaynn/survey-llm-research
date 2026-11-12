"""
exp015 파일럿: 또래집단 대비 이탈(z-score) 기반 프로필 — arm8(raw z) vs arm9(LLM narration).
타겟: SATFIN, PARTYLR (HELPFUL 제외).
peer_def=A1(연령±5+성별+학력), N=3, n_heldout=100 — exp009/exp011과 동일 split/sample seed 재사용
(같은 100명에 대한 arm0/arm1/arm2 결과를 기존 CSV에서 그대로 재사용하기 위함, 신규 LLM 호출은 arm8/arm9만).
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import random
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pyreadstat
from dotenv import load_dotenv

load_dotenv(_REPO + "/.env")

sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation"
PROGRESS = f"{OUTDIR}/progress.txt"

YEAR = 2023
N_HELDOUT = 100
SPLIT_SEED = 42
SAMPLE_SEED = 7
N_EXAMPLES = 3
PEER_DEF = "A1"
N_DEV_FEATS = 3
TEMPERATURE = 0.3

TARGETS = ["SATFIN", "PARTYLR"]
CORE9 = L.CORE9

ANTI_DRIFT_RULES = """## 답변 규칙 (반드시 지킬 것)
1. 위 인물 본인의 시점에서 1인칭으로 답하라.
2. 본인의 실제 배경에 근거해 자연스럽게 답하라. 일반적/교과서적 지식을 끌어와 답하지 말 것.
3. 자기 의견을 단언적으로 말하라. 조건·회피·분석체 금지.
4. 당신은 AI가 아니라 위 인물 본인이다.
5. "비슷한 사람들"은 참고 예시일 뿐, 당신은 어디까지나 "기본 정보"에 묘사된 인물 본인이다.
"""
_CIRCLE = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧']
CIRCLE_NUMS = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5, '⑥': 6, '⑦': 7, '⑧': 8}


def profile9(row):
    p = {}
    for v in CORE9:
        dec = L.decode(v, row[v])
        p[v] = dec if dec else "정보 없음"
    return p


def short_desc(p9):
    return f"{p9['AGE']} {p9['SEX']}, {p9['OCC']}, 월 가구소득 {p9['INCOME']}"


def build_system_prompt(basic_info_dict_or_text, examples, extra_narrative=None):
    ex_lines = [f"- {short_desc(p9)}: \"{a}\"라고 답함" for p9, a in examples]
    if isinstance(basic_info_dict_or_text, dict):
        basic_info = json.dumps(basic_info_dict_or_text, ensure_ascii=False, indent=2)
    else:
        basic_info = basic_info_dict_or_text
    extra_block = f"\n## 추가 특징\n{extra_narrative}\n" if extra_narrative else ""
    return f"""당신은 아래 "기본 정보"에 묘사된 인물 본인입니다. 이 인물로서 1인칭으로 답하세요.
2023년 한국종합사회조사(KGSS)에 실제로 응답하는 상황입니다.

## 비슷한 사람들 (참고 — 같은 질문에 실제로 답한 다른 사람들)
{chr(10).join(ex_lines)}

## 기본 정보 (당신)
{basic_info}
{extra_block}{ANTI_DRIFT_RULES}"""


def build_user_prompt(target):
    options = TARGET_OPTIONS[target]
    options_str = "\n".join(f"  {_CIRCLE[i]} {opt}" for i, opt in enumerate(options))
    return f"""다음 질문에 답해주세요.

[질문 - {L.VARDICT[target]['label']}]
{L.VARDICT[target]['question']}

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


TARGET_OPTIONS = {}
all_results = []
audit_rows = []

for TARGET in TARGETS:
    print(f"\n===== {TARGET} =====", flush=True)
    TARGET_OPTIONS[TARGET] = [v for k, v in L.VARDICT[TARGET]["values"].items() if float(k) not in L.missing_set(TARGET)]

    ordinal_vars, audit_df = L.get_ordinal_candidates(TARGET, audit_path=f"{OUTDIR}/audit_ordinal_{TARGET}.csv")
    print(f"ordinal candidates: {len(ordinal_vars)} / {len(L.get_candidate_pool(TARGET))}", flush=True)

    cands = L.get_candidate_pool(TARGET)
    usecols = ["YEAR", "RESPID"] + CORE9 + [TARGET] + cands
    usecols = list(dict.fromkeys(usecols))
    print(f"loading {len(usecols)} columns...", flush=True)
    df, meta = pyreadstat.read_sav(SAV, usecols=usecols)
    df = df[df["YEAR"] == YEAR].copy()
    df["target_decoded"] = df[TARGET].apply(lambda v: L.decode(TARGET, v))
    pool = df[df["target_decoded"].notna()].copy()

    rng_split = random.Random(SPLIT_SEED)
    all_idx = list(pool.index)
    rng_split.shuffle(all_idx)
    n_retrieval = int(len(all_idx) * 0.7)
    retrieval_pool = pool.loc[all_idx[:n_retrieval]]
    heldout_pool = pool.loc[all_idx[n_retrieval:]]
    print(f"retrieval_pool={len(retrieval_pool)}, heldout_pool={len(heldout_pool)}", flush=True)

    random.seed(SAMPLE_SEED)
    sample_idx = random.sample(list(heldout_pool.index), min(N_HELDOUT, len(heldout_pool)))
    sample = heldout_pool.loc[sample_idx]
    print(f"evaluating n={len(sample)}", flush=True)

    # arm_mode: retrieval_pool 최빈값 (LLM 미사용)
    mode_val = retrieval_pool["target_decoded"].mode()[0]

    # leakage 필터 통과한 ordinal 후보만 이탈특징 후보로 사용
    eligible = L.recompute_eligible_pool(TARGET, retrieval_pool, candidate_vars=ordinal_vars)
    eligible_set = set(eligible["candidate"])
    print(f"eligible(ordinal, leakage-filtered) candidates: {len(eligible_set)}", flush=True)

    for i, (idx, row) in enumerate(sample.iterrows()):
        p9 = profile9(row)
        ex_idx = random.Random(1000 + i).sample(list(retrieval_pool.index), N_EXAMPLES)
        examples = [(profile9(retrieval_pool.loc[j]), retrieval_pool.loc[j, "target_decoded"]) for j in ex_idx]

        zscores, zmeta = L.compute_person_zscore(row, retrieval_pool, ordinal_vars, peer_def=PEER_DEF)
        feats = L.extract_deviation_features(row, zscores, eligible_set, n_top=N_DEV_FEATS)
        audit_rows.append({
            "target": TARGET, "respid": row.get("RESPID", idx), "peer_n": zmeta["peer_n"],
            "fallback_level": zmeta["fallback_level"], "n_features": len(feats),
            "features": json.dumps(feats, ensure_ascii=False),
        })

        # ---- arm_mode (LLM 미사용) ----
        actual = row["target_decoded"]
        options = TARGET_OPTIONS[TARGET]
        idx_map = {o: k for k, o in enumerate(options)}
        pred_mode = mode_val
        all_results.append({
            "target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm_mode",
            "actual": actual, "predicted": pred_mode, "match_method": "mode",
            "correct": pred_mode == actual,
            "distance": abs(idx_map.get(pred_mode, len(options) - 1) - idx_map[actual]),
            "raw_response": None,
        })

        # ---- arm8: raw z-score (narration 없음) ----
        arm8_profile = dict(p9)
        if feats:
            arm8_profile["또래_대비_이탈_문항"] = [
                {"문항": f["label"], "본인응답": f["respondent_value"], "또래평균": f["peer_mean_label"], "이례성|z|": abs(f["z"])}
                for f in feats
            ]
        messages8 = [
            {"role": "system", "content": build_system_prompt(arm8_profile, examples)},
            {"role": "user", "content": build_user_prompt(TARGET)},
        ]
        raw8 = L.call_luxia(messages8, temperature=TEMPERATURE)
        parsed8 = parse_objective_response_robust(raw8, options) if raw8 else {"selected_option": None, "match_method": "api_fail"}
        pred8 = parsed8["selected_option"]
        all_results.append({
            "target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm8_zscore_raw",
            "actual": actual, "predicted": pred8, "match_method": parsed8["match_method"],
            "correct": pred8 == actual,
            "distance": abs(idx_map[pred8] - idx_map[actual]) if pred8 in idx_map else len(options) - 1,
            "raw_response": raw8,
        })

        # ---- arm9: LLM narration ----
        narrative = L.narrate(feats, temperature=TEMPERATURE) if feats else None
        messages9 = [
            {"role": "system", "content": build_system_prompt(p9, examples, extra_narrative=narrative)},
            {"role": "user", "content": build_user_prompt(TARGET)},
        ]
        raw9 = L.call_luxia(messages9, temperature=TEMPERATURE)
        parsed9 = parse_objective_response_robust(raw9, options) if raw9 else {"selected_option": None, "match_method": "api_fail"}
        pred9 = parsed9["selected_option"]
        all_results.append({
            "target": TARGET, "respid": row.get("RESPID", idx), "arm": "arm9_zscore_narrative",
            "actual": actual, "predicted": pred9, "match_method": parsed9["match_method"],
            "correct": pred9 == actual,
            "distance": abs(idx_map[pred9] - idx_map[actual]) if pred9 in idx_map else len(options) - 1,
            "raw_response": raw9, "narrative": narrative,
        })

        if (i + 1) % 5 == 0 or (i + 1) == len(sample):
            msg = f"[{TARGET}][{i+1}/{len(sample)}] respid={row.get('RESPID', idx)} done"
            print(msg, flush=True)
            with open(PROGRESS, "w") as f:
                f.write(msg + "\n")
            pd.DataFrame(all_results).to_csv(f"{OUTDIR}/output_predictions_new_arms.csv", index=False)
            pd.DataFrame(audit_rows).to_csv(f"{OUTDIR}/output_deviation_audit.csv", index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_predictions_new_arms.csv", index=False)
pd.DataFrame(audit_rows).to_csv(f"{OUTDIR}/output_deviation_audit.csv", index=False)

# ---- 기존 arm0/arm1/arm2 결과 재사용 (동일 seed/heldout이므로 respid 매칭 가능) ----
OLD_CSV = {
    "SATFIN": _REPO + "/01_exploration/experiments/exp009_phase1_satfin_n100/output_predictions.csv",
    "PARTYLR": _REPO + "/01_exploration/experiments/exp011_phase1_partylr_n100/output_predictions.csv",
}
old_frames = []
for t, path in OLD_CSV.items():
    d = pd.read_csv(path)
    d["target"] = t
    old_frames.append(d[d.arm.isin(["arm0_baseline", "arm1_corr", "arm2_embed"])])
old_df = pd.concat(old_frames, ignore_index=True)
old_df = old_df[["target", "respid", "arm", "actual", "predicted", "match_method", "correct", "distance", "raw_response"]]

combined = pd.concat([res_df, old_df], ignore_index=True)
combined.to_csv(f"{OUTDIR}/output_predictions_combined.csv", index=False)

print("\n\n=== exp015 파일럿 최종 요약 (n=100 x 2 targets) ===", flush=True)
for target in TARGETS:
    print(f"\n--- {target} ---")
    for arm in ["arm0_baseline", "arm_mode", "arm1_corr", "arm2_embed", "arm8_zscore_raw", "arm9_zscore_narrative"]:
        sub = combined[(combined.target == target) & (combined.arm == arm)]
        if len(sub) == 0:
            continue
        acc = sub["correct"].mean()
        mae = sub["distance"].mean()
        n_valid = sub["predicted"].notna().sum()
        print(f"{arm}: acc={acc:.3f}  MAE={mae:.3f}  n={len(sub)} (valid={n_valid})")

print("\n--- pooled (2 targets) ---")
for arm in ["arm0_baseline", "arm_mode", "arm1_corr", "arm2_embed", "arm8_zscore_raw", "arm9_zscore_narrative"]:
    sub = combined[combined.arm == arm]
    if len(sub) == 0:
        continue
    print(f"{arm}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")
