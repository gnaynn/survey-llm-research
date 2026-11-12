"""
exp015 N-스윕: 이탈 특징 개수 N=1/2/5에 대해 arm9(narration)만 재실행.
N=3은 script_pilot.py 결과를 재사용(동일 seed/heldout이므로 그대로 곡선에 포함).
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import random
import re
import sys

import pandas as pd
import pyreadstat
from dotenv import load_dotenv

load_dotenv(_REPO + "/.env")

sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation"
PROGRESS = f"{OUTDIR}/nsweep_progress.txt"

YEAR = 2023
N_HELDOUT = 100
SPLIT_SEED = 42
SAMPLE_SEED = 7
N_EXAMPLES = 3
PEER_DEF = "A1"
TEMPERATURE = 0.3
N_VALUES = [1, 2, 5]  # N=3은 script_pilot.py(arm9_zscore_narrative)에서 재사용

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


def build_system_prompt(basic_info_dict, examples, extra_narrative=None):
    ex_lines = [f"- {short_desc(p9)}: \"{a}\"라고 답함" for p9, a in examples]
    basic_info = json.dumps(basic_info_dict, ensure_ascii=False, indent=2)
    extra_block = f"\n## 추가 특징\n{extra_narrative}\n" if extra_narrative else ""
    return f"""당신은 아래 "기본 정보"에 묘사된 인물 본인입니다. 이 인물로서 1인칭으로 답하세요.
2023년 한국종합사회조사(KGSS)에 실제로 응답하는 상황입니다.

## 비슷한 사람들 (참고 — 같은 질문에 실제로 답한 다른 사람들)
{chr(10).join(ex_lines)}

## 기본 정보 (당신)
{basic_info}
{extra_block}{ANTI_DRIFT_RULES}"""


def build_user_prompt(target, options):
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


all_results = []

for TARGET in TARGETS:
    print(f"\n===== {TARGET} =====", flush=True)
    options = [v for k, v in L.VARDICT[TARGET]["values"].items() if float(k) not in L.missing_set(TARGET)]

    ordinal_vars, _ = L.get_ordinal_candidates(TARGET)
    cands = L.get_candidate_pool(TARGET)
    usecols = ["YEAR", "RESPID"] + CORE9 + [TARGET] + cands
    usecols = list(dict.fromkeys(usecols))
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

    random.seed(SAMPLE_SEED)
    sample_idx = random.sample(list(heldout_pool.index), min(N_HELDOUT, len(heldout_pool)))
    sample = heldout_pool.loc[sample_idx]

    eligible = L.recompute_eligible_pool(TARGET, retrieval_pool, candidate_vars=ordinal_vars)
    eligible_set = set(eligible["candidate"])

    idx_map = {o: k for k, o in enumerate(options)}

    for n_top in N_VALUES:
        print(f"-- N={n_top} --", flush=True)
        for i, (idx, row) in enumerate(sample.iterrows()):
            p9 = profile9(row)
            ex_idx = random.Random(1000 + i).sample(list(retrieval_pool.index), N_EXAMPLES)
            examples = [(profile9(retrieval_pool.loc[j]), retrieval_pool.loc[j, "target_decoded"]) for j in ex_idx]

            zscores, zmeta = L.compute_person_zscore(row, retrieval_pool, ordinal_vars, peer_def=PEER_DEF)
            feats = L.extract_deviation_features(row, zscores, eligible_set, n_top=n_top)
            narrative = L.narrate(feats, temperature=TEMPERATURE) if feats else None

            messages = [
                {"role": "system", "content": build_system_prompt(p9, examples, extra_narrative=narrative)},
                {"role": "user", "content": build_user_prompt(TARGET, options)},
            ]
            raw = L.call_luxia(messages, temperature=TEMPERATURE)
            actual = row["target_decoded"]
            parsed = parse_objective_response_robust(raw, options) if raw else {"selected_option": None, "match_method": "api_fail"}
            pred = parsed["selected_option"]
            all_results.append({
                "target": TARGET, "N": n_top, "respid": row.get("RESPID", idx),
                "actual": actual, "predicted": pred, "match_method": parsed["match_method"],
                "correct": pred == actual,
                "distance": abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1,
                "n_features_used": len(feats), "raw_response": raw, "narrative": narrative,
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(sample):
                msg = f"[{TARGET}][N={n_top}][{i+1}/{len(sample)}] done"
                print(msg, flush=True)
                with open(PROGRESS, "w") as f:
                    f.write(msg + "\n")
                pd.DataFrame(all_results).to_csv(f"{OUTDIR}/output_nsweep.csv", index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_nsweep.csv", index=False)

# N=3은 기존 파일럿 arm9 결과에서 가져와 곡선에 합침
pilot = pd.read_csv(f"{OUTDIR}/output_predictions_new_arms.csv")
pilot9 = pilot[pilot.arm == "arm9_zscore_narrative"].copy()
pilot9["N"] = 3
res_df = pd.concat([res_df, pilot9[["target", "N", "respid", "actual", "predicted", "correct", "distance"]]], ignore_index=True)
res_df.to_csv(f"{OUTDIR}/output_nsweep_with_n3.csv", index=False)

print("\n\n=== N별 정확도 곡선 (arm9 narration 기준) ===", flush=True)
for target in TARGETS:
    print(f"\n--- {target} ---")
    for n in sorted(set(N_VALUES) | {3}):
        sub = res_df[(res_df.target == target) & (res_df.N == n)]
        if len(sub) == 0:
            continue
        print(f"N={n}: acc={sub['correct'].mean():.3f}  MAE={sub['distance'].mean():.3f}  n={len(sub)}")
