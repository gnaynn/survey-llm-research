"""
exp015 2차 실험 (arm10): arm9(또래 이탈 narration) 프로필에 "r/s 상 타겟과 가장 유사한 문항"에 대한
이 응답자의 실제 답변을 추가로 넣었을 때 예측이 어떻게 바뀌는지 확인.

정보 누출 필터를 끈 조건을 이번 2차 실험에서 확인:
leakage 배제(|r|>=0.30 또는 s>=0.50) 없이, 오히려 그 배제 대상이었던 고유사도 문항의 실제 답을
의도적으로 프로필에 추가한다 (PARTYLR의 경우 twin 문항 LEFTRIGT 포함 가능).

유사 문항 선정: target별 exp002 153개 후보 중 |r| 상위 5개 ∪ s 상위 5개 (중복 제거).
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
PROGRESS = f"{OUTDIR}/exp2_progress.txt"

YEAR = 2023
N_HELDOUT = 100
SPLIT_SEED = 42
SAMPLE_SEED = 7
N_EXAMPLES = 3
PEER_DEF = "A1"
N_DEV_FEATS = 3
K_SIMILAR = 5  # |r| 상위 K ∪ s 상위 K (조정 가능 파라미터)
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


def build_system_prompt(basic_info_dict, examples, extra_narrative=None, similar_answers=None):
    ex_lines = [f"- {short_desc(p9)}: \"{a}\"라고 답함" for p9, a in examples]
    basic_info = json.dumps(basic_info_dict, ensure_ascii=False, indent=2)
    extra_block = f"\n## 추가 특징 (또래 대비 이탈 서술)\n{extra_narrative}\n" if extra_narrative else ""
    sim_block = ""
    if similar_answers:
        sim_lines = "\n".join(f"- [{s['label']}] \"{s['question']}\" → 본인 답변: \"{s['value']}\"" for s in similar_answers)
        sim_block = f"\n## 이 질문과 유사한(통계적으로 관련 높은) 다른 문항에 대한 본인의 실제 답변\n{sim_lines}\n"
    return f"""당신은 아래 "기본 정보"에 묘사된 인물 본인입니다. 이 인물로서 1인칭으로 답하세요.
2023년 한국종합사회조사(KGSS)에 실제로 응답하는 상황입니다.

## 비슷한 사람들 (참고 — 같은 질문에 실제로 답한 다른 사람들)
{chr(10).join(ex_lines)}

## 기본 정보 (당신)
{basic_info}
{extra_block}{sim_block}{ANTI_DRIFT_RULES}"""


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


def get_similar_vars(target, k):
    """leakage 필터 없이 |r| 상위 k ∪ s 상위 k (중복 제거) — twin 문항 포함 가능."""
    s_df = pd.read_csv(L.EXP002_CSV)
    sub = s_df[s_df.target == target].copy()
    top_r = list(sub.reindex(sub["abs_r"].sort_values(ascending=False).index).head(k)["candidate"])
    top_s = list(sub.sort_values("s", ascending=False).head(k)["candidate"])
    seen, out = set(), []
    for v in top_r + top_s:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


all_results = []

for TARGET in TARGETS:
    print(f"\n===== {TARGET} =====", flush=True)
    options = [v for k, v in L.VARDICT[TARGET]["values"].items() if float(k) not in L.missing_set(TARGET)]
    idx_map = {o: k for k, o in enumerate(options)}

    ordinal_vars, _ = L.get_ordinal_candidates(TARGET)
    cands = L.get_candidate_pool(TARGET)
    similar_vars = get_similar_vars(TARGET, K_SIMILAR)
    print("similar_vars (leakage 필터 없음):", similar_vars, flush=True)

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

    for i, (idx, row) in enumerate(sample.iterrows()):
        p9 = profile9(row)
        ex_idx = random.Random(1000 + i).sample(list(retrieval_pool.index), N_EXAMPLES)
        examples = [(profile9(retrieval_pool.loc[j]), retrieval_pool.loc[j, "target_decoded"]) for j in ex_idx]

        zscores, zmeta = L.compute_person_zscore(row, retrieval_pool, ordinal_vars, peer_def=PEER_DEF)
        feats = L.extract_deviation_features(row, zscores, eligible_set, n_top=N_DEV_FEATS)
        narrative = L.narrate(feats, temperature=TEMPERATURE) if feats else None

        similar_answers = []
        for v in similar_vars:
            val = L.decode(v, row.get(v))
            if val is not None:
                similar_answers.append({"var": v, "label": L.VARDICT.get(v, {}).get("label", v),
                                         "question": L.VARDICT.get(v, {}).get("question", ""), "value": val})

        messages = [
            {"role": "system", "content": build_system_prompt(p9, examples, extra_narrative=narrative, similar_answers=similar_answers)},
            {"role": "user", "content": build_user_prompt(TARGET, options)},
        ]
        raw = L.call_luxia(messages, temperature=TEMPERATURE)
        actual = row["target_decoded"]
        parsed = parse_objective_response_robust(raw, options) if raw else {"selected_option": None, "match_method": "api_fail"}
        pred = parsed["selected_option"]
        all_results.append({
            "target": TARGET, "arm": "arm10_peerdev_plus_similar", "respid": row.get("RESPID", idx),
            "actual": actual, "predicted": pred, "match_method": parsed["match_method"],
            "correct": pred == actual,
            "distance": abs(idx_map[pred] - idx_map[actual]) if pred in idx_map else len(options) - 1,
            "n_similar_used": len(similar_answers), "raw_response": raw, "narrative": narrative,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(sample):
            msg = f"[{TARGET}][{i+1}/{len(sample)}] done"
            print(msg, flush=True)
            with open(PROGRESS, "w") as f:
                f.write(msg + "\n")
            pd.DataFrame(all_results).to_csv(f"{OUTDIR}/output_experiment2_arm10.csv", index=False)

res_df = pd.DataFrame(all_results)
res_df.to_csv(f"{OUTDIR}/output_experiment2_arm10.csv", index=False)

# arm9(기존 파일럿) 대비 비교
pilot = pd.read_csv(f"{OUTDIR}/output_predictions_new_arms.csv")
arm9 = pilot[pilot.arm == "arm9_zscore_narrative"]

print("\n\n=== 2차 실험 결과: arm9(이탈만) vs arm10(이탈+유사문항답변) ===", flush=True)
for target in TARGETS:
    a9 = arm9[arm9.target == target]
    a10 = res_df[res_df.target == target]
    print(f"\n--- {target} ---")
    print(f"arm9_zscore_narrative : acc={a9['correct'].mean():.3f}  MAE={a9['distance'].mean():.3f}  n={len(a9)}")
    print(f"arm10_peerdev_plus_similar: acc={a10['correct'].mean():.3f}  MAE={a10['distance'].mean():.3f}  n={len(a10)}")
