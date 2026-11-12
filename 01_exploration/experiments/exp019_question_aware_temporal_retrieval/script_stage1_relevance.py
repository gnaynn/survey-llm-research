"""
exp019 Stage 1 — 타겟 문항당 1회: 관련 문항 Top-K 선정 + 관련성 점수(1~10).
후보 풀은 ordinal(순서형)만(exp015 get_ordinal_candidates 재사용, z-score 거리 계산과 호환되도록).
배터리 쏠림 방지 + n_years는 관련성 다음 2차 기준으로만 사용하도록 프롬프트에 명시.
결과는 VARDICT에 실존하는지 검증 후 저장.
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[3].as_posix()  # repo root
import json
import random
import re
import sys

import pyreadstat
from dotenv import load_dotenv

load_dotenv(_REPO + "/.env")
sys.path.insert(0, _REPO + "/01_exploration/experiments/exp015_axis1_peer_deviation")
import lib_peer_deviation as L  # noqa: E402

SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUTDIR = _REPO + "/01_exploration/experiments/exp019_question_aware_temporal_retrieval"
TARGETS = ["SATFIN", "PARTYLR"]
K = 8
YEAR = 2023
SPLIT_SEED = 42


# 2026-08-09 정책 변경: 광범위한 r/s 통계 필터 대신, target별 확인된 twin 문항 1개만 수동 제외.
MANUAL_TWIN_EXCLUDE = {
    "PARTYLR": ["LEFTRIGT"],  # r=0.401, s=0.544 — 정치성향을 다른 문구로 재확인하는 twin 문항
    "SATFIN": [],             # 확인된 twin 없음(FINPROS는 "10년 뒤 전망"으로 시제가 달라 twin 아님) — 전부 포함
}


def get_eligible_ordinal(target):
    """ordinal 후보 전체 - target별 확인된 twin 문항만 수동 제외."""
    ordinal_vars, _ = L.get_ordinal_candidates(target)
    exclude = set(MANUAL_TWIN_EXCLUDE.get(target, []))
    return sorted(v for v in ordinal_vars if v not in exclude)

RELEVANCE_SYSTEM = """당신은 사회조사 설계 전문가입니다. 특정 문항에 대한 응답을 예측하는 데
어떤 다른 문항들이 의미적으로 관련 있는지 판단합니다."""


def build_prompt(target_label, question, candidates):
    # candidates: list of (var, label, n_years)
    lines = [f"{v}: {lab} (조사된 연도 수: {ny})" for v, lab, ny in candidates]
    cand_text = "\n".join(lines)
    return f"""[타겟 문항 — {target_label}]
{question}

[후보 문항 목록 ({len(candidates)}개, 문항명: 라벨 (조사된 연도 수))]
{cand_text}

이 타겟 문항에 대한 응답을 예측하는 데 실제로 도움이 될 문항 {K}개를 고르세요.

규칙:
1. 관련성이 최우선 기준입니다. 단순 통계적 상관이 아니라, 이 사람의 가치관·경험·상황을 통해
   타겟 문항 응답을 추론하는 데 실제로 의미 있는 연결이 있는지 판단하세요.
2. 같은 계열(예: 여러 기관에 대한 신뢰, 여러 조직 가입 여부, 여러 대상에 대한 신뢰)에서는
   대표 1~2개만 고르고, 서로 다른 주제의 문항을 다양하게 선택하세요.
3. 관련성이 비슷한 후보들 사이에서만, "조사된 연도 수"가 많은 문항을 우선하세요
   (이후 여러 연도 데이터를 검색 풀로 확장할 계획이라 다년치 조사된 문항이 유리합니다).
   단, 이건 관련성이 동률일 때만 쓰는 2차 기준입니다 — 관련성 없는 문항을 연도 수만 보고 고르지 마세요.

각 선택에 1~10 관련성 점수(10이 가장 관련 높음)와 한 줄 이유를 붙이세요.

형식(반드시 이 형식으로, {K}개):
문항명 | 점수 | 이유"""


results = {}
for TARGET in TARGETS:
    print(f"\n===== {TARGET} =====", flush=True)
    ordinal_vars = get_eligible_ordinal(TARGET)
    print(f"leakage 필터 통과한 ordinal 후보: {len(ordinal_vars)}개", flush=True)

    candidates = []
    for v in ordinal_vars:
        label = L.VARDICT[v]["label"]
        n_years = L.META.get(v, {}).get("n_years", 0)
        candidates.append((v, label, n_years))

    prompt = build_prompt(L.VARDICT[TARGET]["label"], L.VARDICT[TARGET]["question"], candidates)
    resp = L.call_luxia([{"role": "system", "content": RELEVANCE_SYSTEM}, {"role": "user", "content": prompt}])
    print(resp, flush=True)

    # 파싱: "문항명[: 라벨] | 점수 | 이유" — 마크다운 표/굵게 등 다양한 포맷 허용
    parsed = []
    for line in (resp or "").splitlines():
        m = re.match(r"^\s*[\|\*]*\s*([A-Z][A-Z0-9_]{2,})\b[^|]*\|\s*(\d+)\s*\|\s*(.+?)\s*\**\s*\|?\s*$", line)
        if m:
            var, score, reason = m.group(1), int(m.group(2)), m.group(3).strip()
            if var in L.VARDICT and var in ordinal_vars:
                parsed.append({"var": var, "score": score, "reason": reason})

    # 검증: VARDICT에 실존 + ordinal 후보에 속함 (이미 위에서 필터링됨) + 중복 제거
    seen = set()
    final = []
    for p in parsed:
        if p["var"] not in seen:
            seen.add(p["var"])
            final.append(p)
    final = final[:K]

    print(f"\n검증된 최종 선택 ({len(final)}개):", flush=True)
    for f in final:
        print(f"  {f['var']} (score={f['score']}): {f['reason']}", flush=True)

    results[TARGET] = {"raw_response": resp, "selected": final}
    with open(f"{OUTDIR}/stage1_relevance_{TARGET}.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

with open(f"{OUTDIR}/stage1_relevance_all.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n\n=== Stage 1 완료 ===")
for t, r in results.items():
    print(f"{t}: {len(r['selected'])}개 선택됨")
