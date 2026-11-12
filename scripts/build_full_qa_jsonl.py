"""
(9)kgss_all_respondents_full_qa.jsonl 생성 스크립트.
(8)kgss_all_respondents_core_profile.jsonl과 동일하게 응답자×연도 1줄 구조를 확장해,
9개 핵심 인구통계뿐 아니라 (2)변수사전의 전체 문항(QUALPRES 등 no_valid_values 제외)에 대해
그 해에 실제 조사된 문항만 담는다.

결측 두 종류 구분:
- 그 해 미조사(문항 자체가 그 연도에 없었음): 행에서 아예 생략 (문항의 "years" 필드로 판별,
  (2)kgss_variable_dictionary.json과 대조하면 100% 복원 가능 — 2007/2013/2014년 SATFIN 전원 IAP였음을 검증 완료)
- 조사는 됐지만 응답자가 무응답/DK: "무응답"으로 명시
"""
_REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()  # repo root
import json

import pyreadstat

BASE = _REPO + "/data/KGSS_결과파일"
SAV = _REPO + "/data/KGSS_원본파일/kor_data_CUM0074_V2.sav"
OUT = f"{BASE}/(9)kgss_all_respondents_full_qa.jsonl"

with open(f"{BASE}/(2)kgss_variable_dictionary.json", encoding="utf-8") as f:
    VARDICT = json.load(f)
with open(f"{BASE}/(7)kgss_variable_metadata.json", encoding="utf-8") as f:
    META = json.load(f)

ALLVARS = [v for v in VARDICT.keys() if META.get(v, {}).get("response_format") != "no_valid_values"]
print(f"포함 문항 수: {len(ALLVARS)} (전체 {len(VARDICT)}개 중 no_valid_values 제외)")

MISSING_SETS = {}
YEARS_SETS = {}
for v in ALLVARS:
    ms = set()
    for c in VARDICT.get(v, {}).get("missing_codes", []):
        try:
            ms.add(float(c))
        except ValueError:
            pass
    MISSING_SETS[v] = ms
    YEARS_SETS[v] = set(VARDICT.get(v, {}).get("years", []))

print("loading .sav (전체 문항 컬럼)...", flush=True)
df, meta = pyreadstat.read_sav(SAV, usecols=["YEAR", "RESPID"] + ALLVARS)
print(f"loaded shape: {df.shape}", flush=True)

n_written = 0
with open(OUT, "w", encoding="utf-8") as f:
    for year, g in df.groupby("YEAR"):
        year_int = int(year)
        year_str = str(year_int)
        # 그 해에 실제 조사된 문항만 미리 추려 반복 비용을 줄임
        fielded_vars = [v for v in ALLVARS if year_str in YEARS_SETS[v]]
        print(f"{year_int}: n={len(g)}, fielded_vars={len(fielded_vars)}", flush=True)

        values_lookup = {v: VARDICT[v]["values"] for v in fielded_vars}
        missing_lookup = {v: MISSING_SETS[v] for v in fielded_vars}

        for row in g.itertuples(index=False):
            row_d = row._asdict()
            respid = row_d["RESPID"]
            profile = {}
            for v in fielded_vars:
                val = row_d[v]
                if val is None or val != val or val in missing_lookup[v]:  # val!=val -> NaN 체크(pandas)
                    profile[v] = "무응답"
                    continue
                label = values_lookup[v].get(str(int(val)))
                profile[v] = label if label is not None else str(val)
            record = {
                "respondent_id": f"{year_int}{int(respid):05d}",
                "year": year_int,
                "profile": profile,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

print(f"\n완료: {n_written}줄 작성 -> {OUT}", flush=True)
