# exp001 config

- 데이터: `Survey-LLM-Lab/KGSS데이터/KGSS_원본파일/kor_data_CUM0074_V2.sav`
- 참조 산출물: `(2)kgss_variable_dictionary.json`, `(7)kgss_variable_metadata.json`
- 연도: 2023, n=1230
- 타겟 7개: CANTRUST, BEFAIR, HELPFUL, PARTYLR, CURGOV, SATPOL, SATFIN
- 후보 풀: 타겟과 같은 18파트 837개 + 배경 대조군(다른 파트 무작위 300개, seed=42)
- 지표: Spearman ρ, pairwise dropna, 최소 n=30
- 실행 환경: 로컬 venv (`scratchpad/venv`), pandas 3.0.5 / pyreadstat 1.3.5 / scipy 1.17.1
- 스크립트: `script.py` (재실행 시 경로만 맞으면 그대로 재현 가능)
