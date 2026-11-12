# exp002 config

- 입력: `exp001_leakage_calibration`의 1071개 target×candidate 쌍 (r 이미 계산됨)
- 임베딩 모델: `nlpai-lab/KURE-v1` (고려대 개발, 한국어 검색 특화, 무료·오픈소스, Hugging Face 배포) — API 키 없이 로컬 실행
- 텍스트 소스: `(2)kgss_variable_dictionary.json`의 `label` 필드 (문항 요약 라벨)
- 지표: 코사인 유사도 (정규화된 임베딩의 내적)
- 실행 환경: 로컬 venv (`scratchpad/venv`), sentence-transformers 5.6.1 / torch 2.13.0, CPU
- 스크립트: `script.py`
