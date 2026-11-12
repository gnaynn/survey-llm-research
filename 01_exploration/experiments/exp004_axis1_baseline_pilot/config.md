# exp004 config

- 조건: 실험 설계의 **arm 0 (baseline, 9개 핵심 프로필만)** — RAG/few-shot 없이 순수 zero-shot
- 타겟 문항: `SATFIN` (가계 상태 만족도, 5점 척도) — 2023년 KGSS
- 표본: 2023년 SATFIN 유효 응답자 1227명 중 무작위 20명 (seed=7)
- 프로필: 9개 핵심 변수(SEX/AGE/MARITAL/EDUC/INCOME/EMPLY/REGION/URBAN/OCC) 실제 값 디코딩
- 프롬프트 구조: `persona-dashboard`(sechun4686) `backend/prompts.py`의 system/user 프롬프트 패턴 + 1인칭 답변 규칙 그대로 적용, persona 서술형 필드(직업일상/가치관 등)는 KGSS에 없어 제외
- 응답 파싱: `persona-dashboard` `backend/postprocess.py`의 `parse_objective_response_robust` 이식 (원문자 기호 우선 매칭)
- LLM: Luxia `luxia3-llm-32b-0731`, temperature 등 파라미터 미지정(API 기본값)
- 스크립트: `script.py`, 원 결과: `output_predictions.csv` (프로필/실제값/예측값/원문 응답 전부 포함)
