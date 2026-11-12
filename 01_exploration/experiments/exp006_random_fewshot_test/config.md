# exp006 config

- `exp004_axis1_baseline_pilot`와 동일 20명(seed=7)·동일 타겟(SATFIN)·동일 프로필 형태(JSON) — **딱 하나 추가: "무작위 검색" few-shot 예시 3명**
- axis1 goal.md의 "실험 통제 원칙"(axis1 테스트 시 axis2는 무작위 검색으로 고정)을 처음으로 실제 반영
- 예시 추출: 타겟 본인을 제외한 SATFIN 유효 응답자 풀에서 매 타겟마다 무작위 3명(seed=123), 그들의 프로필 요약 + 실제 응답을 시스템 프롬프트 "## 비슷한 사람들 (참고)" 섹션에 추가
- temperature 등 다른 변수는 exp004와 동일(미지정) — "few-shot 유무" 하나만 격리해서 비교
- 스크립트: `script.py`, 원 결과: `output_predictions.csv` (예시 3명 목록도 포함)
