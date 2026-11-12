# exp011 config — Phase 1 세 번째 재현 (PARTYLR, n=100, arm0~4)

- exp009/exp010과 동일 파이프라인, 타겟만 PARTYLR(정치적 진보-보수 성향, 정치 인식 영역)로 교체
- 5점 척도(매우 진보적~매우 보수적)가 원래 코드 순서 그대로 이념 스펙트럼이라 재정렬 불필요
- retrieval pool 834 / held-out 358 (2023년 PARTYLR 유효 응답자 1192명 기준), n=100, temperature=0.3
- SATFIN(exp009)·HELPFUL(exp010) 결과가 서로 반대 방향이라 "타이브레이커" 목적으로 실행, 3개 타겟 확보 후 pooled 분석 예정
- 스크립트: `script.py`, 원 결과: `output_predictions.csv`
