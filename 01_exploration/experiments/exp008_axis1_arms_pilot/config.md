# exp008 config — axis1 arm0~4 실제 구현 첫 실행

- **처음으로 axis1의 실제 조건(arm1~4)을 구현**: `build_arms.py`가 `exp002_embedding_calibration`의 r/s 값을 재사용해 eligible pool(leakage 필터 통과, 138개) 구성 → 응답자별로 상관관계(arm1)/임베딩(arm2)/이론적개념군(arm3)/무작위(arm4) 방식으로 top-10 추가 문항 선정
- 이전 파일럿(exp004~007)은 전부 arm0(baseline)이었음 — 이번이 진짜 axis1 비교의 첫 실행
- 통제변수는 exp006/007에서 확정한 값 그대로: 무작위 3명 few-shot, temperature=0.3, JSON 프로필
- 타겟: SATFIN, n=20(seed=7, exp004~007과 동일 응답자) × 5개 arm = 100회 Luxia 호출
- 지표: 정확도, MAE
- 스크립트: `build_arms.py`(arm 구성 로직) + `script.py`(실행), 원 결과: `output_predictions.csv`
