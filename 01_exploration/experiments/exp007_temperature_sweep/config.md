# exp007 config — 실험 계획 0-5 캘리브레이션

- 베이스: `exp006_random_fewshot_test`와 동일(JSON 프로필 + 무작위 3명 few-shot), 동일 20명(seed=7), 동일 예시추출 seed(123, 온도별로 동일하게 재사용해 예시 자체는 통제)
- 변경: `temperature`를 명시적으로 0.3 / 0.7 / 1.0 세 값으로 지정 (기존 exp004~006은 API 기본값, 미지정 상태였음)
- 지표: 정확도, 최다 예측값 비중(max_share), 정규화 엔트로피(bits, 최대 log2(5)=2.322)
- 스크립트: `script.py`, 원 결과: `output_predictions.csv` (온도별 60행)
