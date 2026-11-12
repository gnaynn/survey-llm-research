# exp013 config — arm5(중앙값/최빈값)·arm6(kNN) imputation control

- 실험 설계의 arm5·arm6를 처음 실행. LLM/RAG 미사용, 순수 통계 계산이라 API 불필요, 수 초 내 완료.
- exp009~011과 **동일한 retrieval/held-out 분할(SPLIT_SEED=42) 및 동일 held-out 100명(SAMPLE_SEED=7)** 사용 — 같은 사람들이라 LLM arm들과 직접 페어드 비교 가능
- arm5: retrieval pool에서 최빈값(mode) 계산 → held-out 전원에게 동일하게 적용
- arm6: 9개 핵심 프로필을 원핫 인코딩 후 kNN(k=15) 분류기를 retrieval pool로 학습 → held-out 예측
- 대상: SATFIN, HELPFUL, PARTYLR (기존 3개 타겟과 동일)
- 스크립트: `script.py`, 원 결과: `output_predictions.csv`
