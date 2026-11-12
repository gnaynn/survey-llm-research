# exp014 config — axis2 검색 전략 비교 (인구통계 kNN / 임베딩 유사도)

- axis1과 동일 3개 타겟(SATFIN/HELPFUL/PARTYLR), 동일 held-out 100명, 동일 retrieval/held-out 분할(SPLIT_SEED=42), 동일 LLM 설정(Luxia, temperature=0.3)
- **arm_demo_knn**: 9개 핵심 프로필을 원핫 인코딩 후 유클리드 최근접 K=3명을 retrieval pool에서 검색 (exp013 arm6과 같은 인코딩, 용도만 "답 다수결"이 아니라 "사람 자체를 few-shot 예시로 제시")
- **arm_embed_sim**: 각 응답자의 9개 프로필을 서술형 문장으로 변환 → `nlpai-lab/KURE-v1`로 임베딩 → 코사인 유사도 top-3 검색
- 비교 대상: axis1 arm0(무작위 3명, `exp009_phase1_satfin_n100` 등) / arm5(최빈값 imputation, `exp013_imputation_baselines`)
- 스크립트: `script.py`, 원 결과: `output_predictions.csv`
