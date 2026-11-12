# exp012 config — 3개 타겟 pooled 통계 분석

- `exp009_phase1_satfin_n100`, `exp010_phase1_helpful_n100`, `exp011_phase1_partylr_n100`의 원 결과(각 n=100)를 하나로 합쳐(n=300) 검정력을 높인 사후 분석
- (타겟, 응답자) 쌍을 관측 단위(uid)로 삼아 동일한 paired McNemar/Wilcoxon 검정 재실행
- 스크립트: `script.py`, 원 결과: `output_pooled_stats.txt`
