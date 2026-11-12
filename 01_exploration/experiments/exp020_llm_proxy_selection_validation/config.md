# exp020 config — LLM의 역할을 "Proxy Selector"로 좁히고 그 기여를 검증

## 배경

exp017~019를 거쳐 정리된 가설:

> **LLM = Response Generator가 아니라 Proxy Selector.**
> LLM은 답을 직접 만들지 않습니다. 여러 응답 변수의 조합을 종합해 "타겟에게 가장 informative한 실제
> 응답자"를 골라내고, 최종 예측은 그 응답자의 실제 관측 응답으로 합니다.

지금까지 확인된 것:
- 단순 인구통계 유사도만으로는 부족 (exp014, arm_demo_top8이 최빈값보다도 못함)
- 관련 문항 "선정"에 LLM을 쓰는 것은 근거 부족 (exp018 4조건: LLM-8 ≈ Random-8 ≈ Stat-8)
- 직접 답변 생성은 명백히 성능이 낮음 (exp015 arm8/9, exp016 CANTRUST)
- 다속성 응답 패턴을 종합한 응답자 랭킹은 유의미하게 잘함 (exp018 v3, v2-extended)

**목적**: "LLM이 여러 설문 응답 변수의 조합을 이용해 좋은 proxy를 고른다"는 주장이 학습 기반
통계/ML 베이스라인(Ridge, Logistic, RF, GBM)과 비교해도 유지되는지 검증합니다. 동일 가중치 z-score
거리와의 비교만으로는 약한 비교(strawman)가 될 수 있기 때문입니다.

## 설계

- exp018 `arm_neighbor_target_enriched`(n=100×2)의 결과를 재사용합니다. LLM이 고른 후보 풀과 최종 답은
  그대로 두고, 같은 K=8 변수로 학습한 통계/ML 모델이 같은 30명 후보를 얼마나 잘 랭킹하는지 비교합니다.
  신규 LLM 호출은 없습니다 (`script_A_llm_vs_ml.py`).
- 교란 요인 점검: Ridge/Logistic을 LLM이 고르지 않은 Random-8/Stat-8 변수로 재학습해 다시 비교합니다
  (`script_confound_check.py`).
- 사후 해석(post-hoc) 위험 점검: 기존 표본과 겹치지 않는 새 held-out 50명/타겟에서 재현성을 확인합니다
  (`script_mechanism_holdout_replication.py`).
- 후보 수(Top-N)와 최종 선택 수(Top-K) 그리드 비교 (`script_D_candidates_vs_final_sweep.py`).

## 결과

- **LLM vs ML 베이스라인**: LLM이 Ridge/Logistic/RF/GBM 중 어느 것도 유의하게 이기지 못했습니다
  (McNemar p=0.62~0.76).
- **교란 요인 점검**: Random-8 변수에서는 LLM이 Logistic을 유의하게 이겼습니다(p=0.031). 변수 선택에 의한
  착시는 아닌 것으로 잠정 확인했습니다(n=30/타겟, 확정 아님).
- **홀드아웃 재현**: LLM 선택의 "행동 패턴"(K=8 매칭 트레이드오프)은 재현되었지만(p<0.000001),
  "정확도 개선"은 재현되지 않았습니다(방향 반전, p=0.399). → **정확도 개선 주장은 철회하고, 서술적인
  행동 패턴만 결론으로 유지했습니다.**

## 실행 시 주의

LLM API 장애로 배치 대부분이 빈 응답으로 조용히 실패하고 기본값으로 채워진 적이 있습니다. 신규 LLM
호출이 있는 실험은 완료 후 `n_picked==0` 비율과 예외 로그 수를 반드시 확인하고, 실패율이 높으면(예: >5%)
결과를 쓰지 않고 다시 실행합니다.
