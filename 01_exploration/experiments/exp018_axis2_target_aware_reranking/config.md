# exp018 config — 축2: "일반적으로 비슷한 사람" vs "이 문항 예측에 informative한 사람"

> **2026-08-09 갱신**: v1(아래 원 설계) → v2(enriched, 관련문항 추가) → v2-extended(n=100 확대,
> mode 대비 신호 후퇴 확인) → v3(대조실험, 랭킹 방식만 분리) 순서로 확장됨. 결과 요약은 맨 아래
> "결과 히스토리" 참고.

exp017과 candidate filtering(demo_knn)·기본 데이터 구조는 그대로 재사용. exp017이 "축1 프로필
형태(raw vs axis1 narration)가 재랭킹에 기여하는가"를 봤다면, exp018은 **재랭킹 기준 자체**를
비교한다 — "타겟과 전반적으로 비슷한 사람"을 고르는 것과 "이 특정 문항 예측에 도움이 되는 사람"을
고르는 것 중 뭐가 더 나은가.

## 핵심 비교

- **arm_neighbor_raw**: 타겟 문항이 뭔지 전혀 언급하지 않고, "타겟 응답자와 전반적으로 비슷한 순서"로 8명 선택.
- **arm_neighbor_target**: 타겟 문항의 질문 텍스트+보기(옵션)는 보여주되, "이 문항에 대한 타겟의 응답을
  예측하는 데 가장 informative한 순서"로 8명 선택.

**leakage 금지(구조적)**: target-aware 방식에서도 타겟/후보의 **실제 target 문항 답변은 절대 보여주지
않음** — 질문과 보기만 보여줌. LLM은 답을 생성하지 않고 오직 순위+근거만 반환. (2026-08-08 leakage
필터 정책 변경과 별개로, 이건 애초에 이 실험 설계 자체에 내장된 leakage 방지 장치.)

## 파이프라인 (exp017 재사용 부분 표시)

1. **후보 30명 필터링** (exp017과 동일, demo_knn — CORE9 원핫 + NearestNeighbors) — *재사용*
2. **LLM 재랭킹 (응답자당 1회 호출 × 2 arm)**:
   - raw: 타겟 문항 언급 없이 일반적 유사도만으로 8명
   - target: 타겟 문항 텍스트+보기 제공, "예측에 informative한 사람" 기준 8명, 답 생성 금지
3. **실제 답변 집계** (exp017과 동일, median) — *재사용*

## 비교 arm

arm_mode / arm_demo_top8(demo_knn 상위 8명 그대로, LLM 없음) / arm_neighbor_raw / arm_neighbor_target

## 추가 로깅

raw와 target 두 arm이 뽑은 top8의 겹침(overlap count, 0~8)을 응답자마다 기록 — "질문을 알려주면
실제로 다른 사람을 고르는가"를 직접 확인.

## 스코프

- 타겟: SATFIN, PARTYLR
- N_CANDIDATES(1단계)=30, N_FINAL(2단계)=8
- 스모크: SATFIN n=2 → 통과 확인 후 SATFIN+PARTYLR n=15 파일럿

## v2 — Enriched (2026-08-09, `script_pilot_v2_enriched.py`)

exp017/018가 계속 "구조는 좋은데 신호가 약하다"는 상태였던 것을 개선: 후보/타겟 프로필에 인구통계
9개뿐 아니라, `exp019_question_aware_temporal_retrieval` Stage1에서 LLM이 뽑은
"타겟 문항과 의미적으로 관련 있는 문항 8개"의 **실제 응답값**(narration 없이 그대로)을 추가.
leakage는 target별 확인된 twin 1개만 수동 제외(PARTYLR→LEFTRIGT, SATFIN은 twin 없어 전부 포함).

**결과 (n=50×2=100)**: arm_neighbor_target_enriched가 정확도·MAE 둘 다, 두 타겟 모두에서 1등
(pooled acc 0.420 vs arm_mode 0.330, MAE 0.710 vs 0.750). arm_demo_top8 대비 MAE 유의(p=0.033),
arm_mode 대비는 근소하게 비유의(acc p=0.064, 문턱 바로 아래). **지금까지 중 가장 강한 신호.**

## v2-extended — n=100×2로 확장 (2026-08-09, `script_pilot_v2_enriched_extend.py`)

v2의 arm_mode 대비 유의성(p=0.064, 문턱 바로 아래)을 확정 지으려고 held-out 50명씩 추가 평가
(기존 50명 결과는 재사용, 새 50명만 추가 LLM 호출 → 합쳐서 n=100×2). 첫 시도는 LLM API DNS
resolution 장애로 신규 배치의 97~98%가 실패(빈 응답 → arm_demo_top8로 폴백)해 결과 폐기하고
백업에서 복구 후 재실행, 재실행은 예외 0건으로 정상 완료.

**결과 (n=100×2=200, pooled)**: arm_neighbor_target_enriched acc=0.410 MAE=0.740
(arm_mode acc=0.350 MAE=0.735).

**arm_mode 대비 유의성 — n=50 대비 오히려 후퇴**:
- McNemar(acc) p=0.073 (n=50 때 0.064보다 문턱에서 더 멀어짐)
- Wilcoxon(MAE) p=0.893 — MAE 우위가 거의 소멸(n=50: diff=+0.04 → n=100: diff=-0.005, 방향 반전 직전)

**즉 n=50 파일럿의 "거의 유의" 신호는 표본이 늘며 사라짐 — 전형적 파일럿 과대추정.**
"arm_neighbor_target_enriched가 mode baseline을 이긴다"는 주장은 아직 미확정(오히려 근거 약화).

**단, arm_demo_top8 대비 우위는 n=100에서도 견고**: McNemar(acc) p=0.0045, Wilcoxon(MAE) p=0.0052
(MAE diff=+0.12) — "관련문항 실제값 + LLM 종합판단이 인구통계 단순 필터보다 낫다"는 결론은 유지.

### 진단 — arm_neighbor_target_enriched가 mode에 왜 못 이기는지 (n=100×2 기준)

응답자를 typical(actual == pool 최빈값, arm_mode가 정의상 100% 맞힘)/atypical(그 외, arm_mode가
정의상 0%)로 나눠 분석:

- **typical에서 새는 곳**: SATFIN 28명 중 6명(21%), PARTYLR 42명 중 7명(17%)을 LLM이 놓침 —
  전부 방향이 무작위로 갈림(진보 4/보수 3, 만족 4/불만족 2 등 특정 편향 없음). "예측에 informative한
  이웃을 고르라"는 지시가 은연중에 뚜렷한 입장을 가진 이웃을 선호하게 만들어, 정작 타겟이 무난한
  사람일 때 median이 살짝 옆으로 밀리는 것으로 보임 — mode가 공짜로 얻는 승리를 LLM이 자진 반납.
- **atypical에서 잘하는 곳**: SATFIN 72명 중 11명(15.3%), PARTYLR 58명 중 14명(24.1%)을 mode는
  원천적으로 못 맞히는데 LLM은 맞힘 — 무작위(옵션 4~5개 기준 20~25%) 대비 진짜 신호.
- PARTYLR은 atypical 이득(24.1%)이 typical 손실(17%)을 넘어서 mode를 이기고, SATFIN은 이득
  (15.3%)이 손실(21%)을 못 넘어서 짐 — pooled 유의성이 안 잡히는 이유.

### v2c — 프롬프트로 쏠림 방지 시도 (2026-08-09, `script_pilot_v2c_balance_prompt.py`) — **실패**

위 진단에 따라, target-aware 프롬프트에 "타겟의 관련문항 응답이 뚜렷한 방향 없이 중간적/혼재돼
있으면 이웃도 한쪽으로 쏠리지 않게 다양하게 고르라"는 지시를 추가해 코드가 아닌 LLM 스스로
typical-slip을 줄일 수 있는지 테스트. v2-extended와 동일 respid·동일 후보풀(demo_knn 30명)
재사용, target-aware arm만 새로 호출(`arm_neighbor_target_v2c_balance`).

**결과 (같은 n=100×2 대상, uid 기준 페어 비교)**:
- typical-slip 13건 중 6건 회복 — 그러나 멀쩡하던 typical 7건을 새로 틀리게 만들어 **순변화 -1**
  (57→56/70)
- atypical은 오히려 악화: 25→22/130 — "쏠리지 말라"는 지시가 **진짜 비주류 의견을 가진 사람까지
  중도/무난 쪽으로 끌어당김**(예: 실제 "다소 보수적"인데 v2old는 맞혔던 걸 v2c가 "중도"로 오답
  처리, SATFIN도 동일 패턴)
- 전체 acc 0.410→0.390, mode 대비 McNemar p=0.073→**0.243**(후퇴), Wilcoxon MAE p=**0.889**
  (거의 완전 무의미)

**결론: LLM은 "이 타겟이 진짜 중도인지 vs 진짜 비주류 의견인지"를 프로필만으로 구분하지 못함.**
쏠림 방지 지시는 typical/atypical 양쪽에 동시에 영향을 줘서 고친 만큼 새로 깨짐 — 프롬프트로
이 문제를 풀 수 없다는 negative result. 이 typical 판별은 **LLM 판단이 아니라 외부 신호(코드
쪽 hybrid 규칙)로 해야 할 가능성이 높음** — 다음 후보: K=8 관련문항에서 타겟의 응답이 얼마나
극단적/일관적인지(z-score 분산 등)를 정량 계산해서, 무난해 보이면 LLM 결과 대신 mode로
되돌리는 규칙.

## v3 — 대조실험: 후보군 고정, 랭킹 방식만 분리 (2026-08-09, `script_pilot_v3_ablation.py`)

v2의 성공이 "후보군을 demo_knn 30명으로 좁혀서"인지 "그 안에서 LLM이 종합판단해서"인지 분리
검증. 후보 30명·프로필(인구통계+관련문항8) 완전히 고정하고 랭킹 방식만:
- ① `arm_llm_rank_30`: LLM이 target-aware로 8명 선택 (v2와 동일 방식)
- ② `arm_distance_rank_30`: 같은 K=8개 문항을 z-score 동일가중치 거리로 계산해 8명 선택
  (`exp019_question_aware_temporal_retrieval`의 거리 계산 로직 재사용, 검색 범위만
  전체 retrieval_pool이 아니라 이 30명으로 제한)

**결과 (n=50×2=100)**: ①LLM랭킹이 ②거리계산을 정확도·MAE 둘 다, 두 타겟 모두에서 앞섬
(pooled acc 0.410 vs 0.340, MAE 0.720 vs 0.780). ②거리계산은 arm_demo_top8과 거의 차이 없음
(0.340 vs 0.330). 유의성은 아직 안 잡힘(p>0.2, n 더 필요)이나 방향은 일관됨.

**결론: exp018 v2의 성공 요인은 "후보군 제한(demo_knn)" 자체가 아니라, "제한된 후보군 안에서
LLM이 여러 정보(인구통계+관련문항)를 종합 판단하는 것" — 같은 정보를 정량적 거리로 대체하면
효과가 거의 사라짐.** 이건 exp019 전체(정량 거리 계산으로 LLM 노이즈를 없애려던 시도)가 예상과
반대로 더 나쁜 성능을 보였던 것과 같은 결의 증거.

## 결과 히스토리 요약

| 버전 | 핵심 변경 | pooled 결과 (최고 arm) | 유의성 |
|---|---|---|---|
| v1 (n=15) | raw vs target-aware, 인구통계만 | 방향은 target-aware 우위 | 미확정 |
| v2 (n=50, enriched) | + 관련문항 8개 실제값 | acc 0.420, MAE 0.710 (vs mode 0.330/0.750) | demo_top8 대비 MAE 유의(p=0.033), mode 대비 근소 비유의(p=0.064) |
| v2-extended (n=100, enriched) | v2와 동일, n만 확대 | acc 0.410, MAE 0.740 (vs mode 0.350/0.735) | demo_top8 대비 여전히 유의(p=0.0045/0.0052), **mode 대비는 n=50보다 후퇴**(acc p=0.073, MAE p=0.893 — MAE 우위 거의 소멸) |
| v2c (n=100, 쏠림방지 프롬프트) | target-aware 프롬프트에 balance 지시 추가 | acc 0.390, MAE 0.730 (vs v2old 0.410/0.740) | **실패**: typical-slip 13건 중 6건 회복했지만 typical 7건·atypical 다수 새로 깨짐, mode 대비 McNemar p=0.243(후퇴) |
| v3 (n=50, 대조실험) | 랭킹 방식만 LLM vs 거리 분리 | LLM랭킹 0.410/0.720 vs 거리랭킹 0.340/0.780 | 방향 일관, 유의성 미확정 |
