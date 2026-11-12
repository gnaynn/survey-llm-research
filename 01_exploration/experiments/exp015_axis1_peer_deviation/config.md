# exp015 config — axis1 신규 방식: 또래집단 대비 이탈(z-score) 기반 프로필

## 배경

지도교수 피드백: 기존 axis1(상관계수/임베딩/이론적 개념군 기반 선별)은 "타겟과 통계적으로 관련된 변수를 고르는 것"이라
원리상 통계적 변수선택과 다르지 않다는 지적. 파이프라인 전체가 최빈값 baseline에도 밀림
(축1 결과 정리 5절). 새 접근은 타겟 관련성이 아니라 **"이 응답자가 또래집단 대비 어디서
튀는가"**를 기준으로 프로필을 구성 — z-score 계산 자체는 통계적이지만, "왜 의미 있는 특징인지"를
narration으로 풀어 예측에 쓰는 것은 LLM만 할 수 있다는 게 차별점.

## 실험 범위

- **타겟 2개만**: SATFIN(가계 상태 만족도), PARTYLR(정치적 진보/보수 성향). HELPFUL 제외.
- **leakage 필터**: 메인 파이프라인(arm8/arm9)은 정보 누출 기준의
  기존 배제 규칙(|r|≥0.30 또는 s≥0.50)을 이탈특징 후보 추출(Stage 3) 단계에서 그대로 적용함.
  필터를 끈 경우의 효과는 **2차 실험(arm10)**에서 따로 확인 — arm9 프로필에 유사문항(배제 필터 미적용,
  twin 문항 포함 가능) 실제 답변을 추가로 넣어 비교.

## 파이프라인 (4단계, `lib_peer_deviation.py`)

1. **척도형 필터**: 후보 문항(target당 exp002 사전계산 153개 — same-part ∪ 배경 랜덤 300 풀에서
   n≥30, 유효 상관계수 조건 통과한 것들) 중 유효 응답 수준 3~11개 + (중간 라벨이 전부 숫자이거나
   라벨에 척도 키워드["매우","다소","전혀","찬성","만족" 등] 포함) 휴리스틱으로 순서형만 필터링.
   완벽한 의미분류가 아니라 휴리스틱 — `audit_ordinal_{TARGET}.csv`에 전체 분류 결과(라벨 포함) 저장,
   검토 가능. SATFIN/PARTYLR 각각 153개 후보 중 98개가 순서형으로 분류됨.
   부수 효과: soft-missing 라벨("Not Applicable" 등 missing_codes에 안 잡힌 것)도 이 단계에서 결측 처리.
2. **또래집단 + z-score**: 연령 ±5세 슬라이딩 윈도우 + peer_def(A1=성별+학력 / A2=성별만 /
   A3=성별+학력+지역) 조건으로 retrieval_pool(YEAR=2023, 70% split) 내 또래집단을 구성,
   z=(본인 코드값-또래평균)/또래표준편차. 또래 n<10이면 학력/지역→넓은 연령창(±10,±20)→성별만
   순으로 fallback (충족 시 사용 레벨을 함께 기록).
3. **이탈특징 추출**: 2단계 z-score 중 leakage 필터(위 참고) 통과한 것만 |z| 상위 N개.
4. **LLM narration**: 문항별 [본인 응답 라벨] vs [또래평균에 해당하는 근사 라벨]을 직접 비교해
   서술하도록 프롬프트 설계 — **z의 부호만으로 방향(더 동의/덜 동의 등)을 추론하지 말라고 명시**.
   (초기 버전에서 z 부호를 코드 방향과 혼동해 "매우 동의"인데 "신뢰가 낮다"고 반대로 서술하는
   confabulation 버그 발견 → 라벨 대 라벨 비교 방식으로 수정, 재검증 완료.)

## Arm 구성

| arm | 프로필 |
|---|---|
| arm0_baseline | 인구통계 9개만 (exp009/exp011 기존 결과 재사용) |
| arm_mode | retrieval_pool 최빈값 (LLM 미사용) |
| arm1_corr / arm2_embed | 기존 상관계수/임베딩 기반 10개 확장 (exp009/exp011 기존 결과 재사용) |
| arm8_zscore_raw | 인구통계 9개 + 이탈문항 raw z-score 구조 (narration 없음) |
| arm9_zscore_narrative | 인구통계 9개 + 이탈문항 LLM narration 텍스트 |
| arm10_peerdev_plus_similar (2차 실험) | arm9 + \|r\| 상위5 ∪ s 상위5 유사문항의 실제 답변(leakage 필터 미적용) |

arm0/arm1/arm2는 exp009(SATFIN)·exp011(PARTYLR)과 **동일한 SPLIT_SEED=42/SAMPLE_SEED=7/N_HELDOUT=100**을
써서 같은 held-out 100명을 재현 — 신규 LLM 호출 없이 기존 CSV에서 respid 매칭으로 재사용(비용 절감).

## 실험 변수 (그리드서치)

- 또래집단 정의: A1(기본) — A2/A3는 후속 과제로 보류(1차는 A1+N=3 파일럿을 먼저 확인한 뒤 확장).
- 이탈 특징 개수 N: 파일럿은 N=3, 이후 N=1/2/5 스윕(`script_nsweep.py`)해 N=3과 합쳐 정확도 곡선 산출.
- K_SIMILAR(2차 실험 유사문항 개수): 5로 설정(문서에 명시 없어 임의 지정 — 조정 가능 파라미터로 스크립트 상단에 노출).

## 산출 파일

- `lib_peer_deviation.py` — 4단계 공용 라이브러리
- `audit_ordinal_{TARGET}.csv` — Stage1 척도형 분류 감사 로그
- `script_pilot.py` → `output_predictions_new_arms.csv`(arm_mode/arm8/arm9 신규) + `output_predictions_combined.csv`(+ 기존 arm0/1/2) + `output_deviation_audit.csv`(또래 n·fallback 레벨 등)
- `script_nsweep.py` → `output_nsweep.csv`, `output_nsweep_with_n3.csv`
- `script_experiment2.py` → `output_experiment2_arm10.csv`
