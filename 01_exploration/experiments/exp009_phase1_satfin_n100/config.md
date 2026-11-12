# exp009 config — Phase 1 (SATFIN, n=100, arm0~4)

- 실험 계획 Phase 1의 첫 실행. exp008(n=20) 로직을 정식화 + 확장.
- **retrieval pool 70%(858명) / held-out 30%(369명) 분할** 최초 적용 — eligible pool(r/s)을 retrieval pool로만 재계산해 leakage 방지 (`build_arms_v2.py`)
- held-out에서 n=100 무작위 표본 평가
- few-shot 예시 3명은 retrieval pool에서만 추출 (held-out 오염 방지)
- 통제변수: temperature=0.3, JSON 프로필, Luxia `luxia3-llm-32b-0731` — exp006/007에서 확정한 값
- arm0~4만 실행 (imputation control arm5/6은 미포함, 후속)
- 스크립트: `script.py` + `build_arms_v2.py`, 원 결과: `output_predictions.csv` (500행 = 100명×5arm)
