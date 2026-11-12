# data/

이 폴더의 내용은 저장소에 포함되지 않습니다(`.gitignore`). KGSS 원자료는 재배포가 제한될 수 있으므로 직접 받아서 아래 구조로 두세요.

```
data/
├── KGSS_원본파일/
│   └── kor_data_CUM0074_V2.sav        # KGSS 누적 원자료 (SPSS)
├── KGSS_결과파일/                       # 변수 사전·메타데이터·응답자 가공본
│   ├── (2)kgss_variable_dictionary.json
│   ├── (4)kgss_full_year_variables.json
│   ├── (5)kgss_필수input.json
│   ├── (7)kgss_variable_metadata.json
│   ├── (8)kgss_all_respondents_core_profile.jsonl
│   └── (9)kgss_all_respondents_full_qa.jsonl   # scripts/build_full_qa_jsonl.py로 생성
└── questions/
    ├── 2025_questions.json            # 2단계 실험 후보 문항 풀(188문항) 메타데이터
    └── 2025_pilot50_questions.json
```
