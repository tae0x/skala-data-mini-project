# 신약 후보물질 독성 예측 (SKALA 데이터분석 Mini Project)

화합물의 구조 정보와 물성 수치로 **독성이 있는지 없는지**를 맞히는 이진 분류 과제입니다. **Day1 EDA와 Day2 개별 챌린지(Modeling & Prediction)의 제출 4회를 모두 마쳤습니다.** 최고 리더보드 F1은 0.8351(제출 3)이고, 모델링 코드는 `src/`와 `notebooks/02~06`에 있습니다.

- 평가 지표: **F1 Score**
- Positive 클래스: `label` = 1 = **독성 없음** (0 = 독성 있음). 직관과 반대이니 지표를 해석할 때 주의하세요.

## EDA 핵심 결과

- 클래스 비율은 독성 없음 54.4%(4,542건), 독성 있음 45.6%(3,807건)로 크게 쏠리지 않았습니다.
- **전부 1(독성 없음)로 예측하는 기준선의 F1이 이미 0.705**입니다. 모델은 이 값을 넘어야 의미가 있습니다.
- 변수 하나의 설명력은 약합니다. 가장 큰 `clogp`도 Gini 0.325이고, 극성 관련 변수(`TPSA`, `NOCount` 등)가 뒤를 잇습니다. 여러 변수를 합쳤을 때 좋아지는지는 모델링에서 확인해야 합니다.

자세한 내용은 [`reports/EDA_보고서.md`](reports/EDA_보고서.md)에 있습니다.

## 시작하기

데이터(`data/train.csv`)는 용량과 배포 자료 문제로 저장소에 올리지 않습니다. 과제 배포본을 받아 `data/train.csv` 위치에 직접 넣어 주세요.

```bash
# 가상환경 생성과 설치 (Python 3.11)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# EDA 노트북 재실행 (프로젝트 루트에서)
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

- 노트북은 루트 기준 상대경로를 쓰므로 반드시 루트에서 실행합니다.
- 그림은 `reports/figures/`, 표는 `reports/tables/`에 덮어써집니다.
- `requirements.txt`의 모든 버전은 실제로 실행해서 확인한 값입니다(`lightgbm`, `xgboost`, `catboost`, `optuna`, `shap` 포함).
- Day2 노트북 `02~07`은 앞 번호의 결과(`data/cache/`, `reports/tables/`)를 이어 쓰므로 번호 순서로 실행합니다. `05`(Optuna 튜닝)가 가장 오래 걸리고 진행 로그는 `data/cache/optuna_progress.log`에 남습니다. `data/cache/`는 git에 없으며 처음 실행할 때 만들어집니다.

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `ROADMAP.md` | Day2 일정, 제출 계획, 실험 로그, 제출 체크리스트 |
| `notebooks/01_eda.ipynb` | EDA 전체(구조, 품질, 단변량, 변수별 label 설명력, RDKit 파생변수) |
| `notebooks/02~07` | Day2 모델링: 02 기준선·임계값·다중공선성, 03 모델 6종 비교, 04 피처군·피처 선택, 05 Optuna 튜닝·앙상블, 06 홀드아웃·SHAP, 07 제출 3·4 재현과 리더보드 해석 |
| `src/` | `data.py`(분할·캐시), `features.py`(RDKit 기술자·추가 지문), `preprocess.py`(상관·VIF 필터, 파이프라인), `models.py`(Tanimoto SVM), `validate_submission.py`(제출 파일 검증) |
| `.claude/agents/` | 누수 점검용 서브에이전트(`leakage-auditor`) |
| `submissions/` | 제출 파일(`predict_output_*.csv`)과 양식. git 제외 |
| `reports/` | EDA 보고서, 그림(`figures/`), 표(`tables/`) |
| `study/` | 데이터 분석 입문자용 학습 자료 |
| `.mcp.json` | Canva MCP 서버 설정(project 스코프) |
| `CLAUDE.md` | Claude Code용 프로젝트 안내 |
| `docs/` | 교수님 안내 메모(`professor_notes.md`)와 참고 자료(`reference/titanic/`: 타이타닉 실습, Optuna 예제 포함)는 추적. 저작권 문구가 있는 PDF만 git 제외 |
| `data/`, `.venv/` | 로컬에만 있음(git 제외) |

## 문서 읽는 순서

1. [`study/01_프로젝트_개요와_진행상황.md`](study/01_프로젝트_개요와_진행상황.md): 프로젝트 개요, 전체 단계, 진행 상황
2. [`study/02_Day1_핵심_개념과_발표QA.md`](study/02_Day1_핵심_개념과_발표QA.md): 필요한 개념, 퀴즈, 발표 Q&A
3. [`study/03_Day2_모델링_개념과_결과.md`](study/03_Day2_모델링_개념과_결과.md): Day2 모델링의 뼈대와 핵심 개념(비유 중심)
4. [`reports/EDA_보고서.md`](reports/EDA_보고서.md) → `notebooks/01_eda.ipynb`

## 팀 작업

발표자료는 팀원들이 Canva MCP로 같이 만듭니다. 처음 열면 서버 승인 프롬프트가 뜨고, `/mcp`에서 각자 Canva 계정으로 인증합니다. 토큰은 저장소에 들어 있지 않습니다. `docs/`의 PDF는 저작권 문구가 있으므로 Canva에 올리지 않습니다.

## 주의사항

- **데이터 누수**: EDA의 라벨 기반 통계는 전체 train으로 계산한 탐색용입니다. 이 결과로 변수를 고르지 말고, 데이터를 먼저 분할한 뒤 학습 데이터에서 특성 선택을 다시 하세요.
- 전처리 기준(대체값, 스케일러 등)은 모두 학습 데이터에서만 정하고, 분류이므로 분할에는 `stratify=y`를 씁니다.
- `docs/`의 PDF(저작권 문구가 있는 교육 자료)와 `data/`는 git에 올리지 않습니다.

## Day2 진행 (모델링과 예측 제출)

진행 순서는 분할 → 전처리 → 기준선 모델(Dummy, 로지스틱, 결정트리) → 트리 앙상블 비교(5-겹 교차검증) → 튜닝 → 시험용 데이터로 최종 평가와 SHAP 해석 → `predict_output_<작성자>.csv`(열: `SMILES`, `output`) 제출이었고, 모두 끝났습니다.

| 제출 | 구성 | 교차검증 F1(OOF) | 리더보드 F1 |
|---|---|---|---|
| 1 | HistGB, 임계값 0.5 | 0.820 | 0.8284 |
| 2 | LightGBM + XGBoost + Tanimoto SVM 평균(튜닝), 임계값 0.43 | 0.8328 | 0.8287 |
| 3 | SVM + ExtraTrees + RandomForest 평균, 임계값 0.44 | 0.8339 | **0.8351** |
| 4 | 제출 2·3 구성 평균, 임계값 0.45 | 0.8357 | 0.8312 |

네 점수의 폭(0.0067)은 927건 시험 세트의 우연(F1 표준편차 약 0.0126)보다 작아서, 모델 구성의 차이는 구별되지 않습니다. 다중공선성은 상관(|Spearman| > 0.95)과 VIF(> 10) 필터로 수치형 207개를 134개로 줄였고, 피처 영향력은 SHAP으로 정리했습니다(`reports/tables/final_shap_*.csv`).

Day2는 **개별 챌린지**(Modeling & Prediction)였습니다. 일정, 제출 계획, 실험 기록은 [`ROADMAP.md`](ROADMAP.md)를 보세요.
