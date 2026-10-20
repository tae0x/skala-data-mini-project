# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

SKALA 데이터분석 Mini Project: 신약 후보물질 독성 예측(이진 분류). 현재 단계는 **Day1 EDA까지**이며 모델링은 하지 않았다.
빌드, 테스트, 린트 설정은 없다.

## Git 규칙

- 커밋 메시지는 한국어 `type: 설명` 형식(`chore`, `feat`, `docs` 등)으로 쓴다. `Co-Authored-By` 트레일러와 Claude 표기는 넣지 않는다.
- 커밋과 푸시는 사용자가 요청할 때만 한다.

## 환경과 실행

- 가상환경은 `.venv`(Python 3.11, `uv`로 생성)이고 `.gitignore` 대상이다. Python 3.14 시스템 파이썬에는 분석 라이브러리가 없다.
- `.gitignore`에는 `.venv/` 외에 `docs/*.pdf`, `data/`, `__pycache__/`, `.DS_Store`, `reports/presentation/`, `submissions/`도 있다. `data/train.csv`는 git에 없으므로 과제 배포본을 직접 넣어야 한다.
- 설치: `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt`
  - EDA에 필요한 패키지(pandas, scipy, statsmodels, scikit-learn, rdkit, matplotlib, seaborn, ipykernel, nbconvert)만 설치해서 검증했다. `xgboost`, `lightgbm`, `catboost`, `optuna`, `shap`는 `requirements.txt`에는 있지만 아직 설치/검증하지 않았다.
- EDA 노트북 재실행(프로젝트 루트에서): `.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb`
  - 그림은 `reports/figures/`, 표는 `reports/tables/`에 덮어써진다.

## MCP (Canva)

발표자료는 팀원들이 Canva MCP로 같이 만든다. 서버는 저장소 루트의 `.mcp.json`(project 스코프)에 등록돼 있고, `.mcp.json`에는 URL만 있고 토큰은 넣지 않는다.
처음 열면 서버 승인 프롬프트가 뜨고, `/mcp`에서 각자 Canva 계정으로 인증한다. 주소(`https://mcp.canva.com/mcp`)는 `claude mcp get canva`에서 Connected로 확인했다. 연결에 실패하면 인증 상태부터 확인한다.
발표 PDF와 대본은 `reports/presentation/`에 로컬로만 두고(`.gitignore` 대상) 원본은 Canva다. 수정은 Canva에서 하고, 점검 전에 PDF를 다시 내보내 덮어쓴다.
`docs/`의 PDF는 저작권 문구가 있으므로 Canva에 올리지 않는다. 그림은 `reports/figures/`를 쓴다.

## 코드 구조

- `README.md`: 팀원용 첫 안내(진행 상황, 설치와 실행, 폴더 구조, 주의사항). 진행 단계나 핵심 수치가 바뀌면 함께 갱신한다.
- `ROADMAP.md`: Day2 일정, 제출 계획(최대 4회), 실험 로그, 제출 전 체크리스트. 일정이 바뀌면 여기만 고친다. 제출 폼 URL은 저장소에 적지 않는다(슬랙 공지 참고).
- `notebooks/01_eda.ipynb`: EDA 전체(구조, 품질, 단변량, 변수별 label 설명력, RDKit 파생변수). 첫 셀이 프로젝트 루트(`data/`가 있는 폴더)를 찾아 `ROOT / "data/train.csv"`, `ROOT / "reports"`를 쓰고 `src/`를 `sys.path`에 추가하므로 루트와 `notebooks/` 어디서 실행해도 된다. 새 노트북도 같은 방식으로 시작한다.
- `src/features.py`: `compute_rdkit_descriptors(smiles)`가 SMILES에서 RDKit 기술자 217개를 만든다. 컬럼 이름에 `rd_` 접두사를 붙여 원본 컬럼(`MolWt`, `qed`)과 겹치지 않게 한다. 상수열 제거와 결측 대체는 하지 않는다(분할 후 학습 데이터 기준으로 해야 하므로). `predict_input.csv`에도 같은 함수를 쓴다.
- `reports/EDA_보고서.md`: 변수별 설명력 중심의 보고서. 수치는 노트북 출력에서 가져왔다.
- `study/`: 데이터 분석 입문자(사용자)를 위한 학습 자료. `01_프로젝트_한눈에_보기.md`(개요와 진행 상황), `02_핵심_개념.md`(개념과 퀴즈), `03_발표_예상질문.md`(발표 Q&A). 프로젝트 숫자가 바뀌면 보고서와 함께 갱신한다.

## 데이터 스키마 (`data/train.csv`, 약 52MB, 8,349건 + 헤더)

- 총 3,078열: `SMILES` / `ecfp_0..1023` / `fcfp_0..1023` / `ptfp_0..1023` (핑거프린트 각 1,024비트, 0/1) / `MolWt`, `clogp`, `sa_score`, `qed` / `label`
- `label`: **1 = 독성 없음(Positive), 0 = 독성 있음(Negative)**. 직관과 반대이므로 지표 계산과 해석 시 양성 클래스 방향에 주의한다. 분포는 1이 54.4%, 0이 45.6%다.
- 결측값, SMILES 중복, RDKit 파싱 실패는 모두 0건이다. `ptfp`에는 상수열이 49개 있다.
- 기존 `MolWt`, `clogp`, `qed`는 RDKit의 `MolWt`, `MolLogP`, `qed`와 같은 값이다(Spearman 1.0). 파생변수에서 중복으로 취급한다.
- `predict_input.csv`(927건)는 `label`만 없는 동일 구조(3,077열)이며 `data/`에 있다. 제출 파일 `predict_output_<작성자>.csv`의 열은 `SMILES`, `output` 두 개이고 값은 0/1이다. 열 이름은 바꾸지 않는다(양식: `submissions/predict_output_홍길동.csv`). 만든 제출 파일도 `submissions/`에 저장한다(git 제외).
- 핑거프린트 열이 많아서 `head`/`cat`으로 전체를 출력하지 말고 컬럼 단위로 읽는다(`usecols`). 핑거프린트 열은 `uint8`로 읽으면 메모리가 206MB에서 27MB로 준다.

## 평가 및 분석 원칙

- 주 지표는 F1 Score(Positive = 독성 없음). **전부 1로 예측하는 기준선의 F1이 이미 0.705**라서 모델 F1은 이 값과 비교한다.
- 분석 범위는 EDA까지다. 모델링은 Day2 이후에 하며 사용자가 요청할 때만 시작한다. Day2는 개별 챌린지(Modeling & Prediction)이고 일정, 제출 계획, 실험 로그는 `ROADMAP.md`에서 관리한다.
- 데이터 분할을 전처리보다 먼저 수행한다. 대체값, 스케일러, 인코딩 등 모든 전처리 기준은 학습 데이터에서만 학습한다. 분류이므로 `stratify=y`를 적용한다. `notebooks/01_eda.ipynb`의 라벨 기반 통계는 전체 train 기준 탐색용이므로 특성 선택은 분할 후 다시 한다.
- 전처리와 모델은 scikit-learn `Pipeline`/`ColumnTransformer`로 묶고, 교차검증은 Stratified 5-Fold를 쓴다. 단순 baseline(Dummy, Logistic, 단일 Decision Tree)을 먼저 만든 뒤 트리 앙상블(RF, XGBoost, LightGBM, CatBoost)을 비교한다.
- Day2 평가 기준(10/20 구두 안내): 다중공선성 컬럼을 어떻게 찾고 해소했는지, 사용한 피처별 설명력(영향력), RDKit 파생변수 활용을 보여 줘야 한다. 대응 체크리스트는 `ROADMAP.md` 7절이다.
- SMILES는 식별자에 가까워서(8,349개 전부 고유) 변수로 쓰지 않고, 파생변수를 만든 뒤 버려도 된다는 안내를 받았다.

## 참고 문서

`docs/`의 PDF는 저작권 문구(SK C&C 콘텐츠) 때문에 `.gitignore`에 들어 있어 git에는 올라가지 않고 로컬에만 있다. `professor_notes.md`는 추적한다.

- `docs/professor_notes.md`: 교수님 공지와 분석 가이드(EDA → 분할 → 전처리 → 파이프라인 → 모델링 → 튜닝/앙상블 → 해석), Day2 과제 안내, 10/20 평가 기준(구두 전달 메모)
- `docs/데이터분석 및 AIOps_7.데이터분석 Mini Project_박병선p.pdf`: 과제 배경, 컬럼 정의, 평가지표 슬라이드 (git 제외)
