# 교수님 슬랙 안내 (박병선 교수님)

## Day1 시간표

- 09:00~09:50 Project Briefing
- 10:00~11:50 Group Activity: EDA
- 13:10~15:30 Group Activity: EDA
- 15:30~17:30 EDA Results Presentation
- 17:30~18:00 Wrap-up & Day 2 Preview

## 공지

- 학습 데이터: train.csv, 조별로 EDA 진행
- 오늘은 EDA까지만 진행
- 발표는 15:30부터, 발표 순서는 13:50에 정함
- 교수님이 알려주신 라이브러리(RDKit)로 SMILES 코드에서 파생변수를 만들어볼 것
- 발표자료는 슬랙에 올리기

## Day2 시간표

"데이터 분석 Mini-project" Day2 Time Table 안내

- 09:00~09:30 Challenge Briefing
- 09:30~11:50 Individual Challenge : Modeling & Prediction
- 13:10~16:30 Individual Challenge : Modeling & Prediction
- 16:30~18:00 Jam Study: Review & Wrap-up

## Day2 과제 안내 (Individual Challenge)

[데이터 분석 Mini Project – Individual Challenge 안내]

어제 배포된 Train.csv를 활용하여 독성 예측 모델링을 진행합니다.
작성한 모델로 predict_input.csv의 독성을 예측하고, 예측값(0/1)을 output 칼럼에 입력하여 predict_output_작성자.csv 형식의 결과파일을 만든 뒤 아래 제출 링크를 통해 제출해주세요.
Challenge는 16:30에 종료합니다.

- 제출 링크: 슬랙 공지 참고 (저장소 공유 시 노출되지 않도록 URL은 적지 않는다)

유의사항

- 예시 파일 predict_output_홍길동.csv를 참고하여 제출 형식을 지켜주세요.
- 제출파일의 칼럼명은 변경하지 말아주세요. (SMILES, output)
- 개인별 제출 횟수는 최대 4회입니다.
- 각 제출 결과는 Leaderboard를 통해 확인할 수 있습니다.
- Leaderboard 반영 시간: 11:00 / 11:30 / 13:30 / 14:30 / 15:00 / 15:30 / 16:00 / 16:10 / 16:20 / 16:30
- 위 시간을 기준으로 일괄 채점하므로, 반영을 원하는 시간보다 조금 여유 있게 제출해주세요.

이번 Challenge에서는 단순히 한 번의 높은 점수를 만드는 것보다, 모델을 개선하면서 성능이 어떻게 변하는지 직접 확인해보는 과정에 집중해보세요.
여러 가지 가설을 세우고 실험해보면서, 어제 EDA에서 발견한 내용을 모델링에 잘 연결해보시기 바랍니다.
Leaderboard도 확인해보면서 본인의 모델을 끝까지 개선해보세요.
마지막 제출까지 좋은 실험과 도전 이어가시기 바랍니다!

## 평가 기준 (10/20, 구두 전달 메모)

들은 내용을 적은 메모이므로 정확한 표현은 교수님께 확인한다.

- 다중공선성이 있는 컬럼을 **어떻게 찾아서 어떻게 해소했는지** 나타낼 것. 다중공선성이 있어도 성능은 좋을 수 있다. 그러나 변수 간의 관계 해석에 장애가 될 수 있다.
- 학습에 사용한 피처들이 종속변수에 대해 **각각 어느 정도의 영향력(설명력)** 을 가지는지 나타낼 것. 불필요한 컬럼이 많다고 좋은 것이 아니다. 아이리스 데이터도 사실 2개 컬럼만 있어도 된다.
- RDKit 라이브러리로 만든 파생변수를 활용하는 것은 좋은 방법이다.
- 내일(10/21) 한 번에 중간 제출 자료를 업데이트해서 같이 제출해도 된다.
  - 확인 필요: "중간 제출 자료"가 Day1 EDA 발표자료의 갱신본인지, 모델링 결과를 담은 자료인지 메모만으로는 알 수 없다.

현재 프로젝트 상태 (EDA 기준, 2026-10-20)

| 평가 기준 | 지금까지 한 것 | 남은 것 |
|---|---|---|
| 다중공선성 | 연속형 `MolWt`와 `qed`의 Spearman 상관 -0.76, RDKit 파생변수 204개 중 \|Spearman\| > 0.95 쌍 66개를 **식별만** 함 | 해소 방법(상관 군집별 대표 변수 선택, VIF 등)과 근거를 정해 적용하고 기록. 분할 후 학습 데이터 기준으로 수행 |
| 피처별 설명력 | 3,231개 변수의 Gini와 FDR 보정 q값을 계산함(`reports/tables/feature_explanatory_power.csv`). 단, **전체 train 기준 탐색용** | 모델에 실제로 쓴 피처에 대해 분할 후 학습 데이터로 다시 계산하거나 SHAP, Permutation Importance로 제시. 불필요한 피처 제거 근거 포함 |
| RDKit 파생변수 | 217개를 계산하고 204개로 정리함. 설명력 상위는 `clogp`와 극성 계열(`rd_TPSA` 등) | 모델에 포함했을 때의 효과를 변수군별로 비교 |
| 중간 제출 자료 | Day1 EDA 발표자료 제출 완료 | 갱신 범위를 교수님께 확인 |

## 일반적인 머신러닝 프로세스 (딥러닝 X, 교수님 참고자료)

1. 독립변수와 종속 변수 분리
2. Train Set과 Test Set 분리
3. 탐색적 데이터 분석 (EDA)
4. 이상치 진단 및 처리
5. 데이터 전처리 (결측치 대체, 스케일링, 인코딩)
6. 클래스 불균형 처리 (SMOTE, Under-sampling, class_weight) - 불균형 정도나 상태에 따라 선택
7. 상관관계 분석 및 다중공선성(VIF) 진단/제거
8. 추가 특성 선택 (Filter, Wrapper, Embedded)
9. 모델 선택 및 하이퍼파라미터 튜닝 (GridSearchCV / RandomizedSearchCV / Optuna(제 최애입니다))
10. 최적 모델 도출 (best_estimator_)
11. 분류 임계값 최적화 (Threshold Tuning via PR-Curve)
12. 최종 예측 (.predict (클래스 예측), .predict_proba (각 클래스별 확률 추출))
13. 다면적 성능 평가 (Confusion Matrix, F1, ROC-AUC, Brier Score)
14. 모델 해석 및 설명 (SHAP, Permutation Importance)

프로젝트때 마다 이 모든 과정을 수행해야 하는 것은 아닙니다. 데이터 상태나 프로젝트의 목적에 따라 중간에 빠지는 과정도 있습니다.

## 분석 가이드 (윤재성 교수님 제공)

1. 탐색적 데이터 분석 (EDA, Exploratory Data Analysis)

데이터의 형태, 통계적 특성, 분포, 잠재적 문제점을 파악하는 단계입니다.

- 데이터 구조 및 기본 정보 파악:
- 행/열 크기, 데이터 타입('dtypes'), 메모리 사용량 확인
- 타깃 변수의 유형(수치형 vs 범주형) 및 클래스 불균형(Class Imbalance) 여부 점검

- 단변량 분석 (Univariate Analysis):
- 수치형: 평균, 중앙값, 왜도(Skewness), 첨도(Kurtosis), 이상치 점검 (히스토그램, 박스플롯)
- 범주형: 고유값 개수(Cardinality), 빈도수 분포 (막대 차트)

- 다변량 및 상관관계 분석 (Multivariate Analysis):
- 독립변수 간 및 독립변수-종속변수 간 상관계수(피어슨, 스피어만) 히트맵
- 산점도(Scatter Plot), 페어플롯(Pairplot)을 통한 비선형 패턴 및 군집 형태 탐색

- 결측치 및 이상치 식별:
- 변수별 결측 비율(%) 산출 및 결측 패턴(MCAR, MAR, MNAR) 파악

1. 데이터 분할 (Data Splitting)

> 데이터 누수(Data Leakage) 방지 원칙:
> 결측치 대체 기준값(평균/중앙값), 스케일러 파라미터, 인코딩 맵핑 등 모든 전처리 기준은 오직 학습 데이터(Train Set)로부터만 학습되어야 하므로 전처리 전에 분할을 우선 수행합니다.

- 분할 방식 결정:
- 일반적인 경우: 'train_test_split' (통상 7:3 또는 8:2)
- 분류 문제: 타깃 클래스 비율을 유지하는 층화 추출('stratify=y')
- 시계열 데이터: 미래 시점의 데이터를 누수하지 않도록 시간순 분할('TimeSeriesSplit')

1. 데이터 전처리 및 특성 공학 (Feature Engineering)

EDA에서 발견한 데이터를 모델이 학습하기 적합한 수치 행렬 형태로 가공합니다.

- 결측치 및 이상치 처리:
- 결측치: 단순 대체(평균, 중앙값, 최빈값), 모델 기반 대체(KNN, IterativeImputer), 결측 지시자 플래그 추가
- 이상치: IQR 기반 클리핑(Winsorization), 로그 변환을 통한 정규화 유도

- 인코딩 (Categorical Encoding):
- 순서형(Ordinal): 순서형 정수 인코딩
- 명목형(Nominal): 원-핫 인코딩(One-Hot Encoding, 저차원), 타깃 인코딩(Target Encoding) 또는 빈도 인코딩(High Cardinality)

- 스케일링 및 변환 (Scaling & Transformation):
- 거리/선형 기반 모델(SVM, KNN, Logistic, NN): 'StandardScaler', 'MinMaxScaler', 'RobustScaler'
- 왜도가 심한 수치형: 로그 변환('log1p'), Box-Cox 변환

- 파생 변수 생성 및 특성 선택 (Feature Selection):
- 도메인 지식 기반 변수 조합 (비율, 집계값, 날짜 분해 등)
- 저분산 변수 제거, 상관계수/VIF 기반 다중공선성 해소, RFE 또는 트리 기반 중요도 필터링

1. 파이프라인 구축 및 베이스라인 모델 수립

- 엔드투엔드 파이프라인 구성:
- 전처리(ColumnTransformer)와 모델을 Scikit-learn 'Pipeline'으로 결합하여 평가/서빙 시 일관성 확보

- 베이스라인(Baseline) 모델 수립:
- 복잡한 알고리즘 이전에 단순한 휴리스틱 모델(더미 모델)이나 기본 Linear/Logistic Regression, 단일 Decision Tree로 최소 성능 기준선 확보

1. 모델 탐색 및 학습 (Model Training)

문제의 도메인과 데이터 크기, 해석력 요구 수준에 맞는 후보 알고리즘을 선정하고 교차 검증을 통해 학습합니다.

- 후보 모델군 선정:
- 정형 데이터 기준: 선형 모델(Ridge, Lasso, ElasticNet), 트리 앙상블(Random Forest, XGBoost, LightGBM, CatBoost)

- 교차 검증 (Cross Validation, CV):
- $K$-Fold 또는 Stratified $K$-Fold (통상 5-Fold)로 데이터 전반에 대한 일반화 성능 평가

- 평가 지표 정의:
- 회귀: RMSE, MAE, R², MAPE
- 분류: Balanced Accuracy, F1-Score, ROC-AUC, PR-AUC (불균형 데이터일 때 Accuracy 배제)

1. 하이퍼파라미터 튜닝 및 앙상블 (Optimization)

성능이 우수한 상위 2~3개 모델을 선별하여 최적 파라미터를 탐색합니다.

- 튜닝 전략:
- 무작위 탐색: 'RandomizedSearchCV' (넓은 탐색 공간)
- 베이지안 최적화: 'Optuna', 'BayesSearchCV' (목적 함수 기반 효율적 수렴)

- 앙상블 기법 적용:
- 보팅(Voting / Stacking): 상호 보완적인 이종 모델(예: LightGBM + Logistic + CatBoost)의 예측값을 메타 모델로 결합

1. 최종 검증 및 모델 해석 (Evaluation & Interpretation)

- 테스트 셋 최종 평가:
- 전처리 및 튜닝 과정에 전혀 관여하지 않은 보류 데이터(Test Set)를 통해 최종 일반화 성능 산출

- 모델 해석력 확보:
- Permutation Importance, Feature Importance 분석
- SHAP(SHapley Additive exPlanations) 값을 통한 개별/글로벌 변수 기여도 시각화
