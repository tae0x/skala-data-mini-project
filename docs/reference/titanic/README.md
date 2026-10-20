# 타이타닉 실습 (교수님 제공 참고 자료)

머신러닝 전체 흐름과 **Optuna 하이퍼파라미터 튜닝** 예제를 보여 주는 참고용 노트북이다. 우리 프로젝트(독성 예측)의 코드가 아니고, 필요한 부분만 참고해서 `notebooks/`에 새로 작성한다. 출처의 저작권 표기는 노트북에서 발견되지 않았으나 사용 범위는 확인하지 않았다.

## 구성

| 노트북 | 내용 | 우리 프로젝트에서 볼 곳 |
|---|---|---|
| `01_타이타닉 사망 생존 데이터 정보 확인` | 크기, dtype, 기본 정보 | |
| `02_타이타닉 사망 생존 데이터 EDA` | 단변량, 다변량 시각화 | |
| `03_타이타닉 사망 생존 데이터 정제` | 결측치, 이상치 처리 | |
| `04_타이타닉 사망 생존 데이터 변환` | 인코딩, 스케일링 | |
| `05_타이타닉 사망 생존 데이터 상관관계 분석` | 상관 분석 | 다중공선성 평가 기준 |
| `06_타이타닉 사망 생존 데이터 피처 선택` | 피처 선택 | 피처별 영향력 평가 기준 |
| `07_타이타닉 사망 생존 데이터 학습` | 모델 10종 Optuna 튜닝과 비교 | **Optuna 참고** |
| `08_타이타닉 사망 생존 데이터 예측` | 저장한 모델로 test 예측, 결과 저장 | 제출 파일 만들기 |

`07`의 모델: KNN, LogisticRegression, SVC, DecisionTree, RandomForest, GradientBoosting, LightGBM, XGBoost, GaussianNB, CatBoost. 각각 `objective(trial)`을 정의하고 `TPESampler(seed=42)`, `MedianPruner`, `n_trials=50`, `StratifiedKFold(5)`와 `cross_val_score`로 점수를 구한다.

## 이 폴더만으로는 실행되지 않는다

노트북이 읽는 `data/titanic_train6.csv`, `data/titanic_test6.csv`, `model/project1.dat`는 저장소에 없다. 출력이 저장돼 있어서 읽는 용도로만 쓴다.

## 우리 프로젝트에 가져올 때 주의할 점

1. **지표**: 실습은 `scoring='accuracy'`다. 우리는 F1(Positive = `label` 1)이므로 `'f1'`로 바꾼다.
2. **누수**: `07`은 전체 X에 `StandardScaler`를 fit한 뒤 교차검증한다. 우리는 분할 후 학습 데이터로만 학습해야 하므로 `Pipeline` 안에서 fit한다. 트리 계열은 스케일링이 필요 없다.
3. **시간**: 열이 3,000개가 넘고 Day2 작업 시간이 제한되어 있다. 10개 모델 × 50 trial은 과하므로 상위 2~3개 모델만 튜닝하고 trial 수를 줄인다.
4. **폰트**: `Malgun Gothic`(Windows)로 설정돼 있다. mac은 `AppleGothic`(주석으로 있음).
5. **저장**: `08`은 모델과 스케일러를 따로 pickle로 저장한다. 우리는 `Pipeline` 하나를 저장하면 된다.
6. **패키지**: `optuna`, `lightgbm`, `xgboost`, `catboost`는 `requirements.txt`에 있지만 아직 설치와 검증 전이다.
