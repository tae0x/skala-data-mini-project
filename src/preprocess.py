"""모델링용 전처리 모듈: 상관 기반 중복 변수 제거와 파이프라인 생성."""
import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CorrFilter(BaseEstimator, TransformerMixin):
    """|Spearman| 상관이 threshold를 넘는 변수를 하나만 남기는 필터(다중공선성 완화).

    라벨을 쓰지 않으므로 Pipeline 안에서 fit 데이터(학습 fold)만으로 학습해도 누수가 없다.
    열 순서대로 훑으면서, 이미 남긴 변수와 상관이 threshold 이하일 때만 남긴다.
    그래서 앞쪽 열이 우선한다(원본 수치형 컬럼을 앞에 두면 파생변수보다 먼저 남는다).
    """

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = np.abs(np.nan_to_num(np.corrcoef(rankdata(X, axis=0), rowvar=False)))
        keep = []
        for j in range(corr.shape[0]):
            if not keep or corr[j, keep].max() <= self.threshold:
                keep.append(j)
        self.keep_ = np.array(keep)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


class VifFilter(BaseEstimator, TransformerMixin):
    """VIF가 threshold를 넘는 변수를 가장 큰 것부터 하나씩 제거하는 필터(선형 결합 형태의 다중공선성 해소).

    VIF는 상관행렬 역행렬의 대각이다. 정확한 선형 종속(예: 고리 수 = 방향족 + 포화)이 있으면
    역행렬이 불안정하므로 유사 역행렬(pinv)을 쓴다. 라벨을 쓰지 않아 fold 안에서 fit해도 누수가 없다.

    protect: 앞의 protect개 열은 제거하지 않는다(해석이 쉬운 원본 수치형을 지키고 파생변수 쪽을 먼저 제거).
    """

    def __init__(self, threshold=10.0, protect=0):
        self.threshold = threshold
        self.protect = protect

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = np.nan_to_num(np.corrcoef(X, rowvar=False))
        keep = list(range(corr.shape[0]))

        def vifs():
            return np.diag(np.linalg.pinv(corr[np.ix_(keep, keep)]))

        # 제거 가능한 변수 중 VIF가 가장 큰 것부터 제거. 보호한 변수는 VIF가 커도 남긴다(해석 우선).
        while True:
            removable = [i for i, j in enumerate(keep) if j >= self.protect]
            if not removable:
                break
            vif = vifs()
            worst = max(removable, key=lambda i: vif[i])
            if vif[worst] <= self.threshold:
                break
            keep.pop(worst)
        self.keep_ = np.array(keep)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


def build_pipeline(model, group_cols, groups, scale=False, corr=None, vif=None):
    """피처군(cont, rdkit, fp, maccs, morgan_cnt, ap_cnt, tt_cnt)을 골라 전처리 + 모델 Pipeline을 만든다.

    수치형(cont, rdkit): 중앙값 대체 → 상수열 제거 → (상관 필터) → (VIF 필터) → (표준화)
    그 밖의 그룹(fp, maccs, morgan_cnt 등): 상수열 제거 → (상관 필터)
    모든 단계는 fit 데이터에서만 학습된다.
    """
    num_cols = [c for g in groups if g != "fp" for c in group_cols[g]]
    transformers = []
    if num_cols:
        steps = [("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0))]
        if corr:
            steps.append(("corr", CorrFilter(corr)))
        if vif:
            # 수치형 분기의 앞쪽 열(원본 수치형)은 지키고 RDKit 파생변수를 먼저 제거한다.
            # cont의 마지막 열(MolWt)은 RDKit 질량 기술자와 구조적으로 중복돼 보호하지 않는다.
            n_protect = len(group_cols["cont"]) - 1 if "cont" in groups else 0
            steps.append(("vif", VifFilter(vif, protect=n_protect)))
        if scale:
            steps.append(("sc", StandardScaler()))
        transformers.append(("num", Pipeline(steps), num_cols))
    for g in groups:
        if g in ("cont", "rdkit"):
            continue
        # 핑거프린트와 추가 분자 표현(비트, 카운트): 상수열 제거 → (상관 필터)
        steps = [("var", VarianceThreshold(0.0))]
        if corr:
            steps.append(("corr", CorrFilter(corr)))
        transformers.append((g, Pipeline(steps), group_cols[g]))
    return Pipeline([("prep", ColumnTransformer(transformers, remainder="drop")), ("model", model)])


def build_selected_pipeline(model, group_cols, groups, k=1000, seed=42):
    """전처리 → LightGBM gain 중요도 상위 k개 선택 → 모델. 선택도 fit 데이터(학습 fold)에서만 수행한다."""
    base = build_pipeline(model, group_cols, groups)
    selector = SelectFromModel(
        LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3, subsample=0.8,
                       subsample_freq=1, importance_type="gain", random_state=seed, n_jobs=3, verbose=-1),
        threshold=-np.inf, max_features=k)
    return Pipeline([("prep", base.named_steps["prep"]), ("sel", selector), ("model", model)])
