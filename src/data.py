"""모델링용 데이터 로딩과 분할. 02, 03 노트북이 같은 분할(SEED 42, stratify)을 쓰도록 한 곳에 둔다."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.model_selection import train_test_split

from features import compute_extra_fingerprints, compute_rdkit_descriptors

# 순서 주의: 앞쪽 3개는 VIF 필터에서 보호하고 마지막 MolWt는 RDKit 질량 기술자와 중복돼 제거 대상이 될 수 있다
CONT = ["clogp", "qed", "sa_score", "MolWt"]
# 기존 컬럼과 같은 값으로 확인된 RDKit 기술자(EDA에서 Spearman 1.0)
DUP = ["rd_MolWt", "rd_MolLogP", "rd_qed"]


def _cached(fn, smiles, cache_path):
    if cache_path.exists():
        return pd.read_pickle(cache_path)
    out = fn(smiles)
    out.to_pickle(cache_path)
    return out


EXTRA_GROUPS = ["maccs", "morgan_cnt", "ap_cnt", "tt_cnt"]


def load_all(root: Path, seed=42, test_size=0.2, extra=False):
    """train/predict_input을 읽고 RDKit 파생변수를 붙인 뒤 학습/홀드아웃으로 분할한다.

    extra=True이면 추가 분자 표현(MACCS, Morgan/AtomPair/TopologicalTorsion 카운트)을 X에 붙이고
    groups에 maccs, morgan_cnt, ap_cnt, tt_cnt 키를 추가한다.
    """
    cache = root / "data/cache"
    cache.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(root / "data/train.csv")
    pred = pd.read_csv(root / "data/predict_input.csv")
    fp = [c for c in train.columns if c.split("_")[0] in ("ecfp", "fcfp", "ptfp") and "_" in c]
    for df_ in (train, pred):
        df_[fp] = df_[fp].astype("uint8")

    rd_train = _cached(compute_rdkit_descriptors, train["SMILES"], cache / "rdkit_train.pkl").drop(columns=DUP)
    rd_pred = _cached(compute_rdkit_descriptors, pred["SMILES"], cache / "rdkit_pred.pkl").drop(columns=DUP)
    assert list(rd_train.columns) == list(rd_pred.columns)

    X = pd.concat([train[CONT], rd_train, train[fp]], axis=1)
    y = train["label"]
    X_pred = pd.concat([pred[CONT], rd_pred, pred[fp]], axis=1)
    assert list(X.columns) == list(X_pred.columns)

    groups = {"cont": CONT, "rdkit": list(rd_train.columns), "fp": fp}
    if extra:
        ex_train = _cached(compute_extra_fingerprints, train["SMILES"], cache / "extra_train.pkl")
        ex_pred = _cached(compute_extra_fingerprints, pred["SMILES"], cache / "extra_pred.pkl")
        X, X_pred = pd.concat([X, ex_train], axis=1), pd.concat([X_pred, ex_pred], axis=1)
        assert list(X.columns) == list(X_pred.columns)
        for g in EXTRA_GROUPS:
            groups[g] = [c for c in ex_train.columns if c.startswith(g + "_")]
    X_tr, X_ho, y_tr, y_ho = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    return SimpleNamespace(X=X, y=y, X_pred=X_pred, smiles_pred=pred["SMILES"], groups=groups,
                           X_tr=X_tr, X_ho=X_ho, y_tr=y_tr, y_ho=y_ho)
