"""SMILES에서 RDKit 분자 기술자(파생변수)를 만드는 모듈."""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

PREFIX = "rd_"  # 원본 컬럼(MolWt, qed 등)과 이름이 겹치지 않도록 붙이는 접두사


def compute_rdkit_descriptors(smiles: pd.Series) -> pd.DataFrame:
    """SMILES 시리즈 -> RDKit 기술자 데이터프레임(행 순서와 인덱스 유지).

    파싱에 실패한 SMILES는 전체 NaN 행이 되고, inf는 NaN으로 바꾼다.
    상수열 제거나 결측 대체는 하지 않는다(분할 후 학습 데이터 기준으로 해야 하므로).
    """
    names = [name for name, _ in Descriptors._descList]
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({PREFIX + n: np.nan for n in names})
            continue
        row = {}
        for name, func in Descriptors._descList:
            try:
                row[PREFIX + name] = func(mol)
            except Exception:
                row[PREFIX + name] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows, index=smiles.index)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_extra_fingerprints(smiles: pd.Series) -> pd.DataFrame:
    """SMILES에서 추가 분자 표현을 만든다: MACCS 167비트, Morgan/AtomPair/TopologicalTorsion 카운트.

    행 단위 계산이라 fit이 없다. 카운트는 uint16으로 저장한다(AtomPair는 255를 넘을 수 있다).
    파싱에 실패한 SMILES는 0으로 채운다.
    """
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    atom_pair = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
    torsion = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
    sizes = {"maccs": 167, "morgan_cnt": 2048, "ap_cnt": 2048, "tt_cnt": 1024}
    blocks = {k: np.zeros((len(smiles), n), dtype=np.uint16) for k, n in sizes.items()}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        blocks["maccs"][i] = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.uint16)
        blocks["morgan_cnt"][i] = morgan.GetCountFingerprintAsNumPy(mol)
        blocks["ap_cnt"][i] = atom_pair.GetCountFingerprintAsNumPy(mol)
        blocks["tt_cnt"][i] = torsion.GetCountFingerprintAsNumPy(mol)
    cols = {f"{k}_{j}": blocks[k][:, j] for k, n in sizes.items() for j in range(n)}
    return pd.DataFrame(cols, index=smiles.index)
