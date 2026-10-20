"""제출 파일 형식 검증 스크립트.

사용법(프로젝트 루트에서): .venv/bin/python src/validate_submission.py submissions/predict_output_<작성자>.csv
행 수, 열 이름, 값 범위, SMILES 순서를 검사하고 하나라도 어긋나면 종료 코드 1로 끝난다.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def validate(path: Path) -> list[str]:
    errors = []
    if not (path.name.startswith("predict_output_") and path.suffix == ".csv"):
        errors.append(f"파일 이름이 predict_output_<작성자>.csv 형식이 아니다: {path.name}")
    sub = pd.read_csv(path)
    ref = pd.read_csv(ROOT / "data/predict_input.csv", usecols=["SMILES"])

    if list(sub.columns) != ["SMILES", "output"]:
        errors.append(f"열 이름이 SMILES, output이 아니다: {list(sub.columns)}")
        return errors
    if len(sub) != len(ref):
        errors.append(f"행 수가 다르다: 제출 {len(sub)} / 입력 {len(ref)}")
    elif not (sub["SMILES"].values == ref["SMILES"].values).all():
        errors.append("SMILES 순서나 값이 predict_input.csv와 다르다")
    if sub["output"].isna().any():
        errors.append("output에 결측이 있다")
    elif not set(sub["output"].unique()) <= {0, 1}:
        errors.append(f"output 값이 0/1이 아니다: {sorted(sub['output'].unique())}")
    elif not pd.api.types.is_integer_dtype(sub["output"]):
        errors.append(f"output이 정수형이 아니다: {sub['output'].dtype}")
    return errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("사용법: python src/validate_submission.py <제출 파일 경로>")
    target = Path(sys.argv[1])
    problems = validate(target)
    if problems:
        print("검증 실패")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    out = pd.read_csv(target)["output"]
    print(f"검증 통과: {len(out)}행, label 1 비율 {out.mean():.3f}")
    if out.nunique() == 1:
        print("주의: 예측값이 전부 같다")
