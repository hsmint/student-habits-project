import os

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit("파일을 찾을 수 없습니다.")
    df = pd.read_csv(path)
    print(f"데이터 로드 완료: {df.shape[0]}행 x {df.shape[1]}열")
    return df


def explore_structure(data: pd.DataFrame) -> None:
    print(f"---------------------\n전체 행 열\n\n{data.shape[0]}행 x {data.shape[1]}열")
    print(f"---------------------\n컬럼별 데이터 타입\n\n{data.dtypes.to_string()}")
    print(f"---------------------\n상위 5행 출력\n\n{data.head(5)}")
    print("---------------------")


def show_statistics(data: pd.DataFrame) -> None:
    print(
        data.describe().round(2)
    )  # count : 데이터 개수, mean : 평균, std : 표준편차, min : 최솟값, 25% : 하위 25%, 50% : 중앙값, 75% : 상위 25%, max : 최댓값
    print(
        f"---------------------\n수치형 컬럼 평균\n\n{data.select_dtypes(include='number').mean().round(2).to_string()}"
    )
    print("---------------------")


def check_missing(data: pd.DataFrame) -> dict:
    total = len(data)
    missing_num = [data[col].isnull().sum() for col in data.columns]
    missing = {
        col: missing_num[i] for i, col in enumerate(data.columns) if missing_num[i] > 0
    }
    print("결측치 확인\n")
    for col, num in missing.items():
        ratio = num / total
        print(
            f"{col}:\t 결측치 수 {num}개 결측치 비율 {num / total:.2%} {'심각' if ratio > 0.2 else '주의' if ratio > 0.05 else '양호'}"
        )
    print("---------------------")
    return missing


def numpy_stats(data: pd.DataFrame, col_name: str) -> None:
    if data[col_name].dtype not in [np.float64, np.int64]:
        raise SystemExit("수치형 데이터가 아닙니다.")
    col_data = data[col_name].dropna().values
    print("컬럼 통계 (numpy 활용)\n")
    print(f"평균\t\t\t\t{np.mean(col_data):.2f}")
    print(f"표준편차\t\t\t{np.std(col_data):.2f}")
    print(f"중앙값\t\t\t\t{np.median(col_data):.2f}")
    print(f"최솟값\t\t\t\t{np.min(col_data):.2f}")
    print(f"최댓값\t\t\t\t{np.max(col_data):.2f}")
    print(f"6시간 이상 공부하는 학생 수 \t{len(col_data[col_data >= 6])}명\n\n")

    print("컬럼 통계 (describe 활용)\n")
    print(data[col_name].describe().round(2).to_string())


def main() -> None:
    data = load_data("data/student_habits.csv")
    explore_structure(data)
    show_statistics(data)
    check_missing(data)
    numpy_stats(data, "study_hours")


if __name__ == "__main__":
    main()
