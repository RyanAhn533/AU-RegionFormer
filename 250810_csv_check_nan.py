import pandas as pd
import numpy as np

csv_path = "/mnt/hdd/ajy_25/Aihub_data_AU_Croped/index_train.csv"
df = pd.read_csv(csv_path)

# 기본 정보
print(df.info())
print(df.head())

# NaN, inf 값 체크
print("NaN 포함 여부:\n", df.isna().sum())
print("inf 포함 여부:\n", np.isinf(df.select_dtypes(include=[np.number])).sum())

# 경로가 실제 존재하는지 확인
from pathlib import Path
missing_files = [p for p in df['path_column_name'] if not Path(p).exists()]
print("존재하지 않는 파일 개수:", len(missing_files))
