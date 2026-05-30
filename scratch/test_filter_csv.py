import glob
import pandas as pd

csv_files = glob.glob("*.csv")
gyeonggi_files = [f for f in csv_files if "경기" in f]
csv_file_path = gyeonggi_files[0] if gyeonggi_files else csv_files[0]

print(f"Reading CSV from: {csv_file_path}")

# Target columns
cols = ['상가업소번호', '상호명', '지점명', '상권업종대분류명', '상권업종중분류명', '상권업종소분류명', '도로명주소', '위도', '경도', '영업상태명']

try:
    # Read chunk to see categories
    chunk = pd.read_csv(csv_file_path, usecols=cols, nrows=10000, dtype=str, encoding='utf-8')
    print("Columns:", chunk.columns.tolist())
    print("\nSample unique major categories:")
    print(chunk['상권업종대분류명'].unique())
    print("\nSample unique sub-categories:")
    print(chunk['상권업종소분류명'].unique()[:20])
except Exception as e:
    print("Error:", e)
