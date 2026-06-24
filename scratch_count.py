import sys
import glob
import pandas as pd
from categories import CATEGORY_DESCRIPTIONS

sys.stdout.reconfigure(encoding='utf-8')
cats = list(CATEGORY_DESCRIPTIONS.keys())
total = 0
matched = 0

csv_files = glob.glob('data/소상공인시장진흥공단_상가*.csv')
print(f"Found {len(csv_files)} CSV files.")

for f in csv_files:
    try:
        df = pd.read_csv(f, usecols=['상권업종소분류명', '상권업종대분류명'], encoding='utf-8', encoding_errors='replace', dtype=str)
        total += len(df)
        c_col = '상권업종소분류명' if '상권업종소분류명' in df.columns else '상권업종대분류명'
        matched_rows = df[df[c_col].isin(cats)]
        matched += len(matched_rows)
        print(f"  - {f}: {len(df)} rows, {len(matched_rows)} matched")
    except Exception as e:
        print(f"  - Error reading {f}: {e}")

print(f"Total rows: {total}")
print(f"Total matched: {matched}")
