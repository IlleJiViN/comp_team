import sys
import glob
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

for f in glob.glob('data/소상공인시장진흥공단_상가*.csv'):
    encoding = 'cp949' if '강원' in f else 'utf-8'
    try:
        df = pd.read_csv(f, encoding=encoding, encoding_errors='replace', dtype=str)
        matched = df[df['상호명'].astype(str).str.contains('블링블링', na=False)]
        if len(matched) > 0:
            print(f"File: {f}")
            print(matched[['상가업소번호', '상호명', '도로명주소']].head(5))
            print("-" * 50)
    except Exception as e:
        print(f"Error reading {f}: {e}")
