import sys
import glob
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
csv_files = glob.glob("*.csv")
gyeonggi_files = [f for f in csv_files if "경기" in f]
csv_file_path = gyeonggi_files[0] if gyeonggi_files else csv_files[0]

print(f"Testing pandas UTF-8 load on: {csv_file_path}")
try:
    df = pd.read_csv(csv_file_path, sep=',', encoding='utf-8', encoding_errors='replace', nrows=10, dtype=str)
    print("Success! Columns:")
    print(df.columns.tolist())
    print("\nSample values for '상호명':")
    name_col = [c for c in df.columns if "상호명" in c][0]
    print(df[name_col].tolist())
except Exception as e:
    print("Error:", e)
