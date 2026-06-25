import sys
import glob
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

for f in glob.glob('data/소상공인시장진흥공단_상가*.csv'):
    # Try reading as UTF-8
    try:
        df = pd.read_csv(f, nrows=1, encoding='utf-8')
        if '상가업소번호' in df.columns:
            print(f"{f}: UTF-8")
        else:
            print(f"{f}: UTF-8 (but columns: {df.columns.tolist()})")
    except UnicodeDecodeError:
        # Try reading as CP949
        try:
            df = pd.read_csv(f, nrows=1, encoding='cp949')
            if '상가업소번호' in df.columns:
                print(f"{f}: CP949")
            else:
                print(f"{f}: CP949 (but columns: {df.columns.tolist()})")
        except Exception as e:
            print(f"{f}: Failed CP949 too: {e}")
    except Exception as e:
        print(f"{f}: Other exception on UTF-8: {e}")
