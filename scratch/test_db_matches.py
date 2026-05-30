import sys
import glob
import pandas as pd
from sqlalchemy import create_engine, text

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    
    engine = create_engine("postgresql://postgres:postgres@localhost:5432/spotsync")
    with engine.connect() as conn:
        p_ids = set([r[0] for r in conn.execute(text('SELECT place_id FROM places WHERE embedding_vector IS NOT NULL')).all()])
        
    path = glob.glob("*경기*.csv")[0]
    df = pd.read_csv(path, usecols=['상가업소번호', '상권업종대분류명', '상권업종중분류명', '상권업종소분류명'], dtype=str, encoding='utf-8')
    matched = df[df['상가업소번호'].isin(p_ids)]
    
    sub_cats = sorted(matched['상권업종소분류명'].dropna().unique())
    print(f"Total Unique Sub-Categories in 6750 rows: {len(sub_cats)}")
    for i in range(0, len(sub_cats), 5):
        print(", ".join(sub_cats[i:i+5]))

if __name__ == "__main__":
    main()
