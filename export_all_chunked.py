import os
import psycopg2
import pandas as pd
import re

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Export all enriched places
    cur.execute("""
        SELECT id, name, category, COALESCE(description, '') as description
        FROM places 
        WHERE is_enriched = TRUE AND description != ''
    """)
    rows = cur.fetchall()
    
    chunked_data = []
    
    for row in rows:
        pid, name, category, desc = row
        
        desc = desc.replace("| [네이버 블로그 리뷰]", " ")
        desc = desc.replace("| [카카오 블로그 리뷰]", " ")
        
        raw_chunks = [c.strip() for c in desc.split("...") if len(c.strip()) > 10]
        
        valid_chunks = []
        for c in raw_chunks:
            if len(re.sub(r'[^가-힣a-zA-Z]', '', c)) > 5:
                valid_chunks.append(c)
                
        if not valid_chunks:
            fallback_text = f"{name} {category}"
            chunked_data.append({
                "id": pid,
                "text_val": fallback_text,
                "name": name,
                "category": category
            })
        else:
            for chunk in valid_chunks:
                chunked_data.append({
                    "id": pid,
                    "text_val": chunk,
                    "name": name,
                    "category": category
                })

    df = pd.DataFrame(chunked_data)
    
    os.makedirs('data', exist_ok=True)
    out_path = 'data/all_chunked_for_bge.csv'
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Original places: {len(rows)}")
    print(f"Total chunks generated: {len(df)}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
