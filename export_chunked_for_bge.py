import os
import psycopg2
import pandas as pd
import re

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # We will export only mapo-gu for the quick local test again.
    cur.execute("""
        SELECT id, name, category, COALESCE(description, '') as description
        FROM places 
        WHERE address LIKE '%마포구%'
    """)
    rows = cur.fetchall()
    
    chunked_data = []
    
    for row in rows:
        pid, name, category, desc = row
        
        # Clean up the description
        # It contains " | [네이버 블로그 리뷰] " etc.
        desc = desc.replace("| [네이버 블로그 리뷰]", " ")
        desc = desc.replace("| [카카오 블로그 리뷰]", " ")
        
        # Split into chunks based on "..." which is the typical snippet delimiter from Naver/Kakao
        # Or if not present, just split by common boundaries if needed, but Naver always ends snippets with ...
        raw_chunks = [c.strip() for c in desc.split("...") if len(c.strip()) > 10]
        
        # Filter out junk chunks that don't have enough meaning
        valid_chunks = []
        for c in raw_chunks:
            # If it's just random dates or numbers without much text, skip
            if len(re.sub(r'[^가-힣a-zA-Z]', '', c)) > 5:
                valid_chunks.append(c)
                
        if not valid_chunks:
            # Fallback: No valid reviews found! Use Name + Category as requested.
            fallback_text = f"{name} {category}"
            chunked_data.append({
                "id": pid,
                "text_val": fallback_text,
                "name": name,
                "category": category
            })
        else:
            # Insert a row for each chunk! ONLY the review text.
            for chunk in valid_chunks:
                chunked_data.append({
                    "id": pid,
                    "text_val": chunk,
                    "name": name,
                    "category": category
                })

    df = pd.DataFrame(chunked_data)
    
    os.makedirs('data', exist_ok=True)
    out_path = 'data/mapo_chunked_for_bge.csv'
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Original places: {len(rows)}")
    print(f"Total chunks generated: {len(df)}")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
