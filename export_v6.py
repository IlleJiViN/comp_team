import pandas as pd
from sqlalchemy import create_engine, text
from categories import CATEGORY_DESCRIPTIONS
import os

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

def chunk_category_desc(desc: str):
    """
    Applies the user's chunking logic:
    '4단어인데 한 단어씩 잇기로 하고'
    This means chunks of 4 words, with 1 word overlap (stride = 3).
    """
    if not desc:
        return ""
    words = desc.split()
    chunks = []
    # 4 words per chunk, sliding by 3 (1 word overlap)
    for i in range(0, len(words), 3):
        chunk = words[i:i+4]
        chunks.append(" ".join(chunk))
        if i + 4 >= len(words):
            break
    # Combine the chunks into a structured text or just space-separated?
    # Usually we just join them to form the embedding text
    return " | ".join(chunks)

def generate_v6_data():
    engine = create_engine(DATABASE_URL)
    
    print("[1] Fetching data from DB...")
    # We will fetch a subset for prototyping to avoid 1 hour Colab wait, OR fetch all?
    # The user says "이번에 넣는 프리미엄 상가 제외한 상가들에 대해...". 
    # Let's fetch everything, but since we are just prototyping the pipeline,
    # let's only export Mapo-gu (sigungu=13) to keep Colab time under 5 minutes.
    # If the user wants all 1.14M, they can change the WHERE clause.
    
    query = """
    SELECT id, name, category, description, is_premium 
    FROM places 
    WHERE address LIKE '%%마포구%%' AND name IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    
    print(f"[2] Processing {len(df)} places...")
    
    texts = []
    for _, row in df.iterrows():
        name = str(row['name'])
        cat = str(row['category']) if row['category'] else ""
        
        # Check if premium (has TourAPI description)
        if row['is_premium'] and row['description']:
            # Premium: Name + Category + Full TourAPI Description
            text_val = f"{name} {cat} {row['description']}"
        else:
            # Non-Premium: Name + Category + Chunked Category Description
            cat_desc = CATEGORY_DESCRIPTIONS.get(cat, "")
            chunked_desc = chunk_category_desc(cat_desc)
            text_val = f"{name} {cat} {chunked_desc}"
            
        texts.append(text_val)
        
    df['text_val'] = texts
    
    # Export only needed columns for Colab
    export_df = df[['id', 'text_val', 'is_premium']]
    
    os.makedirs('data', exist_ok=True)
    export_path = 'data/places_for_bge_v6.csv'
    export_df.to_csv(export_path, index=False, encoding='utf-8')
    print(f"[DONE] Exported {len(export_df)} rows to {export_path}")

if __name__ == "__main__":
    generate_v6_data()
