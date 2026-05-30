import os
import time
import sys
import glob
import pandas as pd
from sqlalchemy import create_engine, text
import torch
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEVICE = "cpu"

def main():
    print("="*80)
    print("      SpotSync AI - Migration & V2 Embedding Generator")
    print("="*80)
    
    engine = create_engine(DATABASE_URL)
    
    # 1. Add embedding columns if not exists
    print("[DB] Ensuring 'embedding_text_v2' & 'embedding_vector_v2' columns exist...")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_text_v2 Text;"))
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_vector_v2 float8[];"))
        
    # 2. Get rows with existing V1 embeddings
    print("[DB] Fetching rows with existing V1 embeddings...")
    with engine.connect() as conn:
        existing_rows = conn.execute(text(
            "SELECT id, place_id, name, category, address FROM places WHERE embedding_vector IS NOT NULL"
        )).all()
    
    print(f"[DB] Found {len(existing_rows)} rows with existing V1 embeddings.")
    if not existing_rows:
        print("[INFO] No existing V1 embeddings found to migrate. Please load data first.")
        return
        
    existing_place_ids = {row[1] for row in existing_rows}
    
    # 3. Read CSV to get category hierarchy (major, mid, branch)
    print("[CSV] Locating target Gyeonggi-do CSV file...")
    csv_files = glob.glob("*.csv")
    gyeonggi_files = [f for f in csv_files if "경기" in f]
    csv_file_path = gyeonggi_files[0] if gyeonggi_files else csv_files[0]
    
    print(f"[CSV] Reading metadata mapping from: {csv_file_path}")
    t0 = time.perf_counter()
    
    # Read only required columns to save memory
    cols_to_read = []
    # Peek at columns first
    peek_df = pd.read_csv(csv_file_path, nrows=5)
    id_col = [col for col in peek_df.columns if "상가업소번호" in col][0]
    branch_col = [col for col in peek_df.columns if "지점명" in col][0]
    major_col = [col for col in peek_df.columns if "상권업종대분류명" in col][0]
    mid_col = [col for col in peek_df.columns if "상권업종중분류명" in col][0]
    
    print(f"[CSV] Parsing required columns: {id_col}, {branch_col}, {major_col}, {mid_col}")
    
    # Chunked read to find matching place_ids quickly
    mapping = {}
    chunk_size = 50000
    for chunk in pd.read_csv(csv_file_path, usecols=[id_col, branch_col, major_col, mid_col], dtype=str, chunksize=chunk_size):
        chunk_filtered = chunk[chunk[id_col].isin(existing_place_ids)]
        for _, row in chunk_filtered.iterrows():
            mapping[row[id_col]] = {
                "branch": str(row[branch_col]).strip() if pd.notna(row[branch_col]) else "",
                "major": str(row[major_col]).strip() if pd.notna(row[major_col]) else "",
                "mid": str(row[mid_col]).strip() if pd.notna(row[mid_col]) else ""
            }
            
    print(f"[CSV] Parsed metadata for {len(mapping)} matching records in {time.perf_counter() - t0:.2f} seconds.")
    
    # 4. Generate new V2 text descriptions
    print("[MIGRATE] Generating V2 ultra-compact descriptions...")
    updated_records = []
    for row in existing_rows:
        db_id, place_id, name, cat, address = row
        meta = mapping.get(place_id, {"branch": "", "major": "", "mid": ""})
        
        full_name = f"{name} {meta['branch']}".strip()
        cat_hierarchy = []
        if meta['major']:
            cat_hierarchy.append(meta['major'])
        if meta['mid']:
            cat_hierarchy.append(meta['mid'])
            
        hierarchy_str = " > ".join(cat_hierarchy)
        
        if hierarchy_str:
            desc_v2 = f"{full_name}은 {cat} ({hierarchy_str})"
        else:
            desc_v2 = f"{full_name}은 {cat}"
            
        updated_records.append({
            "id": db_id,
            "desc_v2": desc_v2
        })
        
    # Write V2 text descriptions to PostgreSQL
    print("[DB] Updating 'embedding_text_v2' in database...")
    t_update = time.perf_counter()
    with engine.begin() as conn:
        for r in updated_records:
            conn.execute(
                text("UPDATE places SET embedding_text_v2 = :desc WHERE id = :id"),
                {"desc": r["desc_v2"], "id": r["id"]}
            )
    print(f"[DB] Updated embedding_text_v2 for {len(updated_records)} rows in {time.perf_counter() - t_update:.2f} seconds.")
    
    # 5. Compute V2 Embeddings
    print(f"[MODEL] Loading '{MODEL_NAME}' on {DEVICE}...")
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    
    print("[MODEL] Computing V2 embeddings for the updated rows...")
    t_emb = time.perf_counter()
    
    ids = [r["id"] for r in updated_records]
    texts = [r["desc_v2"] for r in updated_records]
    
    # Compute in batches
    batch_size = 250
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            batch_embs = model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()
            embeddings.extend(batch_embs)
            
    print(f"[MODEL] Computed {len(embeddings)} embeddings in {time.perf_counter() - t_emb:.2f} seconds.")
    
    # Save V2 embeddings to PostgreSQL
    print("[DB] Saving V2 embeddings to 'embedding_vector_v2' column...")
    t_save = time.perf_counter()
    with engine.begin() as conn:
        for db_id, emb in zip(ids, embeddings):
            conn.execute(
                text("UPDATE places SET embedding_vector_v2 = :emb WHERE id = :db_id"),
                {"emb": emb, "db_id": db_id}
            )
            
    print(f"[DB] V2 embeddings saved in {time.perf_counter() - t_save:.2f} seconds.")
    print("\n🎉 V2 Migration & Embedding Generation Completed Successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
