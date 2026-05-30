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
    print("      SpotSync AI - Robust Direct V2 Embedding Generator")
    print("="*80)
    
    engine = create_engine(DATABASE_URL)
    
    # 1. Fetch V1 embedded rows
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, embedding_text_v2 FROM places WHERE embedding_text_v2 IS NOT NULL"
        )).all()
    print(f"Total rows with V2 text: {len(rows)}")
    if not rows:
        print("No V2 text found in DB. Please run data pipeline first.")
        return
        
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    
    # 2. Load model
    print(f"Loading SentenceTransformer: {MODEL_NAME}...")
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    
    # 3. Compute and save in chunks of 500
    chunk_size = 500
    total = len(ids)
    
    t_start = time.perf_counter()
    print(f"Processing V2 vector embeddings in chunks of {chunk_size}...")
    
    for i in range(0, total, chunk_size):
        chunk_ids = ids[i:i+chunk_size]
        chunk_texts = texts[i:i+chunk_size]
        
        # Compute embeddings
        with torch.no_grad():
            embeddings = model.encode(
                chunk_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()
            
        # Update database in a dedicated transaction chunk
        t0 = time.perf_counter()
        try:
            with engine.begin() as conn:
                for db_id, emb in zip(chunk_ids, embeddings):
                    conn.execute(
                        text("UPDATE places SET embedding_vector_v2 = :emb WHERE id = :id"),
                        {"emb": emb, "id": db_id}
                    )
            elapsed = time.perf_counter() - t0
            print(f"[CHUNK] Updated {i + len(chunk_ids)}/{total} | DB write time: {elapsed*1000:.1f}ms")
        except Exception as e:
            print(f"[ERROR CHUNK] Failed to save batch {i}-{i+chunk_size}: {e}")
            sys.exit(1)
            
    print(f"\n🎉 Successfully computed and saved V2 vector embeddings in {time.perf_counter() - t_start:.2f} seconds!")
    print("="*80)

if __name__ == "__main__":
    main()
