import os
import time
import sys
import numpy as np
import torch
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

# Reconfigure stdout/stderr encoding for UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEVICE = "cpu"
BATCH_SIZE = 250  # Balanced batch size for CPU

def run_embedding_phase(engine, model, text_col, vector_col, total_remaining):
    processed = 0
    t_start = time.perf_counter()
    
    try:
        while True:
            # Fetch a batch of places that don't have embeddings for this column
            with engine.connect() as conn:
                result = conn.execute(text(
                    f"SELECT id, {text_col} FROM places WHERE {vector_col} IS NULL LIMIT :batch_size"
                ), {"batch_size": BATCH_SIZE}).all()
                
            if not result:
                break
                
            ids = [row[0] for row in result]
            texts = [row[1] for row in result]
            
            # Filter out any None or empty values to prevent model failure
            valid_pairs = [(idx, txt) for idx, txt in zip(ids, texts) if txt is not None and str(txt).strip() != ""]
            if not valid_pairs:
                break
                
            valid_ids = [p[0] for p in valid_pairs]
            valid_texts = [p[1] for p in valid_pairs]
            
            # Compute embeddings
            with torch.no_grad():
                embeddings = model.encode(
                    valid_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).tolist()  # Convert numpy array to list for PostgreSQL float8[] compatibility
                
            # Update database using batch update in transaction
            t_batch_start = time.perf_counter()
            with engine.begin() as conn:
                # We execute batch updates
                for place_id, emb in zip(valid_ids, embeddings):
                    conn.execute(
                        text(f"UPDATE places SET {vector_col} = :emb WHERE id = :id"),
                        {"emb": emb, "id": place_id}
                    )
                    
            processed += len(valid_ids)
            elapsed = time.perf_counter() - t_start
            batch_time = time.perf_counter() - t_batch_start
            
            rate = processed / elapsed if elapsed > 0 else 0
            est_remaining_seconds = (total_remaining - processed) / rate if rate > 0 else 0
            
            print(f"[BATCH] {vector_col} Processed: {processed}/{total_remaining} | "
                  f"Batch write: {batch_time*1000:.1f}ms | "
                  f"Speed: {rate:.1f} rows/s | "
                  f"EST Remaining: {est_remaining_seconds/60:.1f} minutes")
    except KeyboardInterrupt:
        print(f"\n[INFO] Generation for {vector_col} paused by user.")
        raise KeyboardInterrupt
    except Exception as e:
        print(f"\n[ERROR] Generation for {vector_col} failed: {str(e)}")
        raise e

def main():
    print("="*80)
    print("      SpotSync AI - Background Embedding Generator")
    print("="*80)
    
    engine = create_engine(DATABASE_URL)
    
    # 1. Add embedding_vector columns if not exists
    print("[DB] Ensuring 'embedding_vector', 'embedding_vector_v2' & 'embedding_vector_v3' columns exist...")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_vector float8[];"))
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_vector_v2 float8[];"))
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_text_v2 Text;"))
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_vector_v3 float8[];"))
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_text_v3 Text;"))
    
    # 2. Count rows remaining for V1, V2 and V3
    with engine.connect() as conn:
        total_remaining_v1 = conn.execute(text("SELECT count(*) FROM places WHERE embedding_vector IS NULL")).scalar()
        total_remaining_v2 = conn.execute(text("SELECT count(*) FROM places WHERE embedding_vector_v2 IS NULL")).scalar()
        total_remaining_v3 = conn.execute(text("SELECT count(*) FROM places WHERE embedding_vector_v3 IS NULL")).scalar()
        total_rows = conn.execute(text("SELECT count(*) FROM places")).scalar()
    
    print(f"[DB] Total rows in DB: {total_rows}")
    print(f"[DB] V1 Rows remaining: {total_remaining_v1}")
    print(f"[DB] V2 Rows remaining: {total_remaining_v2}")
    print(f"[DB] V3 Rows remaining: {total_remaining_v3}")
    
    if total_remaining_v1 == 0 and total_remaining_v2 == 0 and total_remaining_v3 == 0:
        print("[SUCCESS] All places have already been successfully embedded for V1, V2, and V3!")
        return
        
    # 3. Load Model
    print(f"[MODEL] Loading '{MODEL_NAME}' on {DEVICE}...")
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print("[MODEL] Model loaded successfully.")
    
    try:
        # 4. Phase 1: Compute V1 Embeddings
        if total_remaining_v1 > 0:
            print(f"\n[PROCESS] Starting Phase 1: V1 Embedding Generation ({total_remaining_v1} rows)...")
            run_embedding_phase(engine, model, "embedding_text", "embedding_vector", total_remaining_v1)
            
        # 5. Phase 2: Compute V2 Embeddings
        if total_remaining_v2 > 0:
            print(f"\n[PROCESS] Starting Phase 2: V2 Embedding Generation ({total_remaining_v2} rows)...")
            run_embedding_phase(engine, model, "embedding_text_v2", "embedding_vector_v2", total_remaining_v2)
            
        # 6. Phase 3: Compute V3 Embeddings
        if total_remaining_v3 > 0:
            print(f"\n[PROCESS] Starting Phase 3: V3 Embedding Generation ({total_remaining_v3} rows)...")
            run_embedding_phase(engine, model, "embedding_text_v3", "embedding_vector_v3", total_remaining_v3)
            
        print("\n🎉 Background embedding generation finished successfully!")
    except KeyboardInterrupt:
        print("\n[INFO] Background generation paused by user.")
    except Exception as e:
        print(f"\n[ERROR] Background generation failed: {str(e)}")
    print("="*80)

if __name__ == "__main__":
    main()
