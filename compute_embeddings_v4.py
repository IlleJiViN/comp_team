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
BATCH_SIZE = 1000  # Balanced batch size for CPU

def run_embedding_v4(engine, model, total_remaining):
    processed = 0
    t_start = time.perf_counter()
    
    try:
        queries = [
            "SELECT id, name || ' ' || category AS text_val FROM places WHERE embedding_vector_v4 IS NULL AND address LIKE '서울특별시%' LIMIT :batch_size",
            "SELECT id, name || ' ' || category AS text_val FROM places WHERE embedding_vector_v4 IS NULL AND address NOT LIKE '서울특별시%' LIMIT :batch_size"
        ]
        
        for current_query in queries:
            while True:
                with engine.connect() as conn:
                    result = conn.execute(text(current_query), {"batch_size": BATCH_SIZE}).all()
                    
                if not result:
                    break
                
                ids = [row[0] for row in result]
                texts = [row[1] for row in result]
                
                valid_pairs = [(idx, txt) for idx, txt in zip(ids, texts) if txt is not None and str(txt).strip() != ""]
                invalid_ids = [idx for idx, txt in zip(ids, texts) if txt is None or str(txt).strip() == ""]
                
                # Update invalid ids with zeros so they aren't fetched again
                if invalid_ids:
                    with engine.begin() as conn:
                        conn.execute(
                            text("UPDATE places SET embedding_vector_v4 = :emb WHERE id = :id"),
                            [{"emb": [0.0]*768, "id": place_id} for place_id in invalid_ids]
                        )
                
                if not valid_pairs:
                    continue
                
                valid_ids = [p[0] for p in valid_pairs]
                valid_texts = [p[1] for p in valid_pairs]
                
                with torch.no_grad():
                    embeddings = model.encode(valid_texts, convert_to_numpy=True, normalize_embeddings=True).tolist()
                
                t_batch_start = time.perf_counter()
                with engine.begin() as conn:
                    conn.execute(
                        text("UPDATE places SET embedding_vector_v4 = :emb WHERE id = :id"),
                        [{"emb": emb, "id": place_id} for place_id, emb in zip(valid_ids, embeddings)]
                    )
                processed += len(result)
                elapsed = time.perf_counter() - t_start
                batch_time = time.perf_counter() - t_batch_start
                
                rate = processed / elapsed if elapsed > 0 else 0
                est_remaining_seconds = (total_remaining - processed) / rate if rate > 0 else 0
                
                print(f"[BATCH] V4 Processed: {processed}/{total_remaining} | "
                      f"Batch write: {batch_time*1000:.1f}ms | "
                      f"Speed: {rate:.1f} rows/s | "
                      f"EST Remaining: {est_remaining_seconds/60:.1f} minutes")

    except KeyboardInterrupt:
        print("\n[INFO] Generation for V4 paused by user.")
        raise KeyboardInterrupt
    except Exception as e:
        print(f"\n[ERROR] Generation for V4 failed: {str(e)}")
        raise e

def main():
    print("="*80)
    print("      SpotSync AI - V4 Name+Category Embedding Generator")
    print("="*80)
    
    print(f"[MODEL] Loading '{MODEL_NAME}' on {DEVICE}...")
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print("[MODEL] Model loaded successfully.")

    engine = create_engine(DATABASE_URL)
    
    print("[DB] Ensuring 'embedding_vector_v4' column exists...")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_vector_v4 float8[];"))
    
    with engine.connect() as conn:
        total_rows = conn.execute(text("SELECT count(*) FROM places")).scalar()
        total_remaining_v4 = conn.execute(text("SELECT count(*) FROM places WHERE embedding_vector_v4 IS NULL")).scalar()
    
    print(f"[DB] Total rows in DB: {total_rows}")
    print(f"[DB] V4 Rows remaining: {total_remaining_v4}")
    
    if total_remaining_v4 == 0:
        print("[SUCCESS] All places have already been successfully embedded for V4!")
        return
        
    try:
        print(f"\n[PROCESS] Starting V4 Embedding Generation ({total_remaining_v4} rows)...")
        run_embedding_v4(engine, model, total_remaining_v4)
        print("\n🎉 V4 embedding generation finished successfully!")
    except KeyboardInterrupt:
        print("\n[INFO] Background generation paused by user.")
    except Exception as e:
        print(f"\n[ERROR] Background generation failed: {str(e)}")
    print("="*80)

if __name__ == "__main__":
    main()
