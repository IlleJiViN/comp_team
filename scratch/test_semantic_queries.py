import os
import sys
import numpy as np
import torch
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEVICE = "cpu"

def test_query(engine, model, query_str):
    print(f"\nQuery: '{query_str}'")
    print("-" * 80)
    
    # 1. Generate query vector
    with torch.no_grad():
        q_vector = model.encode([query_str], convert_to_numpy=True, normalize_embeddings=True).astype('float32').squeeze().tolist()
        
    # 2. Query DB using cosine similarity (dot product since normalized)
    # We query places where embedding_vector_v3 is populated
    sql = text("""
        SELECT name, category, address, embedding_text_v3,
               (SELECT sum(x*y) FROM unnest(embedding_vector_v3) WITH ORDINALITY AS u1(x, i) 
                JOIN unnest(:q_vector) WITH ORDINALITY AS u2(y, j) ON u1.i = u2.j) AS similarity
        FROM places
        WHERE embedding_vector_v3 IS NOT NULL
        ORDER BY similarity DESC
        LIMIT 5
    """)
    
    with engine.connect() as conn:
        results = conn.execute(sql, {"q_vector": q_vector}).all()
        
    for i, r in enumerate(results, 1):
        print(f"#{i} Match: {r[0]} ({r[1]})")
        print(f"   Similarity Score: {r[4]:.4f}")
        print(f"   Address: {r[2]}")
        print(f"   V3 Text: {r[3][:150]}...")
        print()

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("="*80)
    print("      SpotSync AI - V3 Real-Time Semantic Query Diagnostic Test")
    print("="*80)
    
    engine = create_engine(DATABASE_URL)
    
    # Check how many rows are embedded
    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM places WHERE embedding_vector_v3 IS NOT NULL")).scalar()
    print(f"[DB] Found {count} rows with active V3 embeddings.")
    
    if count == 0:
        print("[ERROR] No V3 embeddings found. Please wait for generation script to process some batches.")
        return
        
    print(f"[MODEL] Loading '{MODEL_NAME}' on {DEVICE}...")
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    
    test_queries = [
        "노트북 들고 공부하기 좋은 조용하고 편안한 카페",
        "안주가 맛있고 시원한 맥주 마시기 좋은 감성 술집",
        "가족 외식하기 좋은 갈비 삼겹살 숯불 고기집 한식 맛집"
    ]
    
    for q in test_queries:
        test_query(engine, model, q)

if __name__ == "__main__":
    main()
