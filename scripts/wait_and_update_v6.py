import os
import time
import shutil
import numpy as np
from sqlalchemy import create_engine, text

DOWNLOADS_DIR = os.path.expanduser('~/Downloads')
NPZ_FILENAME = 'embeddings_v6.npz'
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

def update_db(npz_path):
    print(f"\n[1] Found {npz_path}! Loading embeddings...")
    data = np.load(npz_path)
    ids = data['ids']
    embeddings = data['embeddings']
    
    print(f"[2] Normalizing {len(ids)} embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1e-10, norms)
    
    print(f"[3] Updating PostgreSQL DB (places.embedding_vector_v6)...")
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        for idx, db_id in enumerate(ids):
            emb_list = embeddings[idx].tolist()
            # float8[] format in postgres: ARRAY[...]
            conn.execute(
                text("UPDATE places SET embedding_vector_v6 = :emb WHERE id = :id"),
                {"emb": emb_list, "id": int(db_id)}
            )
            if (idx + 1) % 1000 == 0:
                print(f"    Updated {idx + 1}/{len(ids)} rows...")
                
    print("[DONE] Successfully updated PostgreSQL DB with BGE-M3 vectors!")

def run_performance_test():
    print("\n[4] Running Automatic Performance Test...")
    from sentence_transformers import SentenceTransformer
    import torch
    
    print("Loading BGE-M3 locally for test...")
    model = SentenceTransformer('BAAI/bge-m3', device='cpu')
    
    test_queries = [
        "조용하고 분위기 좋은 야경 명소",
        "비 오는 날 따뜻한 국물이 있는 맛집",
        "혼자서 집중하기 좋은 조용한 카페"
    ]
    
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        for query in test_queries:
            print(f"\n=== Query: {query} ===")
            query_emb = model.encode(query, normalize_embeddings=True).tolist()
            
            # Array dot product for Cosine Similarity
            # Note: We can write a custom function or just fetch all and dot product in numpy
            # For simplicity of test, let's just fetch all float8[] and do dot product in Python
            res = conn.execute(text("SELECT id, name, category, description, embedding_vector_v6 FROM places WHERE embedding_vector_v6 IS NOT NULL AND address LIKE '%%마포구%%'")).fetchall()
            
            scores = []
            for r in res:
                v = np.array(r[4])
                score = np.dot(query_emb, v)
                scores.append({
                    "name": r[1],
                    "category": r[2],
                    "desc": r[3] if r[3] else "None",
                    "score": score
                })
            
            top_5 = sorted(scores, key=lambda x: x['score'], reverse=True)[:5]
            for i, place in enumerate(top_5):
                print(f"{i+1}. {place['name']} ({place['category']}) - Score: {place['score']:.4f}")

def wait_for_npz():
    print(f"Monitoring '{DOWNLOADS_DIR}' for '{NPZ_FILENAME}'...")
    target_path = os.path.join(DOWNLOADS_DIR, NPZ_FILENAME)
    
    while True:
        if os.path.exists(target_path):
            # Wait a few seconds to ensure the file is fully downloaded
            time.sleep(5)
            # Copy to data/ folder
            local_path = os.path.join('data', NPZ_FILENAME)
            shutil.copy2(target_path, local_path)
            print(f"Copied to {local_path}")
            
            # Start the update process
            update_db(local_path)
            run_performance_test()
            break
        time.sleep(5)

if __name__ == "__main__":
    wait_for_npz()
