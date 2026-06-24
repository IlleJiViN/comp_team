import numpy as np
import psycopg2
import psycopg2.extras

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
NPZ_PATH = "data/embeddings_v6.npz"

def fast_update():
    print(f"\n[1] Loading embeddings from {NPZ_PATH}...")
    data = np.load(NPZ_PATH)
    ids = data['ids']
    embeddings = data['embeddings']
    
    print(f"[2] Normalizing {len(ids)} embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1e-10, norms)
    
    print(f"[3] Updating PostgreSQL DB FAST...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Prepare data
    update_data = []
    for idx, db_id in enumerate(ids):
        emb_list = embeddings[idx].tolist()
        update_data.append((emb_list, int(db_id)))
        
    query = """
    UPDATE places AS p
    SET embedding_vector_v6 = v.emb::float8[]
    FROM (VALUES %s) AS v(emb, id)
    WHERE p.id = v.id
    """
    
    psycopg2.extras.execute_values(cursor, query, update_data, template="(%s, %s)", page_size=1000)
    
    print("[DONE] Successfully updated PostgreSQL DB with BGE-M3 vectors FAST!")
    
    run_performance_test()

def run_performance_test():
    print("\n[4] Running Automatic Performance Test for V6 (BGE-M3)...")
    from sentence_transformers import SentenceTransformer
    import torch
    from sqlalchemy import create_engine, text
    
    print("Loading BGE-M3 locally for test...")
    model = SentenceTransformer('BAAI/bge-m3', device='cpu')
    
    test_queries = [
        "조용하고 분위기 좋은 야경 명소",
        "비 오는 날 따뜻한 국물이 있는 맛집",
        "혼자서 집중하기 좋은 조용한 카페",
        "연인과 데이트하기 좋은 프리미엄 레스토랑"
    ]
    
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        for query in test_queries:
            print(f"\n======================================")
            print(f"🔍 Query: {query}")
            print(f"======================================")
            query_emb = model.encode(query, normalize_embeddings=True).tolist()
            
            res = conn.execute(text("SELECT id, name, category, description, embedding_vector_v6 FROM places WHERE embedding_vector_v6 IS NOT NULL AND address LIKE '%%마포구%%'")).fetchall()
            
            scores = []
            for r in res:
                v = np.array(r[4])
                score = np.dot(query_emb, v)
                scores.append({
                    "name": r[1],
                    "category": r[2],
                    "desc": r[3] if r[3] else "No description (Chunked metadata)",
                    "score": score
                })
            
            top_5 = sorted(scores, key=lambda x: x['score'], reverse=True)[:5]
            for i, place in enumerate(top_5):
                print(f"#{i+1} [{place['score']:.4f}] {place['name']} ({place['category']})")
                desc = place['desc']
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                print(f"   => {desc}")

if __name__ == "__main__":
    fast_update()
