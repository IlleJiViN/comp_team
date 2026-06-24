import json
import torch
import psycopg2
from psycopg2.extras import execute_batch
import time

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
JSON_PATH = "places_data.json"
PT_PATH = "rich_place_embeddings.pt"

def init_postgres():
    print(f"[INFO] Connecting to PostgreSQL at {DATABASE_URL}...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()

    print("[INFO] Creating pgvector extension...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    print("[INFO] Creating places table...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS places (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            category VARCHAR(255),
            address TEXT,
            description TEXT,
            embedding_vector_bge_m3 vector(1024)
        );
    """)

    conn.autocommit = False

    print(f"[INFO] Loading {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"[INFO] Loading {PT_PATH}...")
    data = torch.load(PT_PATH, map_location='cpu', weights_only=False)
    ids = data['ids']
    vectors = data['vectors']

    if len(metadata) != len(ids):
        print("[WARN] Metadata and vectors length mismatch!")

    print("[INFO] Inserting data into PostgreSQL...")
    insert_query = """
        INSERT INTO places (id, name, category, address, description, embedding_vector_bge_m3)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            embedding_vector_bge_m3 = EXCLUDED.embedding_vector_bge_m3;
    """

    batch_data = []
    for i in range(len(metadata)):
        meta = metadata[i]
        place_id = int(ids[i])
        emb_list = vectors[i].tolist()
        vec_str = '[' + ','.join(map(str, emb_list)) + ']'
        
        batch_data.append((
            place_id,
            meta.get('name', ''),
            meta.get('category', ''),
            meta.get('address', ''),
            meta.get('description', ''),
            vec_str
        ))

    start_time = time.time()
    batch_size = 5000
    for i in range(0, len(batch_data), batch_size):
        execute_batch(cur, insert_query, batch_data[i:i+batch_size])
        conn.commit()
        print(f"    Inserted {min(i+batch_size, len(batch_data))}/{len(batch_data)} rows...")

    cur.close()
    conn.close()
    
    elapsed = time.time() - start_time
    print(f"[SUCCESS] Initialized PostgreSQL DB in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    init_postgres()
