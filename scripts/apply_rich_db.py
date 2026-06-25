import torch
import psycopg2
from psycopg2.extras import execute_batch
import time

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
PT_PATH = "rich_place_embeddings.pt"

def update_db():
    print(f"\n[1] Loading embeddings from {PT_PATH}...")
    # Map location cpu to avoid GPU memory issues if any
    data = torch.load(PT_PATH, map_location='cpu', weights_only=False)
    ids = data['ids']
    vectors = data['vectors'] # NumPy array
    
    # Adding dimension validation
    dimension = data.get('dimension', 1024)
    print(f"[INFO] Dimension: {dimension}, Total records: {len(ids)}")
    
    print(f"[2] Connecting to PostgreSQL and creating column if not exists...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()
    
    # Create pgvector extension just in case
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Add column
    col_name = "embedding_vector_bge_m3"
    cur.execute(f"ALTER TABLE places ADD COLUMN IF NOT EXISTS {col_name} vector({dimension});")
    print(f"[INFO] Column '{col_name}' ensures exists with dimension {dimension}.")

    conn.autocommit = False
    
    print(f"[3] Updating PostgreSQL DB (places.{col_name})...")
    
    # Prepare batch data
    # execute_batch is much faster than running single UPDATE queries
    update_query = f"UPDATE places SET {col_name} = %s WHERE id = %s"
    
    batch_data = []
    for idx, db_id in enumerate(ids):
        # Convert numpy array to list, then to pgvector string format
        emb_list = vectors[idx].tolist()
        vec_str = '[' + ','.join(map(str, emb_list)) + ']'
        batch_data.append((vec_str, int(db_id)))
    
    print(f"[INFO] Starting batch update of {len(batch_data)} rows...")
    start_time = time.time()
    
    batch_size = 5000
    for i in range(0, len(batch_data), batch_size):
        execute_batch(cur, update_query, batch_data[i:i+batch_size])
        conn.commit()
        print(f"    Updated {min(i+batch_size, len(batch_data))}/{len(batch_data)} rows...")
        
    cur.close()
    conn.close()
    
    elapsed = time.time() - start_time
    print(f"[DONE] Successfully updated PostgreSQL DB with BGE-M3 vectors in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    update_db()
