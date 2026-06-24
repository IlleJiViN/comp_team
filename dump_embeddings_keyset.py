# -*- coding: utf-8 -*-
import psycopg2
import time
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

print("Connecting to DB...")
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM places WHERE embedding_vector_v4 IS NOT NULL")
total_rows = cur.fetchone()[0]
print(f"Total rows to fetch: {total_rows}")

batch_size = 50000
all_ids = []
all_vectors = []

start_time = time.time()
last_id = -1
rows_fetched = 0

while True:
    cur.execute(f"SELECT id, embedding_vector_v4 FROM places WHERE embedding_vector_v4 IS NOT NULL AND id > {last_id} ORDER BY id ASC LIMIT {batch_size}")
    rows = cur.fetchall()
    if not rows:
        break
    
    # Parse vectors
    batch_ids = np.array([r[0] for r in rows], dtype=np.int32)
    batch_vecs = np.array([r[1] for r in rows], dtype=np.float32)
    
    all_ids.append(batch_ids)
    all_vectors.append(batch_vecs)
    
    last_id = batch_ids[-1]
    rows_fetched += len(rows)
    elapsed = time.time() - start_time
    print(f"Fetched {rows_fetched}/{total_rows} ({rows_fetched/total_rows*100:.1f}%) in {elapsed:.1f}s")

print("Concatenating arrays...")
final_ids = np.concatenate(all_ids)
final_vectors = np.concatenate(all_vectors)

print("Saving to npz...")
np.savez_compressed('embeddings_v4.npz', ids=final_ids, vectors=final_vectors)

print(f"Done in {time.time() - start_time:.1f}s! Saved {len(final_ids)} vectors.")
