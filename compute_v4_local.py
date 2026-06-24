# -*- coding: utf-8 -*-
import pandas as pd
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
import time
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

print("Connecting to DB...")
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()

# Ensure the column exists
cur.execute("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_vector_v4 vector(768);")
conn.commit()

print("Fetching unique texts...")
query = "SELECT DISTINCT name || ' ' || category FROM places WHERE name IS NOT NULL AND embedding_vector_v4 IS NULL"
cur.execute(query)
unique_texts = [r[0] for r in cur.fetchall() if r[0]]
print(f"Found {len(unique_texts)} unique texts to process.")

if not unique_texts:
    print("All done!")
    sys.exit(0)

print("Loading model...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask', device='cpu')

batch_size = 512
total_batches = (len(unique_texts) + batch_size - 1) // batch_size

print(f"Starting embedding generation in {total_batches} batches...")
start_time = time.time()

for i in range(0, len(unique_texts), batch_size):
    batch_texts = unique_texts[i:i+batch_size]
    
    # Generate embeddings
    embeddings = model.encode(batch_texts, normalize_embeddings=True, convert_to_numpy=True)
    
    # Store in a temporary table to join and update
    cur.execute("CREATE TEMP TABLE temp_embs (text_val text, emb vector(768)) ON COMMIT DROP;")
    
    # Prepare data for insertion
    insert_data = []
    for j, text_val in enumerate(batch_texts):
        # Format vector as string
        vec_str = '[' + ','.join(map(str, embeddings[j])) + ']'
        insert_data.append((text_val, vec_str))
        
    execute_values(cur, "INSERT INTO temp_embs (text_val, emb) VALUES %s", insert_data)
    
    # Update main table
    update_query = """
        UPDATE places
        SET embedding_vector_v4 = t.emb
        FROM temp_embs t
        WHERE places.name || ' ' || places.category = t.text_val
          AND places.embedding_vector_v4 IS NULL
    """
    cur.execute(update_query)
    conn.commit()
    
    elapsed = time.time() - start_time
    progress = (i + len(batch_texts)) / len(unique_texts) * 100
    print(f"Processed {i + len(batch_texts)}/{len(unique_texts)} ({progress:.1f}%) in {elapsed:.1f}s")

print(f"Finished completely in {time.time() - start_time:.1f}s")
