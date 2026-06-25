# -*- coding: utf-8 -*-
import psycopg2
import time
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

print("Connecting...")
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()

print("Fetching 100k rows...")
start = time.time()
cur.execute("SELECT id, embedding_vector_v4 FROM places WHERE embedding_vector_v4 IS NOT NULL LIMIT 100000")
rows = cur.fetchall()
print(f"Fetched 100k rows in {time.time() - start:.2f} seconds")

start = time.time()
vectors = [np.array(r[1], dtype=np.float32) for r in rows]
print(f"Converted to numpy in {time.time() - start:.2f} seconds")
