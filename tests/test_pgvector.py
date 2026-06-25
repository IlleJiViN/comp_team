# -*- coding: utf-8 -*-
import psycopg2
import sys
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')
model = SentenceTransformer('jhgan/ko-sroberta-multitask', device='cpu')
query_vector = model.encode("조용하고 분위기 좋은 맥도날드", normalize_embeddings=True)
vec_str = '[' + ','.join(map(str, query_vector)) + ']'

conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()

sql = """
    SELECT name, category, address, 
           1 - (embedding_vector_v4 <=> %s::vector) AS similarity,
           ST_Distance(location::geography, ST_SetSRID(ST_Point(126.9371, 37.5562), 4326)::geography) AS dist
    FROM places
    WHERE ST_DWithin(location, ST_SetSRID(ST_Point(126.9371, 37.5562), 4326), 0.03) -- ~3km
    ORDER BY similarity DESC
    LIMIT 10
"""
cur.execute(sql, (vec_str,))
rows = cur.fetchall()
for r in rows:
    print(r)
