# -*- coding: utf-8 -*-
import psycopg2
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()
try:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    print("pgvector extension created successfully!")
    
    # Check if we can alter column
    print("Altering column to vector type...")
    cur.execute("ALTER TABLE places ALTER COLUMN embedding_vector_v4 TYPE vector(768) USING embedding_vector_v4::text::vector(768);")
    conn.commit()
    print("Column altered successfully!")
except Exception as e:
    print(f"Error: {e}")
