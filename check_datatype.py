# -*- coding: utf-8 -*-
import psycopg2
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()
cur.execute("SELECT data_type FROM information_schema.columns WHERE table_name='places' AND column_name='embedding_vector_v4'")
print(cur.fetchall())
