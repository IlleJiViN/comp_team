# -*- coding: utf-8 -*-
import pandas as pd
import psycopg2
import sys

sys.stdout.reconfigure(encoding='utf-8')

conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
query = "SELECT id, CONCAT(name, ' ', category) AS text_val FROM places WHERE name IS NOT NULL"
df = pd.read_sql(query, conn)
df.to_parquet('places_for_embedding_v4.parquet')
print(f'Exported {len(df)} rows to places_for_embedding_v4.parquet')
