# -*- coding: utf-8 -*-
import psycopg2
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()
cur.execute("SELECT name, similarity(name, '조용하고 분위기 좋은 맥도날드') FROM places WHERE name LIKE '%맥도날드%' LIMIT 5")
print(cur.fetchall())
