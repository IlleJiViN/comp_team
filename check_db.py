# -*- coding: utf-8 -*-
import psycopg2
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()
cur.execute("SELECT name, address, ST_Distance(location::geography, ST_SetSRID(ST_Point(126.9371, 37.5562), 4326)::geography) FROM places WHERE name LIKE '%맥도날드%' ORDER BY 3 ASC LIMIT 5")
print(cur.fetchall())
