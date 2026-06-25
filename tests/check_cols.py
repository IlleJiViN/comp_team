# -*- coding: utf-8 -*-
import psycopg2
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='places' AND column_name LIKE 'embedding%'")
print(cur.fetchall())
