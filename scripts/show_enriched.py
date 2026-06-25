import psycopg2

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute("SELECT name, description FROM places WHERE is_enriched = TRUE AND description IS NOT NULL LIMIT 3")
for r in cur.fetchall():
    print(f"\n[{r[0]}]\n{str(r[1])[:300]}...")
