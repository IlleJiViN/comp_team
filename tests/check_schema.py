import psycopg2
conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'places';")
print(cur.fetchall())
cur.execute("SELECT id, blog_metadata FROM places WHERE blog_metadata IS NOT NULL AND blog_metadata::text != '[]' LIMIT 5;")
print(cur.fetchall())
