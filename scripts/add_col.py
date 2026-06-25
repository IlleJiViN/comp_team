import psycopg2

try:
    conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/spotsync")
    cur = conn.cursor()
    cur.execute("ALTER TABLE places ADD COLUMN IF NOT EXISTS description TEXT;")
    conn.commit()
    print("Column added successfully.")
    cur.close()
    conn.close()
except Exception as e:
    print("Error:", e)
