import psycopg2
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/spotsync")
cur = conn.cursor()

# 1. 전체 places 수
cur.execute("SELECT COUNT(*) FROM places")
print(f"DB places count: {cur.fetchone()[0]}")

# 2. 카테고리 분포 (상위 20개)
cur.execute("SELECT category, COUNT(*) FROM places GROUP BY category ORDER BY COUNT(*) DESC LIMIT 20")
print("\n--- Category Distribution ---")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")

# 3. 주소 패턴 (동네 분포)
cur.execute("""SELECT SUBSTRING(address FROM '마포구 ([^ ]+)') as dong, COUNT(*) 
              FROM places WHERE address LIKE '%%마포구%%' 
              GROUP BY dong ORDER BY COUNT(*) DESC LIMIT 15""")
print("\n--- Dong Distribution ---")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]}")

# 4. 상호명 샘플
cur.execute("SELECT name, category, address FROM places ORDER BY RANDOM() LIMIT 10")
print("\n--- Sample Places ---")
for r in cur.fetchall():
    print(f"  {r[0]} | {r[1]} | {r[2]}")

# 5. 소상공인 CSV 확인
import os
csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]
print(f"\n--- CSV files in data/ ---")
for f in csv_files:
    size_mb = os.path.getsize(f"data/{f}") / 1024 / 1024
    print(f"  {f}: {size_mb:.1f} MB")

conn.close()
