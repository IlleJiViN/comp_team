import psycopg2

conn = psycopg2.connect('postgresql://postgres:postgres@localhost:5432/spotsync')
cur = conn.cursor()

# 1. Total places in Seoul
cur.execute("SELECT COUNT(*) FROM places WHERE address LIKE '서울%'")
total_seoul = cur.fetchone()[0]

# 2. Enriched places in Seoul
cur.execute("SELECT COUNT(*) FROM places WHERE address LIKE '서울%' AND is_enriched = TRUE")
enriched_seoul = cur.fetchone()[0]

not_enriched_seoul = total_seoul - enriched_seoul
reviews_needed = not_enriched_seoul * 3

print(f'Total Seoul places: {total_seoul}')
print(f'Enriched Seoul places: {enriched_seoul}')
print(f'Remaining Seoul places: {not_enriched_seoul}')
print(f'Reviews needed: {reviews_needed}')
