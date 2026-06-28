import psycopg2
import numpy as np
import json
from psycopg2.extras import execute_values

DB_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

print("Connecting to DB...")
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

print("Creating places_v10 table...")
cursor.execute("""
    DROP TABLE IF EXISTS places_v10;
    CREATE TABLE places_v10 (
        id SERIAL PRIMARY KEY,
        place_id VARCHAR(255),
        name VARCHAR(255) NOT NULL,
        category VARCHAR(255),
        address VARCHAR(255),
        description TEXT,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        blog_metadata JSONB,
        location GEOMETRY(Point, 4326),
        embedding halfvec(512)
    );
""")
conn.commit()

print("Fetching existing places from 'places' table...")
cursor.execute("SELECT id, place_id, name, category, address, description, latitude, longitude, blog_metadata, ST_X(location::geometry), ST_Y(location::geometry), embedding_vector_v6::text FROM places WHERE embedding_vector_v6 IS NOT NULL")
rows = cursor.fetchall()
print(f"Fetched {len(rows)} rows.")

print("Truncating embeddings to 512-dim and re-normalizing...")
new_rows = []
for r in rows:
    _id, pid, name, cat, addr, desc, lat, lng, blog, st_x, st_y, emb_str = r
    
    # Parse embedding string
    emb = np.array(json.loads(emb_str))
    
    # Truncate
    emb_512 = emb[:512]
    
    # Normalize
    norm = np.linalg.norm(emb_512)
    if norm > 0:
        emb_512 = emb_512 / norm
        
    emb_512_str = f"[{','.join(map(str, emb_512))}]"
    
    new_rows.append((_id, pid, name, cat, addr, desc, lat, lng, json.dumps(blog) if blog else None, f"SRID=4326;POINT({st_x} {st_y})", emb_512_str))

print("Inserting into places_v10...")
execute_values(cursor, """
    INSERT INTO places_v10 (id, place_id, name, category, address, description, latitude, longitude, blog_metadata, location, embedding)
    VALUES %s
""", new_rows, page_size=1000)
conn.commit()

print("Creating indexes on places_v10...")
cursor.execute("CREATE INDEX ON places_v10 USING gist (location);")
cursor.execute("CREATE INDEX ON places_v10 USING hnsw (embedding halfvec_cosine_ops);")
conn.commit()

print("Renaming tables...")
cursor.execute("ALTER TABLE places RENAME TO places_old;")
cursor.execute("ALTER TABLE places_v10 RENAME TO places;")
conn.commit()

cursor.close()
conn.close()
print("Migration complete!")
