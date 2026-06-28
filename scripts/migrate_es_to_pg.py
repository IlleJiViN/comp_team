import psycopg2
import requests
import json
import numpy as np
from psycopg2.extras import execute_values

DB_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
ES_URL = "http://localhost:9200/spotsync_chunks/_search?scroll=2m"

print("Connecting to DB...")
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

print("Creating places_chunks table...")
cursor.execute("""
    DROP TABLE IF EXISTS places_chunks;
    CREATE TABLE places_chunks (
        id SERIAL PRIMARY KEY,
        place_id INT,
        chunk_id VARCHAR(255),
        text TEXT,
        embedding halfvec(512)
    );
""")
conn.commit()

print("Fetching chunks from Elasticsearch using Scroll API...")
query = {
    "size": 1000,
    "query": {"match_all": {}},
    "_source": ["place_id", "chunk_id", "text", "embedding"]
}

response = requests.post(ES_URL, json=query).json()
scroll_id = response['_scroll_id']
hits = response['hits']['hits']

total_migrated = 0
while len(hits) > 0:
    new_rows = []
    for hit in hits:
        source = hit['_source']
        pid = source.get('place_id')
        if pid is None:
            continue
        cid = source.get('chunk_id', '')
        text = source.get('text', '')
        emb = source.get('embedding')
        
        if emb:
            emb_np = np.array(emb)
            emb_512 = emb_np[:512]
            norm = np.linalg.norm(emb_512)
            if norm > 0:
                emb_512 = emb_512 / norm
            emb_512_str = f"[{','.join(map(str, emb_512))}]"
            new_rows.append((pid, cid, text, emb_512_str))

    if new_rows:
        execute_values(cursor, """
            INSERT INTO places_chunks (place_id, chunk_id, text, embedding)
            VALUES %s
        """, new_rows, page_size=1000)
        conn.commit()
    
    total_migrated += len(hits)
    print(f"Migrated {total_migrated} rows...")
    
    # Get next batch
    scroll_query = {
        "scroll": "2m",
        "scroll_id": scroll_id
    }
    response = requests.post("http://localhost:9200/_search/scroll", json=scroll_query).json()
    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']

print("Creating HNSW Index on places_chunks...")
cursor.execute("CREATE INDEX ON places_chunks USING hnsw (embedding halfvec_cosine_ops);")
# Create index for joins
cursor.execute("CREATE INDEX ON places_chunks (place_id);")
conn.commit()

cursor.close()
conn.close()
print("Migration from ES to PG V10 complete!")
