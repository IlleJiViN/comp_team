import os
import json
import pandas as pd
import numpy as np
import psycopg2
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Connect to ES
es = Elasticsearch("http://localhost:9200", request_timeout=60)

print(es.info())

INDEX_NAME = "spotsync_chunks"

if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)
    print(f"Deleted existing index {INDEX_NAME}")

mapping = {
    "mappings": {
        "properties": {
            "place_id": {"type": "integer"},
            "chunk_id": {"type": "keyword"},
            "name": {"type": "text", "analyzer": "nori"},
            "category": {"type": "keyword"},
            "region": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "nori"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "dot_product"
            }
        }
    }
}

es.indices.create(index=INDEX_NAME, body=mapping)
print(f"Created index {INDEX_NAME}")

print("Loading data...")
df = pd.read_csv("data/all_chunked_for_bge.csv")

print("Combining embedding parts...")
part_files = [
    "embeddings_part_1.npz",
    "embeddings_part_2.npz",
    "embeddings_part_3.npz",
    "embeddings_part_4.npz",
    "embeddings_part_5.npz"
]

embeddings_list = []
for p in part_files:
    data = np.load(p, allow_pickle=True)
    embeddings_list.append(data['embeddings'])
    
embeddings = np.vstack(embeddings_list)
print(f"Total embeddings loaded: {embeddings.shape}")

print("Loading metadata from Postgres...")
conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/spotsync")
cur = conn.cursor()
cur.execute("SELECT id, name, category, address FROM places")
places = {row[0]: {"name": row[1], "category": row[2], "address": row[3]} for row in cur.fetchall()}

def generate_actions():
    for i, row in df.iterrows():
        place_id = int(row['id'])
        place_info = places.get(place_id, {"name": "", "category": "", "address": ""})
        
        addr = place_info['address'] or ""
        region = ""
        if "마포구" in addr: region = "마포구"
        elif "홍대" in addr: region = "홍대"
        
        yield {
            "_index": INDEX_NAME,
            "_id": f"{place_id}_{i}",
            "_source": {
                "place_id": place_id,
                "chunk_id": f"{place_id}_{i}",
                "name": place_info['name'],
                "category": place_info['category'],
                "region": region,
                "text": str(row['text_val']),
                "embedding": [float(x) for x in embeddings[i]]
            }
        }

print("Indexing documents into Elasticsearch...")
success, failed = bulk(es, generate_actions(), chunk_size=500, request_timeout=60)
print(f"Successfully indexed {success} documents.")
if failed:
    print(f"Failed to index {failed} documents.")
