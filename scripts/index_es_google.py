import os
import sys
import json
import pandas as pd
import numpy as np
import psycopg2
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, parallel_bulk

# Configure stdout to use UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Connection and index configuration
ES_URL = "http://localhost:9200"
INDEX_NAME = "spotsync_chunks"
CSV_PATH = "data/all_chunked_for_bge.csv"
EMBEDDINGS_PATH = "google_embeddings.npz"
DB_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

def main():
    # Connect to ES
    es = Elasticsearch(ES_URL, request_timeout=60)
    try:
        info = es.info()
        print(f"[INFO] Connected to Elasticsearch: {info['name']} (version {info['version']['number']})")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Elasticsearch at {ES_URL}: {e}")
        return

    # Delete existing index
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"[INFO] Deleted existing index '{INDEX_NAME}'")

    # Define new 768-dimension mapping for Google text-embedding-004
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
                    "dims": 768,          # Google text-embedding-004 is 768 dimensions
                    "index": True,
                    "similarity": "dot_product" # Normalized vector dot product is equivalent to cosine similarity
                }
            }
        }
    }

    # Create new index
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"[SUCCESS] Created index '{INDEX_NAME}' with 768 dimensions.")

    # Load metadata CSV
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV file not found at {CSV_PATH}")
        return
    print(f"[INFO] Loading CSV data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    print(f"[INFO] Loaded {total_rows} rows.")

    # Load Google embeddings
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"[ERROR] Embeddings file not found at {EMBEDDINGS_PATH}. Please run generate_google_embeddings.py first.")
        return
    print(f"[INFO] Loading Google embeddings from {EMBEDDINGS_PATH}...")
    embeddings_data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    embeddings = embeddings_data['embeddings']
    print(f"[INFO] Successfully loaded embeddings array of shape {embeddings.shape}")

    if len(embeddings) != total_rows:
        print(f"[ERROR] Mismatch in row count! CSV has {total_rows} rows, but embeddings file has {len(embeddings)} vectors.")
        return

    # Load places metadata from PostgreSQL
    print("[INFO] Connecting to PostgreSQL database...")
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT id, name, category, address FROM places")
        places = {row[0]: {"name": row[1], "category": row[2], "address": row[3]} for row in cur.fetchall()}
        conn.close()
        print(f"[INFO] Loaded metadata for {len(places)} places from PostgreSQL.")
    except Exception as e:
        print(f"[WARN] PostgreSQL connection failed: {e}. Indexing will proceed with CSV-only metadata.")
        places = {}

    def generate_actions():
        for i, row in df.iterrows():
            place_id = int(row['id'])
            
            # Fetch metadata from database if available, else fallback to CSV values
            place_info = places.get(place_id, {})
            name = place_info.get("name") or str(row.get("name", ""))
            category = place_info.get("category") or str(row.get("category", ""))
            address = place_info.get("address") or ""
            
            region = ""
            if "마포구" in address or "마포" in address: 
                region = "마포구"
            elif "홍대" in address or "서교동" in address or "동교동" in address or "연남동" in address or "합정동" in address: 
                region = "홍대"
                
            yield {
                "_index": INDEX_NAME,
                "_id": f"{place_id}_{i}",
                "_source": {
                    "place_id": place_id,
                    "chunk_id": f"{place_id}_{i}",
                    "name": name,
                    "category": category,
                    "region": region,
                    "text": str(row['text_val']),
                    "embedding": embeddings[i].tolist()
                }
            }

    print("[INFO] Parallel bulk indexing documents into Elasticsearch...")
    start_time = time.time()
    success = 0
    failed = []
    try:
        for ok, item in parallel_bulk(es, generate_actions(), chunk_size=1000, thread_count=4, request_timeout=120, raise_on_error=False, raise_on_exception=False):
            if ok:
                success += 1
            else:
                failed.append(item)
            if success % 20000 == 0:
                print(f"[PROGRESS] Indexed {success} documents...")
        elapsed = time.time() - start_time
        print(f"[SUCCESS] Successfully indexed {success} documents in {elapsed:.2f}s.")
        if failed:
            print(f"[WARN] Failed to index {len(failed)} documents. First 5 failures: {failed[:5]}")
    except Exception as e:
        print(f"[ERROR] Indexing failed: {e}")

if __name__ == "__main__":
    import time
    main()
