import os
import sys
import json
import time
import torch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, parallel_bulk

sys.stdout.reconfigure(encoding='utf-8')

# Configuration
ES_URL = "http://localhost:9200"
INDEX_NAME = "spotsync_rich_places"
JSON_PATH = "embedding_pipeline/places_data.json"
PT_PATH = "rich_place_embeddings.pt"

def main():
    print(f"[INFO] Connecting to Elasticsearch at {ES_URL}...")
    es = Elasticsearch(ES_URL, request_timeout=60)
    try:
        info = es.info()
        print(f"[INFO] Connected: {info['name']} (version {info['version']['number']})")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return

    # Delete existing index if present
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"[INFO] Deleted existing index '{INDEX_NAME}'")

    # Load Embeddings from .pt
    print(f"[INFO] Loading embeddings from {PT_PATH}...")
    data = torch.load(PT_PATH, map_location='cpu', weights_only=False)
    ids = data['ids']
    vectors = data['vectors']
    dimension = data.get('dimension', 1024)
    print(f"[INFO] Loaded {len(ids)} vectors of dimension {dimension}.")

    # Define ES Mapping
    mapping = {
        "mappings": {
            "properties": {
                "place_id": {"type": "integer"},
                "name": {"type": "text", "analyzer": "nori"},
                "category": {"type": "keyword"},
                "address": {"type": "text", "analyzer": "nori"},
                "description": {"type": "text", "analyzer": "nori"},
                "region": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dimension,
                    "index": True,
                    "similarity": "dot_product" 
                }
            }
        }
    }

    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"[SUCCESS] Created index '{INDEX_NAME}'.")

    print(f"[INFO] Loading JSON metadata from {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        places_metadata = json.load(f)
    print(f"[INFO] Loaded {len(places_metadata)} metadata rows.")

    if len(ids) != len(places_metadata):
        print(f"[WARN] Vectors count ({len(ids)}) != Metadata count ({len(places_metadata)})")
        
    # To quickly map place_id to metadata if order differs
    # But since they were generated sequentially, they probably match exactly.
    # To be safe, we'll zip them, assuming index-to-index match if ids are the same.
    
    # We will build a generator for parallel_bulk
    def generate_actions():
        for i in range(len(ids)):
            place_id = int(ids[i])
            meta = places_metadata[i]
            
            # Sanity check
            if meta['id'] != place_id:
                # Should not happen if they are aligned
                continue
                
            name = meta.get("name", "")
            category = meta.get("category", "")
            address = meta.get("address", "")
            description = meta.get("description", "")
            
            region = ""
            if "마포구" in address or "마포" in address: 
                region = "마포구"
            elif "홍대" in address or "서교동" in address or "동교동" in address or "연남동" in address or "합정동" in address: 
                region = "홍대"
                
            yield {
                "_index": INDEX_NAME,
                "_id": str(place_id),
                "_source": {
                    "place_id": place_id,
                    "name": name,
                    "category": category,
                    "address": address,
                    "description": description,
                    "region": region,
                    "embedding": vectors[i].tolist()
                }
            }

    print("[INFO] Parallel bulk indexing into Elasticsearch...")
    start_time = time.time()
    success = 0
    failed = []
    try:
        # thread_count=4, chunk_size=1000
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
            print(f"[WARN] Failed to index {len(failed)} documents.")
    except Exception as e:
        print(f"[ERROR] Indexing failed: {e}")

if __name__ == "__main__":
    main()
