import os
import sys
import time
import json
import psycopg2
import numpy as np
from elasticsearch import Elasticsearch
from google import genai
from google.genai import errors

# Configure stdout/stderr for UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Configurations
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
ES_URL = "http://localhost:9200"
INDEX_NAME = "spotsync_chunks"
PROJECT_ID = "spotsync-500217"
LOCATION = "us-central1"
MODEL_NAME = "text-embedding-004"

def get_google_client():
    return genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

def chunk_text(text_val: str, chunk_size=200):
    """Chunks long review texts into smaller pieces for optimal semantic embedding."""
    if not text_val:
        return []
    words = text_val.split()
    chunks = []
    current_chunk = []
    current_len = 0
    for w in words:
        current_chunk.append(w)
        current_len += len(w) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def embed_texts_with_retry(client, texts, max_retries=5):
    """Generates 768-dimensional embeddings using Vertex AI text-embedding-004 via modern google-genai SDK."""
    if not texts:
        return []
    
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=MODEL_NAME,
                contents=texts
            )
            return [emb.values for emb in response.embeddings]
        except errors.APIError as e:
            if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
                wait_time = (2 ** attempt) + np.random.uniform(0.5, 1.5)
                print(f"  - [WARN] Rate limited. Retrying in {wait_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  - [ERROR] Vertex AI API Error: {e}")
                time.sleep(1)
        except Exception as e:
            print(f"  - [ERROR] Unexpected embedding error: {e}. Retrying...")
            time.sleep(2)
            
    # Return zero vectors on failure
    print(f"  - [FATAL] Embedding generation failed for {len(texts)} texts. Falling back to zero-vectors.")
    return [np.zeros(768).tolist() for _ in texts]

def main():
    print("=" * 80)
    print("     SpotSync AI - On-Demand Batch Embedding & Elasticsearch Sync")
    print("=" * 80)

    # 1. Connect to PostgreSQL
    print("[1] Connecting to PostgreSQL database...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
    except Exception as e:
        print(f"[ERROR] Failed to connect to PostgreSQL: {e}")
        return

    # Find places that are enriched (have blog data) but not yet embedded
    cur.execute("""
        SELECT id, name, category, address, description
        FROM places 
        WHERE is_enriched = TRUE AND is_embedded = FALSE AND description IS NOT NULL AND description != '';
    """)
    rows = cur.fetchall()

    if not rows:
        print("🎉 No pending blogs to embed! All enriched places are fully synchronized.")
        conn.close()
        return

    print(f"[INFO] Found {len(rows)} places with scraped blogs pending embedding.")

    # 2. Initialize Clients
    try:
        google_client = get_google_client()
        print("[INFO] Google GenAI Client successfully authenticated.")
    except Exception as e:
        print(f"[ERROR] Google GenAI Authentication failed: {e}")
        conn.close()
        return

    es = Elasticsearch(ES_URL, request_timeout=30)
    if not es.ping():
        print("[ERROR] Elasticsearch is not reachable.")
        conn.close()
        return
    print("[INFO] Elasticsearch connection verified.")

    # 3. Process, Chunk, Embed, and Index
    print("\n[2] Processing places and generating embeddings...")
    
    total_chunks_indexed = 0
    success_place_ids = []

    for pid, name, category, address, desc in rows:
        # Extract reviews part if description contains " | [카카오 블로그] "
        reviews_text = desc
        if " | [카카오 블로그] " in desc:
            reviews_text = desc.split(" | [카카오 블로그] ")[-1]

        chunks = chunk_text(reviews_text)
        if not chunks:
            print(f"  - Place '{name}' (ID: {pid}): No valid chunks to embed. Marking as embedded.")
            success_place_ids.append(pid)
            continue

        print(f"  - Generating {len(chunks)} embeddings for place '{name}' (ID: {pid})...")
        embeddings = embed_texts_with_retry(google_client, chunks)

        # Bulk Index Chunks into Elasticsearch
        bulk_actions = []
        for idx, (chunk_text_val, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{pid}_sync_{int(time.time())}_{idx}"
            
            # Region extraction
            region = ""
            addr_str = address or ""
            if "마포구" in addr_str or "마포" in addr_str: 
                region = "마포구"
            elif "홍대" in addr_str or "서교동" in addr_str or "동교동" in addr_str or "연남동" in addr_str or "합정동" in addr_str: 
                region = "홍대"

            bulk_actions.append({
                "index": {
                    "_index": INDEX_NAME,
                    "_id": chunk_id
                }
            })
            bulk_actions.append({
                "place_id": pid,
                "chunk_id": chunk_id,
                "name": name,
                "category": category,
                "region": region,
                "text": chunk_text_val,
                "embedding": emb
            })

        if bulk_actions:
            try:
                res = es.bulk(operations=bulk_actions)
                if res.get('errors'):
                    print(f"    - [WARN] ES bulk indexing encountered some errors for place {pid}.")
                else:
                    total_chunks_indexed += len(chunks)
                    success_place_ids.append(pid)
                    print(f"    - [SUCCESS] Indexed {len(chunks)} chunks into Elasticsearch index '{INDEX_NAME}'.")
            except Exception as e:
                print(f"    - [ERROR] Failed to index chunks into ES for place {pid}: {e}")

    # 4. Update is_embedded flag in PostgreSQL
    if success_place_ids:
        print(f"\n[3] Updating is_embedded = TRUE in PostgreSQL for {len(success_place_ids)} places...")
        try:
            cur.execute("""
                UPDATE places 
                SET is_embedded = TRUE 
                WHERE id = ANY(%s);
            """, (success_place_ids,))
            conn.commit()
            print("  - [SUCCESS] PostgreSQL update completed successfully!")
        except Exception as e:
            print(f"  - [ERROR] Failed to update PostgreSQL is_embedded flags: {e}")
            conn.rollback()

    conn.close()
    print("\n" + "=" * 80)
    print(f"🎉 Batch execution completed! Total chunks indexed: {total_chunks_indexed}")
    print("=" * 80)

if __name__ == "__main__":
    main()
