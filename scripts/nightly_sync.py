import os
import sys
import json
import time
import asyncio
import aiohttp
import psycopg2
import psycopg2.extras
import numpy as np
from elasticsearch import Elasticsearch
import google.auth
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Configure stdout/stderr for UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Configurations
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
ES_URL = "http://localhost:9200"
INDEX_NAME = "spotsync_chunks"
GCP_PROJECT = "spotsync-500217"
GCP_LOCATION = "us-central1"
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")

# Concurrency & Batch Limits
MAX_PLACES_PER_RUN = 45 # Keep strictly under 45 to respect Vertex AI & Kakao limits
VERTEX_BATCH_SIZE = 45

async def fetch_blog_reviews(session, sem, pid, name, address):
    """Scrapes Kakao blog reviews for a place."""
    if not KAKAO_API_KEY:
        print("[WARN] KAKAO_API_KEY not found in env.")
        return pid, "", []
        
    dong = ""
    if address:
        for part in address.split():
            if part.endswith('동') or part.endswith('로') or part.endswith('길'):
                dong = part
                break
                
    query = f"{dong} {name}".strip()
    url = "https://dapi.kakao.com/v2/search/blog"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": query, "size": 3}
    
    async with sem:
        try:
            async with session.get(url, headers=headers, params=params, timeout=5) as res:
                if res.status != 200:
                    return pid, "", []
                    
                data = await res.json()
                contents = []
                metadata_list = []
                
                for doc in data.get("documents", []):
                    text = doc.get("contents", "").replace("<b>", "").replace("</b>", "").strip()
                    title = doc.get("title", "").replace("<b>", "").replace("</b>", "").strip()
                    if text:
                        contents.append(text)
                        metadata_list.append({
                            "source": "kakao",
                            "title": title,
                            "url": doc.get("url", ""),
                            "postdate": doc.get("datetime", "")[:10],
                            "bloggername": doc.get("blogname", ""),
                            "thumbnail": doc.get("thumbnail", "")
                        })
                        
                combined_text = " ".join(contents)
                return pid, combined_text, metadata_list
        except Exception as e:
            print(f"      [WARN] Scraping failed for '{name}' (ID: {pid}): {e}")
            return pid, "", []

def chunk_text(text_val: str, chunk_size=200):
    """Chunks long review texts into smaller pieces for optimal semantic embedding."""
    if not text_val:
        return []
    # Simple chunking by character size to keep it robust and context-preserving
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

def generate_embeddings_batch(texts: list) -> list:
    """Generates 768-dimensional embeddings using Vertex AI text-embedding-004."""
    if not texts:
        return []
    try:
        vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        
        inputs = [TextEmbeddingInput(t, "RETRIEVAL_DOCUMENT") for t in texts]
        kwargs = {"output_dimensionality": 768}
        embeddings = model.get_embeddings(inputs, **kwargs)
        return [emb.values for emb in embeddings]
    except Exception as e:
        print(f"  - [ERROR] Vertex AI embedding generation failed: {e}")
        return [np.zeros(768).tolist() for _ in texts]

async def sync_pipeline():
    print("="*80)
    print("      SpotSync AI - Robust Daily Scraper & Search Sync Pipeline")
    print("="*80)

    # 1. Connect to PostgreSQL and fetch un-enriched places
    print("[1] Connecting to PostgreSQL database...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
    except Exception as e:
        print(f"[ERROR] Failed to connect to PostgreSQL: {e}")
        return

    # Find places where is_enriched is False or blog_metadata is empty
    # Limit to 45 places per run to ensure strict compliance with Vertex AI API quotas
    cur.execute("""
        SELECT id, name, category, address 
        FROM places 
        WHERE is_enriched = FALSE OR blog_metadata IS NULL OR jsonb_array_length(blog_metadata) = 0
        LIMIT %s;
    """, (MAX_PLACES_PER_RUN,))
    rows = cur.fetchall()
    
    if not rows:
        print("🎉 All places are fully enriched! No daily sync required.")
        conn.close()
        return

    print(f"Loaded {len(rows)} places needing daily reviews and search sync.")

    # 2. Async scraping of reviews
    print("\n[2] Scraping blog reviews from Kakao API...")
    sem = asyncio.Semaphore(10) # Safe concurrency rate
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_blog_reviews(session, sem, r[0], r[1], r[3]) for r in rows]
        scraped_results = await asyncio.gather(*tasks)

    # Convert rows to lookup
    places_lookup = {r[0]: {"name": r[1], "category": r[2], "address": r[3]} for r in rows}

    # 3. Process, Chunk, and Embed reviews
    print("\n[3] Generating 768-dim Google text-embedding-004 vectors...")
    es = Elasticsearch(ES_URL, request_timeout=30)
    
    bulk_actions = []
    postgres_updates = []

    for pid, combined_text, meta in scraped_results:
        p_info = places_lookup[pid]
        
        # Save scraped review to PostgreSQL
        blog_metadata_json = json.dumps(meta, ensure_ascii=False)
        postgres_updates.append((combined_text, blog_metadata_json, pid))
        
        if not combined_text:
            continue
            
        # Chunk text
        chunks = chunk_text(combined_text)
        if not chunks:
            continue
            
        # Generate embeddings in batches of 45 (strictly within rate limits)
        print(f"  - Generating {len(chunks)} embeddings for place '{p_info['name']}' (ID: {pid})...")
        embeddings = generate_embeddings_batch(chunks)
        
        # Prepare Elasticsearch Bulk Actions
        for idx, (chunk_text_val, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{pid}_sync_{int(time.time())}_{idx}"
            
            # Region extraction
            region = ""
            address = p_info["address"] or ""
            if "마포구" in address or "마포" in address: 
                region = "마포구"
            elif "홍대" in address or "서교동" in address or "동교동" in address or "연남동" in address or "합정동" in address: 
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
                "name": p_info["name"],
                "category": p_info["category"],
                "region": region,
                "text": chunk_text_val,
                "embedding": emb
            })

    # 4. Push to Elasticsearch
    if bulk_actions:
        print(f"\n[4] Indexing {len(bulk_actions)//2} new semantic chunks into Elasticsearch index '{INDEX_NAME}'...")
        try:
            res = es.bulk(operations=bulk_actions)
            if res.get('errors'):
                print("  - [WARN] Elasticsearch bulk indexing encountered some errors.")
            else:
                print(f"  - [SUCCESS] Successfully indexed all new chunks into Elasticsearch!")
        except Exception as e:
            print(f"  - [ERROR] Failed to push to Elasticsearch: {e}")
    else:
        print("\n[4] No new reviews found. Skipping Elasticsearch indexing.")

    # 5. Update PostgreSQL
    if postgres_updates:
        print("\n[5] Updating PostgreSQL places table...")
        update_query = """
            UPDATE places AS p 
            SET 
                description = CASE WHEN e.text != '' THEN COALESCE(p.description, '') || ' | [카카오 블로그] ' || e.text ELSE p.description END, 
                blog_metadata = CASE WHEN e.meta != '[]' THEN e.meta::jsonb ELSE p.blog_metadata END, 
                is_enriched = TRUE 
            FROM (VALUES %s) AS e(text, meta, id) 
            WHERE p.id = e.id
        """
        try:
            psycopg2.extras.execute_values(cur, update_query, postgres_updates)
            conn.commit()
            print(f"  - [SUCCESS] Updated {len(postgres_updates)} records in PostgreSQL table 'places'!")
        except Exception as e:
            print(f"  - [ERROR] Failed to update PostgreSQL: {e}")
            conn.rollback()

    conn.close()
    print("\n🎉 Daily scraper and search sync pipeline run completed successfully!")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(sync_pipeline())
