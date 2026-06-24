import asyncio
import aiohttp
import psycopg2
import psycopg2.extras
import os
import json
import time
import itertools
from dotenv import load_dotenv

load_dotenv()

class APIKeyRotator:
    def __init__(self, env_key):
        keys_str = os.getenv(env_key, "")
        self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        self.key_iterator = itertools.cycle(self.keys) if self.keys else None
        self.current_key = next(self.key_iterator) if self.keys else None
        self.exhausted_keys = set()
        
    def get_key(self):
        return self.current_key
        
    def rotate(self):
        if not self.keys:
            return None
        self.exhausted_keys.add(self.current_key)
        if len(self.exhausted_keys) >= len(self.keys):
            self.current_key = None
            return None # All exhausted
        self.current_key = next(self.key_iterator)
        return self.current_key

naver_id_rotator = APIKeyRotator("NAVER_CLIENT_ID")
naver_secret_rotator = APIKeyRotator("NAVER_CLIENT_SECRET")
kakao_rotator = APIKeyRotator("KAKAO_API_KEY")

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
req_count = 0
start_time = time.time()

async def fetch_naver_blog(session, sem, query):
    global req_count
    while True:
        nid = naver_id_rotator.get_key()
        nsec = naver_secret_rotator.get_key()
        if not nid or not nsec:
            return "EXHAUSTED", []
            
        url = "https://openapi.naver.com/v1/search/blog.json"
        headers = {
            "X-Naver-Client-Id": nid,
            "X-Naver-Client-Secret": nsec
        }
        params = {"query": query, "display": 3}
        
        async with sem:
            req_count += 1
            try:
                async with session.get(url, headers=headers, params=params, timeout=5) as res:
                    if res.status == 429:
                        print(f"Naver Key {nid} TPS rate limited, sleeping 2 seconds...")
                        await asyncio.sleep(2.0)
                        return "RETRY", []
                    if res.status != 200:
                        return "ERROR", []
                        
                    data = await res.json()
                    contents = []
                    metadata_list = []
                    for doc in data.get("items", []):
                        text = doc.get("description", "").replace("<b>", "").replace("</b>", "").strip()
                        title = doc.get("title", "").replace("<b>", "").replace("</b>", "").strip()
                        if text:
                            contents.append(text)
                            metadata_list.append({
                                "source": "naver",
                                "title": title,
                                "url": doc.get("link", ""),
                                "postdate": doc.get("postdate", ""),
                                "bloggername": doc.get("bloggername", "")
                            })
                    return " ".join(contents), metadata_list
            except Exception:
                return "ERROR", []

async def fetch_kakao_blog(session, sem, query):
    global req_count
    while True:
        kkey = kakao_rotator.get_key()
        if not kkey:
            return "EXHAUSTED", []
            
        url = "https://dapi.kakao.com/v2/search/blog"
        headers = {"Authorization": f"KakaoAK {kkey}"}
        params = {"query": query, "size": 3}
        
        async with sem:
            req_count += 1
            try:
                async with session.get(url, headers=headers, params=params, timeout=5) as res:
                    if res.status == 429:
                        print(f"Kakao Key {kkey} TPS rate limited, sleeping 2 seconds...")
                        await asyncio.sleep(2.0)
                        return "RETRY", []
                    if res.status != 200:
                        return "ERROR", []
                        
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
                                "bloggername": doc.get("blogname", "")
                            })
                    return " ".join(contents), metadata_list
            except Exception:
                return "ERROR", []

async def process_place(session, sem, place):
    pid, name, address, dong = place
    query = f"{dong} {name}".strip()
    if not query:
        query = name
        
    # 1. Try Naver
    text, meta = await fetch_naver_blog(session, sem, query)
    if text == "EXHAUSTED" or not text:
        # 2. Try Kakao
        text, meta = await fetch_kakao_blog(session, sem, query)
        if text == "EXHAUSTED":
            return "ALL_EXHAUSTED"
            
    if not text:
        # Retry with just name if dong failed
        query = name
        text, meta = await fetch_naver_blog(session, sem, query)
        if text == "EXHAUSTED" or not text:
            text, meta = await fetch_kakao_blog(session, sem, query)
            if text == "EXHAUSTED":
                return "ALL_EXHAUSTED"
    if text == "RETRY":
        await asyncio.sleep(1.0)
        text, meta = await fetch_kakao_blog(session, sem, query)

    if text and text != "ERROR" and text != "EXHAUSTED" and text != "RETRY":
        return (pid, text, json.dumps(meta, ensure_ascii=False))
    return (pid, "", "[]")

async def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    cur.execute("SELECT id, name, address FROM places WHERE is_enriched = FALSE OR is_enriched IS NULL")
    rows = cur.fetchall()
    
    if not rows:
        print("No more places to enrich!")
        return
        
    print(f"Loaded {len(rows)} places to scrape. Starting Hybrid Async workers...")
    
    places_to_scrape = []
    for r in rows:
        pid, name, address = r
        dong = ""
        if address:
            for part in address.split():
                if part.endswith('동') or part.endswith('로') or part.endswith('길'):
                    dong = part
                    break
        places_to_scrape.append((pid, name, address, dong))
        
    sem = asyncio.Semaphore(15) # Safe TPS limit for combined API
    
    async with aiohttp.ClientSession() as session:
        batch_size = 500
        for i in range(0, len(places_to_scrape), batch_size):
            batch = places_to_scrape[i:i+batch_size]
            tasks = [process_place(session, sem, p) for p in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            update_data = []
            exhausted = False
            for res in results:
                if isinstance(res, tuple) and len(res) == 3:
                    pid, text, meta = res
                    update_data.append((text, meta, pid))
                elif res == "ALL_EXHAUSTED":
                    exhausted = True
            
            if update_data:
                query = """
                    UPDATE places AS p 
                    SET 
                        description = CASE WHEN e.text != '' THEN COALESCE(p.description, '') || ' | ' || e.text ELSE p.description END, 
                        blog_metadata = CASE WHEN e.meta != '[]' THEN e.meta::jsonb ELSE p.blog_metadata END, 
                        is_enriched = TRUE,
                        is_embedded = FALSE
                    FROM (VALUES %s) AS e(text, meta, id) 
                    WHERE p.id = e.id
                """
                psycopg2.extras.execute_values(cur, query, update_data)
                conn.commit()
            
            elapsed = time.time() - start_time
            print(f"Processed {i + len(batch)} / {len(places_to_scrape)} places... (Speed: {req_count / elapsed:.1f} calls/sec)")
            
            if exhausted:
                print("FATAL: ALL API KEYS EXHAUSTED. Exiting.")
                break
                
    conn.close()
    print("Batch complete.")

if __name__ == "__main__":
    asyncio.run(main())
