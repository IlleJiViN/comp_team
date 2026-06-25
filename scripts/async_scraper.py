import asyncio
import aiohttp
import psycopg2
import psycopg2.extras
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

# Global rate limit counters
req_count = 0
start_time = time.time()

async def fetch_kakao_blog(session, sem, place):
    global req_count
    pid, name, address, dong = place
    
    query = f"{dong} {name}".strip()
    if not query:
        return (pid, "", "[]")
        
    url = "https://dapi.kakao.com/v2/search/blog"
    headers = {
        "Authorization": f"KakaoAK {KAKAO_API_KEY}",
        "Origin": "http://localhost:5173",
        "KA": "sdk/1.0 os/javascript lang/en-US device/Win32 origin/http%3A%2F%2Flocalhost%3A5173"
    }
    params = {"query": query, "size": 3}
    
    async with sem:
        try:
            req_count += 1
            async with session.get(url, headers=headers, params=params, timeout=5) as res:
                if res.status == 429:
                    return "RATE_LIMIT"
                if res.status != 200:
                    return (pid, "", "[]")
                    
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
                if not combined_text:
                    # Retry with just the name if Dong+Name failed
                    params["query"] = name
                    req_count += 1
                    async with session.get(url, headers=headers, params=params, timeout=5) as res2:
                        if res2.status == 200:
                            data2 = await res2.json()
                            for doc in data2.get("documents", []):
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
                            
                return (pid, combined_text, json.dumps(metadata_list, ensure_ascii=False))
        except Exception as e:
            return (pid, "", "[]")

async def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Query up to 50,000 places that are not yet enriched
    cur.execute("""
        SELECT id, name, address 
        FROM places 
        WHERE is_enriched = FALSE OR is_enriched IS NULL
        LIMIT 50000;
    """)
    rows = cur.fetchall()
    
    if not rows:
        print("No more places to enrich!")
        return
        
    print(f"Loaded {len(rows)} places to scrape. Starting Async workers...")
    
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
        
    sem = asyncio.Semaphore(50) # 50 concurrent requests
    
    async with aiohttp.ClientSession() as session:
        batch_size = 500
        for i in range(0, len(places_to_scrape), batch_size):
            batch = places_to_scrape[i:i+batch_size]
            tasks = [fetch_kakao_blog(session, sem, p) for p in batch]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            update_data = []
            rate_limit_hit = False
            for res in results:
                if isinstance(res, tuple) and len(res) == 3:
                    pid, text, meta = res
                    update_data.append((text, meta, pid))
                elif res == "RATE_LIMIT":
                    rate_limit_hit = True
            
            if update_data:
                # Fast batch update. If text is empty, we don't append to description.
                query = """
                    UPDATE places AS p 
                    SET 
                        description = CASE WHEN e.text != '' THEN COALESCE(p.description, '') || ' | [카카오 블로그] ' || e.text ELSE p.description END, 
                        blog_metadata = CASE WHEN e.meta != '[]' THEN e.meta::jsonb ELSE p.blog_metadata END, 
                        is_enriched = TRUE,
                        is_embedded = FALSE
                    FROM (VALUES %s) AS e(text, meta, id) 
                    WHERE p.id = e.id
                """
                psycopg2.extras.execute_values(cur, query, update_data)
                conn.commit()
            
            elapsed = time.time() - start_time
            print(f"Processed {i + len(batch)} / {len(places_to_scrape)} places... (Speed: {req_count / elapsed:.1f} API calls/sec)")
            
            if rate_limit_hit:
                print("FATAL: KAKAO API RATE LIMIT EXHAUSTED. Exiting.")
                break
                
    conn.close()
    print("Batch complete.")

if __name__ == "__main__":
    asyncio.run(main())
