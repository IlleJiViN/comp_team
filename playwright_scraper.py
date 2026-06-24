import asyncio
import aiohttp
import psycopg2
import psycopg2.extras
import re
import sys
import time
from playwright.async_api import async_playwright

sys.stdout.reconfigure(encoding='utf-8')

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

async def get_kakao_place_id(session, query):
    url = f"https://search.map.kakao.com/mapsearch/map.daum"
    params = {"q": query, "msFlag": "A", "page": 1, "sort": 0}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://map.kakao.com/"
    }
    try:
        async with session.get(url, params=params, headers=headers, timeout=5) as res:
            if res.status == 200:
                text = await res.text()
                match = re.search(r'"confirmid":"(\d+)"', text)
                if match:
                    return match.group(1)
                else:
                    # Print when query returned no place ID (but HTTP status was 200)
                    pass
            else:
                print(f"Search API returned HTTP {res.status} for query '{query}'")
    except Exception as e:
        print(f"Search API request failed for query '{query}': {e}")
    return None

async def abort_unnecessary_requests(route):
    if route.request.resource_type in ["image", "media", "font"]:
        await route.abort()
    else:
        await route.continue_()

async def scrape_reviews(page, place_id):
    url = f"https://place.map.kakao.com/{place_id}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        try:
            await page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass
        reviews = await page.locator(".desc_review").all_text_contents()
        return " | [카카오 리뷰] ".join([r.strip() for r in reviews if r.strip()][:5])
    except Exception as e:
        print(f"Error in scrape_reviews for {place_id}: {e}")
        return ""

async def worker(queue, browser, session, db_queue):
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    page = await context.new_page()
    await page.route("**/*", abort_unnecessary_requests)
    
    while True:
        try:
            place = await queue.get()
            pid, name, address, dong = place
            
            query = f"{dong} {name}".strip() if dong else name
            kakao_id = await get_kakao_place_id(session, query)
            if not kakao_id:
                kakao_id = await get_kakao_place_id(session, name)
                
            combined_text = ""
            if kakao_id:
                import random
                await asyncio.sleep(random.uniform(0.5, 1.5))
                combined_text = await scrape_reviews(page, kakao_id)
                
            await db_queue.put((pid, combined_text))
        except Exception as e:
            print(f"Error in worker for pid {pid} ({name}): {e}")
            await db_queue.put((pid, ""))
        finally:
            queue.task_done()

async def db_writer(db_queue, total_places):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    processed = 0
    batch = []
    
    while True:
        pid, combined_text = await db_queue.get()
        batch.append((combined_text, pid))
        processed += 1
        
        if len(batch) >= 50 or processed == total_places:
            query = """
                UPDATE places AS p 
                SET 
                    description = CASE WHEN e.text != '' THEN COALESCE(p.description, '') || ' | [카카오 리뷰] ' || e.text ELSE p.description END, 
                    is_enriched = TRUE,
                    is_embedded = FALSE
                FROM (VALUES %s) AS e(text, id) 
                WHERE p.id = e.id
            """
            psycopg2.extras.execute_values(cur, query, batch)
            conn.commit()
            print(f"Processed {processed} / {total_places} places...")
            batch = []
            
        db_queue.task_done()
        if processed == total_places:
            break
            
    conn.close()

async def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, address 
        FROM places 
        WHERE (is_enriched = FALSE OR is_enriched IS NULL)
          AND address LIKE '서울%'
    """)
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        print("No more places to enrich!")
        return
        
    print(f"Loaded {len(rows)} places to scrape. Starting Playwright workers...")
    
    queue = asyncio.Queue()
    for r in rows:
        pid, name, address = r
        dong = ""
        if address:
            for part in address.split():
                if part.endswith('동') or part.endswith('로') or part.endswith('길'):
                    dong = part
                    break
        queue.put_nowait((pid, name, address, dong))
        
    db_queue = asyncio.Queue()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        async with aiohttp.ClientSession() as session:
            db_task = asyncio.create_task(db_writer(db_queue, len(rows)))
            
            num_workers = 40  # Reverted to 40 workers at user request
            workers = [asyncio.create_task(worker(queue, browser, session, db_queue)) for _ in range(num_workers)]
                
            await queue.join()
            await db_queue.join()
            
            for w in workers:
                w.cancel()
                
            await db_task
            
        await browser.close()
    print("Scraping complete.")

if __name__ == "__main__":
    start_t = time.time()
    asyncio.run(main())
    print(f"Time taken: {time.time() - start_t:.1f}s")
