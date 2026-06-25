import os
import psycopg2
import requests
import time
from dotenv import load_dotenv

load_dotenv()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"

def search_naver_blog(query):
    url = "https://openapi.naver.com/v1/search/blog.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {"query": query, "display": 3}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.get(url, headers=headers, params=params)
            
            if res.status_code == 429:
                print("Rate limited! Sleeping for 2 seconds...")
                time.sleep(2)
                continue
                
            res.raise_for_status()
            data = res.json()
            
            contents = []
            for doc in data.get("items", []):
                text = doc.get("description", "").replace("<b>", "").replace("</b>", "").strip()
                if text:
                    contents.append(text)
                    
            return " ".join(contents)
            
        except Exception as e:
            print(f"Error fetching {query}: {e}")
            time.sleep(1)
            
    return None # Return None if it completely failed so we don't mark it as enriched

def main():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id, name, category, address 
        FROM places 
        WHERE is_premium = FALSE 
        AND is_enriched = FALSE 
        AND address LIKE '%마포구%'
    """)
    rows = cur.fetchall()
    
    if not rows:
        print("No more places to enrich in Mapo-gu!")
        return
        
    print(f"Found {len(rows)} places to enrich using Naver Blog Search API.")
    
    for count, row in enumerate(rows, 1):
        pid, name, category, address = row
        
        address_parts = address.split()
        dong = ""
        for part in address_parts:
            if part.endswith('동') or part.endswith('로') or part.endswith('길'):
                dong = part
                break
                
        search_query = f"{dong} {name} {category}".strip()
        
        # Don't print every single line to prevent log flooding. Print every 100.
        if count % 100 == 0:
            print(f"[{count}/{len(rows)}] Searching: {search_query}")
        
        blog_text = search_naver_blog(search_query)
        
        if blog_text is not None:
            if blog_text:
                cur.execute("""
                    UPDATE places 
                    SET description = COALESCE(description, '') || ' | [네이버 블로그 리뷰] ' || %s,
                        is_enriched = TRUE 
                    WHERE id = %s
                """, (blog_text, pid))
            else:
                # API call succeeded but no results
                cur.execute("UPDATE places SET is_enriched = TRUE WHERE id = %s", (pid,))
        else:
            # API call failed (e.g., rate limit, network issue). Skip updating so it retries next time.
            print(f"[{count}] Skipping {search_query} due to repeated API errors.")
            
        time.sleep(0.15)

if __name__ == "__main__":
    main()
