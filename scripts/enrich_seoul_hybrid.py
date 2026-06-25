import os
import psycopg2
import requests
import time
from dotenv import load_dotenv

load_dotenv()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
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
                return "NAVER_RATE_LIMIT", []
                
            res.raise_for_status()
            data = res.json()
            
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
                        "bloggername": doc.get("bloggername", ""),
                        "thumbnail": ""
                    })
                    
            return " ".join(contents), metadata_list
            
        except Exception as e:
            time.sleep(1)
            
    return None, []

def search_kakao_blog(query):
    url = "https://dapi.kakao.com/v2/search/blog"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": query, "size": 3}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.get(url, headers=headers, params=params)
            
            if res.status_code == 429:
                return "KAKAO_RATE_LIMIT", []
                
            res.raise_for_status()
            data = res.json()
            
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
                        "postdate": doc.get("datetime", "")[:10], # Keep YYYY-MM-DD
                        "bloggername": doc.get("blogname", ""),
                        "thumbnail": doc.get("thumbnail", "")
                    })
                    
            return " ".join(contents), metadata_list
            
        except Exception as e:
            time.sleep(1)
            
    return None, []

def main():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()
    
    # Query for all of Seoul, excluding Mapo-gu since it's already done
    cur.execute("""
        SELECT id, name, category, address 
        FROM places 
        WHERE is_premium = FALSE 
        AND is_enriched = FALSE 
        AND address LIKE '%서울%'
        AND address NOT LIKE '%마포구%'
    """)
    rows = cur.fetchall()
    
    if not rows:
        print("No more places to enrich in Seoul!")
        return
        
    print(f"Found {len(rows)} places to enrich in Seoul. Starting Hybrid strategy...")
    
    use_naver = bool(NAVER_CLIENT_ID and NAVER_CLIENT_SECRET)
    
    for count, row in enumerate(rows, 1):
        pid, name, category, address = row
        
        address_parts = address.split()
        dong = ""
        for part in address_parts:
            if part.endswith('동') or part.endswith('로') or part.endswith('길'):
                dong = part
                break
                
        search_query = f"{dong} {name} {category}".strip()
        
        if count % 500 == 0:
            print(f"[{count}/{len(rows)}] Searching: {search_query} (Using {'Naver' if use_naver else 'Kakao'})")
        
        blog_text = None
        blog_metadata = []
        source_name = ""
        
        if use_naver:
            blog_text, blog_metadata = search_naver_blog(search_query)
            if blog_text == "NAVER_RATE_LIMIT":
                print(f"[{count}] Naver rate limit hit! Switching to Kakao API...")
                use_naver = False
                blog_text, blog_metadata = search_kakao_blog(search_query)
                source_name = "[카카오 블로그 리뷰]"
            else:
                source_name = "[네이버 블로그 리뷰]"
        else:
            blog_text, blog_metadata = search_kakao_blog(search_query)
            if blog_text == "KAKAO_RATE_LIMIT":
                print(f"[{count}] BOTH APIs EXHAUSTED for today. Stopping.")
                break
            source_name = "[카카오 블로그 리뷰]"
            
        import json
        
        if blog_text not in [None, "NAVER_RATE_LIMIT", "KAKAO_RATE_LIMIT"]:
            if blog_text:
                cur.execute("""
                    UPDATE places 
                    SET description = COALESCE(description, '') || ' | ' || %s || ' ' || %s,
                        blog_metadata = COALESCE(blog_metadata, '[]'::jsonb) || %s::jsonb,
                        is_enriched = TRUE 
                    WHERE id = %s
                """, (source_name, blog_text, json.dumps(blog_metadata, ensure_ascii=False), pid))
            else:
                cur.execute("UPDATE places SET is_enriched = TRUE WHERE id = %s", (pid,))
        
        # Avoid getting rate limited prematurely
        time.sleep(0.15)

if __name__ == "__main__":
    main()
